# Hailo Vehicle Counting App

Counts unique vehicles (COCO classes: car 2, motorcycle 3, bus 5, truck 7) in an
RTSP stream using YOLOv8m on a Hailo-10H, ByteTrack for persistent IDs, and
Postgres for periodic count persistence.

```
RTSPSource (20 FPS) -> HailoYoloDetector (yolov8m HEF, on-chip NMS)
                    -> VehicleTracker (ByteTrack, unique-ID count)
                    -> debug/*.jpg snapshot per new vehicle
                    -> CountRecorder -> Postgres (every 5 min)
```

## Modules

| File | Class | Role |
|---|---|---|
| `ingest.py` | `RTSPSource` | RTSP capture, re-timed to exactly 20 FPS (drop/duplicate), bounded queue, raises `SourceLost` |
| `detector.py` | `HailoYoloDetector` | HEF on shared VDevice via `HailoInfer`, letterbox preprocess, NMS decode, class filter {2,3,5,7} |
| `tracker.py` | `VehicleTracker` | `supervision.ByteTrack` wrapper; `seen_ids` set, `total_count`, reports new IDs |
| `persistence.py` | `CountRecorder` | APScheduler background job inserting `(ts, total_count)` via psycopg pool; survives DB outages |
| `main.py` | `VehicleCounterApp` | Composition + hardening: RTSP auto-reconnect (backoff), detector reset on inference failure, SIGTERM/SIGINT flush |

## Setup (once per host)

```bash
# 1. Python env (HailoRT SDK must be installed; pyhailort wheel built from
#    https://github.com/hailo-ai/hailort v5.3.0 if not provided by the SDK)
python3 -m venv venv_hailo_apps
./venv_hailo_apps/bin/pip install -r requirements.txt -e .

# 2. Postgres (direct install)
sudo apt-get install postgresql
sudo -u postgres psql -c "CREATE USER vehicle_app WITH PASSWORD 'vehicle_pass';"
sudo -u postgres psql -c "CREATE DATABASE vehicle_counts OWNER vehicle_app;"
sudo -u postgres psql -d vehicle_counts -f schema.sql

# 3. Model (already present in the model zoo cache on the target device)
ls /usr/local/hailo/resources/models/hailo10h/yolov8m.hef
```

Copy `.env.example` to `.env` and adjust values.

## Run

```bash
./venv_hailo_apps/bin/python -m src.main \
    --rtsp 'rtsp://admin:PASSWORD@192.168.105.120:554/cam/realmonitor?channel=1&subtype=1' \
    --hef /usr/local/hailo/resources/models/hailo10h/yolov8m.hef \
    --dsn postgresql://vehicle_app:vehicle_pass@localhost:5432/vehicle_counts
```

Useful flags: `--interval-minutes` (default 5), `--debug-dir` (default `debug/`),
`--score-threshold` (default 0.3), `--duration N` (stop after N seconds, for tests),
`--mock-detector` (simulated detections for hosts without a Hailo device).

## Tests

```bash
# Unit/integration tests (no Hailo device needed; Postgres tests need local DB)
./venv_hailo_apps/bin/python -m pytest tests/vehicle_counting/ -v

# Phase B ingest FPS test (needs an RTSP stream)
./venv_hailo_apps/bin/python -m src.ingest --rtsp <uri> --duration 60

# Phase C on-device test (needs Hailo-10H; verify with hailortcli monitor)
./venv_hailo_apps/bin/python -m src.detector --hef <yolov8m.hef> --image <vehicles.jpg>

# Phase G hardening tests (needs local mediamtx test stream, see script header)
bash tests/vehicle_counting/harden_test.sh
```

## On-target validation checklist (requires Hailo-10H + camera LAN)

1. `hailortcli scan` shows the device.
2. `python -m src.detector --hef ... --image ...` -> detections non-empty, median latency <= 50 ms, `hailortcli monitor` shows activity.
3. `python -m src.ingest --rtsp <camera uri> --duration 60` -> min FPS >= 20.
4. 10-minute run of `src.main` without `--mock-detector`.
