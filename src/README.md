# Hailo Vehicle Counting App

Counts unique vehicles (COCO classes: car 2, motorcycle 3, bus 5, truck 7) in an
RTSP stream using YOLOv8m on a Hailo-10H, ByteTrack for persistent IDs, and
Postgres for periodic count persistence.

```
RTSPSource (20 FPS) -> HailoYoloDetector (yolov8m HEF, on-chip NMS)
                    -> VehicleTracker (ByteTrack, unique-ID count,
                                       direction of travel)
                    -> debug/<id>-<timestamp>-<direction>.jpg per new vehicle
                    -> CountRecorder -> Postgres
                         - vehicle_counts: (ts, total_count) every 5 min
                         - vehicle_events: (vehicle_id, ts, direction)
                           immediately per new vehicle
```

## Direction of travel

Each vehicle's direction is the dominant axis of its net centroid movement in
frame coordinates: `UP`, `DOWN`, `LEFT`, or `RIGHT` (y grows downward, so a
vehicle moving toward the bottom of the frame is `DOWN`). A new vehicle stays
pending until it has moved at least `min_displacement_px` (default 20 px);
if it never moves enough within `max_pending_frames` (default 60 ≈ 3 s at
20 FPS) it is recorded as `UNKNOWN`. When the direction resolves:

- a snapshot is saved as `debug/<vehicle_id>-<YYYYmmdd_HHMMSS>-<direction>.jpg`
- a `(vehicle_id, ts, direction, angle)` row is inserted into `vehicle_events`
  (buffered and retried if Postgres is down)

The `angle` column stores the exact angle of travel in degrees on the same
conceptual plane: 0° = UP, 90° = RIGHT, 180° = DOWN, 270° = LEFT (clockwise).
It is NULL when the direction is UNKNOWN. The angle is for debugging only
and is not part of the snapshot filename.

## Double-count / false-positive protection

- **Cross-class NMS** (`detector.class_agnostic_nms`): the HEF's on-chip NMS
  runs per class, so one vehicle detected as both e.g. "car" and "truck"
  yields two overlapping boxes; the lower-scoring one is dropped.
- **Track confirmation** (`minimum_consecutive_frames=3`): a detection must
  persist 3 consecutive frames before it is confirmed and counted, so
  single-frame flickers and ghost boxes never become counts.
- **Lost-track persistence** (`lost_track_buffer=90` → ~3 s at 20 FPS): a
  vehicle that flaps out of detection briefly re-matches its old id instead
  of getting a new one. Per-track state is a few hundred bytes and is purged
  after the buffer expires, so long runs stay lightweight.
- **Duplicate-id guard** (`duplicate_iou_threshold=0.55`): if a brand-new
  track's box overlaps an already-counted visible track, it is flagged as a
  duplicate id for the same vehicle and excluded from the count.
- **Score threshold raised to 0.45** (from 0.3) to suppress low-confidence
  ghost boxes (`--score-threshold` to tune).

## Modules

| File | Class | Role |
|---|---|---|
| `ingest.py` | `RTSPSource` | RTSP capture, re-timed to exactly 20 FPS (drop/duplicate), bounded queue, raises `SourceLost` |
| `detector.py` | `HailoYoloDetector` | HEF on shared VDevice via `HailoInfer`, letterbox preprocess, NMS decode, class filter {2,3,5,7} |
| `tracker.py` | `VehicleTracker` | `supervision.ByteTrack` wrapper; `seen_ids` set, `total_count`, per-track motion history, emits `VehicleEvent(vehicle_id, direction, ts)` per new vehicle |
| `persistence.py` | `CountRecorder` | APScheduler background job inserting `(ts, total_count)`; `record_vehicle()` inserts per-vehicle `(vehicle_id, ts, direction)` rows immediately; survives DB outages |
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
