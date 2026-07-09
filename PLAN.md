Hailo Vehicle Counting App - Agent Dev Plan
🤖 AGENT INSTRUCTIONS
You are an autonomous AI coding agent. Your goal is to complete the project according to the Definition of Done (DoD) below.Execute in an iterative loop:

Read the "Current State" below.
Plan the immediate next steps for the current phase.
Write/edit the necessary code.
Run the build/tests.
If tests pass, update "Current State" to the next phase.
If tests fail, read the error, fix the code, and test again.
Do not stop until all DoD criteria are met.

📍 Current State
Current Phase: Phase B (INGEST)
Last Action: Phase A complete. Built HailoRT 5.3.0 from source (libhailort + hailortcli + pyhailort wheel), created venv_hailo_apps, installed deps (requirements.txt). Postgres 16 installed directly (not docker), DB vehicle_counts + schema.sql applied, user vehicle_app created. yolov8m.hef (hailo10h, MZ v5.3.0) downloaded to /usr/local/hailo/resources/models/hailo10h/ and parses OK (640x640 in, NMS out). Import test `python -c "import hailo_platform, supervision, psycopg2"` PASSED. pg_isready healthy.
Blockers: No physical Hailo device in this dev VM (`hailortcli scan` -> none) — on-device inference tests (Phase C latency/monitor) must run on the target Hailo-10H host. RTSP camera 192.168.105.120:554 accepts TCP connects but does not answer RTSP handshakes from this VM (likely LAN-only ACL) — Phase B live-stream test needs the target network.
🏁 Definition of Done (DoD)
Environment has HailoRT SDK + HEF compiled for yolov8m.
Reads RTSP stream and maintains ≥20 FPS ingestion.
(you can use http://admin:clancy252629@192.168.105.120:554/cam/realmonitor?channel=1&subtype=1 to test this all)
Inference runs on Hailo 10H (not CPU).
Detects COCO classes 2 (car), 3 (motorcycle), 5 (bus), 7 (truck).
ByteTrack assigns persistent unique IDs across frames.
Maintains running total_count = number of unique IDs ever seen.
Saves to .jpg the video frame every time a new car is detected (preferably in a /debug directory)
Every 5 min, row (timestamp, total_count) inserted into Postgres.
App survives a 10-minute continuous run without crash.

🔄 The Phases
Phase A — ENV
Plan: Verify Hailo hardware (hailortcli scan). Create venv. Install deps (hailo-apps-infra, opencv-python, supervision, psycopg[binary], apscheduler). yolov8m is already download in the model zoo. Stand up Postgres via direct install. Create schema.sql.
Build: requirements.txt, schema.sql, .env.example.
Test: python -c "import hailo_platform, supervision, psycopg2" succeeds. docker compose ps shows postgres healthy.
Phase B — INGEST
Plan: Create src/ingest.py with RTSPSource(uri, target_fps=20). Use OpenCV backed by GStreamer. Drop/duplicate frames to maintain exactly 20 FPS. Use a bounded queue.
Build: src/ingest.py
Test: Print measured ingest FPS every second for 60s; min ≥ 20. Killing source raises SourceLost exception.
Phase C — INFER (Hailo yolov8m)
Plan: Load yolov8m.hef onto VDevice. Use Hailo's InferencePipeline. Filter detections to classes {2,3,5,7}.
Build: src/detector.py exposing HailoYoloDetector(hef_path, classes={2,3,5,7}).
Test: Non-empty detections on test frames. hailortcli monitor shows activity. Latency ≤ 50ms.
Phase D — TRACK + COUNT
Plan: Wrap supervision.ByteTrack. Maintain seen_ids: set[int]. total_count = len(seen_ids).
Build: src/tracker.py exposing VehicleTracker.
Test: Drive a short test clip. Assert total_count == 3 at end. Assert ID remains constant for a single car crossing frames.
Phase E — PERSIST
Plan: Background scheduler thread (APScheduler IntervalTrigger(minutes=5)). Write to Postgres vehicle_counts table. Use psycopg.ConnectionPool.
Build: src/persistence.py exposing CountRecorder.
Test: Lower interval to 5s for unit test. Expect ≥ 3 rows after 20s. Kill Postgres mid-run; pipeline survives and resumes.
Phase F — INTEGRATE
Plan: Compose components: RTSPSource -> Queue -> Detector -> Tracker -> (Logger + CountRecorder).
Build: src/main.py accepting --rtsp, --hef, --dsn.
Test: 10-minute soak test. FPS ≥ 20. total_count increments correctly. DB shows rows ~5 min apart.
Phase G — HARDEN
Plan: Add RTSP auto-reconnect. Hailo device reset on exception. SIGTERM handler flushes final count.
Test: Unplug network 30s mid-run; app recovers. kill -TERM triggers final DB write.
