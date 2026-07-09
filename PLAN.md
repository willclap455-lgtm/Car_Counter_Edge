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
Current Phase: Phase A (ENV)
Last Action: Project initialized.
Blockers: None
🏁 Definition of Done (DoD)
Environment has HailoRT SDK + HEF compiled for yolov8m.
Reads RTSP stream and maintains ≥20 FPS ingestion.
Inference runs on Hailo-8 (not CPU).
Detects COCO classes 2 (car), 3 (motorcycle), 5 (bus), 7 (truck).
ByteTrack assigns persistent unique IDs across frames.
Maintains running total_count = number of unique IDs ever seen.
Every 5 min, row (timestamp, total_count) inserted into Postgres.
App survives a 10-minute continuous run without crash.

🔄 The Phases
Phase A — ENV
Plan: Verify Hailo hardware (hailortcli scan). Create venv. Install deps (hailo-apps-infra, opencv-python, supervision, psycopg[binary], apscheduler). Download yolov8m.hef. Stand up Postgres via docker-compose. Create schema.sql.
Build: requirements.txt, docker-compose.yml, schema.sql, .env.example.
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
