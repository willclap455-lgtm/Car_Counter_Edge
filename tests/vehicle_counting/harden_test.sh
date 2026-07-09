#!/bin/bash
# Phase G hardening tests (run manually; requires local mediamtx RTSP server
# publishing rtsp://localhost:8554/test and local Postgres from Phase A).
#
#   Test 1: RTSP outage mid-run -> app survives and resumes counting.
#   Test 2: kill -TERM -> final total_count row flushed to Postgres.
#
# Usage: bash tests/vehicle_counting/harden_test.sh
set -u

cd "$(dirname "$0")/../.."
PY=./venv_hailo_apps/bin/python
DSN="postgresql://vehicle_app:vehicle_pass@localhost:5432/vehicle_counts"
RTSP="rtsp://localhost:8554/test"
LOG=/tmp/harden.log
FAIL=0

restart_publisher() {
    tmux -f /exec-daemon/tmux.portal.conf send-keys -t "rtsp-publisher:0.0" \
        "ffmpeg -re -stream_loop -1 -i /tmp/test_traffic.mp4 -c copy -f rtsp -rtsp_transport tcp rtsp://localhost:8554/test 2>/tmp/publisher.log" C-m
}

echo "=== Test 1: RTSP outage recovery ==="
sudo -u postgres psql -d vehicle_counts -qc "DELETE FROM vehicle_counts;"
$PY -m src.main --rtsp "$RTSP" --dsn "$DSN" --mock-detector \
    --interval-minutes 0.25 --duration 150 --debug-dir /tmp/debug_harden \
    > "$LOG" 2>&1 &
APP_PID=$!

sleep 30
echo "[harden] killing RTSP publisher for 30s..."
pkill -f stream_loop
sleep 30
echo "[harden] restoring RTSP publisher..."
restart_publisher

wait "$APP_PID"
APP_EXIT=$?
RECONNECTS=$(grep -c "reconnect attempt" "$LOG" || true)
FPS_AFTER=$(tail -20 "$LOG" | grep -c "pipeline FPS" || true)
echo "app exit: $APP_EXIT, reconnect attempts logged: $RECONNECTS, FPS lines near end: $FPS_AFTER"
if [[ "$APP_EXIT" -eq 0 && "$RECONNECTS" -ge 1 && "$FPS_AFTER" -ge 1 ]]; then
    echo "Test 1 PASS: app survived outage and resumed"
else
    echo "Test 1 FAIL"
    FAIL=1
fi

echo
echo "=== Test 2: SIGTERM flushes final count ==="
sudo -u postgres psql -d vehicle_counts -qc "DELETE FROM vehicle_counts;"
$PY -m src.main --rtsp "$RTSP" --dsn "$DSN" --mock-detector \
    --interval-minutes 60 --debug-dir /tmp/debug_harden \
    > "$LOG" 2>&1 &
APP_PID=$!
sleep 25   # long enough for at least one mock vehicle, no periodic tick (60m)
kill -TERM "$APP_PID"
wait "$APP_PID"
APP_EXIT=$?

ROWS=$(sudo -u postgres psql -d vehicle_counts -tAc "SELECT count(*) FROM vehicle_counts;")
LAST=$(sudo -u postgres psql -d vehicle_counts -tAc "SELECT total_count FROM vehicle_counts ORDER BY ts DESC LIMIT 1;")
echo "app exit: $APP_EXIT, rows: $ROWS, last total_count: $LAST"
if [[ "$APP_EXIT" -eq 0 && "$ROWS" -ge 1 && "$LAST" -ge 1 ]]; then
    echo "Test 2 PASS: SIGTERM triggered final DB write"
else
    echo "Test 2 FAIL"
    FAIL=1
fi

exit "$FAIL"
