"""Phase E tests: CountRecorder against a live local Postgres.

Requires the Phase A database (vehicle_counts DB, vehicle_app user) and a
running Postgres on localhost. Tests use a 5-second interval per the plan.

The outage test restarts the postgres service via sudo; it is skipped when
passwordless sudo is unavailable.
"""

import subprocess
import time

import psycopg
import pytest

from src.persistence import CountRecorder

DSN = "postgresql://vehicle_app:vehicle_pass@localhost:5432/vehicle_counts"
INTERVAL_MINUTES_5S = 5.0 / 60.0


def _postgres_available() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=3):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _postgres_available(), reason="local Postgres not available")


@pytest.fixture()
def clean_table():
    with psycopg.connect(DSN) as conn:
        conn.execute("DELETE FROM vehicle_counts")
        conn.execute("DELETE FROM vehicle_events")
    yield
    with psycopg.connect(DSN) as conn:
        conn.execute("DELETE FROM vehicle_counts")
        conn.execute("DELETE FROM vehicle_events")


def _row_count() -> int:
    with psycopg.connect(DSN) as conn:
        return conn.execute("SELECT count(*) FROM vehicle_counts").fetchone()[0]


def test_writes_at_least_3_rows_in_20s(clean_table):
    counter = {"n": 0}

    def get_count() -> int:
        counter["n"] += 1
        return counter["n"]

    with CountRecorder(DSN, get_count, interval_minutes=INTERVAL_MINUTES_5S):
        time.sleep(20)
    # stop() flushes one final row, so >= 3 periodic + 1 flush.
    rows = _row_count()
    assert rows >= 3, f"expected >= 3 rows after 20s at 5s interval, got {rows}"

    with psycopg.connect(DSN) as conn:
        data = conn.execute("SELECT ts, total_count FROM vehicle_counts ORDER BY ts").fetchall()
    counts = [c for _, c in data]
    assert counts == sorted(counts), "total_count must be monotonically non-decreasing"


def _sudo_available() -> bool:
    return subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _sudo_available(), reason="needs passwordless sudo to restart postgres")
def test_survives_postgres_outage_and_resumes(clean_table):
    recorder = CountRecorder(DSN, lambda: 42, interval_minutes=INTERVAL_MINUTES_5S)
    recorder.start()
    try:
        time.sleep(7)  # at least one successful insert
        rows_before = recorder.rows_written
        assert rows_before >= 1

        subprocess.run(["sudo", "service", "postgresql", "stop"], check=True, capture_output=True)
        time.sleep(12)  # a couple of ticks fail while postgres is down
        assert recorder.failures >= 1, "expected failed inserts during outage"

        subprocess.run(["sudo", "service", "postgresql", "start"], check=True, capture_output=True)
        deadline = time.monotonic() + 30
        while recorder.rows_written <= rows_before and time.monotonic() < deadline:
            time.sleep(1)
        assert recorder.rows_written > rows_before, "recorder did not resume after outage"
    finally:
        subprocess.run(["sudo", "service", "postgresql", "start"], capture_output=True)
        time.sleep(2)
        recorder.stop()


def test_record_vehicle_inserts_event_row(clean_table):
    recorder = CountRecorder(DSN, lambda: 0, interval_minutes=60.0)
    recorder.start()
    try:
        ts = time.time()
        assert recorder.record_vehicle(17, "LEFT", ts=ts, angle=270.0) is True
        assert recorder.record_vehicle(18, "UP", angle=2.5) is True
        assert recorder.record_vehicle(19, "UNKNOWN") is True  # no angle -> NULL
        with psycopg.connect(DSN) as conn:
            rows = conn.execute(
                "SELECT vehicle_id, direction, angle, extract(epoch from ts) "
                "FROM vehicle_events ORDER BY vehicle_id"
            ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(17, "LEFT"), (18, "UP"), (19, "UNKNOWN")]
        assert rows[0][2] == pytest.approx(270.0)
        assert rows[1][2] == pytest.approx(2.5)
        assert rows[2][2] is None
        assert abs(float(rows[0][3]) - ts) < 1.0, "stored ts should match event ts"
    finally:
        recorder.stop(flush=False)


@pytest.mark.skipif(not _sudo_available(), reason="needs passwordless sudo to restart postgres")
def test_record_vehicle_buffers_during_outage_and_drains(clean_table):
    recorder = CountRecorder(DSN, lambda: 0, interval_minutes=INTERVAL_MINUTES_5S)
    recorder.start()
    try:
        subprocess.run(["sudo", "service", "postgresql", "stop"], check=True, capture_output=True)
        time.sleep(1)
        assert recorder.record_vehicle(99, "RIGHT") is False, "insert should fail during outage"
        assert len(recorder._event_buffer) == 1

        subprocess.run(["sudo", "service", "postgresql", "start"], check=True, capture_output=True)
        deadline = time.monotonic() + 30
        while recorder.events_written < 1 and time.monotonic() < deadline:
            time.sleep(1)  # scheduler tick drains the buffer
        assert recorder.events_written == 1, "buffered event was not drained after outage"
        with psycopg.connect(DSN) as conn:
            row = conn.execute("SELECT vehicle_id, direction FROM vehicle_events").fetchone()
        assert row == (99, "RIGHT")
    finally:
        subprocess.run(["sudo", "service", "postgresql", "start"], capture_output=True)
        time.sleep(2)
        recorder.stop(flush=False)


def test_flush_writes_immediately(clean_table):
    recorder = CountRecorder(DSN, lambda: 7, interval_minutes=60.0)  # tick far in the future
    recorder.start()
    try:
        assert recorder.flush() is True
        assert _row_count() == 1
        with psycopg.connect(DSN) as conn:
            val = conn.execute("SELECT total_count FROM vehicle_counts").fetchone()[0]
        assert val == 7
    finally:
        recorder.stop(flush=False)
