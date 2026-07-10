"""Persistence to Postgres (Phase E + per-vehicle events).

CountRecorder runs a background APScheduler job that inserts
``(timestamp, total_count)`` into the ``vehicle_counts`` table every
``interval_minutes`` (default 5). It also writes one row per newly detected
vehicle — ``(vehicle_id, timestamp, direction)`` — into ``vehicle_events``
via record_vehicle(), immediately when the event occurs.

Connections come from a psycopg_pool.ConnectionPool; a Postgres outage never
propagates into the pipeline — failed count inserts are retried on the next
tick, and failed vehicle-event inserts are buffered and drained on ticks.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from psycopg_pool import ConnectionPool

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

INSERT_SQL = "INSERT INTO vehicle_counts (ts, total_count) VALUES (now(), %s)"
INSERT_EVENT_SQL = (
    "INSERT INTO vehicle_events (vehicle_id, ts, direction, angle) "
    "VALUES (%s, to_timestamp(%s), %s, %s)"
)

# Cap the retry buffer so a very long DB outage cannot grow memory unbounded.
MAX_BUFFERED_EVENTS = 10_000


class CountRecorder:
    """Persists the running vehicle count to Postgres on a fixed interval.

    Args:
        dsn: Postgres connection string (postgresql://user:pass@host/db).
        get_count: Callable returning the current total_count.
        interval_minutes: Minutes between inserts (tests may pass fractions,
            e.g. 5 / 60 for a 5-second interval).
    """

    def __init__(
        self,
        dsn: str,
        get_count: Callable[[], int],
        interval_minutes: float = 5.0,
    ) -> None:
        self._get_count = get_count
        self._pool = ConnectionPool(
            dsn,
            min_size=0,
            max_size=2,
            open=False,
            # Fail fast on checkout if Postgres is down; scheduler retries next tick.
            timeout=5.0,
            reconnect_timeout=5.0,
        )
        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._persist_once,
            IntervalTrigger(seconds=interval_minutes * 60.0),
            id="persist_count",
            max_instances=1,
            coalesce=True,
        )
        self.rows_written = 0
        self.failures = 0
        self.events_written = 0
        self._event_buffer: deque[tuple[int, float, str, float | None]] = deque(
            maxlen=MAX_BUFFERED_EVENTS
        )

    def start(self) -> CountRecorder:
        """Open the pool (non-blocking) and start the scheduler."""
        self._pool.open(wait=False)
        self._scheduler.start()
        logger.info("CountRecorder started")
        return self

    def _persist_once(self) -> None:
        self._drain_event_buffer()
        count = int(self._get_count())
        try:
            with self._pool.connection() as conn:
                conn.execute(INSERT_SQL, (count,))
            self.rows_written += 1
            logger.info("Persisted total_count=%d (row %d)", count, self.rows_written)
        except Exception as e:
            # Never propagate into the pipeline; retry on the next tick.
            self.failures += 1
            logger.error("Failed to persist count (failure %d): %s", self.failures, e)

    def record_vehicle(
        self,
        vehicle_id: int,
        direction: str,
        ts: float | None = None,
        angle: float | None = None,
    ) -> bool:
        """Insert one (vehicle_id, timestamp, direction, angle) row into vehicle_events.

        Called from the pipeline whenever a new vehicle's direction resolves.
        On failure the row is buffered and retried on scheduler ticks.

        Args:
            vehicle_id: Persistent tracker id of the vehicle.
            direction: One of UP / DOWN / LEFT / RIGHT / UNKNOWN.
            ts: Event time (unix epoch seconds); defaults to now.
            angle: Angle of travel in degrees (0=UP, 90=RIGHT, 180=DOWN,
                270=LEFT); None when the direction is UNKNOWN.

        Returns:
            True if the row was written immediately, False if buffered.
        """
        row = (
            int(vehicle_id),
            float(ts if ts is not None else time.time()),
            str(direction),
            float(angle) if angle is not None else None,
        )
        try:
            with self._pool.connection() as conn:
                conn.execute(INSERT_EVENT_SQL, row)
            self.events_written += 1
            logger.info(
                "Recorded vehicle %d direction=%s angle=%s (event %d)",
                row[0],
                row[2],
                row[3],
                self.events_written,
            )
            return True
        except Exception as e:
            self._event_buffer.append(row)
            logger.error(
                "Failed to record vehicle %d (buffered %d): %s", row[0], len(self._event_buffer), e
            )
            return False

    def _drain_event_buffer(self) -> None:
        """Retry buffered vehicle-event rows (called on scheduler ticks)."""
        while self._event_buffer:
            row = self._event_buffer[0]
            try:
                with self._pool.connection() as conn:
                    conn.execute(INSERT_EVENT_SQL, row)
                self._event_buffer.popleft()
                self.events_written += 1
            except Exception as e:
                logger.warning(
                    "Still cannot flush %d buffered vehicle events: %s",
                    len(self._event_buffer),
                    e,
                )
                break

    def flush(self) -> bool:
        """Write the current count immediately (e.g. on SIGTERM).

        Returns:
            True if the row was written.
        """
        before = self.rows_written
        self._persist_once()
        return self.rows_written > before

    def stop(self, flush: bool = True) -> None:
        """Stop the scheduler, optionally flushing a final row, close the pool."""
        try:
            self._scheduler.shutdown(wait=False)
        except Exception as e:
            logger.warning("Scheduler shutdown error: %s", e)
        if flush:
            self.flush()
        self._pool.close()
        logger.info(
            "CountRecorder stopped (rows_written=%d, events_written=%d, failures=%d)",
            self.rows_written,
            self.events_written,
            self.failures,
        )

    def __enter__(self) -> CountRecorder:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
