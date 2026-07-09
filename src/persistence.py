"""Periodic count persistence to Postgres (Phase E).

CountRecorder runs a background APScheduler job that inserts
``(timestamp, total_count)`` into the ``vehicle_counts`` table every
``interval_minutes`` (default 5). Connections come from a
psycopg_pool.ConnectionPool; a Postgres outage never propagates into the
pipeline — failed inserts are logged and retried on the next tick.
"""

from __future__ import annotations

from collections.abc import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from psycopg_pool import ConnectionPool

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

INSERT_SQL = "INSERT INTO vehicle_counts (ts, total_count) VALUES (now(), %s)"


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

    def start(self) -> CountRecorder:
        """Open the pool (non-blocking) and start the scheduler."""
        self._pool.open(wait=False)
        self._scheduler.start()
        logger.info("CountRecorder started")
        return self

    def _persist_once(self) -> None:
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
            "CountRecorder stopped (rows_written=%d, failures=%d)",
            self.rows_written,
            self.failures,
        )

    def __enter__(self) -> CountRecorder:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
