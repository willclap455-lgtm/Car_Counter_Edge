-- Vehicle Counting App — Postgres schema
-- Applied with: psql "$DATABASE_URL" -f schema.sql

CREATE TABLE IF NOT EXISTS vehicle_counts (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_count BIGINT      NOT NULL CHECK (total_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_vehicle_counts_ts ON vehicle_counts (ts);

-- One row per newly detected vehicle: its tracker id, when it was first
-- confirmed, and its direction of travel relative to the frame.
CREATE TABLE IF NOT EXISTS vehicle_events (
    id         BIGSERIAL PRIMARY KEY,
    vehicle_id BIGINT      NOT NULL,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    direction  TEXT        NOT NULL
        CHECK (direction IN ('UP', 'DOWN', 'LEFT', 'RIGHT', 'UNKNOWN'))
);

CREATE INDEX IF NOT EXISTS idx_vehicle_events_ts ON vehicle_events (ts);
