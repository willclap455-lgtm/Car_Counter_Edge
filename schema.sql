-- Vehicle Counting App — Postgres schema
-- Applied with: psql "$DATABASE_URL" -f schema.sql

CREATE TABLE IF NOT EXISTS vehicle_counts (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_count BIGINT      NOT NULL CHECK (total_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_vehicle_counts_ts ON vehicle_counts (ts);

-- One row per newly detected vehicle: its tracker id, when it was first
-- confirmed, its direction of travel relative to the frame, and the exact
-- angle of travel in degrees (0 = UP, 90 = RIGHT, 180 = DOWN, 270 = LEFT;
-- NULL when the vehicle never moved enough to resolve a direction).
CREATE TABLE IF NOT EXISTS vehicle_events (
    id         BIGSERIAL PRIMARY KEY,
    vehicle_id BIGINT      NOT NULL,
    ts         TIMESTAMPTZ NOT NULL DEFAULT now(),
    direction  TEXT        NOT NULL
        CHECK (direction IN ('UP', 'DOWN', 'LEFT', 'RIGHT', 'UNKNOWN')),
    angle      REAL
        CHECK (angle IS NULL OR (angle >= 0 AND angle < 360))
);

-- In-place upgrade for databases created before the angle column existed.
ALTER TABLE vehicle_events ADD COLUMN IF NOT EXISTS angle REAL
    CHECK (angle IS NULL OR (angle >= 0 AND angle < 360));

CREATE INDEX IF NOT EXISTS idx_vehicle_events_ts ON vehicle_events (ts);
