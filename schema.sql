-- Vehicle Counting App — Postgres schema
-- Applied with: psql "$DATABASE_URL" -f schema.sql

CREATE TABLE IF NOT EXISTS vehicle_counts (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    total_count BIGINT      NOT NULL CHECK (total_count >= 0)
);

CREATE INDEX IF NOT EXISTS idx_vehicle_counts_ts ON vehicle_counts (ts);
