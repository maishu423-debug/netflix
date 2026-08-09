"""
db.py — Postgres connection + schema. Replaces the Drive-backed
parquet/sqlite/json files from the Colab version. Set DATABASE_URL to
Railway's provided Postgres connection string.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL not set — add Railway's Postgres connection string.")

# Railway's Postgres URL sometimes starts with 'postgres://', which
# SQLAlchemy 2.x no longer accepts — normalize it.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, poolclass=NullPool)

SCHEMA = """
CREATE TABLE IF NOT EXISTS ground_truth (
    week_start DATE NOT NULL,
    week_end   DATE NOT NULL,
    rank       INTEGER NOT NULL,
    show_title TEXT NOT NULL,
    season_title TEXT,
    cum_weeks_top10 INTEGER,
    PRIMARY KEY (week_start, show_title)
);

CREATE TABLE IF NOT EXISTS title_resolution (
    title   TEXT PRIMARY KEY,
    article TEXT,
    source  TEXT
);

CREATE TABLE IF NOT EXISTS overrides (
    title   TEXT PRIMARY KEY,
    article TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pageviews (
    article TEXT NOT NULL,
    day     DATE NOT NULL,
    views   INTEGER NOT NULL,
    PRIMARY KEY (article, day)
);

CREATE TABLE IF NOT EXISTS candidates_live (
    week    DATE NOT NULL,
    title   TEXT NOT NULL,
    article TEXT,
    PRIMARY KEY (week, title)
);

CREATE TABLE IF NOT EXISTS title_to_tmdbid (
    title   TEXT PRIMARY KEY,
    tmdb_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS model_weights (
    id            SERIAL PRIMARY KEY,
    trained_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    feature_cols  JSONB NOT NULL,
    weights       JSONB NOT NULL,
    mean_         JSONB NOT NULL,
    std_          JSONB NOT NULL,
    l2            DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_results (
    week_start       DATE PRIMARY KEY,
    n_candidates     INTEGER,
    top1_hit         INTEGER,
    top3_hit         INTEGER,
    reciprocal_rank  DOUBLE PRECISION,
    logloss          DOUBLE PRECISION,
    kendall_tau      DOUBLE PRECISION,
    evaluated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS nowcast_history (
    id              SERIAL PRIMARY KEY,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    week_start      DATE NOT NULL,
    title           TEXT NOT NULL,
    p_number_one    DOUBLE PRECISION,
    kalshi_yes_bid  DOUBLE PRECISION,
    kalshi_yes_ask  DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS tmdb_popularity (
    tmdb_id    INTEGER NOT NULL,
    day        DATE NOT NULL,
    title      TEXT,
    popularity DOUBLE PRECISION,
    PRIMARY KEY (tmdb_id, day)
);

CREATE TABLE IF NOT EXISTS job_runs (
    id          SERIAL PRIMARY KEY,
    job_name    TEXT NOT NULL,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status      TEXT NOT NULL DEFAULT 'running',
    log         TEXT
);
"""


def init_schema() -> None:
    with engine.begin() as conn:
        for statement in SCHEMA.strip().split(";\n\n"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))


def get_conn():
    """Use as: with db.get_conn() as conn: conn.execute(...)"""
    return engine.begin()


if __name__ == "__main__":
    init_schema()
    print("Schema created/verified.")
