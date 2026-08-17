"""
gdelt.py — GDELT DOC 2.0 API news-coverage-volume, a low-latency
complement to Wikipedia's attention signal (attention.py). GDELT
updates roughly every 15 minutes vs Wikipedia's pageviews API's
1-2 day lag — attention.py's MIN_DAYS_FOR_MODEL gate exists specifically
because Wikipedia can't say anything about "today" yet; GDELT often can.

RATE LIMIT: confirmed live against the real API — GDELT throttles to
about one request per 5 seconds per client, no documented daily quota
but a hard per-request pace limit. That's too slow to call from a
request handler that's polled every 60s (see main.py's /api/prediction),
so this module's live-fetching functions (fetch_news_volume/fetch_many)
are only ever called from the background jobs (pipeline/sync_news_volume.py).
nowcast.py and build_features.py only ever read the cache (load_cached/
cached_baseline/weekly_news_features) — zero HTTP calls on that path.

RETENTION: the DOC 2.0 API only searches roughly the last 3 months of
articles, unlike Wikipedia's pageviews (backfilled from 2015). A
candidate's cached history — and therefore its baseline and any
historical training weeks older than ~3 months — will legitimately stay
empty; treated the same as any other missing-history case (0.0 baseline,
NaN feature, gracefully dropped/ignored like every other optional
feature in this codebase).
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests
from sqlalchemy import text

import attention
import db

GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_REQUEST_DELAY = 5.5  # GDELT enforces ~1 request/5s per client — confirmed live via 429s
MAX_RETRIES = 3
REQUEST_TIMEOUT = 15
# Hard ceiling on fetch_many's total runtime. Without this, a slow or
# unreachable GDELT endpoint means every title independently retries
# MAX_RETRIES times (each retry: REQUEST_TIMEOUT + GDELT_REQUEST_DELAY)
# before giving up — no exception ever gets raised (timeouts are caught
# and retried, not propagated), so a background job's try/except never
# catches it and the job just sits at "running" indefinitely instead of
# failing fast. This caps it: once the budget is spent, stop starting
# new fetches and return whatever's been gathered so far.
MAX_SYNC_SECONDS = 180


def _get(params: dict) -> dict | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(GDELT_API, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            time.sleep(GDELT_REQUEST_DELAY)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(GDELT_REQUEST_DELAY)
            continue
        r.raise_for_status()
        time.sleep(GDELT_REQUEST_DELAY)
        try:
            return r.json()
        except ValueError:
            return None
    return None


def fetch_news_volume(title: str, start: date, end: date, use_cache: bool = True) -> pd.DataFrame:
    """
    Daily news-coverage-volume for `title`, query-scoped to '"<title>"
    Netflix' to cut down on unrelated-topic collisions (see MOURINHO in
    attention.py's history). LIVE CALL — only use from a background job.
    """
    if use_cache:
        with db.engine.begin() as conn:
            cached = pd.read_sql_query(
                text("SELECT day, value FROM news_volume WHERE title = :t AND day BETWEEN :s AND :e"),
                conn, params={"t": title, "s": start, "e": end},
            )
        expected = (end - start).days + 1
        if len(cached) == expected:
            cached["title"] = title
            cached["day"] = pd.to_datetime(cached["day"]).dt.date
            return cached[["day", "title", "value"]]

    data = _get({
        "query": f'"{title}" Netflix', "mode": "timelinevol", "format": "json",
        "startdatetime": start.strftime("%Y%m%d") + "000000",
        "enddatetime": end.strftime("%Y%m%d") + "235959",
    })

    rows = []
    if data:
        points = (data.get("timeline") or [{}])[0].get("data", [])
        rows = [
            {"day": datetime.strptime(p["date"], "%Y%m%dT%H%M%SZ").date(), "value": p["value"]}
            for p in points
        ]

    df = pd.DataFrame(rows, columns=["day", "value"])
    full = pd.DataFrame({"day": pd.date_range(start, end, freq="D").date})
    df = full.merge(df, on="day", how="left")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
    df["title"] = title

    with db.engine.begin() as conn:
        for d, v in zip(df["day"], df["value"]):
            conn.execute(
                text("""INSERT INTO news_volume (title, day, value) VALUES (:t, :d, :v)
                        ON CONFLICT (title, day) DO UPDATE SET value = :v"""),
                {"t": title, "d": d, "v": float(v)},
            )

    return df[["day", "title", "value"]]


def fetch_many(titles: Iterable[str], start: date, end: date) -> pd.DataFrame:
    """
    Serial, not parallel — GDELT's rate limit means N titles takes
    roughly N * 5.5s when the cache is cold. LIVE CALL — only use from
    a background job (pipeline/sync_news_volume.py), never from a
    request handler. Stops early past MAX_SYNC_SECONDS total elapsed —
    titles not reached this run just stay whatever they were (missing,
    or stale from a prior sync) and get picked up next time.
    """
    titles = [t for t in titles if t]
    if not titles:
        return pd.DataFrame()
    started = time.monotonic()
    frames = []
    for t in titles:
        if time.monotonic() - started > MAX_SYNC_SECONDS:
            break
        frames.append(fetch_news_volume(t, start, end))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cached(titles: Iterable[str], start: date, end: date) -> pd.DataFrame:
    """Read-only — never calls GDELT. Whatever's cached is whatever sync_news_volume has fetched so far."""
    titles = [t for t in titles if t]
    if not titles:
        return pd.DataFrame()
    with db.engine.begin() as conn:
        df = pd.read_sql_query(
            text("SELECT day, title, value FROM news_volume WHERE title = ANY(:titles) AND day BETWEEN :s AND :e"),
            conn, params={"titles": titles, "s": start, "e": end},
        )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


def cached_baseline(titles: Iterable[str], before: date, window_days: int = 90) -> dict[str, float]:
    """
    Median cached daily value per title over the window before `before`
    — read-only, no live GDELT calls. A title with no cached history
    yet gets 0.0, same as attention.fetch_baselines.
    """
    titles = [t for t in titles if t]
    start = before - timedelta(days=window_days)
    end = before - timedelta(days=1)
    if not titles or end < start:
        return {}
    with db.engine.begin() as conn:
        df = pd.read_sql_query(
            text("SELECT title, value FROM news_volume WHERE title = ANY(:titles) AND day BETWEEN :s AND :e"),
            conn, params={"titles": titles, "s": start, "e": end},
        )
    if df.empty:
        return {}
    return df.groupby("title")["value"].median().to_dict()


def weekly_news_features(titles: Iterable[str], week_start: date, as_of: date) -> pd.DataFrame:
    """
    news_share/news_momentum/news_share_wow for `titles` in week_start's
    week — read-only, reuses attention.py's generic panel math (baseline
    subtraction, reporting-cutoff detection, weekly aggregation) on
    GDELT's cached data instead of Wikipedia's.
    """
    empty = pd.DataFrame(columns=["title", "news_share", "news_momentum", "news_share_wow"])
    titles = [t for t in titles if t]
    if not titles:
        return empty

    lookback_start = week_start - timedelta(days=7)
    panel = load_cached(titles, lookback_start, as_of)
    if panel.empty:
        return empty
    panel = panel.rename(columns={"title": "article", "value": "views"})

    cutoff = attention.reported_cutoff(panel)
    effective_as_of = min(as_of, cutoff) if cutoff else as_of
    baselines = cached_baseline(titles, lookback_start)
    panel = attention.apply_baseline(panel, baselines)

    feats = attention.weekly_features(panel, as_of=effective_as_of)
    feats = feats[feats["week"] == week_start].copy() if not feats.empty else feats
    if feats.empty:
        return empty

    return feats.rename(columns={
        "article": "title", "share": "news_share",
        "momentum": "news_momentum", "share_wow": "news_share_wow",
    })[["title", "news_share", "news_momentum", "news_share_wow"]]
