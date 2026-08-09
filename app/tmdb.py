"""
tmdb.py — two things TMDb is good for that Wikipedia can't do:

  1. discover_new_candidates() — a premiere calendar. Finds Netflix TV
     shows first-airing in a given week, via /discover/tv. This is what
     fills the gap flagged earlier: the pipeline can only track a show's
     attention once it's in the candidate set, and nothing else in this
     project knows what's *about* to premiere.

  2. snapshot_popularity()    — TMDb's daily `popularity` float, a
     continuous score that exists from day one, before a Wikipedia
     article necessarily does. NOT backfillable (TMDb doesn't expose
     historical popularity), so it only accrues going forward — every
     day you don't run this is a day of history you can't get back for
     that feature. It's a nowcast supplement, not a backtest feature,
     until enough weeks accumulate.

Auth: TMDb v4 read access token, Bearer auth. Get yours from
themoviedb.org -> Settings -> API, and set it as an environment
variable — NEVER hardcode it in a file you might commit or share:

    export TMDB_READ_TOKEN="eyJhbGciOi..."

Netflix's TMDb provider_id (watch_region=US) is 8. Netflix's TMDb
network_id is 213 — used as a secondary filter since provider data is
sometimes incomplete for very recent premieres.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta

import pandas as pd
import requests
from sqlalchemy import text

import db

TMDB_API = "https://api.themoviedb.org/3"
NETFLIX_PROVIDER_ID = 8
NETFLIX_NETWORK_ID = 213

REQUEST_DELAY = 0.05
MAX_RETRIES = 4


def _token() -> str:
    tok = os.environ.get("TMDB_READ_TOKEN")
    if not tok:
        raise EnvironmentError(
            "Set TMDB_READ_TOKEN before using tmdb.py — never hardcode it. "
            "export TMDB_READ_TOKEN='your read access token'"
        )
    return tok


def _headers() -> dict:
    return {"Authorization": f"Bearer {_token()}", "accept": "application/json"}


def _get(path: str, params: dict | None = None) -> dict | None:
    url = f"{TMDB_API}{path}"
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, headers=_headers(), params=params, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        time.sleep(REQUEST_DELAY)
        return r.json()
    return None


# --------------------------------------------------------------------------
# 1. premiere calendar
# --------------------------------------------------------------------------

def discover_new_candidates(week_start: date, week_end: date) -> pd.DataFrame:
    """
    Netflix TV shows first-airing within [week_start, week_end].
    Returns DataFrame[tmdb_id, title, first_air_date, popularity].

    Use this to auto-populate part of candidates_live.json — still merge
    it with your judgment (a low-profile premiere might not be a real
    contender, and this can miss last-minute schedule changes).
    """
    rows = []
    page = 1
    while True:
        data = _get("/discover/tv", {
            "with_watch_providers": NETFLIX_PROVIDER_ID,
            "watch_region": "US",
            "first_air_date.gte": week_start.isoformat(),
            "first_air_date.lte": week_end.isoformat(),
            "sort_by": "popularity.desc",
            "page": page,
        })
        if not data or not data.get("results"):
            break
        for r in data["results"]:
            rows.append({
                "tmdb_id": r["id"],
                "title": r["name"],
                "first_air_date": r.get("first_air_date"),
                "popularity": r.get("popularity"),
            })
        if page >= data.get("total_pages", 1):
            break
        page += 1

    return pd.DataFrame(rows)


def discover_returning_candidates(week_start: date, week_end: date) -> pd.DataFrame:
    """
    Netflix TV shows with a season/episode airing in the window, for
    already-running titles that aren't 'first air date' events but could
    still spike (e.g. a new season drop). TMDb doesn't expose 'next
    episode air date' via /discover directly, so this uses
    /tv/{id}/season/{n} lookups on known ongoing titles — pass in
    candidates from last week's ground truth as `known_ongoing_tmdb_ids`
    rather than trying to discover blind.
    """
    raise NotImplementedError(
        "TMDb has no clean 'discover shows with new episodes this week' "
        "endpoint. For returning titles, keep tracking them (they stay in "
        "the candidate set via last week's ground-truth top 10 already); "
        "this function is a placeholder if you want to add per-title "
        "next-episode checks via /tv/{id}."
    )


# --------------------------------------------------------------------------
# 2. daily popularity snapshot (nowcast-only, not backfillable)
# --------------------------------------------------------------------------

def search_tv(title: str, year: int | None = None) -> int | None:
    """Best-effort TMDb id lookup for a title, for wiring into candidates."""
    params = {"query": title}
    if year:
        params["first_air_date_year"] = year
    data = _get("/search/tv", params)
    if not data or not data.get("results"):
        return None
    return data["results"][0]["id"]


def resolve_tmdb_ids(titles: list[str]) -> dict[str, int]:
    """
    Cached title -> tmdb_id lookup. Misses are retried every call (cheap
    to skip caching a None — a title search failing once doesn't mean it
    always will, e.g. right after a show's just been added to TMDb).
    """
    with db.engine.begin() as conn:
        rows = conn.execute(
            text("SELECT title, tmdb_id FROM title_to_tmdbid WHERE title = ANY(:titles)"),
            {"titles": titles},
        ).fetchall()
    cache: dict[str, int] = {r[0]: r[1] for r in rows}

    for t in titles:
        if t in cache:
            continue
        tid = search_tv(t)
        if tid:
            cache[t] = tid
            with db.engine.begin() as conn:
                conn.execute(
                    text("""INSERT INTO title_to_tmdbid (title, tmdb_id) VALUES (:t, :i)
                            ON CONFLICT (title) DO UPDATE SET tmdb_id = :i"""),
                    {"t": t, "i": tid},
                )

    return {t: cache[t] for t in titles if t in cache}


def snapshot_popularity(tmdb_ids: dict[str, int], day: date | None = None) -> pd.DataFrame:
    """
    tmdb_ids: {title: tmdb_id}. Call this once a day (e.g. a daily cron)
    to accrue a popularity time series — running it retroactively cannot
    recover past days.
    """
    day = day or date.today()
    rows = []
    for title, tid in tmdb_ids.items():
        data = _get(f"/tv/{tid}")
        if not data:
            continue
        pop = data.get("popularity")
        rows.append({"tmdb_id": tid, "day": day, "title": title, "popularity": pop})

    if rows:
        with db.engine.begin() as conn:
            for r in rows:
                conn.execute(
                    text("""INSERT INTO tmdb_popularity (tmdb_id, day, title, popularity)
                            VALUES (:tid, :day, :title, :pop)
                            ON CONFLICT (tmdb_id, day) DO UPDATE SET popularity = :pop"""),
                    {"tid": r["tmdb_id"], "day": r["day"], "title": r["title"], "pop": r["popularity"]},
                )
    return pd.DataFrame(rows)


def load_popularity_history(tmdb_id: int) -> pd.DataFrame:
    with db.engine.begin() as conn:
        df = pd.read_sql_query(
            text("SELECT day, popularity FROM tmdb_popularity WHERE tmdb_id = :tid ORDER BY day"),
            conn, params={"tid": tmdb_id},
        )
    df["day"] = pd.to_datetime(df["day"]).dt.date
    return df


if __name__ == "__main__":
    # Example: find Netflix TV premieres in the current chart week.
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)
    print(f"Netflix TV premieres {week_start} to {week_end}:")
    print(discover_new_candidates(week_start, week_end))
