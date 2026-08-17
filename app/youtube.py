"""
youtube.py — official-trailer view growth via the YouTube Data API v3.

Trailers often go up weeks before a show premieres — earlier than
Wikipedia has an article worth visiting or GDELT has news coverage —
so this is the one attention source that can say something about a
brand-new candidate before its first day of real airing.

QUOTA: search.list (finding a title's trailer) costs 100 units per
call; videos.list (reading stats for already-known video IDs, up to 50
at once) costs 1 unit regardless of batch size. Daily quota is 10,000
units. Resolving ~15-20 candidates once costs ~1,500-2,000 units —
affordable, but unlike tmdb.resolve_tmdb_ids, a MISS is cached here
(as NULL) rather than retried every call: titles like live sports
rebroadcasts or stand-up specials often have no findable "Official
Trailer" at all, and retrying those forever would burn real quota for
no benefit.

Not rate-limited the way gdelt.py is, but kept to the same
background-job-only shape for consistency: sync_youtube.py does the
live calls, nowcast.py/build_features.py only ever read the cache.
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Iterable

import pandas as pd
import requests
from sqlalchemy import text

import db

YOUTUBE_API = "https://www.googleapis.com/youtube/v3"
MAX_RETRIES = 4


def _token() -> str:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise EnvironmentError("Set YOUTUBE_API_KEY before using youtube.py.")
    return key


def _get(path: str, params: dict) -> dict | None:
    params = {**params, "key": _token()}
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(f"{YOUTUBE_API}{path}", params=params, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    return None


def search_trailer_video_id(title: str) -> str | None:
    """LIVE CALL, 100 quota units — only use from a background job."""
    data = _get("/search", {
        "part": "snippet", "q": f"{title} Official Trailer Netflix",
        "type": "video", "maxResults": 1,
    })
    if not data or not data.get("items"):
        return None
    return data["items"][0]["id"]["videoId"]


def resolve_video_ids(titles: list[str]) -> dict[str, str]:
    """
    Cached title -> trailer video_id. LIVE CALL for cache misses — only
    use from a background job (pipeline/sync_youtube.py).
    """
    with db.engine.begin() as conn:
        rows = conn.execute(
            text("SELECT title, video_id FROM youtube_video_id WHERE title = ANY(:titles)"),
            {"titles": titles},
        ).fetchall()
    cache: dict[str, str | None] = {r[0]: r[1] for r in rows}

    for t in titles:
        if t in cache:
            continue
        vid = search_trailer_video_id(t)
        cache[t] = vid
        with db.engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO youtube_video_id (title, video_id) VALUES (:t, :v)
                        ON CONFLICT (title) DO UPDATE SET video_id = :v"""),
                {"t": t, "v": vid},
            )

    return {t: v for t, v in cache.items() if v}


def snapshot_views(video_ids: dict[str, str], day: date | None = None) -> pd.DataFrame:
    """
    Current cumulative view count for each already-resolved trailer.
    LIVE CALL, 1 quota unit per 50 IDs — only use from a background job.
    Call daily to build a growth time series; not backfillable, same as
    tmdb.snapshot_popularity.
    """
    day = day or date.today()
    rev = {v: t for t, v in video_ids.items()}
    ids = list(rev.keys())
    if not ids:
        return pd.DataFrame()

    rows = []
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = _get("/videos", {"part": "statistics", "id": ",".join(chunk)})
        if not data:
            continue
        for item in data.get("items", []):
            vc = item.get("statistics", {}).get("viewCount")
            if vc is None:
                continue
            rows.append({"video_id": item["id"], "title": rev[item["id"]], "day": day, "view_count": int(vc)})

    df = pd.DataFrame(rows, columns=["video_id", "title", "day", "view_count"])
    with db.engine.begin() as conn:
        for _, r in df.iterrows():
            conn.execute(
                text("""INSERT INTO youtube_views (video_id, day, title, view_count) VALUES (:vid, :d, :t, :vc)
                        ON CONFLICT (video_id, day) DO UPDATE SET view_count = :vc"""),
                {"vid": r["video_id"], "d": r["day"], "t": r["title"], "vc": int(r["view_count"])},
            )
    return df


def load_cached_growth(titles: Iterable[str], lookback_start: date, as_of: date) -> dict[str, float]:
    """
    Views gained on each title's trailer between the earliest and latest
    cached snapshot within [lookback_start, as_of] — read-only, never
    calls YouTube. A title with fewer than 2 snapshots in that window
    has nothing to diff against yet and is left out of the result.
    """
    titles = [t for t in titles if t]
    if not titles:
        return {}
    with db.engine.begin() as conn:
        df = pd.read_sql_query(
            text("SELECT title, day, view_count FROM youtube_views "
                 "WHERE title = ANY(:titles) AND day BETWEEN :s AND :e"),
            conn, params={"titles": titles, "s": lookback_start, "e": as_of},
        )
    if df.empty:
        return {}
    growth = {}
    for title, g in df.groupby("title"):
        g = g.sort_values("day")
        if len(g) >= 2:
            growth[title] = float(g["view_count"].iloc[-1] - g["view_count"].iloc[0])
    return growth
