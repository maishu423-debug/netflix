"""
pipeline/sync_news_volume.py — populates the news_volume cache from
GDELT for this week's candidates. Lives in a background job, not
nowcast.py: GDELT's ~1 request/5s rate limit (~1-2 minutes for a full
candidate set) is too slow for a request handler polled every 60s from
the dashboard. The 90-day baseline window (see gdelt.cached_baseline)
accumulates naturally from repeated daily runs of this job rather than
being fetched all at once each time.
"""

from __future__ import annotations

from datetime import date, timedelta

import attention
import gdelt
from pipeline import build_candidates


def run(today: date | None = None) -> dict:
    today = today or date.today()
    candidates = build_candidates.load_current(today)
    if not candidates:
        return {"error": "No candidates found — run build_candidates first."}

    week = attention.netflix_week(today)
    lookback_start = week - timedelta(days=7)
    panel = gdelt.fetch_many(candidates.keys(), lookback_start, today)
    return {"titles_synced": len(candidates), "rows_cached": len(panel)}
