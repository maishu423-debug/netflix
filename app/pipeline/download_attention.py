"""pipeline/download_attention.py — bulk pageview fetch for all resolved titles."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import text

import attention
import db
from pipeline import ground_truth

PAGEVIEWS_FLOOR = date(2015, 7, 1)


def run(max_workers: int = 10) -> dict:
    gt = ground_truth.load_from_db()
    with db.engine.begin() as conn:
        rows = conn.execute(text("SELECT title, article FROM title_resolution WHERE article IS NOT NULL")).fetchall()
    resolution = {r[0]: r[1] for r in rows}

    articles = sorted(set(resolution.values()))
    if not articles:
        return {"error": "No resolved articles found. Run resolve_titles first."}

    start = max(gt["week_start"].min(), PAGEVIEWS_FLOOR)
    end = min(gt["week_end"].max() + timedelta(days=7), date.today())

    panel = attention.fetch_many(articles, start, end, max_workers=max_workers)

    zero_coverage = []
    if not panel.empty:
        totals = panel.groupby("article")["views"].sum()
        zero_coverage = totals[totals == 0].index.tolist()

    return {
        "articles_fetched": len(articles),
        "date_range": [str(start), str(end)],
        "rows_cached": len(panel),
        "zero_coverage_count": len(zero_coverage),
        "zero_coverage_sample": zero_coverage[:20],
    }
