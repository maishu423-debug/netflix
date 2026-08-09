"""pipeline/build_candidates.py — builds this week's candidate set."""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import text

import attention
import db
import tmdb
from pipeline import ground_truth

PREMIERE_LOOKBACK_DAYS = 21  # covers late-week premieres that spike the following week


def run(today: date | None = None) -> dict:
    today = today or date.today()
    week_start = attention.netflix_week(today)
    week_end = week_start + timedelta(days=6)

    gt = ground_truth.load_from_db()
    past_weeks = [w for w in gt["week_start"].unique() if w < week_start]
    still_contending = set()
    if past_weeks:
        last_week = max(past_weeks)
        still_contending = set(gt.loc[gt["week_start"] == last_week, "show_title"])

    premiere_window_start = week_start - timedelta(days=PREMIERE_LOOKBACK_DAYS)
    premieres_df = tmdb.discover_new_candidates(premiere_window_start, week_end)
    premiere_titles = set(premieres_df["title"]) if not premieres_df.empty else set()

    all_titles = still_contending | premiere_titles
    overrides = attention.load_overrides()

    resolved: dict[str, str | None] = {}
    for t in sorted(all_titles):
        if t in overrides:
            resolved[t] = overrides[t]
            continue
        year_hint = today.year if t in premiere_titles else None
        resolved[t] = attention.resolve_article(t, year=year_hint)

    with db.engine.begin() as conn:
        conn.execute(text("DELETE FROM candidates_live WHERE week = :w"), {"w": week_start})
        for t, a in resolved.items():
            conn.execute(
                text("INSERT INTO candidates_live (week, title, article) VALUES (:w, :t, :a)"),
                {"w": week_start, "t": t, "a": a},
            )

    unresolved = [t for t, a in resolved.items() if not a]
    return {
        "week_start": str(week_start), "week_end": str(week_end),
        "still_contending": len(still_contending), "premieres": len(premiere_titles),
        "total_candidates": len(all_titles), "unresolved": unresolved,
    }


def load_current(today: date | None = None) -> dict[str, str]:
    today = today or date.today()
    week_start = attention.netflix_week(today)
    with db.engine.begin() as conn:
        rows = conn.execute(
            text("SELECT title, article FROM candidates_live WHERE week = :w AND article IS NOT NULL"),
            {"w": week_start},
        ).fetchall()
    return {r[0]: r[1] for r in rows}
