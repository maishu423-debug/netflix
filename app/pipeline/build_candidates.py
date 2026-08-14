"""
pipeline/build_candidates.py — builds this week's candidate set.

CANDIDATE SOURCE: Kalshi's own market ladder, when one is open, instead
of our own TMDb-premiere discovery + last-week's-ground-truth approach.

Why the switch: TMDb discovery had no genre/content-type filter, so it
was pulling in things Kalshi's own market makers don't even consider —
e.g. a documentary series (MOURINHO) that isn't in Kalshi's ladder at
all. Separately, our trained model's core feature (Wikipedia pageviews)
is GLOBAL traffic, not US-specific, so titles genuinely huge outside
the US (large English-reading audiences, e.g. India) could look like a
real US spike in the data even with near-zero actual US Netflix
traction. Kalshi's market makers have effectively already solved both
problems for us — their ladder IS the curated "real US contenders"
list. Using it directly sidesteps both issues at once, at the cost of
being dependent on Kalshi's own listing being complete and timely.

FALLBACK: if no Kalshi market is currently open (e.g. between
settlement and the next week's market being listed), this falls back
to the original TMDb-discovery + last-week's-actual-Top-10 approach
entirely unchanged — so the system still works even without Kalshi.

TITLE NORMALIZATION: Kalshi's titles include suffixes Netflix's own
naming doesn't ("...: Season 3", "...: Limited Series") — these need
stripping before matching against ground_truth's show_title (needed
for previous_rank/is_new) or searching Wikipedia. This is regex-based
and won't be perfect for every case (e.g. stand-up specials with a
date range instead of a season number) — when it's wrong, the failure
is graceful: that candidate just won't match a previous_rank (treated
as new) rather than breaking anything, and Wikipedia resolution is
attempted on both the normalized AND raw title as a fallback.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from sqlalchemy import text

import attention
import db
import kalshi_client
import tmdb
from pipeline import ground_truth

PREMIERE_LOOKBACK_DAYS = 21  # covers late-week premieres that spike the following week

_SUFFIX_PATTERNS = [
    r":\s*Season\s+\d+'?$",
    r":\s*Limited Series$",
    r":\s*Part\s+\d+$",
    r":\s*Volume\s+\d+$",
    r":\s*Collection\s*\d*$",
    r":\s*\d{4}\s*-\s*\w+\s+\d{1,2},\s*\d{4}$",  # comedy-special style, e.g. ": 2026 - August 3, 2026"
]


def normalize_kalshi_title(title: str) -> str:
    """Strips Kalshi's trailing season/format suffixes to approximate Netflix's own naming."""
    t = title.strip()
    for pattern in _SUFFIX_PATTERNS:
        t = re.sub(pattern, "", t, flags=re.IGNORECASE).strip()
    return t


def _resolve_with_fallback(display_title: str, normalized_title: str, overrides: dict,
                            today: date) -> str | None:
    """Tries overrides/resolution on the normalized title first, then the raw Kalshi title."""
    for candidate in (normalized_title, display_title):
        if candidate in overrides:
            return overrides[candidate]
    for candidate in (normalized_title, display_title):
        art = attention.resolve_article(candidate, year=today.year)
        if art:
            return art
    return None


def _build_from_kalshi(today: date, week_start: date) -> dict[str, str | None] | None:
    ladder = kalshi_client.get_current_ladder()
    if ladder is None or not ladder.rows:
        return None

    overrides = attention.load_overrides()
    resolved: dict[str, str | None] = {}
    for row in ladder.rows:
        normalized = normalize_kalshi_title(row.title)
        resolved[normalized] = _resolve_with_fallback(row.title, normalized, overrides, today)
    return resolved


def _build_from_tmdb_fallback(today: date, week_start: date, week_end: date) -> dict[str, str | None]:
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
    return resolved


def run(today: date | None = None) -> dict:
    today = today or date.today()
    week_start = attention.netflix_week(today)
    week_end = week_start + timedelta(days=6)

    resolved = _build_from_kalshi(today, week_start)
    source = "kalshi_ladder"
    if resolved is None:
        resolved = _build_from_tmdb_fallback(today, week_start, week_end)
        source = "tmdb_fallback"

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
        "source": source, "total_candidates": len(resolved), "unresolved": unresolved,
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