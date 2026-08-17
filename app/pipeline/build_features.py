"""pipeline/build_features.py — joins ground truth + attention into the training table.

Unlike the Colab version, this doesn't persist a model_table to its own
DB table — the training table is only ever needed transiently by
train_model.py, so it's built in memory each time run() is called
rather than stored redundantly (everything it's built from — ground_truth
and pageviews — is already durable in Postgres).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from sqlalchemy import text

import attention
import db
from pipeline import ground_truth


def _resolution_map() -> dict[str, str]:
    with db.engine.begin() as conn:
        rows = conn.execute(text("SELECT title, article FROM title_resolution WHERE article IS NOT NULL")).fetchall()
    return {r[0]: r[1] for r in rows}


def _build_week_features(candidate_articles: dict[str, str], week_start, week_end, as_of) -> pd.DataFrame:
    lookback_start = week_start - timedelta(days=7)
    panel = attention.fetch_many(candidate_articles.values(), lookback_start, as_of)
    cutoff = attention.reported_cutoff(panel)
    effective_as_of = min(as_of, cutoff) if cutoff else as_of
    baselines = attention.fetch_baselines(candidate_articles.values(), lookback_start)
    panel = attention.apply_baseline(panel, baselines)
    feats = attention.weekly_features(panel, as_of=effective_as_of)
    if feats.empty:
        return feats
    feats = feats[feats["week"] == week_start].copy()
    rev = {v: k for k, v in candidate_articles.items()}
    feats["title"] = feats["article"].map(rev)
    return feats


def build_table() -> pd.DataFrame:
    gt = ground_truth.load_from_db()
    resolution = _resolution_map()
    all_weeks = sorted(gt["week_start"].unique())
    by_week_titles = {w: set(gt.loc[gt["week_start"] == w, "show_title"]) for w in all_weeks}

    rows = []
    prev_week_titles: set[str] = set()
    prev_rank_map: dict[str, int] = {}
    seen_before: set[str] = set()

    for w in all_weeks:
        this_week_titles = by_week_titles[w]
        candidate_titles = this_week_titles | prev_week_titles
        candidate_articles = {t: resolution[t] for t in candidate_titles if t in resolution}
        if not candidate_articles:
            prev_week_titles = this_week_titles
            continue

        week_end = w + timedelta(days=6)
        feats = _build_week_features(candidate_articles, w, week_end, as_of=week_end)
        if feats.empty:
            prev_week_titles = this_week_titles
            continue

        rank_this_week = dict(
            zip(gt.loc[gt["week_start"] == w, "show_title"], gt.loc[gt["week_start"] == w, "rank"])
        )
        feats["rank"] = feats["title"].map(rank_this_week)
        feats["previous_rank"] = feats["title"].map(prev_rank_map)
        feats["is_new"] = ~feats["title"].isin(seen_before)
        feats["week_start"] = w
        rows.append(feats)

        prev_rank_map = rank_this_week
        prev_week_titles = this_week_titles
        seen_before |= this_week_titles

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run() -> dict:
    table = build_table()
    if table.empty:
        return {"error": "Built 0 rows — check title resolution coverage."}
    return {
        "rows": len(table), "weeks": int(table["week_start"].nunique()),
        "rows_with_rank": int(table["rank"].notna().sum()),
    }
