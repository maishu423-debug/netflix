"""pipeline/nowcast.py — predicts this week's #1, with a TMDb fallback for early-week days."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text

import attention
import db
import tmdb
from model import PlackettLuceRanker, add_derived_features
from pipeline import build_candidates, ground_truth

MIN_DAYS_FOR_MODEL = 2


def _previous_rank_map(gt: pd.DataFrame, before: date) -> dict:
    past_weeks = [w for w in gt["week_start"].unique() if w < before]
    if not past_weeks:
        return {}
    last_week = max(past_weeks)
    sub = gt[gt["week_start"] == last_week]
    return dict(zip(sub["show_title"], sub["rank"]))


def run(today: date | None = None) -> dict:
    today = today or date.today()
    week = attention.netflix_week(today)

    candidates = build_candidates.load_current(today)
    if not candidates:
        return {"week_start": str(week), "days_with_data": 0, "model": [], "fallback": [],
                "note": "No candidates found — run build_candidates first."}

    gt = ground_truth.load_from_db()
    prev_rank = _previous_rank_map(gt, before=week)
    seen_ever = set(gt["show_title"].unique())

    lookback_start = week - timedelta(days=7)
    panel = attention.fetch_many(candidates.values(), lookback_start, today)
    feats = attention.weekly_features(panel, as_of=today)
    feats = feats[feats["week"] == week].copy() if not feats.empty else feats

    days_with_data = int(feats["n_days"].max()) if not feats.empty else 0

    model_rows = []
    if not feats.empty:
        rev = {v: k for k, v in candidates.items()}
        feats["title"] = feats["article"].map(rev)
        feats["previous_rank"] = feats["title"].map(prev_rank)
        feats["is_new"] = ~feats["title"].isin(seen_ever)
        feats = add_derived_features(feats)
        feats["share_wow"] = feats["share_wow"].fillna(0)
        feats["momentum"] = feats["momentum"].fillna(0)
        feats = feats.dropna(subset=["share", "log_previous_rank", "is_new_num"])

        if not feats.empty:
            try:
                model = PlackettLuceRanker.load_latest_from_db()
                feats["p_number_one"] = model.predict_proba(feats)
                scored = feats.sort_values("p_number_one", ascending=False)
                scored = scored[["title", "p_number_one", "share", "momentum",
                                  "share_wow", "previous_rank", "is_new"]]
                # previous_rank is legitimately NaN for brand-new titles (no
                # prior week to have a rank in) — that's valid data, but
                # strict JSON (which FastAPI/Starlette enforce) doesn't allow
                # a literal NaN value, only null. Swap NaN -> None so it
                # serializes as JSON null instead of crashing the response.
                scored = scored.astype(object).where(scored.notna(), None)
                model_rows = scored.to_dict("records")
            except Exception:
                pass  # no usable trained model yet — fall through to the TMDb fallback below

    fallback_rows = []
    if days_with_data < MIN_DAYS_FOR_MODEL or not model_rows:
        ids = tmdb.resolve_tmdb_ids(list(candidates.keys()))
        if ids:
            snap = tmdb.snapshot_popularity(ids)
            if not snap.empty:
                fb = snap.sort_values("popularity", ascending=False)[["title", "popularity"]]
                fb = fb.astype(object).where(fb.notna(), None)
                fallback_rows = fb.to_dict("records")

    result = {
        "week_start": str(week), "days_with_data": days_with_data,
        "model": model_rows, "fallback": fallback_rows,
    }

    with db.engine.begin() as conn:
        for r in model_rows:
            conn.execute(
                text("""INSERT INTO nowcast_history (week_start, title, p_number_one)
                        VALUES (:w, :t, :p)"""),
                {"w": week, "t": r["title"], "p": r["p_number_one"]},
            )

    return result
