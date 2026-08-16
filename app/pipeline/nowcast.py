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


def _daily_best_rank_map(week_start: date, today: date) -> dict:
    """
    Best (lowest) rank each title has hit so far this week in the
    manually-entered daily_top10 table (see sync_daily_top10.py) —
    display-only for now, not a trained model feature, since it can't be
    backfilled for past weeks to retrain against.
    """
    with db.engine.begin() as conn:
        rows = conn.execute(
            text("SELECT title, MIN(rank) AS best_rank FROM daily_top10 "
                 "WHERE day BETWEEN :ws AND :today GROUP BY title"),
            {"ws": week_start, "today": today},
        ).fetchall()
    return {r[0]: r[1] for r in rows}


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
    daily_best_rank = _daily_best_rank_map(week, today)

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
        feats["daily_best_rank_this_week"] = feats["title"].map(daily_best_rank)
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
                                  "share_wow", "previous_rank", "is_new",
                                  "daily_best_rank_this_week"]]
                # previous_rank is legitimately NaN for brand-new titles (no
                # prior week to have a rank in) — that's valid data, but
                # strict JSON (which FastAPI/Starlette enforce) doesn't allow
                # a literal NaN value, only null. Swap NaN -> None so it
                # serializes as JSON null instead of crashing the response.
                scored = scored.astype(object).where(scored.notna(), None)
                model_rows = scored.to_dict("records")
            except Exception:
                pass  # no usable trained model yet — fall through to the TMDb fallback below

    # A daily rank is already an ordering — usable as a fallback signal
    # immediately, unlike the trained model's coefficients, which need
    # weeks of history to fit. It beats TMDb's popularity fallback
    # whenever it's available: it's the real Netflix ranking, not a
    # generic popularity proxy.
    daily_rows = []
    if daily_best_rank:
        candidate_titles = set(candidates.keys())
        daily_rows = sorted(
            ({"title": t, "daily_best_rank_this_week": r}
             for t, r in daily_best_rank.items() if t in candidate_titles),
            key=lambda r: r["daily_best_rank_this_week"],
        )

    fallback_rows = []
    fallback_source = None
    if days_with_data < MIN_DAYS_FOR_MODEL or not model_rows:
        if daily_rows:
            fallback_rows = daily_rows
            fallback_source = "daily_manual"
        else:
            ids = tmdb.resolve_tmdb_ids(list(candidates.keys()))
            if ids:
                snap = tmdb.snapshot_popularity(ids)
                if not snap.empty:
                    fb = snap.sort_values("popularity", ascending=False)[["title", "popularity"]]
                    fb = fb.astype(object).where(fb.notna(), None)
                    fallback_rows = fb.to_dict("records")
                    if fallback_rows:
                        fallback_source = "tmdb_popularity"

    result = {
        "week_start": str(week), "days_with_data": days_with_data,
        "model": model_rows, "fallback": fallback_rows, "fallback_source": fallback_source,
    }

    with db.engine.begin() as conn:
        for r in model_rows:
            conn.execute(
                text("""INSERT INTO nowcast_history (week_start, title, p_number_one)
                        VALUES (:w, :t, :p)"""),
                {"w": week, "t": r["title"], "p": r["p_number_one"]},
            )

    return result
