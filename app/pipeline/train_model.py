"""pipeline/train_model.py — fits the Plackett-Luce model on all history, saves to Postgres."""

from __future__ import annotations

from model import PlackettLuceRanker, add_derived_features, DEFAULT_FEATURES
from pipeline import build_features


def run(l2: float = 1.0) -> dict:
    table = build_features.build_table()
    if table.empty:
        return {"error": "No feature table to train on — run build_features first."}

    table = add_derived_features(table)
    model = PlackettLuceRanker(feature_cols=DEFAULT_FEATURES, l2=l2).fit(table)
    model.save_to_db(l2=l2)

    return {
        "trained_on_rows": len(table),
        "weights": dict(zip(model.feature_cols, model.w_.tolist())),
    }
