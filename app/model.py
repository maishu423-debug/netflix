"""
model.py — Plackett-Luce ranking model for "which title is #1 this week".

Why Plackett-Luce and not a plain classifier: the label isn't independent
per title, it's a competition — probabilities across candidates in the
same week must sum to 1, and we observe a partial ORDER (top 10), not
just a winner. PL is the standard likelihood for exactly this: repeatedly
peel off the top remaining item under a softmax over "strength" scores.

score_i = w . x_i   (linear in the attention/lag features)

Likelihood for one week, true order (i_1 rank1, ..., i_K rank10) out of
candidate set C:

    P = prod_{k=1..K}  exp(s_ik) / sum_{j in R_k} exp(s_j)

    R_k = C minus {i_1, ..., i_{k-1}}   (still-active candidates,
          including ones that never crack the top 10 — they're only ever
          in a denominator, never a numerator, which is exactly the
          right treatment of "was in the running, didn't win this slot")

NLL and its gradient are summed over weeks and minimized with L-BFGS.
Predicting: P(title is #1 | week's candidates) = softmax(scores), the
first-step PL probability — that's the number you want for "who tops
the chart".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

DEFAULT_FEATURES = ["share", "momentum", "share_wow", "log_previous_rank", "is_new_num"]


def _prep_week_matrix(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, list[int]]:
    """
    df: rows for ONE week, one per candidate, with a `rank` column
    (1..10 or NaN). Returns (X, order) where order lists row-indices
    into X sorted by known rank ascending (only the ranked ones).
    """
    X = df[feature_cols].to_numpy(dtype=float)
    ranked = df[df["rank"].notna()].sort_values("rank")
    order = [df.index.get_loc(i) for i in ranked.index]
    return X, order


def _week_nll_grad(X: np.ndarray, order: list[int], w: np.ndarray) -> tuple[float, np.ndarray]:
    s = X @ w
    remaining = list(range(len(s)))
    nll = 0.0
    grad = np.zeros_like(w)
    for i_k in order:
        s_active = s[remaining]
        lse = logsumexp(s_active)
        nll += -(s[i_k] - lse)
        probs = np.exp(s_active - lse)
        grad += -(X[i_k] - probs @ X[remaining])
        remaining.remove(i_k)
    return nll, grad


@dataclass
class PlackettLuceRanker:
    feature_cols: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    l2: float = 1.0
    w_: np.ndarray | None = None
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return (X - self.mean_) / self.std_

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, table: pd.DataFrame) -> "PlackettLuceRanker":
        """
        table: full model table from features.py, one row per
        (week_start, title), with a `rank` column (NaN allowed).
        Rows with any NaN feature are dropped week-locally (kept in
        the denominator is impossible without a value, so we drop them
        entirely — better to impute upstream in features.py if you can).
        """
        weeks_data = []
        skipped_no_rank, skipped_too_few, skipped_all_nan = 0, 0, 0
        for wk, wdf in table.groupby("week_start"):
            wdf_clean = wdf.dropna(subset=self.feature_cols).reset_index(drop=True)
            if wdf["rank"].notna().sum() == 0:
                skipped_no_rank += 1
                continue
            if len(wdf_clean) < 2:
                if len(wdf) >= 2:
                    skipped_all_nan += 1
                else:
                    skipped_too_few += 1
                continue
            weeks_data.append(wdf_clean)

        if not weeks_data:
            raise ValueError(
                f"No usable training weeks after filtering (checked "
                f"{skipped_no_rank + skipped_too_few + skipped_all_nan} weeks): "
                f"{skipped_no_rank} had no known rank at all, "
                f"{skipped_too_few} had <2 candidates, "
                f"{skipped_all_nan} had enough candidates but all had a NaN "
                f"feature (check for a bug like share_wow needing multi-week "
                f"context — see 04_build_features.py). Nothing to fit on."
            )

        all_X = np.concatenate([wdf[self.feature_cols].to_numpy(dtype=float) for wdf in weeks_data], axis=0)
        Xs = self._standardize_fit(all_X)

        # Re-split standardized rows back per week without recomputing.
        splits, start = [], 0
        for wdf in weeks_data:
            n = len(wdf)
            splits.append((Xs[start:start + n], wdf))
            start += n

        def obj(w: np.ndarray):
            total_nll, total_grad = self.l2 * np.sum(w ** 2), self.l2 * 2 * w
            for X, wdf in splits:
                order = [wdf.index.get_loc(i) for i in wdf[wdf["rank"].notna()].sort_values("rank").index]
                nll, grad = _week_nll_grad(X, order, w)
                total_nll += nll
                total_grad += grad
            return total_nll, total_grad

        w0 = np.zeros(len(self.feature_cols))
        res = minimize(obj, w0, jac=True, method="L-BFGS-B")
        self.w_ = res.x
        return self

    def predict_proba(self, week_df: pd.DataFrame) -> pd.Series:
        """P(#1) for each candidate in a single week's DataFrame."""
        X = week_df[self.feature_cols].astype(float).to_numpy()
        Xs = self._standardize_apply(X)
        s = Xs @ self.w_
        p = np.exp(s - logsumexp(s))
        return pd.Series(p, index=week_df.index)

    def coef_table(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.feature_cols, "weight": self.w_}).sort_values(
            "weight", ascending=False
        )

    def save_to_db(self, l2: float) -> None:
        import json
        from sqlalchemy import text
        import db
        with db.engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO model_weights (feature_cols, weights, mean_, std_, l2)
                        VALUES (:fc, :w, :m, :s, :l2)"""),
                {
                    "fc": json.dumps(self.feature_cols),
                    "w": json.dumps(self.w_.tolist()),
                    "m": json.dumps(self.mean_.tolist()),
                    "s": json.dumps(self.std_.tolist()),
                    "l2": l2,
                },
            )

    @classmethod
    def load_latest_from_db(cls) -> "PlackettLuceRanker":
        import numpy as np
        from sqlalchemy import text
        import db
        with db.engine.begin() as conn:
            row = conn.execute(
                text("SELECT feature_cols, weights, mean_, std_, l2 FROM model_weights "
                     "ORDER BY trained_at DESC LIMIT 1")
            ).fetchone()
        if row is None:
            raise ValueError("No trained model found in model_weights table yet.")
        # JSONB columns come back already deserialized (list/dict), not as
        # raw JSON strings — no json.loads() needed here.
        m = cls(feature_cols=row[0], l2=row[4])
        m.w_ = np.array(row[1])
        m.mean_ = np.array(row[2])
        m.std_ = np.array(row[3])
        return m


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """log_previous_rank / is_new_num — call before fit/predict."""
    df = df.copy()
    df["log_previous_rank"] = np.log(df["previous_rank"].fillna(20))
    df["is_new_num"] = df["is_new"].astype(float)
    return df
