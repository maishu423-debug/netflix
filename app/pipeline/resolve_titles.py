"""pipeline/resolve_titles.py — resolves show titles to Wikipedia articles."""

from __future__ import annotations

import attention
from pipeline import ground_truth


def run() -> dict:
    gt = ground_truth.load_from_db()
    titles = sorted(gt["show_title"].dropna().unique().tolist())
    overrides = attention.load_overrides()

    resolved, unresolved = 0, []
    for t in titles:
        if t in overrides:
            resolved += 1
            continue
        art = attention.resolve_article(t)
        if art:
            resolved += 1
        else:
            unresolved.append(t)

    return {"total": len(titles), "resolved": resolved, "unresolved": unresolved}
