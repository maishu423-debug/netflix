"""pipeline/resolve_titles.py — resolves show titles to Wikipedia articles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import attention
from pipeline import ground_truth


def run(max_workers: int = 8) -> dict:
    gt = ground_truth.load_from_db()
    titles = sorted(gt["show_title"].dropna().unique().tolist())
    overrides = attention.load_overrides()

    to_search = [t for t in titles if t not in overrides]
    resolved = len(titles) - len(to_search)  # overrides count as resolved immediately
    unresolved = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(attention.resolve_article, t): t for t in to_search}
        for fut in as_completed(futures):
            t = futures[fut]
            art = fut.result()
            if art:
                resolved += 1
            else:
                unresolved.append(t)

    return {"total": len(titles), "resolved": resolved, "unresolved": unresolved}
