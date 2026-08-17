"""
pipeline/sync_youtube.py — resolves each candidate's official trailer
(cached, including confirmed misses — see youtube.resolve_video_ids)
and snapshots its current view count. Lives in a background job like
sync_news_volume.py, for consistency: nowcast.py only ever reads
cached attention/news/trailer data, never makes a live external call
itself.
"""

from __future__ import annotations

from datetime import date

import youtube
from pipeline import build_candidates


def run(today: date | None = None) -> dict:
    today = today or date.today()
    candidates = build_candidates.load_current(today)
    if not candidates:
        return {"error": "No candidates found — run build_candidates first."}

    video_ids = youtube.resolve_video_ids(list(candidates.keys()))
    snap = youtube.snapshot_views(video_ids, day=today)
    return {"resolved": len(video_ids), "of_candidates": len(candidates), "snapshots": len(snap)}
