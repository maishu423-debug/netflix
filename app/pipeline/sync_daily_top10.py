"""
pipeline/sync_daily_top10.py — pulls the manually-maintained daily Top 10
(typed in from the Netflix app's own Top 10 shelf, which updates daily;
Netflix's official bulk export is weekly-only) from the
'daily_top10_manual' Google Sheet into Postgres.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import text

import db
import sheets_client


def run() -> dict:
    entries = sheets_client.read_daily_top10()
    n_ok, skipped = 0, 0
    with db.engine.begin() as conn:
        for e in entries:
            try:
                day = datetime.strptime(str(e["date"]).strip(), "%Y-%m-%d").date()
                rank = int(e["rank"])
                title = str(e["title"]).strip()
            except (KeyError, ValueError):
                skipped += 1
                continue
            if not title:
                skipped += 1
                continue
            conn.execute(
                text("""INSERT INTO daily_top10 (day, rank, title) VALUES (:d, :r, :t)
                        ON CONFLICT (day, rank) DO UPDATE SET title = :t"""),
                {"d": day, "r": rank, "t": title},
            )
            n_ok += 1
    return {"rows_synced": n_ok, "rows_skipped": skipped}
