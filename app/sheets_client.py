"""
sheets_client.py — appends our prediction + Kalshi's market price to a
Google Sheet, side by side, so you have a running log of both over time.

Requires three env vars:
    GOOGLE_SHEETS_ID            the spreadsheet ID from its URL
    GOOGLE_SERVICE_ACCOUNT_JSON the full service-account JSON, as a string
                                (paste the whole file contents into this
                                env var — don't point to a file path,
                                Railway's filesystem is ephemeral)

The service account's email needs Editor access on the sheet — share it
the same way you'd share with any collaborator.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import gspread
from google.oauth2.service_account import Credentials

SHEET_ID = os.environ.get("GOOGLE_SHEETS_ID")
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

HEADER = [
    "timestamp_utc", "week_start", "title",
    "our_p_number_one", "our_rank",
    "kalshi_yes_bid_cents", "kalshi_yes_ask_cents",
]

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not SHEET_ID or not SERVICE_ACCOUNT_JSON:
        raise EnvironmentError("GOOGLE_SHEETS_ID / GOOGLE_SERVICE_ACCOUNT_JSON not configured.")
    info = json.loads(SERVICE_ACCOUNT_JSON)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    _client = gspread.authorize(creds)
    return _client


def _get_worksheet():
    gc = _get_client()
    sh = gc.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet("predictions_log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="predictions_log", rows=1000, cols=len(HEADER))
        ws.append_row(HEADER)
    return ws


def log_snapshot(week_start, prediction_rows: list[dict], ladder_rows: list[dict]) -> None:
    """
    prediction_rows: our model's output, list of {title, p_number_one}, ranked.
    ladder_rows: Kalshi's ladder, list of {title, yes_bid, yes_ask}.
    Joins on title (best-effort — titles may not match exactly between
    our naming and Kalshi's; unmatched rows still get logged with blank
    Kalshi columns rather than being dropped).
    """
    ws = _get_worksheet()
    now = datetime.now(timezone.utc).isoformat()

    kalshi_by_title = {r["title"]: r for r in ladder_rows}

    rows_to_append = []
    for rank, pred in enumerate(prediction_rows, start=1):
        title = pred["title"]
        k = kalshi_by_title.get(title, {})
        rows_to_append.append([
            now, str(week_start), title,
            pred.get("p_number_one"), rank,
            k.get("yes_bid"), k.get("yes_ask"),
        ])

    if rows_to_append:
        ws.append_rows(rows_to_append, value_input_option="USER_ENTERED")


if __name__ == "__main__":
    # Smoke test: writes one fake row so you can confirm the sheet/creds work.
    log_snapshot(
        week_start="2026-08-03",
        prediction_rows=[{"title": "TEST ROW - safe to delete", "p_number_one": 0.5}],
        ladder_rows=[{"title": "TEST ROW - safe to delete", "yes_bid": 50, "yes_ask": 52}],
    )
    print("Wrote a test row to the predictions_log sheet.")
