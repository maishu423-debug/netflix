"""
kalshi_client.py — Kalshi REST API client.

Market data (events, markets, orderbook) is PUBLIC — no auth needed.
Confirmed via docs.kalshi.com: "Kalshi provides several public endpoints
that don't require API keys." Only trading (orders/positions/balance)
needs the signed RSA-PSS headers. We still implement signing (below)
since KALSHI_KEY_ID / KALSHI_PRIVATE_KEY are configured — costs nothing
to have it working, and it's needed if this ever reads authenticated
endpoints (e.g. higher rate limits, or private-account data later).

BASE_URL is the official production API per docs.kalshi.com's own
curl examples (docs.kalshi.com/api-reference/*, docs.kalshi.com/getting_started/quick_start_market_data).

AUTO-DISCOVERY: rather than hardcoding a week-specific market ticker
(e.g. 'KXNETFLIXRANKSHOW-26AUG10', which changes every week), we query
by series_ticker + status=open. Kalshi always has exactly one open
event per week for a weekly-recurring series, so this naturally always
finds "this week's market" with no date math on our side.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
SERIES_TICKER = os.environ.get("KALSHI_SERIES_TICKER", "KXNETFLIXRANKSHOW")

KEY_ID = os.environ.get("KALSHI_KEY_ID")
PRIVATE_KEY_PEM = os.environ.get("KALSHI_PRIVATE_KEY")  # full PEM contents, not a file path


def _load_private_key():
    if not PRIVATE_KEY_PEM:
        return None
    return serialization.load_pem_private_key(PRIVATE_KEY_PEM.encode(), password=None)


_PRIVATE_KEY = None  # lazy-loaded, only needed for authenticated endpoints


def _signed_headers(method: str, path: str) -> dict:
    """
    path must be the request path only (e.g. '/trade-api/v2/markets'),
    no query string, no host. Not used for the public endpoints below,
    kept here for when authenticated endpoints are needed.
    """
    global _PRIVATE_KEY
    if _PRIVATE_KEY is None:
        _PRIVATE_KEY = _load_private_key()
    if _PRIVATE_KEY is None or not KEY_ID:
        raise EnvironmentError("KALSHI_KEY_ID / KALSHI_PRIVATE_KEY not configured.")

    timestamp_ms = str(int(time.time() * 1000))
    message = f"{timestamp_ms}{method.upper()}{path}".encode()
    signature = _PRIVATE_KEY.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=hashes.SHA256().digest_size),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


@dataclass
class LadderRow:
    ticker: str
    title: str
    yes_bid: float | None  # cents, 1-99 = implied probability
    yes_ask: float | None
    volume: float | None


@dataclass
class MarketLadder:
    event_ticker: str
    headline: str
    subtitle: str | None
    rows: list[LadderRow]


def get_current_ladder(series_ticker: str = SERIES_TICKER) -> MarketLadder | None:
    """
    Auto-discovers this week's open event for the series and returns its
    full ladder (one row per candidate title). Returns None if nothing's
    currently open (e.g. between settlement and the next week's market
    being listed).
    """
    with httpx.Client(timeout=15) as client:
        markets_resp = client.get(
            f"{BASE_URL}/markets",
            params={"series_ticker": series_ticker, "status": "open", "limit": 100},
        )
        markets_resp.raise_for_status()
        markets = markets_resp.json().get("markets", [])
        if not markets:
            return None

        event_ticker = markets[0]["event_ticker"]

        event_resp = client.get(f"{BASE_URL}/events/{event_ticker}")
        event_resp.raise_for_status()
        event = event_resp.json().get("event", {})

        rows = [
            LadderRow(
                ticker=m["ticker"],
                title=m.get("yes_sub_title") or m.get("title") or m["ticker"],
                yes_bid=_cents(m, "yes_bid"),
                yes_ask=_cents(m, "yes_ask"),
                volume=_num(m, "volume"),
            )
            for m in markets
            if m["event_ticker"] == event_ticker
        ]
        rows.sort(key=lambda r: (r.yes_bid or 0), reverse=True)

        return MarketLadder(
            event_ticker=event_ticker,
            headline=event.get("title", event_ticker),
            subtitle=event.get("sub_title"),
            rows=rows,
        )


def _cents(market: dict, prefix: str) -> float | None:
    """Handles both '<prefix>' (integer cents) and '<prefix>_dollars' (string) schema variants."""
    if f"{prefix}_dollars" in market and market[f"{prefix}_dollars"] is not None:
        return float(market[f"{prefix}_dollars"]) * 100
    if prefix in market and market[prefix] is not None:
        return float(market[prefix])
    return None


def _num(market: dict, prefix: str) -> float | None:
    for key in (prefix, f"{prefix}_fp"):
        if key in market and market[key] is not None:
            return float(market[key])
    return None


if __name__ == "__main__":
    ladder = get_current_ladder()
    if ladder is None:
        print("No open market found for this series right now.")
    else:
        print(f"{ladder.headline}\n({ladder.event_ticker})\n")
        for r in ladder.rows:
            print(f"  {r.title:<40} YES {r.yes_bid:>5.0f}c  vol {r.volume}")
