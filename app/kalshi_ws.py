"""
kalshi_ws.py — real-time Kalshi market ladder via their public `ticker`
WebSocket channel, replacing REST polling for /api/market.

Docs (docs.kalshi.com's asyncapi.yaml + WebSocket guide) establish:
  - Endpoint: wss://external-api.kalshi.com/trade-api/ws/v2 (same host
    as the REST client, ws scheme)
  - The WS connection itself is authenticated via the same signed-header
    scheme as REST (KALSHI-ACCESS-KEY/-TIMESTAMP/-SIGNATURE), signed over
    method=GET, path=/trade-api/ws/v2 (the WS path, NOT a REST path, no
    query string) — kalshi_client._signed_headers() already implements
    this exact scheme, reused here as-is
  - Once connected, public channels (ticker, trade, orderbook_delta) need
    no additional per-channel auth
  - The `ticker` channel pushes an update whenever any ticker field
    changes for a subscribed market — exactly what a live ladder needs,
    no polling interval required

CAVEAT: built from Kalshi's public docs, NOT verified against a live
connection — this environment can't open outbound websockets to Kalshi.
The REST client had the same caveat initially and turned out correct on
the first real deploy, but if the ticker payload's field names don't
match what _parse_ticker_msg() assumes, `_state["last_raw_message"]` is
kept around specifically so you can log/inspect it and adjust parsing.

ARCHITECTURE: one persistent background task (started at FastAPI
startup, see main.py) keeps an in-memory dict updated as ticker messages
arrive. /api/market just reads that dict directly — no REST call, no
polling latency, on every page load. If the WS goes quiet for longer
than STALE_FALLBACK_SECONDS (e.g. a parsing mismatch silently
swallowing every message), a REST refresh kicks in as a safety net so
the ladder doesn't go stale forever while that gets diagnosed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import websockets

import kalshi_client

logger = logging.getLogger("kalshi_ws")

WS_URL = "wss://external-api.kalshi.com/trade-api/ws/v2"
WS_PATH = "/trade-api/ws/v2"

RECONNECT_BASE_DELAY = 2
RECONNECT_MAX_DELAY = 60
FULL_REFRESH_INTERVAL_SECONDS = 600  # re-discover event/markets periodically (catches weekly rollover)
STALE_FALLBACK_SECONDS = 120

_state = {
    "headline": None,
    "subtitle": None,
    "event_ticker": None,
    "rows_by_ticker": {},  # ticker -> {title, yes_bid, yes_ask, volume}
    "last_update": 0.0,
    "last_raw_message": None,
}


def get_market_snapshot() -> dict:
    """What /api/market reads — always in-memory, never blocks on network."""
    if _state["event_ticker"] is None:
        return {"error": "No open market found yet."}
    return {
        "headline": _state["headline"],
        "subtitle": _state["subtitle"],
        "event_ticker": _state["event_ticker"],
        "last_update_age_seconds": round(time.time() - _state["last_update"], 1),
        "rows": sorted(_state["rows_by_ticker"].values(),
                        key=lambda r: (r.get("yes_bid") or 0), reverse=True),
    }


def _refresh_from_rest() -> list[str]:
    """(Re)seeds _state from a REST snapshot. Returns the market tickers to subscribe to."""
    ladder = kalshi_client.get_current_ladder()
    if ladder is None:
        _state["event_ticker"] = None
        _state["rows_by_ticker"] = {}
        return []

    _state["headline"] = ladder.headline
    _state["subtitle"] = ladder.subtitle
    _state["event_ticker"] = ladder.event_ticker
    _state["rows_by_ticker"] = {
        r.ticker: {"title": r.title, "yes_bid": r.yes_bid, "yes_ask": r.yes_ask, "volume": r.volume}
        for r in ladder.rows
    }
    _state["last_update"] = time.time()
    return [r.ticker for r in ladder.rows]


def _extract_cents(msg: dict, prefix: str) -> float | None:
    for key in (f"{prefix}_dollars", prefix):
        if key in msg and msg[key] is not None:
            v = float(msg[key])
            return v * 100 if key.endswith("_dollars") else v
    return None


def _parse_ticker_msg(data: dict) -> dict | None:
    """
    Defensive parsing — tries the documented envelope shape
    ({"type": "ticker", "msg": {...}}) and falls back to treating the
    whole payload as the message if that shape doesn't match.
    """
    msg = data.get("msg", data)
    if not isinstance(msg, dict):
        return None
    ticker = msg.get("market_ticker") or msg.get("ticker")
    if not ticker:
        return None
    return {
        "ticker": ticker,
        "yes_bid": _extract_cents(msg, "yes_bid"),
        "yes_ask": _extract_cents(msg, "yes_ask"),
        "volume": msg.get("volume") or msg.get("volume_fp"),
    }


async def _listen_once(market_tickers: list[str]) -> None:
    headers = kalshi_client._signed_headers("GET", WS_PATH)
    async with websockets.connect(WS_URL, additional_headers=headers, ping_interval=None) as ws:
        sub = {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["ticker"], "market_tickers": market_tickers},
        }
        await ws.send(json.dumps(sub))
        logger.info(f"Kalshi WS: subscribed to ticker for {len(market_tickers)} markets.")

        async for raw in ws:
            _state["last_raw_message"] = raw
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue

            parsed = _parse_ticker_msg(data)
            if parsed is None:
                continue

            row = _state["rows_by_ticker"].get(parsed["ticker"])
            if row is None:
                continue  # a market outside our current display scope
            if parsed["yes_bid"] is not None:
                row["yes_bid"] = parsed["yes_bid"]
            if parsed["yes_ask"] is not None:
                row["yes_ask"] = parsed["yes_ask"]
            if parsed["volume"] is not None:
                row["volume"] = parsed["volume"]
            _state["last_update"] = time.time()


async def run_forever() -> None:
    """The background task started at FastAPI startup (main.py). Never returns."""
    delay = RECONNECT_BASE_DELAY
    last_full_refresh = 0.0

    while True:
        try:
            need_refresh = (time.time() - last_full_refresh > FULL_REFRESH_INTERVAL_SECONDS
                             or _state["event_ticker"] is None)
            if need_refresh:
                market_tickers = await asyncio.to_thread(_refresh_from_rest)
                last_full_refresh = time.time()
            else:
                market_tickers = list(_state["rows_by_ticker"].keys())

            if not market_tickers:
                await asyncio.sleep(30)  # no open market right now — check again soon
                continue

            await _listen_once(market_tickers)
            delay = RECONNECT_BASE_DELAY  # reset backoff after a clean connection

        except Exception:
            logger.exception("Kalshi WS connection dropped, reconnecting...")
            # Safety net: if too long has passed without any update (e.g. a
            # parsing mismatch silently eating every message), force a REST
            # refresh so the ladder doesn't go stale forever while that
            # gets diagnosed.
            if time.time() - _state["last_update"] > STALE_FALLBACK_SECONDS:
                try:
                    await asyncio.to_thread(_refresh_from_rest)
                except Exception:
                    logger.exception("REST fallback refresh also failed.")
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)