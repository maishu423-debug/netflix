"""
attention.py — Wikipedia pageviews attention layer, Postgres-backed.

Same math as the Colab version (netflix_week, weekly_features are
byte-for-byte identical — pure pandas, no I/O). Only the caching layer
changed: sqlite-over-Drive -> Postgres, which also means the SQLite
write-lock workaround from the Colab version isn't needed here —
Postgres handles concurrent writes natively.
"""

from __future__ import annotations

import os
import time
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests
from sqlalchemy import text

import db

CONTACT = os.environ.get("WIKI_CONTACT", "netflix-top10-model (contact@example.com)")
UA = {"User-Agent": CONTACT}

PAGEVIEWS_ROOT = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article"
MEDIAWIKI_API = "https://en.wikipedia.org/w/api.php"

REQUEST_DELAY = 0.12
MAX_RETRIES = 4


def _get(url: str, params: dict | None = None) -> requests.Response | None:
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=20)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    return None


def load_overrides() -> dict[str, str]:
    with db.engine.begin() as conn:
        rows = conn.execute(text("SELECT title, article FROM overrides")).fetchall()
    return {r[0]: r[1] for r in rows}


def resolve_article(title: str, year: int | None = None, use_cache: bool = True) -> str | None:
    key = f"{title}|{year or ''}"
    if use_cache:
        with db.engine.begin() as conn:
            row = conn.execute(
                text("SELECT article FROM title_resolution WHERE title = :t"), {"t": title}
            ).fetchone()
        if row:
            return row[0]

    query = f"{title} TV series"
    if year:
        query += f" {year}"

    r = _get(MEDIAWIKI_API, {"action": "query", "list": "search", "srsearch": query,
                             "srlimit": 5, "format": "json"})
    article = None
    if r is not None:
        hits = r.json().get("query", {}).get("search", [])
        if hits:
            article = hits[0]["title"]

    if article:
        r2 = _get(MEDIAWIKI_API, {"action": "query", "titles": article, "redirects": 1, "format": "json"})
        if r2 is not None:
            pages = r2.json().get("query", {}).get("pages", {})
            for _, page in pages.items():
                if "missing" not in page:
                    article = page["title"]
                break

    source = "search" if article else "unresolved"
    with db.engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO title_resolution (title, article, source) VALUES (:t, :a, :s)
                    ON CONFLICT (title) DO UPDATE SET article = :a, source = :s"""),
            {"t": title, "a": article, "s": source},
        )
    time.sleep(REQUEST_DELAY)
    return article


def fetch_pageviews(article: str, start: date, end: date, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        with db.engine.begin() as conn:
            cached = pd.read_sql_query(
                text("SELECT day, views FROM pageviews WHERE article = :a AND day BETWEEN :s AND :e"),
                conn, params={"a": article, "s": start, "e": end},
            )
        expected = (end - start).days + 1
        if len(cached) == expected:
            cached["article"] = article
            cached["day"] = pd.to_datetime(cached["day"]).dt.date
            return cached[["day", "article", "views"]]

    safe = urllib.parse.quote(article.replace(" ", "_"), safe="")
    url = (f"{PAGEVIEWS_ROOT}/en.wikipedia/all-access/user/{safe}/daily/"
           f"{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}")
    r = _get(url)
    time.sleep(REQUEST_DELAY)

    if r is None:
        rows = []
    else:
        rows = [
            {"day": datetime.strptime(it["timestamp"], "%Y%m%d%H").date(), "views": it["views"]}
            for it in r.json().get("items", [])
        ]

    df = pd.DataFrame(rows, columns=["day", "views"])
    full = pd.DataFrame({"day": pd.date_range(start, end, freq="D").date})
    df = full.merge(df, on="day", how="left")
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    df["article"] = article

    with db.engine.begin() as conn:
        for d, v in zip(df["day"], df["views"]):
            conn.execute(
                text("""INSERT INTO pageviews (article, day, views) VALUES (:a, :d, :v)
                        ON CONFLICT (article, day) DO UPDATE SET views = :v"""),
                {"a": article, "d": d, "v": int(v)},
            )

    return df[["day", "article", "views"]]


def fetch_many(articles: Iterable[str], start: date, end: date, max_workers: int = 8) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    articles = [a for a in articles if a]
    if not articles:
        return pd.DataFrame()

    frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fetch_pageviews, a, start, end): a for a in articles}
        for fut in as_completed(futures):
            frames.append(fut.result())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def netflix_week(d: date) -> date:
    return d - timedelta(days=d.weekday())


BASELINE_WINDOW_DAYS = 90


def fetch_baselines(articles: Iterable[str], before: date,
                     window_days: int = BASELINE_WINDOW_DAYS, max_workers: int = 8) -> dict[str, float]:
    """
    Per-article typical daily views in the window strictly before the
    current tracking window starts — median, not mean, so a page that
    was genuinely hot for a week or two within that window doesn't drag
    its own baseline up much. Nets out traffic a page would have gotten
    anyway (an ongoing news story, a famous person's own fame — see
    MOURINHO resolving to Jose Mourinho's own Wikipedia article) from
    traffic actually driven by this week's Netflix performance. An
    article with no cached history before `before` gets 0.0 — nothing
    to net out yet, so its raw views count in full.
    """
    start = before - timedelta(days=window_days)
    end = before - timedelta(days=1)
    if end < start:
        return {}
    panel = fetch_many(articles, start, end, max_workers=max_workers)
    if panel.empty:
        return {}
    return panel.groupby("article")["views"].median().to_dict()


def apply_baseline(panel: pd.DataFrame, baselines: dict[str, float]) -> pd.DataFrame:
    """Nets each article's baseline (see fetch_baselines) out of its daily views, floored at 0."""
    if panel.empty:
        return panel
    df = panel.copy()
    baseline_col = df["article"].map(baselines).fillna(0.0)
    df["views"] = (df["views"] - baseline_col).clip(lower=0)
    return df


def reported_cutoff(panel: pd.DataFrame) -> date | None:
    """
    The same 'Wikipedia hasn't reported this day yet' detection
    weekly_features uses internally (see its docstring below), exposed
    separately so callers that baseline-adjust views before calling
    weekly_features can run it on RAW views first. Baseline-subtracted
    views can legitimately sum to 0 across every candidate on a
    genuinely quiet day (baseline is a median, so ~half of any day is
    below it for a given article) — that would fool the sum>0 check
    into treating a real, already-reported day as unreported.
    """
    if panel.empty:
        return None
    daily_totals = panel.groupby("day")["views"].sum()
    reported_days = daily_totals[daily_totals > 0].index
    return reported_days.max() if len(reported_days) else None


def weekly_features(panel: pd.DataFrame, as_of: date | None = None) -> pd.DataFrame:
    df = panel.copy()
    if df.empty:
        return df
    df["week"] = df["day"].map(netflix_week)
    if as_of is not None:
        df = df[df["day"] <= as_of]

    # Wikipedia's pageviews API lags 1-2 days behind real time. An
    # unreported day gets backfilled as 0 views by fetch_pageviews,
    # indistinguishable at a glance from a genuinely quiet day — which
    # meant live nowcasting's `momentum` always computed to exactly 0
    # (today's unreported row was always the "last day"). Detect the
    # true cutoff as the most recent day where at least one article in
    # the whole candidate set has nonzero views — with several titles
    # tracked at once, a day that's actually been reported will
    # essentially always show up as nonzero for someone. This has no
    # effect on completed historical weeks (training/backtest), where
    # every day already has real data by the time it's used.
    daily_totals = df.groupby("day")["views"].sum()
    reported_days = daily_totals[daily_totals > 0].index
    if len(reported_days):
        df = df[df["day"] <= reported_days.max()]

    g = df.groupby(["week", "article"])["views"]
    out = g.agg(total="sum", peak="max", n_days="count").reset_index()
    out["share"] = out["total"] / out.groupby("week")["total"].transform("sum")

    last = (df.sort_values("day").groupby(["week", "article"])["views"]
              .last().rename("last_day").reset_index())
    out = out.merge(last, on=["week", "article"], how="left")
    out["momentum"] = out["last_day"] / (out["total"] / out["n_days"]).replace(0, pd.NA)

    out = out.sort_values(["article", "week"])
    out["share_wow"] = out.groupby("article")["share"].diff()

    return out