"""pipeline/ground_truth.py — downloads Netflix's official weekly Top 10, writes to Postgres."""

from __future__ import annotations

import re
from datetime import timedelta
from io import BytesIO

import pandas as pd
import requests
from sqlalchemy import text

import db

DATA_URL = "https://www.netflix.com/tudum/top10/data/all-weeks-countries.xlsx"
UA = {"User-Agent": "netflix-top10-model/1.0 (research use)"}


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")


def _is_united_states(country: str) -> bool:
    return str(country).strip().lower() in {"united states", "us", "usa"}


def _is_tv(category: str) -> bool:
    return str(category).strip().lower() == "tv"


def download_and_parse() -> pd.DataFrame:
    r = requests.get(DATA_URL, headers=UA, timeout=60)
    r.raise_for_status()
    df = pd.read_excel(BytesIO(r.content), engine="openpyxl")
    df.columns = [_norm_col(c) for c in df.columns]

    colmap = {}
    for c in df.columns:
        if c in {"country_name", "country"}:
            colmap["country"] = c
        elif c == "category":
            colmap["category"] = c
        elif c == "week":
            colmap["week"] = c
        elif c in {"rank", "weekly_rank"}:
            colmap["rank"] = c
        elif c in {"show_title", "title"}:
            colmap["show_title"] = c
        elif c == "season_title":
            colmap["season_title"] = c
        elif "cumulative_weeks" in c:
            colmap["cum_weeks_top10"] = c

    required = {"country", "category", "week", "rank", "show_title"}
    missing = required - set(colmap)
    if missing:
        raise ValueError(f"Expected columns not found: {missing}. Actual: {list(df.columns)}.")

    out = df.rename(columns={v: k for k, v in colmap.items()})
    keep = ["country", "category", "week", "rank", "show_title"]
    for opt in ("season_title", "cum_weeks_top10"):
        if opt in out.columns:
            keep.append(opt)
    out = out[keep].copy()

    mask = out["country"].map(_is_united_states) & out["category"].map(_is_tv)
    out = out[mask].copy()
    if out.empty:
        raise ValueError(f"Filter produced zero rows. Categories present: {sorted(set(df[colmap['category']].astype(str)))}")

    out["week_end"] = pd.to_datetime(out["week"]).dt.date
    out["week_start"] = out["week_end"].map(lambda d: d - timedelta(days=6))

    bad = [d for d in out["week_start"].unique() if d.weekday() != 0]
    if bad:
        raise ValueError(f"week_start not Monday for {len(bad)} week(s), e.g. {bad[0]}.")

    out["rank"] = out["rank"].astype(int)
    return out


def save_to_db(df: pd.DataFrame) -> int:
    with db.engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(
                text("""INSERT INTO ground_truth
                        (week_start, week_end, rank, show_title, season_title, cum_weeks_top10)
                        VALUES (:ws, :we, :r, :t, :st, :cw)
                        ON CONFLICT (week_start, show_title) DO UPDATE SET rank = :r"""),
                {
                    "ws": row["week_start"], "we": row["week_end"], "r": int(row["rank"]),
                    "t": row["show_title"],
                    "st": row.get("season_title"),
                    "cw": int(row["cum_weeks_top10"]) if pd.notna(row.get("cum_weeks_top10")) else None,
                },
            )
    return len(df)


def run() -> dict:
    df = download_and_parse()
    n = save_to_db(df)
    return {"rows_parsed": n, "weeks": int(df["week_start"].nunique()),
            "date_range": [str(df["week_start"].min()), str(df["week_start"].max())]}


def load_from_db() -> pd.DataFrame:
    with db.engine.begin() as conn:
        return pd.read_sql_query(text("SELECT * FROM ground_truth ORDER BY week_start, rank"), conn)
