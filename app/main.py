"""
main.py — the dashboard's backend.

Runs as ONE persistent Railway web service (not separate cron jobs) so
the live UI, the scheduler, and the manual trigger buttons all share
the same process and the same DB connection pool.
"""

from __future__ import annotations

import asyncio
import traceback
from datetime import date, datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text

import db
import kalshi_client
import kalshi_ws
import sheets_client
from pipeline import build_candidates, build_features, download_attention
from pipeline import ground_truth, nowcast, resolve_titles, train_model

app = FastAPI()


def _job_start(job_name: str) -> int:
    with db.engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO job_runs (job_name, status) VALUES (:n, 'running') RETURNING id"),
            {"n": job_name},
        )
        return result.scalar()


def _job_finish(job_id: int, status: str, log: str) -> None:
    with db.engine.begin() as conn:
        conn.execute(
            text("UPDATE job_runs SET status = :s, log = :l, finished_at = now() WHERE id = :id"),
            {"s": status, "l": log[:8000], "id": job_id},
        )


def _run_weekly_chain() -> str:
    log = []
    log.append(f"ground_truth: {ground_truth.run()}")
    log.append(f"resolve_titles: {resolve_titles.run()}")
    log.append(f"build_candidates: {build_candidates.run()}")
    result = nowcast.run()
    log.append(f"nowcast: {result.get('week_start')}, "
               f"{len(result.get('model', []))} model rows, "
               f"{len(result.get('fallback', []))} fallback rows")

    try:
        ladder = kalshi_client.get_current_ladder()
        ladder_rows = [{"title": r.title, "yes_bid": r.yes_bid, "yes_ask": r.yes_ask} for r in ladder.rows] if ladder else []
        prediction_rows = result.get("model") or result.get("fallback") or []
        sheets_client.log_snapshot(result["week_start"], prediction_rows, ladder_rows)
        log.append("Logged snapshot to Google Sheet.")
    except Exception as e:
        log.append(f"Sheet logging failed (non-fatal): {e}")

    return "\n".join(log)


def _run_monthly_chain() -> str:
    log = []
    log.append(f"download_attention: {download_attention.run()}")
    log.append(f"build_features: {build_features.run()}")
    log.append(f"train_model: {train_model.run()}")
    return "\n".join(log)


def run_weekly_job() -> None:
    job_id = _job_start("weekly")
    try:
        log = _run_weekly_chain()
        _job_finish(job_id, "success", log)
    except Exception:
        _job_finish(job_id, "failed", traceback.format_exc())


def run_monthly_job() -> None:
    job_id = _job_start("monthly")
    try:
        log = _run_monthly_chain()
        _job_finish(job_id, "success", log)
    except Exception:
        _job_finish(job_id, "failed", traceback.format_exc())


def run_daily_snapshot_job() -> None:
    """
    Refreshes candidates + logs prediction/Kalshi to the sheet, once a day.
    Refreshing candidates here (not just in the weekly job) matters: a
    title premiering mid-week would otherwise stay invisible to
    /api/prediction until the next Tuesday auto-run — exactly the kind
    of same-week breakout this dashboard exists to catch.
    """
    job_id = _job_start("daily_snapshot")
    try:
        cand_result = build_candidates.run()
        result = nowcast.run()
        ladder = kalshi_client.get_current_ladder()
        ladder_rows = [{"title": r.title, "yes_bid": r.yes_bid, "yes_ask": r.yes_ask} for r in ladder.rows] if ladder else []
        prediction_rows = result.get("model") or result.get("fallback") or []
        sheets_client.log_snapshot(result["week_start"], prediction_rows, ladder_rows)
        _job_finish(job_id, "success",
                    f"candidates: {cand_result}\nLogged {len(prediction_rows)} rows.")
    except Exception:
        _job_finish(job_id, "failed", traceback.format_exc())


# ---- scheduler ----
scheduler = BackgroundScheduler(timezone="UTC")
scheduler.add_job(run_weekly_job, CronTrigger(day_of_week="tue", hour=6, minute=0), id="weekly")
scheduler.add_job(run_monthly_job, CronTrigger(day=1, hour=6, minute=0), id="monthly")
scheduler.add_job(run_daily_snapshot_job, CronTrigger(hour=13, minute=0), id="daily_snapshot")


@app.on_event("startup")
async def startup():
    db.init_schema()
    scheduler.start()
    asyncio.create_task(kalshi_ws.run_forever())


@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown(wait=False)


# ---- API ----

@app.get("/api/market")
def api_market():
    return kalshi_ws.get_market_snapshot()


@app.get("/api/prediction")
def api_prediction():
    return nowcast.run()


@app.post("/api/trigger/weekly")
def trigger_weekly(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_weekly_job)
    return {"status": "started"}


@app.post("/api/trigger/monthly")
def trigger_monthly(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_monthly_job)
    return {"status": "started"}


@app.get("/api/jobs/latest")
def latest_jobs():
    with db.engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT ON (job_name) job_name, status, started_at, finished_at, log
            FROM job_runs ORDER BY job_name, started_at DESC
        """)).fetchall()
    return {r[0]: {"status": r[1], "started_at": str(r[2]), "finished_at": str(r[3]), "log": r[4]} for r in rows}


app.mount("/", StaticFiles(directory="static", html=True), name="static")