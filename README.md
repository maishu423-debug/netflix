# Netflix #1 predictor — live dashboard

Kalshi's market ladder for the current week's Netflix TV ranking market,
next to this project's own Plackett-Luce model prediction, with manual
weekly/monthly refresh buttons and automatic scheduled runs. Everything
persists in Postgres; logs a snapshot to Google Sheets on trigger + daily.

## Deploy to Railway

1. **Push this folder to a GitHub repo.**
2. **In Railway**: New Project → Deploy from GitHub repo → select it.
3. **Add the Postgres plugin** to the same project (Railway → New → Database
   → PostgreSQL). It auto-injects `DATABASE_URL` into your web service — no
   manual copying needed, just make sure the plugin is attached to this
   service in Railway's project graph.
4. **Set the remaining environment variables** (Settings → Variables on the
   web service) — see `.env.example` for the full list and format notes,
   especially for `KALSHI_PRIVATE_KEY` and `GOOGLE_SERVICE_ACCOUNT_JSON`,
   which need to be pasted as their full raw contents (not file paths —
   Railway's filesystem doesn't persist between deploys).
5. **Deploy.** Railway auto-detects Python via `requirements.txt` and uses
   the `Procfile`'s start command.
6. **First run**: the app starts with an empty database. Open the deployed
   URL and click **Update Weekly**, then **Update Monthly** — this
   populates ground truth, resolves titles, builds candidates, gets a
   nowcast, then backfills pageviews and trains the model. After that,
   the scheduled jobs keep it current automatically (see below).

## What runs when

| Trigger | Runs | Default schedule |
|---|---|---|
| "Update Weekly" button | ground_truth → resolve_titles → build_candidates → nowcast → logs to Sheet | Also auto-runs **Tuesdays 6am UTC** |
| "Update Monthly" button | download_attention → build_features → train_model | Also auto-runs **1st of month, 6am UTC** |
| (automatic, no button) | nowcast + Kalshi snapshot → logs to Sheet | **Daily, 1pm UTC** |

Adjust the cron schedules in `app/main.py`'s `scheduler.add_job(...)` calls
if these times don't suit your timezone or the actual Netflix chart
publish schedule.

## Verifying it actually works

The Kalshi client (`app/kalshi_client.py`) was built against Kalshi's
documented API schema, but **wasn't tested against live data** during
development — the environment it was built in couldn't reach Kalshi's
API directly. First real check: after deploying, open `/api/market`
directly in a browser and confirm the ladder rows look sane (real show
titles, prices between 0-100). If field names come back empty/wrong,
Kalshi's actual response schema differs slightly from docs — check the
raw JSON structure and adjust `_cents()`/`_num()` in `kalshi_client.py`
accordingly (they already handle two known schema variants, but a third
is possible).

## Local development

```bash
cd app
pip install -r ../requirements.txt
export DATABASE_URL="postgresql://localhost/netflix_dev"  # a local Postgres
export WIKI_CONTACT="..." TMDB_READ_TOKEN="..." KALSHI_KEY_ID="..." KALSHI_PRIVATE_KEY="..."
export GOOGLE_SHEETS_ID="..." GOOGLE_SERVICE_ACCOUNT_JSON='...'
uvicorn main:app --reload
```

## Architecture notes

- **One persistent web service**, not separate cron jobs — the live UI,
  scheduler, and manual triggers all share one process, avoiding the
  "is the DB even reachable from this cron run" class of problems.
- **Postgres replaces** the Colab version's Drive-backed parquet/sqlite/
  json files entirely. Schema is in `app/db.py`.
- **The Kalshi ladder auto-discovers** the current week's market by
  querying `series_ticker=KXNETFLIXRANKSHOW&status=open` rather than
  hardcoding a week-specific ticker — no date math needed on our side.
- **Manual triggers and scheduled jobs call the same functions**
  (`run_weekly_job`/`run_monthly_job` in `main.py`), so there's exactly
  one code path to maintain, not two.
