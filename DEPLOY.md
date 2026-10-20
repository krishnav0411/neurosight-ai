# DEPLOY.md — Hosting NeuroSight AI (no Docker)

Two pieces, hosted separately:

| Piece | Host | Why |
|-------|------|-----|
| Dashboard (React/Vite) | **Vercel** | static build, perfect fit, free |
| API (FastAPI + worker) | **Render** (or Fly.io / HF Spaces) | needs a *persistent* process: background worker thread + SQLite + model on disk — serverless (Vercel functions) cannot host it |

The E5 serving bundle committed to git is ~32 MB (5 checkpoints + manifest), so a
normal GitHub push carries everything. No GPU needed to serve — CPU inference is
seconds per visit.

---

## 0. One-time: push this repo to GitHub

```bash
cd neurosight-ai
git remote add origin https://github.com/<you>/neurosight-ai.git
git push -u origin main
```

## 1. Backend on Render

1. Render dashboard → **New → Web Service** → connect the GitHub repo.
2. Settings:
   - **Runtime:** Python 3
   - **Build command:** `pip install -e .`
   - **Start command:** `uvicorn neurosight.api.main:create_app --factory --host 0.0.0.0 --port $PORT`
3. Environment variables:
   - `NEUROSIGHT_API_KEY` = a long random string (the demo key for the dashboard)
   - `NEUROSIGHT_CORS_ORIGINS` = `https://<your-project>.vercel.app` (comma-separated origins parsed by API)
   - `NEUROSIGHT_LOG_DIR` = `/tmp/logs` (Render's disk outside the repo is fine)
4. Health check path: `/health`. Deploy.

Notes: Render's free tier sleeps after inactivity (first request then takes ~30–60 s
to wake — the dashboard's health dot shows this). The SQLite DB persists per
instance but resets on redeploy — acceptable for a demo; use a Render disk (paid)
if you need durable history.

## 2. Dashboard on Vercel

1. Vercel dashboard → **Add New → Project** → import the repository.
2. Configuration options:
   - **Recommended:** In Vercel Project Settings, set **Root Directory** to `frontend`. Framework preset is Vite (auto-detected `npm run build` / `dist`). SPA routing rewrites are configured in `frontend/vercel.json`.
   - **Monorepo root fallback:** If deploying from repository root without setting the Root Directory in dashboard, the root `vercel.json` runs `cd frontend && npm ci && npm run build`, outputs `frontend/dist`, and rewrites to `/index.html`.
3. Environment variables:
   - `VITE_API_BASE` = `https://<your-backend>.onrender.com`
4. Deploy → note the `*.vercel.app` URL → put it in the backend's
   `NEUROSIGHT_CORS_ORIGINS` (step 1.3) and redeploy the backend.

## 3. Verify

1. Open the Vercel URL → enter the API key (the `NEUROSIGHT_API_KEY` value).
2. Health dot top-right → green; model chip shows `neurosight-e5@…[mri_only]`.
3. Upload one visit's `mpr-N.nifti.img + .hdr` (or a `.nii`) → Run analysis →
   results render with the disclaimer. Check **Patient history** saves.

## Local (native, no Docker) — unchanged

```bash
# backend
set NEUROSIGHT_API_KEY=dev-key            # Windows CMD
uvicorn neurosight.api.main:create_app --factory --port 8000
# dashboard
cd frontend && npm run dev                # proxies /api to :8000
```

## Updating the model later

`scripts/bakeoff_e5.py` regenerates `models/manifest.json` from the 5 member
checkpoints; commit any changed checkpoint + manifest and both hosts redeploy on
push. To deploy a *different* model, update `models/manifest.json` (relative member
names) and make sure the named files are committed.
