# RUNBOOK — Operating NeuroSight AI

Day-to-day operating notes for the hosted/local system. Setup: [`README.md`](../README.md).
Hosting: [`DEPLOY.md`](DEPLOY.md).

## Environment

- Windows 11, RTX 4050 6 GB (training) — **serving is CPU-only and fine on any VM**
- Python 3.11+ (3.14 in use), Node 20+, `pip install -e ".[dev]"`
- `NUM_WORKERS=0` in configs — deliberate (Windows); do not raise without testing
- Secrets via env only: `NEUROSIGHT_API_KEY`, `NEUROSIGHT_CORS_ORIGINS` (comma-separated origins)

## Serving

| Task | Command |
|------|---------|
| Start backend | `uvicorn neurosight.api.main:create_app --factory --port 8000` |
| Start dashboard | `cd frontend && npm run dev` (or `npm run build` + any static host) |
| Health | `GET /health` → model version + device |
| Check what's deployed | `models/manifest.json` (members, temperature, metrics digest) |

- Startup runs a forward-probe: a broken/incompatible checkpoint **refuses to boot**
  rather than serving garbage.
- Uploads live in `data/uploads/<job_id>/` and are purged after 7 days (startup +
  daily sweep). Rejected uploads are deleted immediately.
- GPU contention: training and serving can share the 6 GB card, but expect slower
  epochs; for long trainings stop the backend first.

## Training (GPU)

1. **Data audit first** (hard gate): `python -m neurosight.data.audit configs`
   — CSV schema + 373/373 session match; refuses to continue below the floor.
2. **Preprocess cache**: `python scripts/preprocess.py` (2D; add `--include-3d` for 3D).
   Cache keys encode preprocessing params — a policy change is a miss, never stale data.
3. **Smoke before any long run**: `python scripts/train.py --model 3_cnn_lstm --debug`
   — minutes on CPU/GPU; catches config-passing and data errors early (this saved
   two overnight runs).
4. **Full ablation** (overnight): `python scripts/train.py --ablation`.
   2D variants ≈ 5–40 min/fold; 3D ≈ 30–60 min/fold. Early stopping on val AUC.
5. **Seed ensemble member**: `python scripts/train.py --model 7_dual_attn --mode final --seed 43`
   → checkpoint `…_s43.pth` (auto-suffixed, no collisions).
6. **Hyperparameter search**: `python scripts/search.py --model <key>` (bounded grid).
7. **Ensemble/TTA evaluation (CV only)**: `python scripts/excellence.py`.
8. **Held-out evaluation**: `scripts/bakeoff_e5.py` — the test set is touched once per
   declared bake-off. Do not iterate against it.

## Known gotchas (all learned the hard way)

- **Leakage shape**: any "too good" result (e.g. AUC 0.95+ here) is leakage until
  proven otherwise — the LOFO ensemble incident is documented in the plan (07).
- **Analyze pairs**: an `.img` without its `.hdr` twin is unreadable; the loader
  skips unpaired series (the dataset contains one orphan: `OAS2_0056_MR2/mpr-1`).
- **Cache coherence**: cache filenames encode slice/size/strategy; different settings
  coexist — delete `data/processed` only to force a full rebuild (~30–90 min).
- **best-checkpoint init**: `best_val_auc` starts at −1 so a single-class val epoch
  (AUC 0.0) still checkpoints.
- **Test flake (rare)**: `test_upload_and_full_job_lifecycle` can fail on background-
  polling timing under heavy parallel load; rerun in isolation before investigating.
- **Windows**: use junctions (`mklink /J`) not symlinks for `data/raw`; keep
  `num_workers=0`; paths are `pathlib` end-to-end.

## Retraining after the OASIS-3 decision (future work)

The port order is documented in `docs/plan/07-MODEL-EXCELLENCE-PLAN.md` §3 Tier 3:
audit adapter → loader adapter → pretrain → freeze-and-validate on ADNI. Current
code assumes OASIS-2 layout; the audit module is where a second dataset plugs in.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `422 incomplete pair` on upload | Include the `.hdr` for every `.img` (same base name) |
| Dashboard health dot red | Backend down/sleeping (Render free tier wakes ~30–60 s) |
| `401` in dashboard | API key mismatch — re-enter in the modal; check `NEUROSIGHT_API_KEY` |
| `429` | 10 analyses/min/IP cap — by design |
| CORS errors in browser console | Add the frontend origin to `NEUROSIGHT_CORS_ORIGINS` |
| OOM during 3D training | Keep batch 2 + grad-accum 4; stop the backend first |
