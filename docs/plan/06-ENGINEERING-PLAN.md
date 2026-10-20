# 06 — Engineering Plan

**Project:** NeuroSight AI · v1.0 · 2026-09-14
**Format:** tasks in dependency order with **tests** and **done criteria** (per the 6-Document Build System). Estimated effort assumes one engineer + AI agent on the dev box (Windows 11, RTX 4050 6 GB). Training wall-clock is called out separately from build time — the old plan's "30 min per phase" conflated them.

**Legend:** dep = prerequisite tasks. Every task ends with its DoD (definition of done). Where a task ports code from baseline implementations, the source file is named — port means: copy logic, fix the named defects, add types/tests, wire to config (no behavior re-invention).

---

## Milestone M0 — Foundations (est. 0.5 day + no training)

**T0.1 Repo scaffold** — dep: none.
Create repository scaffold exactly as `02-TECHNICAL-DESIGN.md` §3.1; `pyproject.toml` (package `neurosight`, src layout, deps per doc 02 §5); `requirements.lock` via pip-compile; `.gitignore` (data/, models/, mlruns/, uploads, .env, __pycache__, node_modules); `git init` + initial commit.
**DoD:** `pip install -e .` succeeds; `python -c "import neurosight"` works; `git log` has the scaffold commit.

**T0.2 Config system** — dep: T0.1.
Port Task-0.2 intent: `src/neurosight/config.py` with Pydantic models `DataConfig, ModelConfig, TrainingConfig, ServingConfig` + `load_config(config_dir)`. Add the fields the old plan missed: `series_selection: Literal["first","middle","mean"]="first"`, `input_mode`, `min_visits=2`, `deterministic=True`, `cudnn_benchmark=False`, `label_mode="binary_cdr"`.
Write `configs/data.yaml, model.yaml, training.yaml, serving.yaml` with real paths (`data/raw/OAS2_RAW_PART1|PART2`, `data/metadata/oasis_longitudinal_demographics.csv`).
**Tests:** invalid YAML → clear `ValidationError` naming the field; unknown key rejected; defaults stable.
**DoD:** `load_config("configs")` returns 4 typed objects; test green.

**T0.3 Logging, seeding, device utils** — dep: T0.1.
Port `logger/seed/io/device` (old plan Tasks 0.3): rotating file log `logs/neurosight.log` (10 MB × 3), format `[ts][level][module] msg`; `set_seed(seed)` sets python/numpy/torch/cuda + `torch.backends.cudnn.deterministic=True, benchmark=False` when `deterministic`; `get_device()` logs choice.
**Tests:** two `set_seed(42)` runs produce identical `torch.randn(3)`; log file created and rotated name pattern exists.
**DoD:** utilities imported by a smoke test; no `print()` in package (ruff rule T201 enforces).

**T0.4 Quality gates** — dep: T0.1.
`.pre-commit-config.yaml` (ruff check+format, black, end-of-file); `ruff.toml`/`pyproject` config (line 100, py311, isort); mypy on `src/neurosight/` (config-core strict, ML files relaxed); GitHub Actions `ci.yml`: job lint → job test (CPU, `pip install -e .[dev]`, pytest -q) → job frontend (npm ci, npm run build) — pytest job generates `tests/fixtures` mini-dataset in-repo (no dataset download in CI).
**DoD:** CI green on the scaffold commit; a deliberate lint error fails CI (verified once locally with `pre-commit run --all-files`).

## Milestone M1 — Data acquisition & integrity (est. 0.5 day) ⚠ gate for everything

**T1.1 Demographics CSV acquisition** — dep: T0.1.
Obtain the missing CSV: first try WUSTL OASIS-2 official download (bundled with imaging tar), else Kaggle mirror `oasis_longitudinal.csv` (jboysen). Save to `data/metadata/oasis_longitudinal_demographics.csv`; record source+date in `DATA.md`; write `data/metadata/README.md` noting column alias (`MRI ID` etc.), license (OASIS data-use agreement), citation (Marcus et al., 2010).
**DoD:** file exists with 373 data rows; DATA.md names provenance; no CSV content committed to git (gitignored) — checksum recorded in `DATA.md` instead.

**T1.2 CSV schema validator** — dep: T1.1.
`src/neurosight/data/audit.py::validate_csv()` — required columns (alias-tolerant), dtypes, 373 rows, Group distribution within tolerance (Nondemented 72 ± 5, Demented 34 ± 4, Converted 31 ± 4 — from known OASIS-2 composition), Visit ≥ 1, CDR ∈ {0, 0.5, 1, 2} ∪ NaN.
Reference distributions (validated 2026-09-14 against the real `Datasets/oasis_longitudinal.csv`): 373 rows; row-level Group counts Nondemented 190 / Demented 146 / Converted 37; subject-level 72 / 64 / 14; CDR ∈ {0, 0.5, 1, 2}; MMSE NaN = 2, SES NaN = 19; CSV MRI IDs match disk session folders 373↔373 exactly.
**Tests:** fixture CSVs (correct; renamed columns; missing column; wrong row count; corrupted CDR) each assert expected outcome.
**DoD:** validator rejects all corrupt fixtures with specific messages, accepts the real CSV.

**T1.3 Session↔CSV audit** — dep: T1.2.
`audit.py::audit_raw()` — walk `OAS2_RAW_PART1|PART2`, enumerate `OAS2_XXXX_MRY/RAW/mpr-*.img(+.hdr)` (ignore `OLD/`, `.log`, `.ima.gz`; count the known 1 unpaired `.img`), match to CSV `MRI ID` rows, emit `reports/data_audit.json`: `{csv_rows, sessions_on_disk, matched, missing_scan, extra_sessions, series_per_session_hist, unpaired_files}`; hard-fail if matched sessions < 360 (documented floor).
**Tests:** run against real disk in a `@pytest.mark.local` (skipped in CI); CI runs it against a synthetic tree fixture.
**DoD:** `data_audit.json` written; ≥ 360/373 matched; numbers printed; audit file committed.

**T1.4 Loader port + subject-level leakage-safe splits** — dep: T0.2, T1.3.
Port `data_loader.py` → `neurosight/data/loader.py` with the fixes from `00-PLAN-REVIEW.md`: config-driven paths; **subject-level split before** Converted pre/post sequence splitting; `label_mode="binary_cdr"` (keep `per_visit`/`majority` options); `series_selection` policy implemented (`first` = lowest `mpr-N`; `middle`; `mean` = voxel-mean stack); metadata vector unchanged + per-dim mask vector for input modes; no silent skip — every exclusion logged with reason and counted.
**Tests:** (a) **leakage test** — assert zero subject-id overlap across train/val/test **and** across CV folds, including Converted straddle case built from a synthetic 6-subject CSV; (b) series policy returns expected path per fixture tree; (c) metadata mask zeros the right dims per mode; (d) label derivation `CDR==0→0 else 1` matches reference table.
**DoD:** loader builds N sequences from real data (expected range 120–260 incl. Converted splits — record actual); all tests green; printed class distribution committed to `reports/data_audit.json`.

## Milestone M2 — Preprocessing & datasets (est. 1 day)

**T2.1 Preprocessing port (fail-loud)** — dep: T0.3.
Port `src/preprocessing.py` → `neurosight/data/preprocessing.py`. Defect fix: **no exception swallowing** — corrupted volume raises `VolumeLoadError`; `fix_shape` gets an explicit orientation assertion against hdr dims instead of the heuristic transpose when hdr is readable (fallback: heuristic + warning). Determinism: `random`/numpy RNG seeded per-call from `(seed, subject, visit)`.
**Tests:** synthetic volume (64×64×32) → expected output shapes for 2D `(32,128,128)` and 3D `(1,64,128,128)`; z-score mean≈0/std≈1; slice strategies (middle/uniform/entropy) on a known ramp volume; augment reproducibility under fixed seed; corrupted input raises (not zeros).
**DoD:** tests green; docstring notes the old zero-tensor bug is dead (no `except: zeros` anywhere — grep gate in CI: `assert "np.zeros((" not in preprocessing.py` for the except paths).

**T2.2 Datasets, collate, sampler, cache** — dep: T2.1, T1.4.
Port `src/dataset.py` → `neurosight/data/dataset.py` (2D/3D/inference datasets, collate with masks, weighted sampler); add `cache.py`: per-session `.npz` under `data/processed/<subject>/<mri_id>_<series>.npz` + `manifest.json` (shape, checksum) replacing the 1.08 GB monolith; cache hits logged.
**Tests:** collate on lengths [2,3,1] → scans `(3,3,32,128,128)`, mask correct, seq lens `[2,3,1]`; weighted sampler distribution within 5% of inverse-frequency; cache write→read roundtrip equal tensors; cache miss rebuilds.
**DoD:** `preprocess --all` completes over real data (est. 30–90 min wall-clock) and cache manifest has one entry per matched session from `data_audit.json`.

## Milestone M3 — Models (est. 1 day)

**T3.1 Model registry port** — dep: T0.2.
Port model files from `src/models/` into `neurosight/models/` unchanged in math; add `registry.py` (11-variant `ABLATION_VARIANTS` from `src/ablation.py`, incl. ResNet18-LSTM) + `build_model(key, ModelConfig)`; every forward keeps `meta` arg (mode-masked upstream).
**Tests:** forward-shape test per variant (2D: `(2,3,32,128,128)→(2,2)`; 3D variants with their shapes); param counts within ±2% of reference values (530K–1.4M customs, ~11M ResNet-LSTM); masked-LSTM variants produce no NaN with zero-padded batch; registry rejects unknown keys.
**DoD:** all 11 variants instantiate and forward on CPU; test marks total param counts into a committed fixture JSON.

**T3.2 Heads: Status + Conversion** — dep: T3.1.
`neurosight/models/heads.py`: `StatusHead` (masked metadata MLP fusion when enabled — port of MetadataFusionHead concept with **mode mask**) and `ConversionHead` (shares encoder, independent 2-logit head, trained on pre-conversion/non-demented sequences only). Wire into `NeuroSightModel` wrapper exposing `forward(scans, meta, mask) → status_logits, conversion_logits|None`.
**Tests:** mode-mask zeroing (demo dims vs clinical dims); conversion head returns None in `mri_only` single-head config; gradient flows to both heads in joint mode; loss masks conversion term for post-conversion samples.
**DoD:** wrapper forwards all 11 variants via registry; tests green.

## Milestone M4 — Training & experiment tracking (est. 1 day build + training wall-clock)

**T4.1 Losses & trainer port** — dep: T3.2, T0.3.
Port `src/train.py` (`FocalLoss`, `Trainer`, weighted CE) with additions: MLflow per-run logging (params, per-epoch metrics, confusion matrix artifact), determinism flags from config, early stop + model selection on **val AUC** (not accuracy), checkpoint save best+last to `models/`.
**Tests:** 2-epoch overfit on a 6-sample synthetic fixture reaches ≥ 99% train acc (sanity: the loop can learn); focal loss ≥ 0 and matches a hand-computed value on a 2×2 case; checkpoint roundtrip (save→load→identical val acc); MLflow run created with expected param keys (local mark).
**DoD:** `train --model 3_cnn_lstm --debug` (5 patients, 2 epochs) runs end-to-end < 5 min on the dev box.

**T4.2 CV + ablation driver** — dep: T4.1.
Port `src/train_cv.py` + `src/ablation.py` → `neurosight/training/cv.py`, `scripts` entry `train --ablation`; run on the **fixed subject-level split**; per variant × selected mode: CV on train+val (5-fold), then train final on train+val, hold out test; writes `reports/csv/ablation_results.csv`, `reports/ablation_results.json` (metrics incl. CIs), and MLflow parent run.
**Tests:** driver in `--fast` mode (4 variants, 1 fold, fixture data) completes and writes well-formed CSV/JSON with all metric keys; champion-selection function picks max val-AUC with documented tie-break.
**DoD:** full ablation (11 variants × 5 folds, `mri_only`) kicked off overnight on GPU — est. 8–14 h wall-clock (2D ~3–5 min/fold/variant, 3D ~15–25 min/fold/variant); no manual intervention; results files exist next morning.

**T4.3 Bounded hyperparameter search** — dep: T4.2.
Port `src/hyperparameter_search.py`; restrict grid to champion family only: lr {1e-4,3e-4} × dropout {0.3,0.5} × lstm_hidden {32,64} × slices {24,32} = 16 configs × 3 folds (bounded, est. 4–8 h). Winner written to `configs/model.yaml` overrides + MLflow.
**Tests:** search on fixture (2 configs × 2 folds) completes; results CSV well-formed; best config applied to config file by the `apply-best` subcommand.
**DoD:** search artifact + updated config committed.

## Milestone M5 — Evaluation, calibration, explainability (est. 1 day + wall-clock)

**T5.1 Evaluator port + CIs** — dep: T4.2.
Port `src/evaluate.py` (metrics, bootstrap CI, McNemar, plots, report/CSV writers) into `neurosight/evaluation/evaluator.py`; every metrics dict gains `input_mode` + `n_test` fields; add balanced accuracy.
**Tests:** metrics on hand-built label/prob vectors match sklearn references to 1e-6; CI bounds ordered lo ≤ mean ≤ hi; McNemar on a known 2×2 gives documented chi2; report/CSV writers produce files with all keys.
**DoD:** `evaluate --from-run <champion>` produces `reports/final_evaluation.txt`, `csv/final_metrics.csv`, ROC/PR/calibration/confusion PNGs — all with real values.

**T5.2 Calibration** — dep: T5.1.
`neurosight/evaluation/calibration.py`: fit temperature on val logits (LBFGS on NLL), report ECE (15-bin) before/after; write temperature into `models/manifest.json`.
**Tests:** temperature recovery on synthetic overconfident logits (recovers within 0.05); ECE decreases after scaling on fixture; manifest written.
**DoD:** champion ECE-after ≤ 0.10 or documented exception; calibration curve PNG saved.

**T5.3 Grad-CAM + attention viz port** — dep: T5.1.
Port `src/gradcam.py` (hooks conv3, overlays, grids, saliency) + build `attention_viz.py` (per-patient visit-weight bars; population chart by class) using the existing `TemporalAttention` weights.
**Tests:** Grad-CAM on fixture model returns heatmaps `(T,S,128,128)` in [0,1]; hook cleanup (no lingering handles); overlay function on known arrays; attention viz writes PNG in tmp dir.
**DoD:** `explain --subject OAS2_0001 --model <champion>` writes overlay grids + attention PNGs to `reports/figures/gradcam/`.

**T5.4 Conversion-head evaluation (T2)** — dep: T4.2 (joint training), T5.1.
Evaluate `ConversionHead` on pre-conversion sequences (held-out test): AUC-ROC, AUC-PR, n explicitly stated; output section in final evaluation labelled "exploratory — n small". **Feasibility note (measured):** only 4 of 14 Converted subjects have ≥2 pre-conversion visits; the conversion task therefore uses `min_visits=1` for pre-conversion sequences (positives = 14 subjects; 4 multi-visit), negatives = matched Nondemented subjects. n=14 is below any claim threshold — report as exploratory with the caveat, never as a headline.
**Tests:** with synthetic conversion fixtures, evaluator computes AP correctly; when n < 5, output carries the caveat flag and no point-estimate claim.
**DoD:** T2 metrics present in `final_evaluation.txt` with caveat, or documented "not available (n)" if sequences insufficient.

**T5.5 Baselines & leakage quantification** — dep: T1.4, T5.1.
(a) `demo_only` logistic regression (sklearn, same splits) → reference table; (b) champion rerun in `mri+clinical` mode (single CV, no test claim beyond report labeling); (c) results table comparing `mri_only` vs `demo_only` vs `mri+clinical` with the leakage sentence pre-written in the report template.
**DoD:** comparison table committed; report template renders it.

## Milestone M6 — Serving plane (est. 1.5 days)

**T6.1 Predictor + export** — dep: T5.2, T5.3.
Port predictor concept into `neurosight/inference/predictor.py` per doc 05 §6 (manifest, TorchScript-first, mode-masked metadata, temperature, trend labels, conversion field, Grad-CAM on slices [10,16,24]); `export.py` writes `models/export/<model>_ts.pt` + `manifest.json` (git_sha, data_audit_sha, metrics digest).
**Tests:** TorchScript module output equals eager output (atol 1e-5) on fixture; manifest probe fails loudly on missing checkpoint; trend thresholds unit-tested (0.04→Stable, 0.10→Declining, 0.20→Rapidly Declining).
**DoD:** `predict --study tests/fixtures/study_mr1_mr2` returns a valid StudyResult JSON < 10 s CPU.

**T6.2 FastAPI app + jobs + store** — dep: T6.1.
Build per doc 05: routes, SQLModel store, worker, rate limit, API key, CORS, purge sweeper, structured logging (no request bodies). OpenAPI schema frozen into `backend/openapi.json` artifact.
**Tests (TestClient):** health 200; 401 without key; 422 unpaired-file upload; 413 oversize; upload→queued→(mock predictor)→succeeded result JSON schema-valid; cancel during queued; 404 unknown id; purge sweeper removes expired dir + logs PurgeLog row.
**DoD:** full API test suite green; `uvicorn backend.main:app` boots and `/health` returns the manifest model version.

**T6.3 SSE endpoint (optional but in-scope)** — dep: T6.2.
`/api/v1/analyses/{id}/events` streaming stage events per doc 05 §5.
**Tests:** subscribe → trigger stage change with mocked worker → event received; connection closes on terminal state.
**DoD:** curl smoke shows `job.stage_changed` frames.

## Milestone M7 — Frontend (est. 2 days)

**T7.1 App shell** — dep: T0.4.
Vite React-TS scaffold per docs 02/04: Tailwind + shadcn/ui init, tokens (`04` §2), sidebar layout, router (`/analyze /patient /about`), API client (axios + `X-API-Key` interceptor + ApiKeyModal), react-query setup, types generated from `backend/openapi.json`.
**DoD:** shell runs; API key modal gates calls; `npm run build` green in CI.

**T7.2 Analyze wizard** — dep: T7.1, T6.2.
Steps 1–3 per doc 03 §2 exactly: FileDropZone with visit chips + client-side pairing pre-check, demographics panel with input-mode banner + experimental toggle, Run → job poll → results sections A–G, low-confidence and single-visit states, Download JSON.
**Tests:** vitest component tests — visit grouping from fixture filenames; mode switch banner appears when demographic filled; polling mock transitions queued→succeeded; disclaimer always rendered with results.
**DoD:** manual E2E against local backend with a real 3-visit OASIS study succeeds < 60 s; all doc-03 states demonstrable.

**T7.3 Patient history + About** — dep: T7.2.
Per doc 03 §3–4: timeline chart, visit cards, progression alert thresholds (Δ>0.15), export CSV, delete with confirm; About renders `/api/v1/model-card`.
**Tests:** component tests for alert threshold logic and empty/error states; model-card page renders metrics digest.
**DoD:** two saved analyses for one subject produce a correct timeline; alert fires on crafted history.

**T7.4 Accessibility & polish pass** — dep: T7.2, T7.3.
Axe scan (0 critical), focus-visible audit, `aria-live` on processing, tab-trap in dialogs, contrast check of badges vs `04` §7.
**DoD:** axe report clean; keyboard-only walkthrough of all three routes recorded in PR description.

## Milestone M8 — Deployment, docs, trust artifacts (est. 1.5 days)

**T8.1 Docker & compose** — dep: T6.2, T7.1.
`Dockerfile.backend` (slim + libgomp1, non-root, HEALTHCHECK), `Dockerfile.frontend` (node build → nginx, `/api` proxy), `docker-compose.yml` (healthchecks, volumes `data/ models/`, optional mlflow profile), `.env.example`.
**DoD:** `docker compose up --build` serves the working app on localhost; healthchecks pass; image sizes documented.

**T8.2 DATA.md + MODEL_CARD.md** — dep: T1.1, T5.2, T5.5.
DATA.md: provenance, license, citation, checksums, audit numbers. MODEL_CARD.md (CHAI-informed): intended use & out-of-scope, data, input modes + leakage note, metrics table with CIs from champion run, calibration, known limitations (n_test≈22, single-scanner, OASIS-only), ethical considerations, citation.
**Tests:** `model_card.py` fills metrics from `final_metrics.csv` (no hand-typed numbers); unit test asserts table values match the CSV.
**DoD:** both docs committed; About page renders them.

**T8.3 Final report + presentation regeneration** — dep: T5.5, T8.2.
Port `generate_report.py` / `generate_pptx.py` → `neurosight/reporting/`; replace hardcoded placeholder tables with reads from `reports/csv/*.json`; include: ablation table (11 variants, CIs), CV ± std, leakage comparison, calibration, Grad-CAM figures, conversion-head section, honest benchmark paragraph (rigorous OASIS-2 band 60–86%; cite OASIS-1 99% papers as different task), TRIPOD+AI/CLAIM alignment checklist appendix.
**DoD:** `python -m neurosight.reporting.all` produces `reports/final_report.pdf` + `presentation.pptx` with **zero `--` placeholders** (grep gate in the generator); figures match run artifacts.

**T8.4 README + runbook + handoff** — dep: all.
README: architecture diagram, setup (native Windows + Docker), train/evaluate/serve commands, dataset placement, metrics table (auto-included from T8.2), disclaimer. `docs/RUNBOOK.md`: overnight ablation procedure, GPU watch, re-training after config change, adding a variant, troubleshooting (OOM, Windows workers, junction issues).
**DoD:** fresh-clone walkthrough on the dev box: clone → `pip install -e .` → place CSV → `preprocess` → `train --debug` → `uvicorn` → upload in UI succeeds.

## Dependency graph (waves)

```
Wave 0: T0.1 T0.2 T0.3 T0.4
Wave 1: T1.1 T1.2 T1.3            (data acquisition/audit)
Wave 2: T1.4  T2.1
Wave 3: T2.2  T3.1
Wave 4: T3.2  T4.1
Wave 5: T4.2                      (overnight training wall-clock)
Wave 6: T4.3 ‖ T5.1 → T5.2 → T5.3 → T5.4 → T5.5
Wave 7: T6.1 → T6.2 → T6.3 ‖ T7.1 → T7.2 → T7.3 → T7.4
Wave 8: T8.1 ‖ T8.2 ‖ T8.3 → T8.4
```

**Total estimated build effort:** ~9–10 working days + ~1.5 days of unattended training wall-clock (overnight runs). Critical path: data audit → loader → trainer → ablation night → evaluation → serving → UI.

## Pre-registered analysis decisions (anti-p-hacking contract)

1. Champion = highest val AUC across the ablation grid; ties → lower ECE, then fewer params.
2. Test set is evaluated **once** per champion, after temperature fitting; exploratory models (T2, mri+clinical) never re-define the champion.
3. All reported accuracies carry bootstrap 95% CIs; CV mean±std accompanies every table.
4. Negative results are reported verbatim (if ResNet18 transfer underperforms custom CNN, say so — the old data suggests small models win here).
5. No metric is reported without its `input_mode` label.
