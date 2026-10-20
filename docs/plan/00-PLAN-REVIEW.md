# Plan Review — Is `neurosight_new_plan.md` Perfect?

**Verdict: No.** The plan is a strong skeleton (good stack choices, correct patient-level split instinct, sensible phases), but it contains **3 fatal defects that will break the project on day 1**, **5 scientific defects that would produce inflated or unreproducible results**, and **12 engineering gaps** that fall short of the industry bar it promises. It also throws away ~2,000 lines of working, rigorous code that already exists in this repo.

**Recommendation:** do not execute `neurosight_new_plan.md` as written. Use the replacement 6-document set in this folder (`01-PRD.md` … `06-ENGINEERING-PLAN.md`), which follows the 6-Document Build System (PRD → Technical Design → App Flow → UI/UX Brief → Backend Design → Engineering Plan) at the depth required for health-related products.

---

## What the plan gets right (keep these)

- Pydantic + YAML config, logging, seed util, "no hardcoded paths" principle.
- Patient-level (never scan-level) train/val/test splits.
- Spatial CNN → temporal encoder → fusion head decomposition with factories.
- FastAPI + React/TS/Tailwind + Docker + MLflow + pytest stack.
- Grad-CAM and temporal-attention explainability tasks.
- Focal loss / class weights for imbalance; ECE calibration metric.

## Defect log

### A. Fatal — project cannot start or will silently break

| # | Defect | Evidence in repo |
|---|--------|------------------|
| A1 | **Wrong dataset paths.** Plan hardcodes `D:/AI Project/Datasets/OASIS/Raw/OAS2_Raw_1/` and `OAS2_Raw_2/`. Actual disk layout is `D:\AI Project\Datasets\OAS2_RAW_PART1\` (209 sessions, 82 subjects) and `OAS2_RAW_PART2\` (164 sessions, 68 subjects). | `ls Datasets` |
| A2 | **The demographics CSV does not exist.** Every code generation (`data_loader.py` line 8, `app.py`, `utils.py`) reads `oasis_longitudinal_demographics.csv`; the file is nowhere under `Datasets/`. The plan has no acquisition task. Verified available: WUSTL OASIS-2 official download (bundled CSV) or Kaggle mirror `oasis_longitudinal.csv` (jboysen). | `find Datasets -name "*.csv"` → 0 results |
| A3 | **Multi-series-per-session ignored.** Session folders contain up to 4 series (`mpr-1..4.nifti.img/.hdr`), plus `OLD/` legacy files, `.log` files and one stray `.ima.gz`. The plan specifies "all .img files per session" with no selection policy; the current code silently takes `vp[0]` — an arbitrary, undocumented choice that changes results. | e.g. `OAS2_0100_MR1/RAW/` has 4 mpr pairs |

### B. Scientific — results would be inflated, unreproducible, or untrustworthy

| # | Defect | Why it matters |
|---|--------|----------------|
| B1 | **Label leakage via MMSE/CDR in the fused metadata.** MMSE and CDR *define* the dementia label (label derivation: `CDR == 0 → 0 else 1`). Feeding MMSE into the model and reporting accuracy makes it a cognitive-score classifier, not an MRI classifier. The plan fuses MMSE with no MRI-only control. | `data_loader.py:88`, plan Task 1.3/2.3 |
| B2 | **3-class "Converted" head is the wrong labeling design.** ~31/150 subjects are Converted; a 3-class softmax with `final_label = label at last visit` conflates current state with trajectory, starves class 2, and breaks the calibration story. The existing, better design (binary + Converted split into pre/post sequences at the CDR boundary) is discarded by the plan. | `data_loader.py:98-121` |
| B3 | **Stated goals 3–5 have zero tasks.** The master context promises progression detection, future-decline probability, and region attribution — but no phase builds a conversion/progression model. Only a "progression_accuracy" metric appears in evaluation. This is the single biggest gap between promise and plan. | Plan phases 0–8 |
| B4 | **Evaluation rigor regression.** The plan drops the 5-fold stratified CV, bootstrap 95% CIs, McNemar's test, and Brier score that the July code already implements, in favor of a single train/val/test pass. | `train_cv.py`, `evaluate.py` exist and work |
| B5 | **Fabricated baseline.** The plan's "previous project summary" claims ~81% train / ~85% val. Actual artifacts show 71% (68-sample test), 72% (32-sample), 74.8/71.9 train/val. The 78% and 86% numbers come from an unlogged ChatGPT session on a 14-sample test. Every target in the plan inherits this fiction. Literature check: rigorous OASIS-2 longitudinal results run ~60–86%; the 96–99% papers are OASIS-1 cross-sectional or leaky setups. | `report/evaluation_report.txt`, `src/results/*`, ChatGPT JSON msgs 120/276 |
| B6 | **No data-integrity gate.** `dataset.py` swallows every load exception and returns a zero tensor (`except Exception: scans.append(np.zeros(...))`) — silent garbage trains silently. The plan states "no silent failures" as a principle but has no task enforcing it. | `dataset.py:54-55, 101-102` |

### C. Engineering — below the industry bar the plan claims

| # | Defect |
|---|--------|
| C1 | Full rewrite discards working assets: 11-variant ablation registry, CV trainer, hyperparameter search, Grad-CAM/saliency, ensemble, 3D baselines, PDF/PPTX report generators. Migration > rewrite. |
| C2 | No package structure (`src` not importable, flat imports, `sys.path.insert` hacks), no pyproject `src` layout. |
| C3 | No CI (lint/type/test gates), no pre-commit, no requirements lock. |
| C4 | FastAPI with no auth, no upload validation (extension/size/hdr-pair), synchronous 10–30 s inference in request handlers, no job model for long analyses. |
| C5 | No PHI/privacy stance despite uploading patient MRI + demographics (retention, no-logging, local-only). |
| C6 | No model export story (TorchScript/ONNX), no model-version pinning in the API responses/DB. |
| C7 | Tests are thin: no split-leakage test, no training smoke test, no fixture-based CI tests. |
| C8 | No determinism spec (cudnn flags, per-split seeds); no data versioning; no DATA.md / model card / CLAIM- or TRIPOD+AI-aligned reporting for a medical AI project. |
| C9 | Frontend spec has no loading/error/empty states, no accessibility requirements. |
| C10 | Docker story ignores the actual dev machine (Windows + RTX 4050 6 GB — existing configs already tuned batch sizes for it). |
| C11 | The final report generator is never re-run with real numbers — `report/final_report.pdf` currently ships with `--` in every results cell. |
| C12 | Time estimates (30 min–1.5 hr/phase) are fantasy for training runs; no acceptance criteria per task. |

---

## What changed in the replacement plan

1. **Evolve, don't rewrite.** New clean repo (`D:\NeuroSight-AI\`) with a proper package; port the ablation registry, CV trainer, evaluation suite, Grad-CAM, and report generators from `D:\AI Project\src\` with tests, rather than re-inventing them.
2. **M0 starts with data acquisition & integrity**: obtain the CSV (Kaggle/WUSTL), schema-validate it, audit raw sessions vs CSV, deterministic mpr-series selection policy, fail-loud preprocessing.
3. **Labeling redesigned**: primary binary task (CDR-derived, Group-verified, Converted split pre/post) + a second **conversion-risk head** trained on pre-conversion sequences — this actually delivers the plan's promised questions 3 & 4.
4. **Leakage-controlled input modes**: `mri_only` (headline), `demo_only`, `mri+demo`, `mri+clinical` (reported separately, never the headline). Every result table carries the mode.
5. **Honest targets with CIs**: MRI-only held-out test acc ≥ 75% (bootstrap 95% CI), AUC ≥ 0.80, ECE ≤ 0.10 after temperature scaling; 5-fold CV ± std always reported; McNemar for model comparisons; literature framed correctly.
6. **Production hardening**: FastAPI background-job inference, API-key auth, upload validation, retention policy, model-version pinning, TorchScript export, CI, pre-commit, DVC-optional data versioning, MODEL_CARD.md, DATA.md, CLAIM/TRIPOD+AI-aligned final report regenerated with real numbers.
7. **Engineering Plan** (doc 06) gives every task: dependencies, tests, and explicit done criteria — replacing the 30-minute fantasy estimates.
