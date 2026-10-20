# 07 — Model Excellence Plan v2: NeuroSight-E5 (detailed executable)

**Date:** 2026-09-15 · v2 supersedes v1's workstream sketch with task-level detail · **Scope ruling (user):** no new datasets — OASIS-1/ADNI deferred; overnight GPU approved.

---

## 0. Baseline (measured) and target

| | Current (champion v1 `7_dual_attn`) |
|---|---|
| Mean CV AUC | 0.744 ± 0.096 (folds: .772/.636/.868/.817/.629) |
| Held-out test (n=22, touched once) | acc 59.1% [36.4–77.3], AUC 0.785, F1 0.571 |
| Calibration | val ECE 0.118→0.041 (T=0.375); **test ECE 0.30–0.35 — poor** |

**Target system "NeuroSight-E5":** 5-seed ensemble of `7_dual_attn` (seeds 42–46, final-model training) + TTA (identity, h-flip, slice-shift ±2) + temperature fitted per fold on val + split-conformal prediction sets (90% coverage). Success: **mean CV AUC ≥ 0.78** (outside v1's ±0.096 or McNemar p<0.05), **val ECE ≤ 0.10**, conformal coverage ∈ [0.85, 0.95] — then ONE bake-off on the untouched test set. Literature grounding: deep ensembles are strongest in low-data regimes ([Lakshminarayanan 2017](https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles)); TTA +0.5–3 pts with medically-plausible transforms only; conformal sets = clinical deployment standard ([CHEST 2025](https://journal.chestnet.org/article/S0012-3692(25)05184-0/fulltext), [PMC review](https://pmc.ncbi.nlm.nih.gov/articles/PMC9309105/)); temperature-ensemble interaction must be measured ([Rahaman NeurIPS 2021](https://dl.acm.org/doi/10.5555/3540261.3541796)).

## 1. Guardrails (unchanged, binding)

1. Test set touched exactly once more — the declared final bake-off. All selection on CV.
2. Champion/system selection: max mean CV AUC; tie → val ECE → params.
3. Every system is a written hypothesis; gains must hold on ≥3/5 folds or McNemar p<0.05.
4. Fold hygiene: when evaluating fold *i*'s val set, fold *i*'s own model is EXCLUDED from any ensemble (it trained on that data). Enforced in code + tested.
5. Calibration/confconformal fitted per-fold on val only.

## 2. Systems to evaluate (CV protocol, all offline)

| ID | System | Training cost |
|----|--------|---------------|
| S0 | champion v1 single model (baseline, exists) | — |
| S1 | **Same-fold CV baselines** — S1a: champion variant per-fold CV (fold *i* scored ONLY by its own fold-*i* checkpoint); S1b: same for `8_transformer` | 0 (55 checkpoints on disk) |
| S2 | **Same-fold cross-variant ensemble**: mean softmax of the S1 members (fold-*i* checkpoints of champion + `8_transformer`; 2D-only ruling, 3D member deferred) | 0 |
| S3 | S2 + **TTA** (identity, h-flip, shift±2; mean-softmax aggregation) | 0 |
| S4 | **E5**: 5-seed final-model ensemble + TTA + temperature + conformal | 4 new trainings (~2.5 h GPU) |

Plus: per-fold temperature (fit on val ensemble probs) and split-conformal sets on top of the winning system; report raw-ensemble vs temperature-scaled ECE both (Rahaman caveat).

## 3. Engineering tasks (TDD, new package `src/neurosight/excellence/`)

- **T1 Same-fold ensemble core** — `excellence/ensemble.py`:
  - `collect_fold_checkpoints(model_key, models_dir, input_mode, run_suffix) -> dict[fold, Path]` — exact `best_{key}_cv{N}_{key}_{mode}[_{suffix}].pth` match per fold; never a wildcard, so folds/seeds can never be mixed.
  - `samefold_fold_predictions(...)` — fold *i* is scored ONLY by its own fold-*i* checkpoint (in k-fold CV, fold *i*'s val subjects sit inside every other fold's training set, so any cross-fold member would score training data; cross-fold ensembling is banned).
  - `average_variant_probs` + `samefold_cross_variant` — mean of aligned per-fold probs across variants (y_true identity asserted per fold).
  - Tests with stub models writing tiny state dicts to tmp.
- **T2 TTA** — `excellence/tta.py`: `tta_batches(scans) -> list[Tensor]` (identity, `flip(-1)`, roll ±2 on slice axis); `tta_probs(model, scans, meta, mask) -> np.ndarray` (mean softmax across passes). Tests: 4 passes emitted; h-flip of a left-right symmetric volume yields identical probs (sanity); mean aggregation correct on stub outputs.
- **T3 Conformal** — `excellence/conformal.py`: `conformal_threshold(n, coverage=0.9)` (corrected rank `k = min(ceil((n+1)·coverage), n)`; the level is `k/n`, always ≤ 1 — the raw `ceil((n+1)·c)/n` could exceed 1, e.g. n=6, c=0.9 → 7/6), `fit_conformal(val_probs, val_true, coverage)` (nonconformity = 1 − p(true class); threshold = k-th smallest score), `prediction_set(p_demented, threshold) -> set[str]` ⊆ {"Non-Demented","Demented"}, and `split_half_coverage(...)` — even-index rows calibrate, odd-index rows are scored, so coverage is NEVER measured on the rows the threshold was fit on (calibration-set coverage is optimistically biased). Tests: threshold/rank math on known n (incl. the clamp); coverage ≥ target on a synthetic 10k sweep; both-classes set when uncertain, singletons when confident.
- **T4 Runner** — `scripts/excellence.py`: `--systems S1 S2 S3 --cv` runs the CV evaluation, writes `reports/csv/excellence/{system}.json` (per-fold + mean AUC, ECE raw/scaled, conformal empirical coverage) and prints the comparison table incl. McNemar vs S0. Reuses `evaluation/evaluator.py` metrics + `collect_predictions` patterns.
- **T5 Overnight training** — extend `scripts/train.py` with `--seed` override; launch background: 4 × `train --model 7_dual_attn --mode final` with seeds 43–46 (checkpoint names get seed suffix), then W4 champion search `search.py --model 7_dual_attn`.
- **T6 Conformal integration + coverage report** — apply T3 to the winning system's fold-val probs via `split_half_coverage`: the threshold is fit on even-index rows of each fold's val predictions and empirical coverage is measured on the disjoint odd-index rows (never on the calibration rows themselves); per-fold coverage across folds goes into the report.
- **T7 Bake-off (single test touch)** — after T5 completes: declare final system from CV, run `scripts/excellence.py --bake-off --system <winning>` on the held-out test set: acc+CI, AUC, F1, sens/spec, ECE raw/scaled, conformal coverage, McNemar vs S0; update `models/manifest.json` + MODEL_CARD metrics.
- **T8 (parallel, CPU)** — M8 product completion (MODEL_CARD.md template now, Dockerfiles, report regeneration) — independent of model work.

## 4. Compute plan

| Step | GPU | Wall-clock |
|------|-----|-----------|
| T1–T3 implementation + tests | CPU | 1 session |
| T4 S1–S3 CV inference (≈5 folds × 29 seq × ≤4 models × 4 TTA) | ~20–40 min | same session |
| T5 4 seed trainings + W4 search | ~6–9 h | 1–2 overnights |
| T6–T7 | inference only | 1 session after T5 |

## 5. Sources

[Deep ensembles (Lakshminarayanan 2017)](https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles) · [Uncertainty Quantification and Deep Ensembles (NeurIPS 2021)](https://dl.acm.org/doi/10.5555/3540261.3541796) · [Conformal prediction in clinical medicine (PMC 2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9309105/) · [CHEST 2025 — guaranteed coverage in clinical AI](https://journal.chestnet.org/article/S0012-3692(25)05184-0/fulltext) · [MDPI Diagnostics 2025 — leakage bands](https://www.mdpi.com/2075-4418/15/18/2348) · [TTA review](https://pmc.ncbi.nlm.nih.gov/articles/PMC7081314/)
