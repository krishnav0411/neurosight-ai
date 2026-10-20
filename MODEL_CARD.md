# Model Card — NeuroSight-E5

**Model:** 5-seed ensemble of `7_dual_attn` (CNN + spatial/temporal attention + BiLSTM)
**Version:** neurosight-e5 (manifest: `models/manifest.json`) · **Date:** 2026-09-17
**Task:** Binary dementia-status classification (Non-Demented / Demented) from longitudinal brain MRI

---

## 1. Intended use

Research and educational decision-**support** only: demonstrating how longitudinal MRI
analysis, calibrated confidence, and attention-based evidence can be presented for
dementia-status estimation. **Not a medical device. Not a diagnosis.** Must not be used
for clinical decision-making on real patients.

**Out of scope:** diagnosis, prognosis for treatment planning, use on pediatric or
non-MRI data, use as a stand-alone screening instrument.

## 2. Training data

- **OASIS-2** (Washington University Knight ADRC): 150 subjects aged 60–96, 373 MRI
  sessions (`mpr-N` Analyze pairs). Demographics CSV with CDR/MMSE/eTIV/nWBV/ASF.
- **Labels:** binary, derived per visit from CDR (0 → Non-Demented, >0 → Demented),
  Group-verified; Converted subjects split into pre/post-conversion sequences.
- **Splits:** subject-level 70/15/15 + 5-fold subject-level CV (seed 42). Leakage
  is guarded by automated tests (zero subject overlap, Converted pairs never straddle).
- Citation: Marcus et al., *J. Cogn. Neurosci.* 22(12), 2010. Data use agreement applies.

## 3. Architecture

Per-slice CNN (Conv 1→8→16→32 + BatchNorm + SE/CBAM attention) → slice mean-pool →
BiLSTM over visits (hidden 32) → additive temporal attention → FC head.
Preprocessing: z-score, 32 middle axial slices, 128×128. Clinical metadata is
**masked out** in the headline `mri_only` mode (MMSE/CDR define the label — feeding
them would be leakage).

## 4. Measured performance (all numbers from committed run artifacts)

### Cross-validation (primary evidence — 5-fold, subject-level)

| Metric | Value |
|--------|-------|
| Mean CV AUC | **0.744 ± 0.097** (folds: .772/.636/.868/.817/.629) |

### Held-out test (n = 22 — touched exactly once, pre-declared)

| Metric | NeuroSight-E5 | Champion v1 (single model) |
|--------|---------------|----------------------------|
| Accuracy | **59.1%** [95% CI 36.4–77.3] | 59.1% |
| AUC-ROC | 0.777 | 0.785 |
| F1 (Demented) | 0.571 | 0.571 |
| Sensitivity / Specificity | 0.545 / 0.636 | 0.545 / 0.636 |
| ECE (calibration error) | **0.227** | 0.304 |

McNemar E5 vs v1: χ² = 0.0, p = 1.0 (identical predictions on all 22 cases).

## 5. Honest disclosures (read before trusting any number)

1. **Small test set.** n = 22 makes every test statistic noisy; the CI above spans
   21 points. CV is the primary evidence.
2. **Calibration remains imperfect.** ECE 0.227 is improved (v1: 0.304) but above
   the 0.10 target — treat probabilities as *rankings*, not calibrated percentages.
3. **Split-conformal prediction sets were implemented but NOT deployed:** measured
   coverage 54.5% vs the 90% target at this calibration size (~18 sequences, empty
   sets occur). Honest failure, on record in `reports/excellence/`.
4. **Training-bug lineage (all fixed in code; see git history):** the v1 ablation
   trained with a cache bug that silently skipped augmentation; the 3D variants
   trained with broken gradient accumulation; the frozen ResNet backbone's BN stats
   drifted; final-model validation was partly in-sample. E5 members s43–s46 were
   trained **after** these fixes; v1-lineage numbers are retained as baselines.
5. **Negative results (attempted and dropped, per pre-registered rules):**
   cross-variant ensembling (−0.006 AUC), test-time augmentation (−0.013),
   hyperparameter search (existing config already optimal in-grid).
6. **No external validation.** Single cohort, single scanner family. ADNI/AIBL
   validation is the planned next step (see docs/plan/07).
7. **Context band:** subject-wise-split OASIS-2 studies report 66–90% accuracy;
   95%+ claims almost always contain same-patient leakage (MDPI Diagnostics 2025).

## 6. Known limitations & risks

- Misses ~45% of Demented cases at the default 0.5 threshold (sensitivity 0.545).
- Trained/evaluated only on OASIS-2-style 1.5T Analyze-format scans; other formats,
  field strengths, and cohorts are unvalidated.
- Demographics are ignored in the deployed mode by design; adding them (modes
  `mri_demo`/`mri_clinical`) changes comparability and is flagged in the UI.
- Small per-fold variance (±0.10 AUC) means retraining can move numbers by points.

## 7. Deployment facts (for operators)

- **Serving:** FastAPI + background worker; `models/manifest.json` pins the 5 member
  checkpoints, temperature (0.657), and metrics digest; forward-probe on startup.
- **Members:** `best_7_dual_attn_final_…{_s43,_s44,_s45,_s46,}.pth` (~2–3 MB each).
- **Auth:** `X-API-Key` header (env `NEUROSIGHT_API_KEY`); CORS restricted to the
  configured dashboard origins; uploads capped at 512 MB/study, ≤8 visits.
- **Privacy:** uploads purge after 7 days; no PHI in logs or URLs; demo runs local.

## 8. Licensing & Redistribution Rights Notice

- **Software Code:** Licensed under the MIT License.
- **Model Checkpoints & Derived Figures:** Neural network weights (`models/manifest.json`, `models/*.pth`) were trained on the OASIS-2 longitudinal MRI cohort. The repository owner attests to holding confidential written authorization expressly covering public GitHub hosting and non-commercial research presentation of these specific trained checkpoints and derived Grad-CAM figures. This permission is confidential, project-specific, and non-transferable; it does not convey blanket downstream redistribution rights. Model weights and derived figures are not covered by the MIT License. Downstream researchers must independently secure required approvals under the OASIS Data Use Agreement (DUA).

## 9. Citation

```bibtex
@dataset{oasis2_2010,
  author = {Marcus, D.S. and Fotenos, A.F. and Csernansky, J.G. and Morris, J.C. and Buckner, R.L.},
  title  = {Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults},
  journal = {Journal of Cognitive Neuroscience},
  volume = {22}, number = {12}, pages = {2677--2684}, year = {2010}
}
```
