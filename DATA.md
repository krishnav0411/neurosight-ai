# DATA.md — Dataset Provenance, License, and Citation

## Dataset: OASIS-2 (Longitudinal MRI Data in Nondemented and Demented Older Adults)

- **Source:** Washington University at St. Louis, Open Access Series of Imaging Studies.
  Official page: <https://sites.wustl.edu/oasisbrains/home/oasis-2/>
- **Contents on this machine:** 150 subjects aged 60–96, 373 imaging sessions
  (`Datasets/OAS2_RAW_PART1`: 209 sessions / 82 subjects; `OAS2_RAW_PART2`: 164 sessions / 68 subjects),
  each session `OAS2_XXXX_MRY/RAW/mpr-N.nifti.img` + `.hdr` (Analyze 7.5 pair, 1–4 series).
  Exposed to the repo via the `data/raw` junction.
- **Raw data location:** Local dataset storage outside this repository (e.g. `data/raw` or user-configured external folder), mapped to the project via the `data/raw` directory junction or symlink. Raw OASIS files are gitignored by design.

## Demographics CSV

| Field | Value |
|-------|-------|
| Canonical copy in repo | `data/metadata/oasis_longitudinal.csv` |
| Origin | User-provided download of the canonical file (identical to the Kaggle mirror `jboysen/mri-and-alzheimers` → `oasis_longitudinal.csv`, which mirrors the WUSTL-bundled `oasis_longitudinal_demographics.csv`) |
| Installed | 2026-09-14 |
| SHA-256 | `d2f0a15ff35fa4a65c35c064b848fc1b396e9d2f4aec935c88d6e1588d31d40e` |
| Size / rows | 28,290 bytes / 373 rows |
| Columns | `Subject ID, MRI ID, Group, Visit, MR Delay, M/F, Hand, Age, EDUC, SES, MMSE, CDR, eTIV, nWBV, ASF` |
| Row-level groups | Nondemented 190 / Demented 146 / Converted 37 |
| Subject-level groups | Nondemented 72 / Demented 64 / Converted 14 |
| Known missing values | MMSE: 2 · SES: 19 (mean-imputed by the loader, never dropped silently) |
| Session↔disk parity | CSV `MRI ID` ↔ 373 raw session folders: exact 1-to-1 (verified 2026-09-14) |

The checksum above is re-verified by `python -m neurosight.data.audit <csv>` at pipeline start.

## License & Terms

OASIS data is distributed for research under the OASIS Data Use Agreement
(<https://sites.wustl.edu/oasisbrains/> — "DUA" link on the download page). By using this
repository you agree to: non-commercial research use only; no attempt to identify subjects;
no redistribution of the imaging data. The raw scans and the demographics CSV are **not
committed** to version control (`.gitignore`).

### Model Weights & Derived Representations
The repository owner attests to holding confidential written authorization expressly covering public GitHub hosting and non-commercial research sharing of the specific trained neural network checkpoints (`models/*.pth`) and OASIS-derived Grad-CAM visualization figures included in this repository. This authorization is confidential, project-specific, and non-transferable; it does not grant blanket downstream redistribution rights to third parties. Model weights and derived figures are not covered by the repository's MIT License. Users wishing to redistribute or repurpose these model checkpoints must independently verify and obtain authorization under the OASIS Data Use Agreement.

## Citation

If you use this project or its results, cite the dataset paper:

> Marcus, D. S., Fotenos, A. F., Csernansky, J. G., Morris, J. C., & Buckner, R. L. (2010).
> Open Access Series of Imaging Studies (OASIS): Longitudinal MRI data in nondemented and
> demented older adults. *Journal of Cognitive Neuroscience*, 22(12), 2677–2684.
> <https://pmc.ncbi.nlm.nih.gov/articles/PMC2895005/>

## Labeling note (affects every metric in this project)

`Group` reflects the subject's overall trajectory; the per-visit **CDR** field defines the
working binary label used by the pipeline (`CDR == 0 → 0 "Non-Demented", else 1 "Demented"`),
with `Group` used to verify it and to split `Converted` subjects into pre/post-conversion
sequences. MMSE and CDR are **diagnosis-defining clinical scores**: they are excluded from the
headline `mri_only` input mode (see `docs/plan/00-PLAN-REVIEW.md`, defect B1).
