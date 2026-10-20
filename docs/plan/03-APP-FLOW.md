# 03 — App Flow

**Project:** NeuroSight AI · v1.0 · 2026-09-14
**Scope:** the React dashboard served by the FastAPI backend. Routes, journeys, states, and every action/redirect. Visual rules live in `04-UI-UX-BRIEF.md`; endpoint contracts in `05-BACKEND-DESIGN.md`.

---

## 1. Route map

| Route | Page | Purpose |
|-------|------|---------|
| `/` | redirect → `/analyze` | |
| `/analyze` | Analyze wizard | Upload visits + optional demographics → job → results |
| `/patient` | Patient timeline | Pick a pseudo-subject-id, see prediction history across analyses |
| `/about` | About & model card | Model card, dataset provenance, limitations, disclaimer |
| `*` | 404 page | Link back to `/analyze` |

Layout: persistent left sidebar (Analyze, Patient History, About) + top bar (logo, model version badge from `/health`, API-status dot). Desktop-first, min supported width 1280 px.

## 2. Primary journey — Analyze wizard (3 steps + results)

### Step 1 — Upload (`/analyze`, default)

**UI:** drag-and-drop zone accepting `.img`/`.hdr` (multi-select; folders not supported by browsers — hint text explains picking the contents of `RAW/` per visit).

**Client logic:**
1. On file add: parse filenames → group into visits via `OAS2_XXXX_MRY` (from filename or folder-in-name) and series `mpr-N`.
2. Live validation chips per visit: `MR1 — 3 pairs ✓` / `MR2 — unpaired .img ⚠` (client-side pre-check; server re-validates).
3. Visits list shows detected count: "Detected 3 MRI sessions (MR1, MR2, MR3)". Zero visits → Run button disabled.
4. Collapsible **Optional demographics** panel: Age, Sex, Education years, SES, MMSE, CDR, eTIV, nWBV, ASF — each with tooltip. A banner inside the panel states which input mode will be used: default `mri_only`; if any demographic field is filled → mode switches to `mri+demo` (visible, never silent). A separate explicit toggle "Include clinical scores (MMSE/CDR) — experimental mode" → `mri+clinical`, amber-warned ("scores define the diagnosis; results not comparable to MRI-only").
5. Toggle **"Include brain maps (Grad-CAM)"** (default on).
6. Primary button **Run analysis** → `POST /api/v1/analyses` (multipart) → on 201, go to Step 2 with `analysis_id`. On 413/422 → inline error listing the offending file and reason.

**States:** empty (dropzone + example filename hint) · validating (chips) · submitting (button spinner, files locked) · submit-error (banner + keep files).

### Step 2 — Processing

**UI:** full-width card with animated brain-scan SVG pulse; stages tick over from `GET /api/v1/analyses/{id}` polls (2 s interval, exponential backoff to 5 s, react-query): `queued` → `preprocessing visits…` → `running temporal analysis…` → `succeeded/failed`. Show elapsed time and study summary (N visits, mode, model version). **Cancel** button → `DELETE /api/v1/analyses/{id}` (allowed while `queued/running`; removes upload dir).

**States:** queued · running (stage text) · failed (error card with server `message`, retry button → re-POST same files kept in memory) · success → auto-advance to Step 3.

### Step 3 — Results

Sections, top to bottom (visual spec in 04):

A. **Diagnosis card** — badge `NON-DEMENTED` / `DEMENTED`, color-coded; calibrated confidence `87% (calibrated)`; risk-level badge Low/Moderate/High; input-mode chip (`MRI-only` / `MRI+demographics` / `MRI+clinical — experimental`) and model version.
B. **Probability bars** — horizontal, both classes, calibrated probabilities; tooltip with raw (pre-temperature) values.
C. **Longitudinal panel** (≥2 visits uploaded) — progression trend line per visit (`Stable / Declining / Rapidly Declining`) + conversion-risk gauge from T2 head when applicable; caption explains it is a risk score, not a prediction of diagnosis date.
D. **Brain maps** (if requested) — visit tabs (MR1…MRn), slice slider, original/heatmap/overlay triptych, caption "Regions that most influenced this prediction".
E. **Temporal attention strip** — small bar chart: attention weight per visit.
F. **Disclaimer banner** (always rendered): "Research use only. Not a medical device. Not a diagnosis."
G. **Actions:** `Save to patient history` (auto if subject_ref provided) · `Download JSON` (result payload) · `Analyze another study` (resets to Step 1) · `View in Patient History` (→ `/patient?subject=...`).

**States:** result normal · low-confidence (<60% max prob → amber "Low confidence — treat as inconclusive" banner above A) · degenerate upload (<2 slices usable → results shown with "single-visit mode" chip and no longitudinal panel).

## 3. Secondary journey — Patient history (`/patient`)

1. Left: patient selector (searchable list from `GET /api/v1/patients`; empty state: "No saved analyses yet — run one from Analyze"). URL param `?subject=` preselects.
2. Right: **timeline** — recharts line chart of Demented-probability per saved analysis (x = visit label/date collected client-side, y = 0–1); dots colored by predicted class; hover tooltip with full payload.
3. Below: visit cards (newest first): predicted class badge, confidence, mode chip, model version, Grad-CAM thumbnail (click → modal with full triptych), timestamp.
4. **Progression alert:** if latest two Demented-probabilities rise by > 0.15 → amber banner "Increasing risk detected across analyses — consider follow-up"; else green "No significant progression detected".
5. Actions: `Export CSV` (per-patient summary) · `Delete analysis` (confirm dialog → `DELETE /api/v1/analyses/{id}` if stored).

**States:** loading skeleton · empty · error banner with retry · normal.

## 4. About page (`/about`)

Rendered from `GET /api/v1/model-card` (server serves MODEL_CARD.md content + metrics): intended use, training data (OASIS-2, 150 subjects), input modes and the leakage explanation, metrics table with CIs, calibration, known limitations, citation, license, full disclaimer. This page is the product's honesty contract — no page may make claims stronger than it.

## 5. Global behaviors

- **API key:** first-run modal asks for the dev API key, stores in localStorage; every request sends `X-API-Key`; 401 → re-open modal.
- **Health badge:** poll `/health` (30 s); red dot + banner "Backend unreachable" when down (pages remain viewable).
- **Error contract:** every 4xx/5xx shows a human message from the server `detail` field; network errors get a generic retry banner. No console-only errors.
- **File handling:** files never leave memory until Run is pressed; no auto-upload on drop.
- **No PHI in URL:** patient identifiers never in query params except the opaque `subject_ref` generated client-side (UUID) or user-typed pseudonym.
