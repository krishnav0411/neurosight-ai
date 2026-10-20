# Plan — NeuroSight AI Frontend Dashboard

**Date:** 2026-09-15 · **Status:** approved (Phase 1 wireframes + Stitch design system, user-approved)
**Source of truth:** docs/plan/03-APP-FLOW.md (routes/states), 04-UI-UX-BRIEF.md (tokens), docs/plan/05-BACKEND-DESIGN.md §3 (API contracts). This plan maps those to concrete components; no re-derivation.

## Stack (fixed)

React 18 + TypeScript + Vite 5 + Tailwind 3 + @tanstack/react-query + axios + recharts.
Tests: vitest + @testing-library/react (jsdom). **Ruling:** shadcn/ui CLI replaced by hand-rolled components styled directly with the design tokens — same visual contract, one less codegen dependency. Cost: we own the primitives.

## Design tokens (from the captured Stitch "Precision Neurodiagnostic" system — adopted verbatim where they match doc 04)

| Token | Value | Notes |
|---|---|---|
| canvas / bg | `#f8fafc` | page background |
| surface | `#ffffff` | cards, 1px `#e2e8f0` border, 12px radius, flat (no cast shadow) |
| primary | `#1e40af` | sidebar active, primary buttons, links |
| success | `#16a34a` | Non-Demented, stable, ok banners (pill bg `#dcfce7`) |
| warning | `#d97706` | experimental mode, low-confidence, progression alert (pill bg `#fef3c7`) |
| danger | `#dc2626` | Demented, failed jobs (pill bg `#fee2e2`) |
| text | `#0f172a` / `#475569` / `#94a3b8` | primary / secondary / muted |
| MRI viewport | `#000000` bg, 6px inner radius | HUD metadata chips `label-sm` |
| focus | 1.5px `#1e40af` + 2px `#dbeafe` offset ring | all interactive elements |
| figures | `font-variant-numeric: tabular-nums` | every metric |
| radius | cards 12px · controls 8px · pills full | |
| status pill | 9999px, 4px×10px pad, `label-sm` | dot indicator 6px |
| telemetry chip | `#f1f5f9` fill, 1px `#e2e8f0`, text `#475569` | model version, mode chips |

## Design → component mapping

| Approved design element | Component (new file) |
|---|---|
| Sidebar + top bar (model chip, health dot) | `src/components/Layout.tsx` |
| Disclaimer banner (persistent, results surfaces) | `src/components/DisclaimerBanner.tsx` |
| Status pills / telemetry chips | `src/components/Badges.tsx` (`StatusPill`, `TelemetryChip`) |
| Upload dropzone + visit chips + demographics panel + mode banner | `src/components/upload/FileDropZone.tsx`, `src/components/upload/DemographicsPanel.tsx` |
| 3-step wizard header | `src/components/upload/WizardSteps.tsx` |
| Diagnosis card + probability bars | `src/components/results/DiagnosisCard.tsx`, `src/components/results/ProbabilityBars.tsx` |
| Longitudinal prefix-probability line | `src/components/results/LongitudinalPanel.tsx` |
| Grad-CAM visit tabs + slice triptych (black viewports) | `src/components/results/BrainMaps.tsx` |
| Patient timeline chart + visit cards + progression alert | `src/components/patient/Timeline.tsx`, `src/components/patient/VisitCard.tsx` |
| API key modal | `src/components/ApiKeyModal.tsx` |
| Pages | `src/pages/AnalyzePage.tsx`, `src/pages/PatientPage.tsx`, `src/pages/AboutPage.tsx` |
| API layer | `src/api/types.ts`, `src/api/client.ts`, `src/api/hooks.ts` |
| Pure logic (test-first) | `src/lib/visits.ts` (filename→visit grouping, label merging), `src/lib/format.ts` (pct, trend colors) |

## API contract (from backend `test_api.py` / doc 05 — no re-derivation)

- `POST /api/v1/analyses` multipart: `files[]`, `visit_labels_json` (JSON **list parallel to files**), `subject_ref?`, `demographics?` (JSON object), `include_gradcam` → 201 `{analysis_id, status, visit_count, visit_labels, poll_url}`; 422 `{detail:{problems:[...]}}`.
- `GET /api/v1/analyses/{id}` → `{analysis_id, status: queued|running|succeeded|failed|cancelled, stage?, result?{predicted_class, class_labels, probabilities{Non-Demented,Demented}, raw_probabilities, confidence, confidence_band, risk_level, input_mode, model_version, visit_count, visit_labels, longitudinal{trend, per_visit_probability[]}, conversion_risk, gradcam?[{visit, slices, images{slice:b64}}], caveats[]}, error?}`.
- `GET /api/v1/patients` → `{patients:[ref]}`; `GET /api/v1/patients/{ref}/history` → list of analysis dicts; `DELETE /api/v1/analyses/{id}`; `GET /health` (no auth) → `{status, model_version, uptime_s, device}`; `GET /api/v1/model-card` (no auth) → `{content, metrics_digest}`.
- Auth: `X-API-Key` header; key stored in `localStorage["neurosight.apiKey"]`; 401 → reopen `ApiKeyModal`.
- Visit grouping (client-side pre-check) lives in `src/lib/visits.ts`: parse `OAS2_\d{4}_MR\d+` from `webkitRelativePath || name`; fallback `SINGLE_VISIT`; pair `.img`↔`.hdr`; flag orphans; produce the parallel `visit_labels` list aligned with the `File[]` order.
- Base URL: `import.meta.env.VITE_API_BASE ?? ""` (vite dev proxy `/api` + `/health` → `http://localhost:8000`).

## Tasks (TDD: failing test → implement → verify, per task)

- **T1 Foundation** — scaffold configs (`vite.config.ts`, `tsconfig*.json`, `tailwind.config.js`, `postcss.config.js`, `index.html`, `src/index.css` tokens), `package.json` (add vitest/jsdom/testing-library), npm install; `src/lib/visits.ts` + `src/lib/format.ts` with tests FIRST; `npm run build` green; vitest green. Commit.
- **T2 API layer** — `src/api/types.ts` (StudyResult etc.), `src/api/client.ts` (axios instance + key interceptor + 401 event), `src/api/hooks.ts` (useHealth, useModelCard, useCreateAnalysis, useAnalysis polling w/ backoff, usePatients, useHistory, useDeleteAnalysis); tests for client header injection + 401 handling with mocked axios. Commit.
- **T3 Shell + About** — `Layout.tsx`, `Badges.tsx`, `DisclaimerBanner.tsx`, `ApiKeyModal.tsx`, `App.tsx` router, `AboutPage.tsx` (renders model-card markdown as pre-wrapped text + metrics digest table); component tests (disclaimer present, health dot classes, 401 opens modal). Commit.
- **T4 Analyze wizard** — `WizardSteps`, `FileDropZone` (drag/drop + input, visit chips w/ pair status), `DemographicsPanel` (mode banner: mri_only default; any field → `mri_demo`; explicit toggle → `mri_clinical` amber warning), `ResultsPanel` (`DiagnosisCard` low-confidence banner <0.6, `ProbabilityBars`, `LongitudinalPanel` recharts line, `BrainMaps` visit tabs + slice triptych, JSON download, single-visit chip), polling states (queued/running/failed/cancelled + Cancel button); tests: grouping chips render, disclaimer always rendered with results, low-confidence banner at 0.55. Commit.
- **T5 Patient history** — selector (react-query list), recharts timeline of Demented-prob per saved analysis, `VisitCard`, progression alert (Δ>0.15 amber / else green), CSV export client-side; tests: alert threshold logic (pure fn), empty state. Commit.
- **T6 Verify + polish** — `npm run build` (tsc strict), full vitest run, keyboard pass (focus rings, dialog Esc), `prefers-reduced-motion` guard on the one animated bar fill, README section in `frontend/README.md` (dev proxy, API key). Commit.

## Verification (whole plan)

`npm run build` exit 0; `npx vitest run` all green; backend suite untouched and still green (frontend is additive); manual E2E: backend up (`uvicorn neurosight.api.main:create_app --factory` + `NEUROSIGHT_API_KEY`) → upload a real study in the browser → result renders with Grad-CAM. Evidence quoted in Phase 5.
