# NeuroSight AI — Frontend

React 18 + TypeScript + Vite 5 + Tailwind 3 dashboard for the NeuroSight AI API.
Design tokens and interaction rules: `docs/plan/04-UI-UX-BRIEF.md` ("Precision
Neurodiagnostic" design system). Screens: `docs/plan/03-APP-FLOW.md`.

## Develop

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 — proxies /api and /health to :8000
```

The backend must be running (repo root):

```bash
# Windows CMD: set NEUROSIGHT_API_KEY=dev-key
export NEUROSIGHT_API_KEY=dev-key
uvicorn neurosight.api.main:create_app --factory --port 8000
```

On first load the app asks for the API key (stored in `localStorage["neurosight.apiKey"]`,
re-prompted on any 401). Override the API base with `VITE_API_BASE` (e.g. when the
backend runs elsewhere); by default the dev proxy handles it.

## Test / build

```bash
npm test           # vitest run (unit + component)
npm run build      # tsc -b && vite build (strict typecheck + production bundle)
```

## Structure

- `src/api/` — typed client, auth interceptor, react-query hooks
- `src/components/` — shell (Layout), upload (dropzone/demographics/steps), results (diagnosis, bars, longitudinal, Grad-CAM), patient (timeline, cards)
- `src/pages/` — Analyze (wizard), Patient history, About (model card)
- `src/lib/` — pure logic: visit grouping, formatting, mode derivation, progression alert, CSV

## Conventions

- Sentence case, no marketing words; the disclaimer renders on every results surface.
- All metrics use tabular figures (`.tnum`); status pills carry a 6px dot indicator.
- Focus: 2px `#1e40af` ring, offset 2px. `prefers-reduced-motion` disables animation.
