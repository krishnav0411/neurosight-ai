# 04 — UI/UX Brief

**Project:** NeuroSight AI · v1.0 · 2026-09-14
**Applies to:** React dashboard (`frontend/`). Layout/behavior spec: `03-APP-FLOW.md`. Tone: clinical, calm, precise — a instrument panel, not a consumer app.

---

## 1. Design principles

1. **Evidence-forward:** every prediction is displayed with its uncertainty, its input mode, and its model version. A bare "DEMENTED" with no context is a design bug.
2. **Calm severity:** danger colors communicate risk, never panic; no animations on result content, only on progress.
3. **No false precision:** probabilities show whole percentages; CIs live in the About/model-card page, not cluttered into the result card.
4. **Research tool honesty:** disclaimers and input-mode labels are permanent UI residents, not footers.

## 2. Design tokens

| Token | Value | Use |
|-------|-------|-----|
| `--primary` | `#1e40af` (deep blue) | sidebar active, primary buttons, links |
| `--success` | `#16a34a` | Non-demented, stable trend, "ok" banners |
| `--warning` | `#d97706` | Converted/experimental mode, low-confidence, progression alert |
| `--danger` | `#dc2626` | Demented, failed jobs, destructive actions |
| `--bg` | `#f8fafc` | app background |
| `--surface` | `#ffffff` | cards |
| `--border` | `#e2e8f0` | card borders, dividers |
| `--text` | `#0f172a` · secondary `#475569` · muted `#94a3b8` | typography |
| Radius | 12 px cards · 8 px controls · full pills | |
| Shadow | `0 1px 3px rgb(0 0 0 / 0.08)` single elevation only | |
| Font | Inter (400/500/600/700); `tabular-nums` for all metrics | |
| Heat colormap | jet (`#00007f → #7fff7f → #ffff7f → #ff7f00 → #7f0000`) — matches matplotlib overlays; do not restyle Grad-CAM | |

Semantic pill styles (badge): Non-demented = success bg 10% + success text; Demented = danger 10%/danger; Experimental mode = warning 10%/warning with flask icon.

## 3. Layout system

- Sidebar 240 px fixed, collapsible to 64 px icon rail; content max-width 1200 px centered, 32 px page padding.
- 8 px spacing grid (4 px allowed inside chips/badges).
- Cards: white, 1 px border, 12 px radius, 24 px padding; section headers 16 px/600 with optional info-tooltip.
- Analyze wizard uses a 3-step progress header (Upload → Processing → Results) — steps clickable only backwards.

## 4. Component inventory (shadcn/ui base)

`button, card, badge, progress, alert, tabs, table, separator, slider, dialog, dropdown-menu, tooltip, skeleton, sonner (toasts)` + custom: `FileDropZone, VisitChip, ProbabilityBars, TrendLine, ConversionGauge, GradCamTriptych, AttentionStrip, DisclaimerBanner, ModelVersionChip, ApiKeyModal`.

Custom behaviors:
- **FileDropZone:** dashed border `--border`, hover/dragover → `--primary` 8% bg; shows per-visit chips with pair counts; invalid files listed with reason.
- **ProbabilityBars:** horizontal, value labels outside bars, animated width once on mount (300 ms ease-out), raw-vs-calibrated in tooltip.
- **GradCamTriptych:** three equal panels (Original grayscale / Heatmap / Overlay), 512 px max, slice slider 1–32 beneath, visit tabs above; image rendering `image-rendering: pixelated` off (smooth).
- **AttentionStrip:** 8 px-wide bars, visit labels beneath, weight normalized to max=1.
- **Skeletons:** for results sections while polling transitions to success.

## 5. Interaction rules

- Buttons: primary (Run analysis, Save), secondary (Download JSON, Export CSV), destructive (Delete analysis — always confirm dialog, focus on cancel).
- All results sections render server data; no client-side recomputation of probabilities.
- Polling UI must never block input; Cancel is always available during processing.
- Toasts: success on save/export; errors mirror server `detail` text; max 1 visible per type.
- Keyboard: full tab order matches visual order; dropzone reachable via Enter (opens file picker); dialogs trap focus, Esc closes; slider operable via arrows.
- Motion: only progress pulse + bar fill; `prefers-reduced-motion` disables both.

## 6. Content & voice

- Sentence case everywhere; no exclamation marks; no marketing words ("powerful", "smart").
- Metric labels spell out terms once: "MMSE (Mini-Mental State Examination)".
- Fixed strings (do not paraphrase):
  - Disclaimer: "Research use only. Not a medical device. This tool does not provide a diagnosis."
  - Low-confidence: "Low confidence — treat this result as inconclusive."
  - Experimental mode: "Clinical scores were used as input. MMSE/CDR strongly determine the diagnosis — do not compare with MRI-only results."

## 7. Accessibility (WCAG 2.1 AA — required)

- Contrast: all text ≥ 4.5:1 (the `--text`/`--bg` pair is 15:1; badge pairs verified below 3:1 for large text only).
- Every icon-only control has `aria-label`; probability bars expose `role="img"` + `aria-label` with values; Grad-CAM tabs are real `role="tablist"`.
- Status changes during processing announced via `aria-live="polite"` region.
- Focus visible: 2 px `--primary` outline, offset 2 px, never removed.
- Charts have text alternatives: probability values in DOM, trend values in table under chart (collapsed by default).

## 8. Responsive & fallbacks

- 1280 px+ is the design target; ≥ 992 px: sidebar collapses to rail, wizard single-column; < 992 px: functional but unsupported (show "best on desktop" notice once).
- Empty/fallback art: single line-art brain SVG reused in empty states; no stock imagery.
- Error art: none — text banners only.
