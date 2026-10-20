/**
 * Grad-CAM brain maps: one tab per visit and a single black viewport showing
 * the Grad-CAM overlay composite the backend sends per slice (base64 PNG
 * keyed by slice label), with an amber/green importance legend under it and
 * a slice slider row beneath reading "Slice n / total (label)".
 *
 * The payload carries one composite image per slice — no separate
 * original/heatmap/fusion payloads — so exactly one viewport is rendered.
 * No synthetic HUD text is drawn: every string comes from the analysis
 * payload or is a static, honest label.
 */
import { useRef, useState } from "react";
import type { GradcamVisit } from "../../api/types";

/** Stable id of the single active tabpanel (referenced by every tab). */
const PANEL_ID = "brainmaps-active-panel";

function srcOf(b64: string): string {
  // The backend sends bare base64 payloads; tolerate data URLs just in case.
  return b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
}

export function BrainMaps({ gradcam }: { gradcam: GradcamVisit[] }) {
  const [visitIdx, setVisitIdx] = useState(0);
  const [sliceIdx, setSliceIdx] = useState(0);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const visit = gradcam[visitIdx];
  if (!visit) return null;

  const slices = [...visit.slices].sort((a, b) => Number(a) - Number(b));
  const slice = slices[Math.min(sliceIdx, slices.length - 1)];
  const image = slice != null ? visit.images[slice] : undefined;

  const selectVisit = (idx: number) => {
    setVisitIdx(idx);
    setSliceIdx(0); // slider state resets per visit tab change
  };

  // ArrowLeft/ArrowRight move selection (wrapping); focus follows selection.
  const onTabKeyDown = (e: React.KeyboardEvent, idx: number) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const dir = e.key === "ArrowRight" ? 1 : -1;
    const next = (idx + dir + gradcam.length) % gradcam.length;
    selectVisit(next);
    tabRefs.current[next]?.focus();
  };

  return (
    <div className="rounded-card border border-hairline bg-white p-5 shadow-card">
      <h3 className="text-base font-semibold text-ink">
        Interpretable attention maps
      </h3>
      <p className="mt-1 text-[12px] text-ink3">
        Grad-CAM spatial attribution over the analyzed visits
      </p>

      <div
        role="tablist"
        aria-label="Visits with brain maps"
        className="mt-4 flex flex-wrap gap-2"
      >
        {gradcam.map((v, i) => (
          <button
            key={v.visit}
            ref={(el) => {
              tabRefs.current[i] = el;
            }}
            id={`brainmaps-tab-${i}`}
            role="tab"
            aria-selected={i === visitIdx}
            aria-controls={PANEL_ID}
            type="button"
            onClick={() => selectVisit(i)}
            onKeyDown={(e) => onTabKeyDown(e, i)}
            tabIndex={i === visitIdx ? 0 : -1}
            className={`rounded-control px-2.5 py-1 text-[12px] font-medium ${
              i === visitIdx
                ? "bg-blue-50 text-primary"
                : "text-ink2 hover:bg-slate-50"
            }`}
          >
            {v.visit}
          </button>
        ))}
      </div>

      <div
        id={PANEL_ID}
        role="tabpanel"
        aria-labelledby={`brainmaps-tab-${visitIdx}`}
        tabIndex={0}
      >
        <figure className="m-0">
          <div className="relative rounded-lg bg-black p-2">
            <span
              aria-hidden="true"
              className="absolute left-2 top-2 rounded bg-white/10 px-1.5 py-0.5 text-[10px] font-medium leading-4 text-slate-200"
            >
              Grad-CAM overlay
            </span>
            {image ? (
              <img
                src={srcOf(image)}
                alt={`Grad-CAM overlay — ${visit.visit}, slice ${slice}`}
                className="mx-auto max-h-64 object-contain"
              />
            ) : (
              <div
                aria-hidden="true"
                className="flex h-40 items-center justify-center text-[11px] text-slate-500"
              >
                No image
              </div>
            )}
          </div>
          <figcaption className="mt-2 flex flex-wrap items-center gap-2 text-[11px] text-ink2">
            <span className="inline-flex items-center gap-1">
              <span
                aria-hidden="true"
                className="h-1.5 w-1.5 rounded-full bg-amber-500"
              />
              Low importance
            </span>
            <span className="inline-flex items-center gap-1">
              <span
                aria-hidden="true"
                className="h-1.5 w-1.5 rounded-full bg-success"
              />
              High importance
            </span>
          </figcaption>
        </figure>

        {slices.length > 1 ? (
          <div className="mt-4 flex items-center gap-3">
            <label
              htmlFor="slice-slider"
              className="text-[12px] text-ink2"
              title="Slice position through the volume"
            >
              Slice
            </label>
            <input
              id="slice-slider"
              type="range"
              min={0}
              max={slices.length - 1}
              value={Math.min(sliceIdx, slices.length - 1)}
              onChange={(e) => setSliceIdx(Number(e.target.value))}
              className="flex-1 accent-[#1e40af]"
              aria-label={`Slice (${slice})`}
            />
            <span className="tnum text-right text-[12px] text-ink2">
              Slice {Math.min(sliceIdx, slices.length - 1) + 1} / {slices.length}
              {slice != null ? ` (${slice})` : ""}
            </span>
          </div>
        ) : null}
      </div>
    </div>
  );
}
