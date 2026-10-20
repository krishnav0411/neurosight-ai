/**
 * One prior analysis rendered as a white card: diagnosis pill, confidence,
 * telemetry chips, an optional Grad-CAM overlay thumbnail (click opens the
 * full-size image in a modal) and a delete action behind a confirm dialog.
 */
import { useState } from "react";
import type { AnalysisRecord, AnalysisStatus } from "../../api/types";
import { pct } from "../../lib/format";
import { StatusPill, TelemetryChip, type PillTone } from "../Badges";
import { useModalA11y } from "../dialogs/useModalA11y";

/** The backend sends bare base64 payloads; tolerate data URLs just in case. */
function toDataUrl(b64: string): string {
  return b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
}

/** Diagnosis pill; only a succeeded record WITH a result can carry one. */
function diagnosisOf(item: AnalysisRecord): { label: string; tone: PillTone } | null {
  if (item.result?.predicted_class == null) return null;
  return item.result.predicted_class === 1
    ? { label: "Demented", tone: "danger" }
    : { label: "Non-demented", tone: "success" };
}

/** Fallback pill for records without a diagnosis: the job status itself. */
function statusPillOf(status: AnalysisStatus): { label: string; tone: PillTone } {
  switch (status) {
    case "failed":
    case "cancelled":
      return { label: status === "failed" ? "Failed" : "Cancelled", tone: "danger" };
    case "succeeded":
      return { label: "Succeeded", tone: "neutral" };
    default:
      return { label: status === "queued" ? "Queued" : "Running", tone: "neutral" };
  }
}

/** First image entry of the first Grad-CAM visit, as a displayable src. */
function firstOverlaySrc(item: AnalysisRecord): string | null {
  const images = item.result?.gradcam?.[0]?.images;
  if (!images) return null;
  const first = Object.values(images)[0];
  return typeof first === "string" ? toDataUrl(first) : null;
}

function createdLabel(item: AnalysisRecord): string {
  const iso = item.created_at;
  if (iso != null && iso !== "") {
    const d = new Date(iso);
    if (!Number.isNaN(d.getTime())) return d.toLocaleDateString();
  }
  return "—";
}

export function VisitCard({
  item,
  onDelete,
}: {
  item: AnalysisRecord;
  onDelete(id: string): void;
}) {
  const [viewerSrc, setViewerSrc] = useState<string | null>(null);
  const lightboxRef = useModalA11y(
    viewerSrc != null,
    () => setViewerSrc(null),
  );

  const pill = diagnosisOf(item) ?? statusPillOf(item.status);
  const result = item.result;
  const overlaySrc = firstOverlaySrc(item);
  const inputMode = result?.input_mode ?? item.input_mode;
  const modelVersion = result?.model_version ?? item.model_version;

  const handleDelete = () => {
    if (
      window.confirm(
        `Delete analysis ${item.analysis_id}? This cannot be undone.`,
      )
    ) {
      onDelete(item.analysis_id);
    }
  };

  return (
    <div className="rounded-card border border-hairline bg-white p-5 shadow-card">
      <div className="flex items-center justify-between gap-3">
        <StatusPill label={pill.label} tone={pill.tone} />
        <span className="tnum text-[12px] text-ink3">
          {createdLabel(item)}
        </span>
      </div>

      {result ? (
        <div className="mt-3 flex items-baseline gap-2">
          <span className="tnum text-xl font-semibold text-ink">
            {pct(result.confidence)}
          </span>
          <span className="text-[12px] text-ink3">confidence</span>
        </div>
      ) : (
        <p className="mt-3 text-[12px] text-ink3">
          No result available (status: {item.status}).
        </p>
      )}

      <div className="mt-3 flex flex-wrap gap-2">
        {inputMode ? <TelemetryChip>{inputMode}</TelemetryChip> : null}
        {modelVersion ? <TelemetryChip>{modelVersion}</TelemetryChip> : null}
      </div>

      <div className="mt-4 flex items-end justify-between gap-3">
        {overlaySrc ? (
          <button
            type="button"
            onClick={() => setViewerSrc(overlaySrc)}
            aria-label="View Grad-CAM overlay full size"
            className="overflow-hidden rounded focus-visible:outline-offset-2"
          >
            <img
              src={overlaySrc}
              alt="Grad-CAM overlay thumbnail"
              className="h-16 w-auto rounded object-cover"
            />
          </button>
        ) : (
          <span aria-hidden="true" />
        )}

        <button
          type="button"
          onClick={handleDelete}
          aria-label="Delete analysis"
          className="rounded-control px-2 py-1 text-[12px] font-medium text-danger hover:bg-red-50"
        >
          Delete
        </button>
      </div>

      {viewerSrc ? (
        <div
          ref={lightboxRef}
          role="dialog"
          aria-modal="true"
          aria-label="Grad-CAM overlay"
          tabIndex={-1}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-6"
          onClick={() => setViewerSrc(null)}
        >
          <div
            className="rounded-card bg-white p-3 shadow-modal"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={viewerSrc}
              alt="Grad-CAM overlay, full size"
              className="max-h-[70vh] max-w-full rounded"
            />
            <div className="mt-2 flex justify-end">
              <button
                type="button"
                onClick={() => setViewerSrc(null)}
                className="rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
