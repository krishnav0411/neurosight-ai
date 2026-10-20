/**
 * Results dashboard (matches the approved Stitch render): a subject header
 * strip, a two-column top row (Primary Clinical Diagnosis + Longitudinal
 * progression, or a single-visit chip), the interpretable attention maps,
 * the action row and — always last — the research-use disclaimer.
 */
import { Link } from "react-router-dom";
import type { StudyResult } from "../../api/types";
import { TelemetryChip } from "../Badges";
import { DisclaimerBanner } from "../DisclaimerBanner";
import { BrainMaps } from "./BrainMaps";
import { DiagnosisCard } from "./DiagnosisCard";
import { LongitudinalPanel } from "./LongitudinalPanel";

/** Client-side JSON export (kept injectable/observable for tests). */
export function downloadResultJson(result: StudyResult): void {
  if (typeof URL.createObjectURL !== "function") return; // jsdom guard
  const blob = new Blob([JSON.stringify(result, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `neurosight_${result.model_version}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function ActionButton({
  onClick,
  children,
  primary,
}: {
  onClick: () => void;
  children: React.ReactNode;
  primary?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        primary
          ? "rounded-control bg-primary px-3 py-1.5 text-[13px] font-medium text-white hover:bg-primary-dark"
          : "rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
      }
    >
      {children}
    </button>
  );
}

export function ResultsPanel({
  result,
  subjectRef,
  onReset,
}: {
  result: StudyResult;
  /** Pseudonym the analysis was stored under, when provided at upload. */
  subjectRef?: string;
  onReset: () => void;
}) {
  const multiVisit =
    result.visit_count > 1 && result.longitudinal.per_visit_probability.length > 1;

  return (
    <div className="space-y-4">
      {/* Subject header strip. */}
      <div className="rounded-card border border-hairline bg-white px-4 py-3 shadow-card">
        <div className="flex flex-wrap items-center gap-2">
          <TelemetryChip>Subject {subjectRef ?? "—"}</TelemetryChip>
          <TelemetryChip>{result.visit_count} Visits</TelemetryChip>
          <TelemetryChip>Mode: {result.input_mode}</TelemetryChip>
          <TelemetryChip>Model: {result.model_version}</TelemetryChip>
        </div>
      </div>

      {/* Top row: diagnosis card + longitudinal progression. */}
      <div className="grid items-start gap-4 lg:grid-cols-2">
        <DiagnosisCard result={result} />

        {multiVisit ? (
          <LongitudinalPanel
            longitudinal={result.longitudinal}
            visitLabels={result.visit_labels}
            visitCount={result.visit_count}
          />
        ) : (
          <div className="rounded-card border border-hairline bg-white p-5 shadow-card">
            <h3 className="text-base font-semibold text-ink">
              Longitudinal progression
            </h3>
            <div className="mt-3">
              <TelemetryChip>single-visit mode</TelemetryChip>
            </div>
          </div>
        )}
      </div>

      {result.gradcam && result.gradcam.length > 0 ? (
        <BrainMaps gradcam={result.gradcam} />
      ) : null}

      {/* Action row: study actions left, patient-history link right. */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <ActionButton primary onClick={() => downloadResultJson(result)}>
            Download JSON
          </ActionButton>
          <ActionButton onClick={onReset}>Analyze another study</ActionButton>
        </div>
        {subjectRef ? (
          <Link
            to={`/patient?subject=${encodeURIComponent(subjectRef)}`}
            aria-label="View in patient history"
            className="text-[13px] font-medium text-primary hover:text-primary-dark"
          >
            View in patient history <span aria-hidden="true">→</span>
          </Link>
        ) : (
          <p className="text-[12px] text-ink3">
            Patient reference not set — history tracking disabled.
          </p>
        )}
      </div>

      <DisclaimerBanner />
    </div>
  );
}
