/**
 * Headline result card ("Primary Clinical Diagnosis"): diagnosis badge,
 * large calibrated confidence with a "Calibrated" chip, the two slim
 * probability rows and the risk pill. When the calibrated confidence band
 * is low, an amber alert renders above the card marking the result as
 * inconclusive.
 */
import type { StudyResult } from "../../api/types";
import { pct1 } from "../../lib/format";
import { DiagnosisBadge, StatusPill, type PillTone } from "../Badges";
import { ProbabilityBars } from "./ProbabilityBars";

const RISK_TONES: Record<string, PillTone> = {
  High: "danger",
  Moderate: "warning",
  Low: "success",
};

export function DiagnosisCard({ result }: { result: StudyResult }) {
  return (
    <>
      {result.confidence_band === "low" ? (
        <div
          role="alert"
          className="rounded-card border border-amber-300 bg-amber-50 px-4 py-3 text-[13px] text-warning"
        >
          Low confidence — treat this result as inconclusive.
        </div>
      ) : null}

      <div className="rounded-card border border-hairline bg-white p-5 shadow-card">
        <div className="flex items-center justify-between">
          <p className="text-[12px] font-medium uppercase tracking-wide text-ink3">
            Classification Estimate (Research)
          </p>
          <span className="rounded bg-amber-100/70 px-1.5 py-0.5 text-[10px] font-medium text-amber-900">
            Non-Diagnostic
          </span>
        </div>

        <div className="mt-3 flex items-center justify-between gap-3">
          <DiagnosisBadge predictedClass={result.predicted_class} />
          <StatusPill
            label={result.risk_level}
            tone={RISK_TONES[result.risk_level] ?? "neutral"}
          />
        </div>

        <div className="mt-4 flex items-baseline gap-2">
          <span className="tnum text-4xl font-semibold leading-none text-ink">
            {pct1(result.confidence)}
          </span>
          <span className="chip">Calibrated</span>
        </div>

        <div className="mt-5">
          <ProbabilityBars result={result} />
        </div>
      </div>
    </>
  );
}
