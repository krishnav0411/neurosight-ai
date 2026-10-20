/**
 * Two horizontal calibrated-probability bars (Non-Demented green, Demented
 * red) rendered as slim 6px rows: label left, bar center, percent right.
 * Fill widths animate over 300ms (reduced to 0.01ms globally under
 * prefers-reduced-motion). Raw probabilities surface via title tooltips.
 */
import type { StudyResult } from "../../api/types";
import { pct } from "../../lib/format";

export function ProbabilityBars({ result }: { result: StudyResult }) {
  const rows = result.class_labels.slice(0, 2).map((label, i) => {
    const value =
      result.probabilities[label] ?? result.probabilities[String(i)] ?? 0;
    const raw = result.raw_probabilities[label] ?? result.raw_probabilities[String(i)] ?? value;
    return {
      label,
      value,
      raw,
      tone: i === 1 ? "bg-danger" : "bg-success",
    };
  });

  const sentence = `Calibrated probabilities. ${rows
    .map(
      (row) =>
        `${row.label}: raw ${String(row.raw)}, calibrated ${String(
          row.value,
        )} (${pct(row.value)}).`,
    )
    .join(" ")}`;

  return (
    <div className="space-y-3">
      {/* Exact values for screen readers (the bars themselves are visual). */}
      <p className="sr-only">{sentence}</p>
      {rows.map((row) => (
        <div
          key={row.label}
          title={`${row.label}: raw ${row.raw.toFixed(4)} (calibrated ${row.value})`}
          className="flex items-center gap-3"
        >
          <span className="w-28 shrink-0 text-[13px] text-ink2">
            {row.label}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-100">
            <div
              data-fill
              className={`h-full rounded-full ${row.tone} transition-[width] duration-300`}
              style={{ width: `${Math.round(row.value * 100)}%` }}
            />
          </div>
          <span className="tnum w-10 shrink-0 text-right text-[13px] font-medium text-ink">
            {pct(row.value)}
          </span>
        </div>
      ))}
    </div>
  );
}
