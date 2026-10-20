/**
 * Pure helpers for the patient history view: progression alerting, timeline
 * point mapping and CSV export. All take plain analysis records, so they are
 * trivially unit-testable (no React, no network).
 *
 * Note: history arrays arrive newest-first from the API; the helpers below use
 * array order as-is (chronological sorting is the caller's job).
 */
import type { AnalysisRecord } from "../api/types";

/** Minimal structural view of a history record used by these helpers. */
export interface HistoryLike {
  result?: { probabilities?: Record<string, number> };
}

export interface ProgressionAlert {
  level: "warning" | "success";
  message: string;
}

export interface TimelinePoint {
  ref: string;
  label: string;
  prob: number;
}

const DEMENTED_KEY = "Demented";
/** A rise between the last two analyses above this triggers the warning. */
const RISE_THRESHOLD = 0.15;

/** Demented probability of a history item, or undefined when unavailable. */
export function dementedProb(
  item: HistoryLike | undefined | null,
): number | undefined {
  const p = item?.result?.probabilities?.[DEMENTED_KEY];
  return typeof p === "number" && Number.isFinite(p) ? p : undefined;
}

/**
 * Compare the last two usable Demented probabilities: a rise greater than
 * 0.15 warns, otherwise the report is reassuring. Needs >= 2 usable values,
 * else null.
 */
export function progressionAlert(
  history: readonly HistoryLike[],
): ProgressionAlert | null {
  const values = history
    .map(dementedProb)
    .filter((p): p is number => p !== undefined);
  if (values.length < 2) return null;

  const rise = values[values.length - 1] - values[values.length - 2];
  // Tolerance keeps a mathematically exact 0.15 rise (e.g. 0.5 - 0.35, which
  // is 0.15000000000000002 in IEEE-754) from triggering the warning.
  if (rise > RISE_THRESHOLD + 1e-12) {
    return {
      level: "warning",
      message: "Increasing risk detected across analyses — consider follow-up",
    };
  }
  return { level: "success", message: "No significant progression detected" };
}

/** Format an ISO date for display, falling back to `fallback` when invalid. */
function dateLabel(
  iso: string | null | undefined,
  fallback: string,
): string {
  if (iso != null && iso !== "") {
    const d = new Date(iso);
    if (!Number.isNaN(d.getTime())) return d.toLocaleDateString();
  }
  return fallback;
}

/**
 * One chart point per item that carries a Demented probability. `label` is
 * the completed_at date, or the first 8 chars of the analysis id when absent.
 */
export function toTimelinePoints(
  history: readonly (HistoryLike & {
    analysis_id: string;
    completed_at?: string | null;
  })[],
): TimelinePoint[] {
  const points: TimelinePoint[] = [];
  for (const item of history) {
    const prob = dementedProb(item);
    if (prob === undefined) continue;
    points.push({
      ref: item.analysis_id,
      label: dateLabel(item.completed_at, item.analysis_id.slice(0, 8)),
      prob,
    });
  }
  return points;
}

/**
 * Quote a CSV field when it contains a comma, quote or newline, and guard
 * against spreadsheet formula injection: a value starting with `=`, `+`, `-`
 * or `@` is prefixed with `'` so Excel/Sheets treat it as text.
 */
function csvField(value: string): string {
  const guarded = /^[=+\-@]/.test(value) ? `'${value}` : value;
  return /[",\n]/.test(guarded)
    ? `"${guarded.replace(/"/g, '""')}"`
    : guarded;
}

/**
 * Render the history as CSV: `analysis_id,created_at,status,predicted_class,
 * confidence,risk_level,demented_prob`, one row per item, empty strings for
 * missing values, proper escaping for fields containing commas/quotes.
 */
export function historyToCsv(history: readonly AnalysisRecord[]): string {
  const header =
    "analysis_id,created_at,status,predicted_class,confidence,risk_level,demented_prob";

  const rows = history.map((item) => {
    const result = item.result;
    let predicted = "";
    if (result?.predicted_class != null) {
      predicted =
        result.class_labels?.[result.predicted_class] ??
        String(result.predicted_class);
    }
    const prob = dementedProb(item);
    const cells = [
      item.analysis_id ?? "",
      item.created_at ?? "",
      item.status ?? "",
      predicted,
      result?.confidence != null ? String(result.confidence) : "",
      result?.risk_level ?? "",
      prob != null ? String(prob) : "",
    ];
    return cells.map(csvField).join(",");
  });

  return [header, ...rows].join("\n");
}
