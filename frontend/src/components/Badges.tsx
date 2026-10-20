/**
 * Small status indicators shared across the dashboard.
 *
 * - StatusPill: labelled pill with a 6px dot indicator (color = tone).
 * - TelemetryChip: compact chip for metadata (model version, etc.).
 * - DiagnosisBadge: maps the binary predicted class (0/1) to a labelled pill.
 */
export type PillTone = "success" | "warning" | "danger" | "neutral";

const TONE_CLASSES: Record<PillTone, string> = {
  success: "pill-success",
  warning: "pill-warning",
  danger: "pill-danger",
  neutral: "pill-neutral",
};

export function StatusPill({ label, tone }: { label: string; tone: PillTone }) {
  return (
    <span className={`pill ${TONE_CLASSES[tone]}`}>
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 shrink-0 rounded-full bg-current"
      />
      <span>{label}</span>
    </span>
  );
}

export function TelemetryChip({ children }: { children: React.ReactNode }) {
  return <span className="chip tnum">{children}</span>;
}

/** predicted_class 0 = Non-demented, 1 = Demented (see StudyResult). */
export function DiagnosisBadge({
  predictedClass,
}: {
  predictedClass: number;
}) {
  if (predictedClass === 1) {
    return (
      <span className="pill pill-danger">
        <span
          aria-hidden="true"
          className="h-1.5 w-1.5 shrink-0 rounded-full bg-current"
        />
        <span>Demented</span>
      </span>
    );
  }
  return (
    <span className="pill pill-success">
      <span
        aria-hidden="true"
        className="h-1.5 w-1.5 shrink-0 rounded-full bg-current"
      />
      <span>Non-demented</span>
    </span>
  );
}
