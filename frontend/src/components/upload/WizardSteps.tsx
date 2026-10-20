/**
 * Header stepper for the analyze wizard: Upload -> Processing -> Results.
 *
 * Completed steps render in ink2 with a check, the active one in primary with
 * a 2px primary bottom border, future ones in ink2 as well (ink3 failed
 * contrast for meaningful labels). Steps are clickable only backwards
 * (n < current) when an onStepClick handler is provided.
 */

const STEPS = ["Upload", "Processing", "Results"] as const;

export function WizardSteps({
  current,
  onStepClick,
}: {
  current: 1 | 2 | 3;
  onStepClick?: (step: 1 | 2 | 3) => void;
}) {
  return (
    <nav aria-label="Wizard progress">
      <ol className="flex gap-8 border-b border-hairline">
        {STEPS.map((label, i) => {
          const n = (i + 1) as 1 | 2 | 3;
          const isCompleted = n < current;
          const isActive = n === current;
          const toneClass = isCompleted
            ? "text-ink2"
            : isActive
              ? "text-primary"
              : "text-ink2";
          const clickable = onStepClick != null && isCompleted;
          const Inner = (
            <>
              <span className="tnum">{n}</span>
              <span>{label}</span>
              {isCompleted ? (
                <span data-check aria-hidden="true">
                  ✓
                </span>
              ) : null}
            </>
          );

          return (
            <li
              key={label}
              aria-current={isActive ? "step" : "false"}
              className={`-mb-px flex items-center gap-1.5 border-b-2 pb-2.5 text-[13px] font-medium ${
                isActive ? "border-primary" : "border-transparent"
              } ${toneClass}`}
            >
              {clickable ? (
                <button
                  type="button"
                  onClick={() => onStepClick?.(n)}
                  className={`flex items-center gap-1.5 hover:text-primary ${toneClass}`}
                >
                  {Inner}
                </button>
              ) : (
                <span className="flex items-center gap-1.5">{Inner}</span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
