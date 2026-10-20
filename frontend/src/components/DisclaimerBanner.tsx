/**
 * Permanent disclaimer banner. Sentence case, amber warning tone.
 * Rendered at the bottom of every page that hosts clinical content.
 */
export const DISCLAIMER_TEXT =
  "Research use only. Not a medical device. This tool does not provide a diagnosis.";

export function DisclaimerBanner({ className }: { className?: string }) {
  return (
    <div
      role="note"
      className={`rounded-card border border-amber-200 bg-amber-50 px-4 py-3 text-[13px] leading-5 text-amber-900 ${
        className ?? ""
      }`}
    >
      {DISCLAIMER_TEXT}
    </div>
  );
}
