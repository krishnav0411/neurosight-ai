/**
 * Input-mode derivation for the analyze wizard.
 *
 * The backend accepts three input modes:
 *  - mri_only:      MRI scans alone (demographics ignored by the model)
 *  - mri_demo:      MRI + non-clinical demographics (age, sex, ...)
 *  - mri_clinical:  MRI + clinical scores (MMSE/CDR) — experimental toggle
 */

export type InputMode = "mri_only" | "mri_demo" | "mri_clinical";

/**
 * Pure derivation of the input mode from wizard state.
 * The experimental clinical toggle dominates: it is the strongest statement
 * of intent (and the only one that reaches the clinical-score fields).
 */
export function deriveMode(
  hasDemographics: boolean,
  experimental: boolean,
): InputMode {
  if (experimental) return "mri_clinical";
  if (hasDemographics) return "mri_demo";
  return "mri_only";
}

/**
 * True when at least one demographics field has a non-empty value
 * (whitespace-only strings and null/undefined count as empty).
 */
export function hasAnyDemographics(d: object): boolean {
  return Object.values(d).some(
    (v) => (typeof v === "string" ? v.trim() !== "" : v != null),
  );
}
