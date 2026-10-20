/**
 * Shared StudyResult fixture for wizard/results tests.
 */
import type { StudyResult } from "../api/types";

export const STUDY_RESULT_FIXTURE: StudyResult = {
  predicted_class: 1,
  class_labels: ["Non-Demented", "Demented"],
  probabilities: { "Non-Demented": 0.13, Demented: 0.87 },
  raw_probabilities: { "Non-Demented": 0.1298, Demented: 0.8702 },
  confidence: 0.87,
  confidence_band: "adequate",
  risk_level: "High",
  input_mode: "mri_only",
  model_version: "v0.4.0-eval",
  visit_count: 3,
  visit_labels: ["MR1", "MR2", "MR3"],
  longitudinal: {
    trend: "Declining",
    per_visit_probability: [0.21, 0.55, 0.87],
  },
  conversion_risk: { available: false },
  caveats: ["Research use only."],
};

export function makeStudyResult(
  overrides: Partial<StudyResult> = {},
): StudyResult {
  return { ...STUDY_RESULT_FIXTURE, ...overrides };
}
