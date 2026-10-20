/**
 * TypeScript interfaces for the NeuroSight AI backend API contract.
 *
 * Auth: every `/api/*` request carries `X-API-Key` (handled in client.ts).
 * `/health` and `/api/v1/model-card` require no auth.
 */

/** Terminal lifecycle states of an analysis. */
export type AnalysisStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

/** GET /health */
export interface Health {
  status: string;
  model_version: string;
  uptime_s: number;
  device: string;
}

/** GET /api/v1/model-card (no auth) */
export interface ModelCard {
  content: string;
  metrics_digest?: Record<string, unknown> | null;
}

/**
 * Optional demographic fields accepted by POST /api/v1/analyses.
 * All numeric fields may arrive as numbers or strings (e.g. from form inputs);
 * `sex` is the only strictly-string field.
 */
export interface Demographics {
  sex?: string;
  age?: number | string;
  educ?: number | string;
  ses?: number | string;
  mmse?: number | string;
  cdr?: number | string;
  etiv?: number | string;
  nwbv?: number | string;
  asf?: number | string;
}

/** Per-visit Grad-CAM overlays (base64 PNGs keyed by slice label). */
export interface GradcamVisit {
  visit: string;
  slices: string[];
  images: Record<string, string>;
}

/** Longitudinal trend across the submitted visits. */
export interface Longitudinal {
  trend: string;
  per_visit_probability: number[];
}

/** Conversion-risk estimate (may be unavailable depending on model/input). */
export interface ConversionRisk {
  available: boolean;
  score?: number;
}

/** Result payload of a succeeded analysis. */
export interface StudyResult {
  /** 0 = Non-Demented, 1 = Demented (index into class_labels). */
  predicted_class: number;
  class_labels: string[];
  /** Calibrated probabilities, sums to 1. */
  probabilities: Record<string, number>;
  raw_probabilities: Record<string, number>;
  /** 0..1 */
  confidence: number;
  confidence_band: "adequate" | "low";
  /** "Low" | "Moderate" | "High" */
  risk_level: string;
  input_mode: string;
  model_version: string;
  visit_count: number;
  visit_labels: string[];
  longitudinal: Longitudinal;
  conversion_risk: ConversionRisk;
  gradcam?: GradcamVisit[];
  caveats: string[];
}

/**
 * Analysis record as returned by GET /api/v1/analyses/{id},
 * GET /api/v1/analyses and GET /api/v1/patients/{ref}/history.
 * `result` is present only for succeeded analyses (list endpoints usually omit it).
 */
export interface AnalysisRecord {
  analysis_id: string;
  status: AnalysisStatus;
  stage?: string | null;
  subject_ref?: string | null;
  input_mode?: string;
  visit_count?: number;
  visit_labels?: string[] | null;
  model_version?: string | null;
  created_at?: string;
  completed_at?: string | null;
  error?: string | null;
  result?: StudyResult;
}

/** 201 response of POST /api/v1/analyses */
export interface CreateAnalysisResponse {
  analysis_id: string;
  status: "queued";
  visit_count: number;
  visit_labels: string[];
  poll_url: string;
}

/** GET /api/v1/patients */
export interface PatientList {
  patients: string[];
}
