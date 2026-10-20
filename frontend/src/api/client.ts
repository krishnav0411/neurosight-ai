/**
 * Typed axios client for the NeuroSight AI backend.
 *
 * Auth model: every request whose URL contains `/api/` carries
 * `X-API-Key: <key>` read from `localStorage["neurosight.apiKey"]`.
 * `/health` is the only exempt endpoint; `/api/v1/model-card` is public on
 * the backend but still sent with the header (its URL contains `/api/`).
 */
import axios, { type AxiosError } from "axios";
import type {
  AnalysisRecord,
  CreateAnalysisResponse,
  Demographics,
  Health,
  ModelCard,
  PatientList,
} from "./types";

export const API_KEY_STORAGE_KEY = "neurosight.apiKey";
export const UNAUTHORIZED_EVENT = "neurosight:unauthorized";

/** In-memory fallback for storage-restricted modes (privacy mode, quota). */
let volatileApiKey = "";

/**
 * Read the API key from localStorage, falling back to the in-memory copy
 * (empty string when unset/unavailable).
 */
export function getApiKey(): string {
  try {
    return localStorage.getItem(API_KEY_STORAGE_KEY) ?? volatileApiKey;
  } catch {
    return volatileApiKey;
  }
}

/** Persist the API key to localStorage; it stays in memory regardless. */
export function setApiKey(key: string): void {
  volatileApiKey = key;
  try {
    localStorage.setItem(API_KEY_STORAGE_KEY, key);
  } catch {
    // The in-memory key remains available for this session.
  }
}

/** Thrown by the response interceptor when the server answers 401. */
export class ApiKeyError extends Error {
  constructor(message = "Invalid or missing API key") {
    super(message);
    this.name = "ApiKeyError";
  }
}

/** Thrown on 422 responses carrying `{detail: {problems: string[]}}`. */
export class ValidationError extends Error {
  constructor(public problems: string[]) {
    super(problems.join("; "));
    this.name = "ValidationError";
  }
}

export const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE ?? "",
});

api.interceptors.request.use((config) => {
  const url = config.url ?? "";
  if (url.includes("/api/")) {
    config.headers["X-API-Key"] = getApiKey();
  }
  return config;
});

/** Extract `problems` from a FastAPI-style error body, if present. */
function extractProblems(data: unknown): string[] | null {
  if (data === null || typeof data !== "object") return null;
  const detail = (data as { detail?: unknown }).detail;
  if (detail !== null && typeof detail === "object") {
    const problems = (detail as { problems?: unknown }).problems;
    if (Array.isArray(problems)) return problems.map(String);
  }
  if (typeof detail === "string" && detail.length > 0) return [detail];
  return null;
}

api.interceptors.response.use(
  (response) => response,
  (error: unknown) => {
    const axiosError = error as AxiosError | undefined;
    const status = axiosError?.response?.status;
    if (status === 401) {
      window.dispatchEvent(new CustomEvent(UNAUTHORIZED_EVENT));
      return Promise.reject(new ApiKeyError());
    }
    if (status === 422) {
      const problems = extractProblems(axiosError?.response?.data);
      if (problems) return Promise.reject(new ValidationError(problems));
    }
    if (status === 429) {
      const detail = (axiosError?.response?.data as { detail?: string } | undefined)?.detail;
      const msg =
        typeof detail === "string" && detail.length > 0
          ? detail
          : "Rate limit reached. Too many analyses — please wait a moment before retrying.";
      return Promise.reject(new ValidationError([msg]));
    }
    return Promise.reject(error);
  },
);

// ---------------------------------------------------------------------------
// Typed endpoint helpers
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<Health> {
  const { data } = await api.get<Health>("/health");
  return data;
}

export async function getModelCard(): Promise<ModelCard> {
  const { data } = await api.get<ModelCard>("/api/v1/model-card");
  return data;
}

export interface CreateAnalysisInput {
  files: File[];
  /** Visit labels, PARALLEL to the order of `files`. */
  visitLabels: string[];
  subjectRef?: string;
  demographics?: Demographics;
  includeGradcam?: boolean;
}

/**
 * Build the multipart form expected by POST /api/v1/analyses.
 * Exported for testability.
 */
export function buildCreateAnalysisForm(input: CreateAnalysisInput): FormData {
  const form = new FormData();
  for (const file of input.files) {
    form.append("files", file, file.name);
  }
  form.append("visit_labels_json", JSON.stringify(input.visitLabels));
  if (input.subjectRef != null && input.subjectRef !== "") {
    form.append("subject_ref", input.subjectRef);
  }
  if (input.demographics) {
    form.append("demographics", JSON.stringify(input.demographics));
  }
  form.append("include_gradcam", input.includeGradcam ? "true" : "false");
  return form;
}

export async function createAnalysis(
  input: CreateAnalysisInput,
): Promise<CreateAnalysisResponse> {
  const form = buildCreateAnalysisForm(input);
  const { data } = await api.post<CreateAnalysisResponse>(
    "/api/v1/analyses",
    form,
  );
  return data;
}

export async function getAnalysis(id: string): Promise<AnalysisRecord> {
  const { data } = await api.get<AnalysisRecord>(
    `/api/v1/analyses/${encodeURIComponent(id)}`,
  );
  return data;
}

export async function deleteAnalysis(
  id: string,
): Promise<{ analysis_id: string; status: "cancelled" }> {
  const { data } = await api.delete<{ analysis_id: string; status: "cancelled" }>(
    `/api/v1/analyses/${encodeURIComponent(id)}`,
  );
  return data;
}

export async function getPatients(): Promise<string[]> {
  const { data } = await api.get<PatientList>("/api/v1/patients");
  return data.patients;
}

export async function getPatientHistory(ref: string): Promise<AnalysisRecord[]> {
  const { data } = await api.get<AnalysisRecord[]>(
    `/api/v1/patients/${encodeURIComponent(ref)}/history`,
  );
  return data;
}
