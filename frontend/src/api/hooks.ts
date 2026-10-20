/**
 * React Query hooks over the typed API client.
 *
 * Polling contract: the wizard polls GET /api/v1/analyses/{id} every 2s
 * (refetchInterval) until the analysis reaches a terminal state
 * (succeeded | failed | cancelled), at which point polling stops.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  createAnalysis,
  deleteAnalysis,
  getAnalysis,
  getHealth,
  getModelCard,
  getPatientHistory,
  getPatients,
  type CreateAnalysisInput,
} from "./client";
import type { AnalysisStatus } from "./types";

/** States after which no further polling is needed. */
export const TERMINAL_STATUSES: ReadonlySet<AnalysisStatus> = new Set([
  "succeeded",
  "failed",
  "cancelled",
]);

/** Central query keys so mutations can invalidate precisely. */
export const queryKeys = {
  health: ["health"] as const,
  modelCard: ["model-card"] as const,
  analysis: (id: string | null | undefined) => ["analysis", id] as const,
  analyses: (limit?: number) =>
    limit === undefined
      ? (["analyses"] as const)
      : (["analyses", limit] as const),
  patients: ["patients"] as const,
  history: (ref: string | null | undefined) => ["history", ref] as const,
};

/** Backend health/model metadata; refreshed every 30s. */
export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: getHealth,
    refetchInterval: 30_000,
  });
}

/** Model card content (no auth required). */
export function useModelCard() {
  return useQuery({
    queryKey: queryKeys.modelCard,
    queryFn: getModelCard,
  });
}

/**
 * Single analysis record. Polls every 2s until a terminal status; disabled
 * while `id` is null/undefined.
 */
export function useAnalysis(id: string | null | undefined) {
  return useQuery({
    queryKey: queryKeys.analysis(id),
    queryFn: () => getAnalysis(id as string),
    enabled: id != null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status != null && TERMINAL_STATUSES.has(status) ? false : 2000;
    },
  });
}

/** Kick off an analysis (multipart upload). */
export function useCreateAnalysis() {
  return useMutation({
    mutationFn: (input: CreateAnalysisInput) => createAnalysis(input),
  });
}

/** Known patient refs (subject_refs) for autocompletion/history browsing. */
export function usePatients() {
  return useQuery({
    queryKey: queryKeys.patients,
    queryFn: getPatients,
  });
}

/** Prior analyses for one patient; disabled while `ref` is null/undefined. */
export function usePatientHistory(ref: string | null | undefined) {
  return useQuery({
    queryKey: queryKeys.history(ref),
    queryFn: () => getPatientHistory(ref as string),
    enabled: ref != null,
  });
}

/** Cancel an analysis; invalidates the analyses list, patients and that record. */
export function useDeleteAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deleteAnalysis(id),
    onSuccess: (_data, id) => {
      void queryClient.invalidateQueries({ queryKey: ["analyses"] });
      void queryClient.invalidateQueries({ queryKey: ["patients"] });
      void queryClient.invalidateQueries({ queryKey: queryKeys.analysis(id) });
    },
  });
}
