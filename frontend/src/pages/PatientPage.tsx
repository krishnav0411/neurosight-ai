/**
 * Patient history page: searchable subject selector (left, 240px) and the
 * selected subject's analysis history (right) — progression banner, timeline
 * chart, visit cards, CSV export and the mandatory disclaimer.
 *
 * `?subject=<ref>` preselects a patient; otherwise the first known patient
 * is selected once the list loads.
 */
import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Link, useSearchParams } from "react-router-dom";
import type { AnalysisRecord } from "../api/types";
import {
  queryKeys,
  useDeleteAnalysis,
  usePatientHistory,
  usePatients,
} from "../api/hooks";
import { DisclaimerBanner } from "../components/DisclaimerBanner";
import { Timeline } from "../components/patient/Timeline";
import { VisitCard } from "../components/patient/VisitCard";
import {
  historyToCsv,
  progressionAlert,
  toTimelinePoints,
} from "../lib/patient";

/** Timestamp used for chronological ordering (completed_at, else created_at). */
function startedTs(item: AnalysisRecord): number {
  const iso = item.completed_at ?? item.created_at;
  const t = iso ? Date.parse(iso) : NaN;
  return Number.isNaN(t) ? 0 : t;
}

/**
 * The API returns history newest-first; the progression alert and the
 * timeline need oldest-first order (chronological sorting is the caller's
 * job per the progressionAlert contract).
 */
function chronological(items: AnalysisRecord[]): AnalysisRecord[] {
  return [...items].sort((a, b) => startedTs(a) - startedTs(b));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

function Skeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden="true"
      className={`animate-pulse rounded bg-slate-200 ${className ?? ""}`}
    />
  );
}

function LoadingPanel({ label }: { label: string }) {
  return (
    <div role="status" className="space-y-4">
      <span className="sr-only">{label}</span>
      <Skeleton className="h-14 w-full" />
      <Skeleton className="h-[220px] w-full" />
      <div className="grid gap-4 md:grid-cols-2">
        <Skeleton className="h-44" />
        <Skeleton className="h-44" />
      </div>
    </div>
  );
}

function ErrorBanner({
  message,
  onRetry,
}: {
  message: string;
  onRetry(): void;
}) {
  return (
    <div
      role="alert"
      className="rounded-card border border-red-300 bg-red-50 px-4 py-3 text-[13px] text-danger"
    >
      <p>{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-2 rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
      >
        Retry
      </button>
    </div>
  );
}

export function PatientPage() {
  const [searchParams] = useSearchParams();
  const subjectParam = searchParams.get("subject");
  const queryClient = useQueryClient();

  const patients = usePatients();
  const deleteMutation = useDeleteAnalysis();

  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");

  // Auto-select: the URL ?subject= patient wins, else the first known one.
  // Re-runs only when the list or the URL changes, so manual clicks stick.
  useEffect(() => {
    const list = patients.data;
    if (list == null) return;
    if (subjectParam != null && list.includes(subjectParam)) {
      setSelected(subjectParam);
    } else if (selected == null || !list.includes(selected)) {
      setSelected(list.length > 0 ? list[0] : null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patients.data, subjectParam]);

  const history = usePatientHistory(selected);

  const list = patients.data ?? [];
  const normalizedFilter = filter.trim().toLowerCase();
  const filteredPatients = normalizedFilter
    ? list.filter((p) => p.toLowerCase().includes(normalizedFilter))
    : list;

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: () => {
        // The hook invalidates analyses/patients; refresh this history too.
        void queryClient.invalidateQueries({
          queryKey: queryKeys.history(selected),
        });
      },
    });
  };

  const exportCsv = () => {
    if (!history.data) return;
    const blob = new Blob([historyToCsv(history.data)], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `neurosight-history-${selected ?? "export"}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const historyData = history.data;
  // Alert + timeline work in chronological order; cards keep the API's
  // newest-first order.
  const chronologicalHistory = historyData ? chronological(historyData) : [];
  const alert =
    chronologicalHistory.length > 0
      ? progressionAlert(chronologicalHistory)
      : null;
  const points = toTimelinePoints(chronologicalHistory);

  let rightColumn: React.ReactNode;
  if (patients.isPending) {
    rightColumn = <LoadingPanel label="Loading patients…" />;
  } else if (patients.isError) {
    rightColumn = (
      <ErrorBanner
        message={`Could not load patients: ${errorMessage(patients.error)}`}
        onRetry={() => void patients.refetch()}
      />
    );
  } else if (list.length === 0) {
    rightColumn = (
      <div className="rounded-card border border-hairline bg-white p-10 text-center shadow-card">
        <p className="text-[14px] font-medium text-ink">
          No saved analyses yet — run one from Analyze
        </p>
        <Link
          to="/analyze"
          className="mt-4 inline-block rounded-control bg-primary px-3 py-1.5 text-[13px] font-medium text-white hover:bg-primary-dark"
        >
          Go to Analyze
        </Link>
      </div>
    );
  } else if (selected == null) {
    rightColumn = <LoadingPanel label="Loading history…" />;
  } else if (history.isPending) {
    rightColumn = <LoadingPanel label="Loading history…" />;
  } else if (history.isError) {
    rightColumn = (
      <ErrorBanner
        message={`Could not load history: ${errorMessage(history.error)}`}
        onRetry={() => void history.refetch()}
      />
    );
  } else if (historyData != null && historyData.length === 0) {
    rightColumn = (
      <div className="rounded-card border border-hairline bg-white p-10 text-center shadow-card">
        <p className="text-[14px] font-medium text-ink">
          No analyses for this subject yet.
        </p>
      </div>
    );
  } else {
    rightColumn = (
      <>
        {alert ? (
          <div
            role="status"
            className={`rounded-card border px-4 py-3 text-[13px] leading-5 ${
              alert.level === "warning"
                ? "border-warning bg-amber-50 text-warning"
                : "border-success bg-green-50 text-success"
            }`}
          >
            {alert.message}
          </div>
        ) : null}

        <div className="mt-4">
          <Timeline points={points} />
        </div>

        <div className="mt-4 grid gap-4 md:grid-cols-2">
          {(historyData ?? []).map((item) => (
            <VisitCard
              key={item.analysis_id}
              item={item}
              onDelete={handleDelete}
            />
          ))}
        </div>

        <div className="mt-4 flex justify-end">
          <button
            type="button"
            onClick={exportCsv}
            className="rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
          >
            Export CSV
          </button>
        </div>
      </>
    );
  }

  return (
    <section>
      <h1 className="text-2xl font-semibold text-ink">Patient history</h1>

      <div className="mt-4 flex flex-col md:flex-row items-start gap-6">
        <aside aria-label="Patient selector" className="w-full md:w-60 shrink-0">
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter patients"
            aria-label="Filter patients"
            className="w-full rounded-control border border-hairline bg-white px-3 py-2 text-[13px] text-ink placeholder:text-ink3 focus:border-primary focus:outline-none"
          />

          {patients.isPending ? (
            <div className="mt-2 space-y-1" aria-hidden="true">
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
              <Skeleton className="h-9 w-full" />
            </div>
          ) : patients.isError ? (
            <p className="mt-2 px-1 text-[12px] text-danger" role="alert">
              Could not load patients.
            </p>
          ) : filteredPatients.length === 0 ? (
            <p className="mt-2 px-1 text-[12px] text-ink3">
              No patients match the filter.
            </p>
          ) : (
            <ul className="mt-2 space-y-1">
              {filteredPatients.map((ref) => (
                <li key={ref}>
                  <button
                    type="button"
                    onClick={() => setSelected(ref)}
                    aria-current={selected === ref ? true : undefined}
                    className={`w-full truncate rounded-control px-3 py-2 text-left text-[13px] font-medium ${
                      selected === ref
                        ? "bg-blue-50 text-primary"
                        : "text-ink2 hover:bg-slate-50"
                    }`}
                  >
                    {ref}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <div className="min-w-0 flex-1">{rightColumn}</div>
      </div>

      <DisclaimerBanner className="mt-6" />
    </section>
  );
}
