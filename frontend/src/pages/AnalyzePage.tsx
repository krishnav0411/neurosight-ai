/**
 * Analyze wizard: a 3-step state machine over the analysis lifecycle.
 *
 *  Step 1 (Upload):    file drop zone + visit chips, optional demographics,
 *                      patient reference, Grad-CAM toggle, Run analysis.
 *  Step 2 (Processing): poll useAnalysis(jobId) (2s interval, handled by the
 *                      hook); cancel via deleteAnalysis.
 *  Step 3 (Results):   the results dashboard; "Analyze another study" resets.
 *
 * 422 responses from POST /analyses arrive as ValidationError and surface as
 * a problems list while staying on step 1.
 */
import { useEffect, useState } from "react";
import type { AnalysisRecord, Demographics, StudyResult } from "../api/types";
import { useAnalysis, useCreateAnalysis, useDeleteAnalysis } from "../api/hooks";
import { ValidationError } from "../api/client";
import { TelemetryChip } from "../components/Badges";
import { DemographicsPanel } from "../components/upload/DemographicsPanel";
import { FileDropZone } from "../components/upload/FileDropZone";
import { WizardSteps } from "../components/upload/WizardSteps";
import { ResultsPanel } from "../components/results/ResultsPanel";
import { deriveMode, hasAnyDemographics } from "../lib/mode";
import { groupFiles, toVisitLabels } from "../lib/visits";

type Step = 1 | 2 | 3;

function relativePathOf(f: File): string | undefined {
  return (
    (f as File & { webkitRelativePath?: string }).webkitRelativePath ||
    undefined
  );
}

const LABEL_CLASS = "mb-1 block text-[13px] font-medium text-ink2";
const INPUT_CLASS =
  "w-full max-w-xs rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] text-ink focus:border-primary focus:outline-none";

export function AnalyzePage() {
  const [files, setFiles] = useState<File[]>([]);
  const [demographics, setDemographics] = useState<Demographics>({});
  const [experimental, setExperimental] = useState(false);
  const [subjectRef, setSubjectRef] = useState("");
  const [includeGradcam, setIncludeGradcam] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [step, setStep] = useState<Step>(1);
  const [problems, setProblems] = useState<string[] | null>(null);
  const [result, setResult] = useState<StudyResult | undefined>(undefined);
  const [elapsed, setElapsed] = useState(0);

  const createMutation = useCreateAnalysis();
  const deleteMutation = useDeleteAnalysis();
  const analysis = useAnalysis(jobId);

  const record: AnalysisRecord | undefined =
    jobId != null ? analysis.data : undefined;

  const hasDemo = hasAnyDemographics(demographics);
  const mode = deriveMode(hasDemo, experimental);
  const visits = groupFiles(
    files.map((f) => ({ name: f.name, relativePath: relativePathOf(f) })),
  );

  // Step 2 -> 3 when the polled record reaches "succeeded", even if the
  // result payload is missing (step 3 then renders a fallback card).
  useEffect(() => {
    if (step !== 2 || !record) return;
    if (record.status === "succeeded") {
      setResult(record.result);
      setStep(3);
    }
  }, [step, record]);

  // Elapsed-seconds counter while processing.
  useEffect(() => {
    if (step !== 2) return;
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [step, jobId]);

  const run = () => {
    setProblems(null);
    createMutation.mutate(
      {
        files,
        visitLabels: toVisitLabels(
          files.map((f) => ({
            name: f.name,
            relativePath: relativePathOf(f),
          })),
        ),
        subjectRef: subjectRef.trim() === "" ? undefined : subjectRef.trim(),
        demographics: hasDemo ? demographics : undefined,
        includeGradcam,
      },
      {
        onSuccess: (data) => {
          setJobId(data.analysis_id);
          setStep(2);
        },
        onError: (err) => {
          if (err instanceof ValidationError) {
            setProblems(err.problems);
          } else {
            setProblems([
              err instanceof Error ? err.message : "Unexpected error",
            ]);
          }
        },
      },
    );
  };

  const cancel = () => {
    if (jobId == null || deleteMutation.isPending) return; // no double-fire
    deleteMutation.mutate(jobId, {
      onSuccess: () => {
        setJobId(null);
        setStep(1);
      },
      // On error: stay on step 2; deleteMutation.isError renders the alert.
      onError: () => {},
    });
  };

  const tryAgain = () => {
    setJobId(null);
    setStep(1); // files are kept
  };

  const resetAll = () => {
    setFiles([]);
    setDemographics({});
    setExperimental(false);
    setSubjectRef("");
    setIncludeGradcam(true);
    setJobId(null);
    setStep(1);
    setProblems(null);
    setResult(undefined);
  };

  // Backwards navigation only; step 2 is not a meaningful destination from
  // the finished results view. Keeping `jobId` preserves the running job:
  // step 1 shows a banner while the analysis is still in flight.
  const onStepClick = (n: Step) => {
    if (n >= step) return;
    if (step === 3 && n === 2) return;
    setStep(n);
  };

  const jobActive =
    jobId != null &&
    (record?.status === "queued" || record?.status === "running");

  const runDisabled = files.length === 0 || createMutation.isPending;

  return (
    <section className="space-y-6">
      <header className="flex items-baseline justify-between gap-4">
        <h1 className="text-2xl font-semibold text-ink">Analyze</h1>
        <p className="text-[12px] text-ink3">
          Longitudinal brain MRI classification
        </p>
      </header>

      <WizardSteps current={step} onStepClick={onStepClick} />

      {step === 1 && jobActive ? (
        <div
          role="status"
          className="rounded-card border border-blue-200 bg-blue-50 px-4 py-3 text-[13px] text-primary"
        >
          Analysis running in background —{" "}
          <button
            type="button"
            onClick={() => setStep(2)}
            className="font-medium underline hover:text-primary-dark"
          >
            view status
          </button>
        </div>
      ) : null}

      {step === 1 ? (
        <div className="space-y-4">
          {/* Subject header strip: patient reference, Grad-CAM toggle, mode. */}
          <div className="rounded-card border border-hairline bg-white p-4 shadow-card">
            <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-3">
              <div className="flex flex-wrap items-end gap-x-5 gap-y-3">
                <div>
                  <label htmlFor="subject-ref" className={LABEL_CLASS}>
                    Patient reference (pseudonym)
                  </label>
                  <input
                    id="subject-ref"
                    type="text"
                    value={subjectRef}
                    onChange={(e) => setSubjectRef(e.target.value)}
                    aria-label="Patient reference (pseudonym)"
                    title="Free-text pseudonym used to group studies in patient history"
                    placeholder="e.g. SUB-042"
                    className={INPUT_CLASS}
                  />
                </div>
                <label className="flex items-center gap-2 pb-1.5 text-[13px] text-ink2">
                  <input
                    type="checkbox"
                    checked={includeGradcam}
                    onChange={(e) => setIncludeGradcam(e.target.checked)}
                    aria-label="Include brain maps (Grad-CAM)"
                  />
                  Include brain maps (Grad-CAM)
                </label>
              </div>
              <TelemetryChip>Mode: {mode}</TelemetryChip>
            </div>
          </div>

          <FileDropZone files={files} onFiles={setFiles} />

          <DemographicsPanel
            values={demographics}
            onChange={setDemographics}
            mode={mode}
            experimental={experimental}
            onExperimentalChange={setExperimental}
          />

          {problems ? (
            <div
              role="alert"
              className="rounded-card border border-red-300 bg-red-50 px-4 py-3 text-[13px] text-danger"
            >
              <p className="font-medium">The analysis could not be started:</p>
              <ul className="mt-1 list-disc pl-5">
                {problems.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="flex items-center justify-end gap-3">
            <TelemetryChip>Mode: {mode}</TelemetryChip>
            <button
              type="button"
              onClick={run}
              disabled={runDisabled}
              className="rounded-control bg-primary px-4 py-2 text-[13px] font-medium text-white hover:bg-primary-dark disabled:cursor-not-allowed disabled:opacity-50"
            >
              {createMutation.isPending ? "Uploading…" : "Run analysis"}
            </button>
          </div>
        </div>
      ) : null}

      {step === 2 ? (
        analysis.isError ? (
          <div
            role="alert"
            className="rounded-card border border-red-300 bg-red-50 p-5"
          >
            <h2 className="text-[15px] font-semibold text-danger">
              Connection lost
            </h2>
            <p className="mt-1 text-[13px] text-ink2">
              Lost contact with the backend while analyzing.
            </p>
            {analysis.error instanceof Error && analysis.error.message ? (
              <p className="mt-1 text-[12px] text-ink3">
                {analysis.error.message}
              </p>
            ) : null}
            <button
              type="button"
              onClick={tryAgain}
              className="mt-3 rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
            >
              Try again
            </button>
          </div>
        ) : record &&
          (record.status === "failed" || record.status === "cancelled") ? (
          <div className="rounded-card border border-red-300 bg-red-50 p-5">
            <h2 className="text-[15px] font-semibold text-danger">
              Analysis {record.status}
            </h2>
            <p className="mt-1 text-[13px] text-ink2">
              {record.error ?? "The analysis did not complete."}
            </p>
            <button
              type="button"
              onClick={tryAgain}
              className="mt-3 rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
            >
              Try again
            </button>
          </div>
        ) : (
          <div className="space-y-3">
            {deleteMutation.isError ? (
              <div
                role="alert"
                className="rounded-card border border-red-300 bg-red-50 px-4 py-3 text-[13px] text-danger"
              >
                The analysis could not be cancelled:{" "}
                {deleteMutation.error instanceof Error
                  ? deleteMutation.error.message
                  : "unexpected error"}
                .
              </div>
            ) : null}
            <div
              aria-live="polite"
              className="rounded-card border border-hairline bg-white p-8 text-center shadow-card"
            >
              <div
                aria-hidden="true"
                className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-hairline border-t-primary"
              />
              <p className="mt-3 text-[14px] font-medium text-ink">
                Analyzing…
              </p>
              <p className="mt-1 text-[13px] text-ink2 tnum">
                Elapsed {elapsed}s
              </p>
              <p className="mt-1 text-[12px] text-ink3">
                {visits.length} visit{visits.length === 1 ? "" : "s"} ·{" "}
                {files.length} file{files.length === 1 ? "" : "s"} uploaded
              </p>
              {record?.status === "queued" || record?.status === "running" ? (
                <button
                  type="button"
                  onClick={cancel}
                  disabled={deleteMutation.isPending}
                  className="mt-4 rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Cancel
                </button>
              ) : null}
            </div>
          </div>
        )
      ) : null}

      {step === 3 ? (
        result ? (
          <ResultsPanel
            result={result}
            subjectRef={
              subjectRef.trim() === "" ? undefined : subjectRef.trim()
            }
            onReset={resetAll}
          />
        ) : (
          <div className="rounded-card border border-hairline bg-white p-5">
            <p className="text-[13px] text-ink2">
              Result payload is unavailable.
            </p>
            <button
              type="button"
              onClick={tryAgain}
              className="mt-3 rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
            >
              Try again
            </button>
          </div>
        )
      ) : null}
    </section>
  );
}
