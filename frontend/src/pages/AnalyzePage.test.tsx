/**
 * AnalyzePage integration test. API hooks are mocked tightly:
 * - useCreateAnalysis exposes a vi.fn() mutate; tests decide whether the
 *   mutation resolves (onSuccess) or rejects (onError).
 * - useAnalysis returns a controllable record (default: succeeded fixture).
 * - useDeleteAnalysis exposes a vi.fn() mutate.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ValidationError } from "../api/client";
import { STUDY_RESULT_FIXTURE } from "../test/fixtures";
import { AnalyzePage } from "./AnalyzePage";

const h = vi.hoisted(() => ({
  createMutate: vi.fn(),
  deleteMutate: vi.fn(),
  analysis: { data: undefined as any, isError: false, error: null as unknown },
  deletion: { isPending: false, isError: false, error: null as unknown },
}));

vi.mock("../api/hooks", () => ({
  useCreateAnalysis: () => ({
    mutate: h.createMutate,
    isPending: false,
    isError: false,
    error: null,
  }),
  useAnalysis: () => h.analysis,
  useDeleteAnalysis: () => ({ mutate: h.deleteMutate, ...h.deletion }),
}));

function makeFile(name: string, relativePath?: string): File {
  const file = new File(["dummy"], name);
  if (relativePath) {
    Object.defineProperty(file, "webkitRelativePath", { value: relativePath });
  }
  return file;
}

async function uploadOneVisitAndRun(user: ReturnType<typeof userEvent.setup>) {
  const input = document.querySelector(
    'input[type="file"]',
  ) as HTMLInputElement;
  await user.upload(input, [makeFile("a.nii", "OAS2_0001_MR1/a.nii")]);
  await user.click(screen.getByRole("button", { name: "Run analysis" }));
}

beforeEach(() => {
  h.createMutate.mockReset();
  h.deleteMutate.mockReset();
  h.analysis.data = undefined;
  h.analysis.isError = false;
  h.analysis.error = null;
  h.deletion.isPending = false;
  h.deletion.isError = false;
  h.deletion.error = null;
});

describe("AnalyzePage", () => {
  it("disables Run analysis until files are selected", () => {
    render(<AnalyzePage />);
    expect(screen.getByRole("button", { name: "Run analysis" })).toBeDisabled();
    expect(screen.getByText("No visits detected yet")).toBeInTheDocument();
  });

  it("walks the happy path: upload -> run -> results dashboard", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a1",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "/api/v1/analyses/a1",
        }),
    );
    h.analysis.data = {
      analysis_id: "a1",
      status: "succeeded",
      result: STUDY_RESULT_FIXTURE,
    };

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    expect(h.createMutate).toHaveBeenCalledWith(
      expect.objectContaining({
        visitLabels: ["OAS2_0001_MR1"],
        includeGradcam: true,
      }),
      expect.anything(),
    );

    // Results dashboard: confidence, trend and the mandatory disclaimer.
    await waitFor(() =>
      expect(screen.getAllByText("87%").length).toBeGreaterThan(0),
    );
    expect(screen.getByText("Declining")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Research use only\. Not a medical device\./,
      ),
    ).toBeInTheDocument();
    // No subject ref was entered: history tracking is disabled (API has no
    // attach-after endpoint), so a muted note replaces the link.
    expect(
      screen.getByText(/Patient reference not set — history tracking disabled\./),
    ).toBeInTheDocument();
  });

  it("stays on step 1 and lists problems when creation is rejected with 422", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onError: (e: unknown) => void }) =>
        opts.onError(
          new ValidationError([
            "Visit OAS2_0002_MR1 has unpaired .img/.hdr files",
          ]),
        ),
    );

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("unpaired .img/.hdr files");
    // Still on step 1: the Run button (and the drop zone) remain.
    expect(
      screen.getByRole("button", { name: "Run analysis" }),
    ).toBeInTheDocument();
  });

  it("shows an error card with Try again when the analysis fails", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a2",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = {
      analysis_id: "a2",
      status: "failed",
      error: "Inference failed: CUDA out of memory",
    };

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    await waitFor(() =>
      expect(screen.getByText(/CUDA out of memory/)).toBeInTheDocument(),
    );

    await user.click(screen.getByRole("button", { name: "Try again" }));
    // Back on step 1, files kept (visit chip still visible).
    expect(
      screen.getByRole("button", { name: "Run analysis" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/OAS2_0001_MR1 — 1 pair ✓/)).toBeInTheDocument();
  });

  it("cancel during processing calls the delete mutation and returns to step 1", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a3",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = { analysis_id: "a3", status: "running" };
    h.deleteMutate.mockImplementation(
      (_id: string, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({ analysis_id: "a3", status: "cancelled" }),
    );

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    expect(await screen.findByText("Analyzing…")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    expect(h.deleteMutate).toHaveBeenCalledWith("a3", expect.anything());
    expect(
      await screen.findByRole("button", { name: "Run analysis" }),
    ).toBeInTheDocument();
  });

  it("hides Cancel until the poll reports a queued/running status", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a4",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = undefined; // first poll has not landed yet

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    expect(await screen.findByText("Analyzing…")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: "Cancel" }),
    ).not.toBeInTheDocument();
  });

  it("disables Cancel while the delete mutation is pending", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a5",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = { analysis_id: "a5", status: "running" };
    h.deletion.isPending = true;

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    expect(await screen.findByRole("button", { name: "Cancel" })).toBeDisabled();
  });

  it("keeps the wizard on step 2 with a danger alert when cancelling fails", async () => {
    const user = userEvent.setup();
    const { rerender } = render(<AnalyzePage />);
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a6",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = { analysis_id: "a6", status: "running" };
    h.deleteMutate.mockImplementation(
      (
        _id: string,
        opts: { onError: (e: unknown) => void },
      ) => {
        h.deletion.isError = true;
        h.deletion.error = new Error("Request failed with status code 500");
        opts.onError(h.deletion.error);
      },
    );

    await uploadOneVisitAndRun(user);
    expect(await screen.findByText("Analyzing…")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Cancel" }));
    rerender(<AnalyzePage />);

    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent("could not be cancelled");
    expect(alert).toHaveTextContent("Request failed with status code 500");
    // Still on step 2: the processing card remains.
    expect(screen.getByText("Analyzing…")).toBeInTheDocument();
  });

  it("shows a poll-error alert card with Try again when the query errors", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a7",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.isError = true;
    h.analysis.error = new Error("Network Error");

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(
      "Lost contact with the backend while analyzing.",
    );
    expect(alert).toHaveTextContent("Network Error");

    await user.click(screen.getByRole("button", { name: "Try again" }));
    // Back to step 1 (files kept), no poll-error card anymore.
    expect(
      screen.getByRole("button", { name: "Run analysis" }),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("moves to step 3 with a fallback card when succeeded without a result payload", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a8",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = { analysis_id: "a8", status: "succeeded" }; // no result

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);

    expect(
      await screen.findByText("Result payload is unavailable."),
    ).toBeInTheDocument();
    expect(screen.queryByText("Analyzing…")).not.toBeInTheDocument();
  });

  it("backwards nav to step 1 keeps the running job and offers a view-status link", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "a9",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = { analysis_id: "a9", status: "running" };

    render(<AnalyzePage />);
    await uploadOneVisitAndRun(user);
    expect(await screen.findByText("Analyzing…")).toBeInTheDocument();

    // Backwards via the stepper (the "1 Upload ✓" step): the job is NOT
    // cancelled or dropped. (^ anchors away from the drop zone's button.)
    await user.click(screen.getByRole("button", { name: /^1 Upload/ }));
    expect(h.deleteMutate).not.toHaveBeenCalled();

    const banner = screen.getByRole("status");
    expect(banner).toHaveTextContent("Analysis running in background");
    await user.click(
      screen.getByRole("button", { name: "view status" }),
    );
    expect(await screen.findByText("Analyzing…")).toBeInTheDocument();
  });

  it("links to the patient history when a subject ref was provided", async () => {
    const user = userEvent.setup();
    h.createMutate.mockImplementation(
      (_input: unknown, opts: { onSuccess: (d: unknown) => void }) =>
        opts.onSuccess({
          analysis_id: "aa",
          status: "queued",
          visit_count: 1,
          visit_labels: ["OAS2_0001_MR1"],
          poll_url: "",
        }),
    );
    h.analysis.data = {
      analysis_id: "aa",
      status: "succeeded",
      result: STUDY_RESULT_FIXTURE,
    };

    render(
      <MemoryRouter initialEntries={["/analyze"]}>
        <AnalyzePage />
      </MemoryRouter>,
    );
    await user.type(
      screen.getByLabelText("Patient reference (pseudonym)"),
      "SUB-042",
    );
    await uploadOneVisitAndRun(user);

    const link = await screen.findByRole("link", {
      name: "View in patient history",
    });
    expect(link).toHaveAttribute("href", "/patient?subject=SUB-042");
  });
});
