/**
 * PatientPage integration test. API hooks are mocked tightly (usePatients,
 * usePatientHistory, useDeleteAnalysis); queryKeys comes from the real module.
 * Router context is provided by MemoryRouter so useSearchParams and Link work.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalysisRecord } from "../api/types";
import { makeStudyResult } from "../test/fixtures";
import { PatientPage } from "./PatientPage";

const h = vi.hoisted(() => ({
  patients: {
    data: undefined as string[] | undefined,
    isError: false,
    error: null as unknown,
    isPending: true,
    refetch: vi.fn(),
  },
  history: {
    data: undefined as AnalysisRecord[] | undefined,
    isError: false,
    error: null as unknown,
    isPending: true,
    refetch: vi.fn(),
  },
  historyRefs: [] as (string | null | undefined)[],
  deleteMutate: vi.fn(),
}));

vi.mock("../api/hooks", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/hooks")>();
  return {
    ...actual,
    usePatients: () => h.patients,
    usePatientHistory: (ref: string | null | undefined) => {
      h.historyRefs.push(ref);
      return h.history;
    },
    useDeleteAnalysis: () => ({ mutate: h.deleteMutate, isPending: false }),
  };
});

function historyItem(overrides: Partial<AnalysisRecord>): AnalysisRecord {
  return {
    analysis_id: "aaaaaaaa-0000-0000-0000-000000000000",
    status: "succeeded",
    created_at: "2026-01-01T00:00:00Z",
    completed_at: "2026-01-02T00:00:00Z",
    ...overrides,
  };
}

function resultWithProb(prob: number) {
  return makeStudyResult({
    probabilities: { "Non-Demented": 1 - prob, Demented: prob },
    confidence: prob,
    predicted_class: prob >= 0.5 ? 1 : 0,
  });
}

function renderPage(initialEntry = "/patient") {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <PatientPage />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  h.patients.data = undefined;
  h.patients.isError = false;
  h.patients.error = null;
  h.patients.isPending = true;
  h.patients.refetch.mockReset();
  h.history.data = undefined;
  h.history.isError = false;
  h.history.error = null;
  h.history.isPending = true;
  h.history.refetch.mockReset();
  h.historyRefs.length = 0;
  h.deleteMutate.mockReset();
});

describe("PatientPage", () => {
  it("shows the success banner, timeline and visit cards for a patient with history", async () => {
    h.patients.data = ["S1", "S2"];
    h.patients.isPending = false;
    h.history.isPending = false;
    // API returns newest first: the newest analysis (0.2) improved on the
    // oldest one (0.4), so after chronological sorting the rise is negative.
    h.history.data = [
      historyItem({
        analysis_id: "id-new",
        completed_at: "2026-02-01T00:00:00Z",
        result: resultWithProb(0.2),
      }),
      historyItem({
        analysis_id: "id-old",
        completed_at: "2026-01-01T00:00:00Z",
        result: resultWithProb(0.4),
      }),
    ];

    renderPage();

    // First patient auto-selected; history fetched for it.
    expect(
      await screen.findByRole("button", { name: "S1" }),
    ).toHaveAttribute("aria-current", "true");
    expect(h.historyRefs).toContain("S1");

    expect(
      screen.getByText("No significant progression detected"),
    ).toBeInTheDocument();

    // Timeline chart card with its text alternative.
    expect(screen.getByText("Risk over time")).toBeInTheDocument();
    expect(screen.getByText("View as table")).toBeInTheDocument();

    // Two visit cards.
    expect(
      screen.getAllByRole("button", { name: "Delete analysis" }),
    ).toHaveLength(2);

    // Mandatory disclaimer at the bottom.
    expect(
      screen.getByText(/Research use only\. Not a medical device\./),
    ).toBeInTheDocument();
  });

  it("shows the warning banner when risk rises more than 0.15 over time", async () => {
    h.patients.data = ["S1"];
    h.patients.isPending = false;
    h.history.isPending = false;
    // Newest first: newest 0.4, oldest 0.2 -> chronological rise of 0.2.
    h.history.data = [
      historyItem({
        analysis_id: "id-new",
        completed_at: "2026-02-01T00:00:00Z",
        result: resultWithProb(0.4),
      }),
      historyItem({
        analysis_id: "id-old",
        completed_at: "2026-01-01T00:00:00Z",
        result: resultWithProb(0.2),
      }),
    ];

    renderPage();

    expect(
      await screen.findByText(
        "Increasing risk detected across analyses — consider follow-up",
      ),
    ).toBeInTheDocument();
  });

  it("shows the success banner when risk has not risen", async () => {
    h.patients.data = ["S1"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.data = [
      historyItem({ analysis_id: "id-new", result: resultWithProb(0.25) }),
      historyItem({ analysis_id: "id-old", result: resultWithProb(0.2) }),
    ];

    renderPage();

    expect(
      await screen.findByText("No significant progression detected"),
    ).toBeInTheDocument();
  });

  it("preselects the ?subject= patient from the URL", async () => {
    h.patients.data = ["S1", "S2"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.data = [historyItem({ result: resultWithProb(0.4) })];

    renderPage("/patient?subject=S2");

    expect(
      await screen.findByRole("button", { name: "S2" }),
    ).toHaveAttribute("aria-current", "true");
    expect(h.historyRefs).toContain("S2");
  });

  it("shows the empty state when there are no saved analyses at all", () => {
    h.patients.data = [];
    h.patients.isPending = false;

    renderPage();

    expect(
      screen.getByText("No saved analyses yet — run one from Analyze"),
    ).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to Analyze" })).toHaveAttribute(
      "href",
      "/analyze",
    );
  });

  it("shows a per-subject note when the selected patient has no analyses", async () => {
    h.patients.data = ["S1"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.data = [];

    renderPage();

    expect(
      await screen.findByText("No analyses for this subject yet."),
    ).toBeInTheDocument();
  });

  it("shows an error banner with Retry when history fails", async () => {
    const user = userEvent.setup();
    h.patients.data = ["S1"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.isError = true;
    h.history.error = new Error("backend unavailable");

    renderPage();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("backend unavailable");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(h.history.refetch).toHaveBeenCalled();
  });

  it("filters the patient list client-side", async () => {
    const user = userEvent.setup();
    h.patients.data = ["S1", "S2", "OAS2_0001"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.data = [];

    renderPage();

    await user.type(
      await screen.findByRole("textbox", { name: "Filter patients" }),
      "S2",
    );
    expect(screen.getByRole("button", { name: "S2" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "S1" })).not.toBeInTheDocument();
  });

  it("asks for confirmation before deleting an analysis", async () => {
    const user = userEvent.setup();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    h.patients.data = ["S1"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.data = [
      historyItem({ analysis_id: "id-1", result: resultWithProb(0.4) }),
    ];

    renderPage();

    await user.click(
      await screen.findByRole("button", { name: "Delete analysis" }),
    );
    expect(confirmSpy).toHaveBeenCalled();
    expect(h.deleteMutate).toHaveBeenCalledWith("id-1", expect.anything());
    confirmSpy.mockRestore();
  });

  it("keeps the analysis when the confirmation is dismissed", async () => {
    const user = userEvent.setup();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    h.patients.data = ["S1"];
    h.patients.isPending = false;
    h.history.isPending = false;
    h.history.data = [
      historyItem({ analysis_id: "id-1", result: resultWithProb(0.4) }),
    ];

    renderPage();

    await user.click(
      await screen.findByRole("button", { name: "Delete analysis" }),
    );
    expect(h.deleteMutate).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });
});
