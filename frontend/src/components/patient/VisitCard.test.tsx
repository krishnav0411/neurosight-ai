/**
 * VisitCard: the pill must reflect reality — a diagnosis pill only when the
 * record carries a result; otherwise a neutral/danger status pill of the job
 * STATUS (never a fabricated "Non-demented").
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { AnalysisRecord } from "../../api/types";
import { STUDY_RESULT_FIXTURE } from "../../test/fixtures";
import { VisitCard } from "./VisitCard";

function makeRecord(overrides: Partial<AnalysisRecord> = {}): AnalysisRecord {
  return {
    analysis_id: "rec-1",
    status: "succeeded",
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

describe("VisitCard", () => {
  it("shows the diagnosis pill for a succeeded record with a result", () => {
    render(
      <VisitCard
        item={makeRecord({ result: STUDY_RESULT_FIXTURE })}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByText("Demented").parentElement).toHaveClass("pill-danger");
  });

  it("renders a 'Failed' danger status pill and never fabricates a diagnosis", () => {
    render(
      <VisitCard item={makeRecord({ status: "failed" })} onDelete={vi.fn()} />,
    );
    expect(screen.getByText("Failed").parentElement).toHaveClass("pill-danger");
    expect(screen.queryByText("Non-demented")).not.toBeInTheDocument();
    expect(screen.queryByText("Demented")).not.toBeInTheDocument();
  });

  it("renders neutral status pills for queued and running records", () => {
    const { rerender } = render(
      <VisitCard item={makeRecord({ status: "queued" })} onDelete={vi.fn()} />,
    );
    expect(screen.getByText("Queued").parentElement).toHaveClass("pill-neutral");

    rerender(
      <VisitCard item={makeRecord({ status: "running" })} onDelete={vi.fn()} />,
    );
    expect(screen.getByText("Running").parentElement).toHaveClass("pill-neutral");
  });

  it("renders a 'Cancelled' danger status pill", () => {
    render(
      <VisitCard
        item={makeRecord({ status: "cancelled" })}
        onDelete={vi.fn()}
      />,
    );
    expect(screen.getByText("Cancelled").parentElement).toHaveClass("pill-danger");
  });

  it("shows the no-result note instead of a confidence figure when result is absent", () => {
    render(
      <VisitCard item={makeRecord({ status: "failed" })} onDelete={vi.fn()} />,
    );
    expect(screen.getByText(/No result available \(status: failed\)/))
      .toBeInTheDocument();
    expect(screen.queryByText(/confidence/)).not.toBeInTheDocument();
  });
});
