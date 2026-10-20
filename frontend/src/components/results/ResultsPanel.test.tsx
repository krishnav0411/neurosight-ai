/**
 * ResultsPanel: subject header strip, two-column top row (diagnosis +
 * longitudinal), single-visit gating, stat tiles, action row and the
 * disclaimer pinned last.
 */
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { ResultsPanel } from "./ResultsPanel";
import { makeStudyResult } from "../../test/fixtures";

function renderPanel(result = makeStudyResult(), subjectRef?: string) {
  return render(
    <MemoryRouter>
      <ResultsPanel result={result} subjectRef={subjectRef} onReset={vi.fn()} />
    </MemoryRouter>,
  );
}

describe("ResultsPanel", () => {
  it("renders the subject header strip chips (subject, visits, mode, model)", () => {
    renderPanel(makeStudyResult(), "SUB-042");

    expect(screen.getByText("Subject SUB-042")).toBeInTheDocument();
    expect(screen.getByText("3 Visits")).toBeInTheDocument();
    expect(screen.getByText("Mode: mri_only")).toBeInTheDocument();
    expect(screen.getByText("Model: v0.4.0-eval")).toBeInTheDocument();
  });

  it("shows an em-dash subject chip when no patient reference was set", () => {
    renderPanel();

    expect(screen.getByText("Subject —")).toBeInTheDocument();
    expect(
      screen.getByText(
        /Patient reference not set — history tracking disabled\./,
      ),
    ).toBeInTheDocument();
  });

  it("renders the longitudinal stat tiles with a signed, tone-colored delta", () => {
    // Fixture per-visit [0.21, 0.55, 0.87]: rising demented probability.
    renderPanel();

    expect(screen.getByText("Latest")).toBeInTheDocument();
    const delta = screen.getByText("+32%");
    expect(delta).toHaveClass("text-danger");
    expect(screen.getByText("Visits")).toBeInTheDocument();
  });

  it("hides the longitudinal card behind a single-visit mode chip for 1 visit", () => {
    const single = makeStudyResult({
      visit_count: 1,
      visit_labels: ["MR1"],
      longitudinal: { trend: "Stable", per_visit_probability: [0.4] },
    });
    renderPanel(single);

    expect(screen.getByText("single-visit mode")).toBeInTheDocument();
    expect(screen.queryByText("Δ latest visit")).toBeNull();
  });

  it("links to the patient history when a subject ref is set", () => {
    renderPanel(makeStudyResult(), "SUB-042");

    const link = screen.getByRole("link", {
      name: "View in patient history",
    });
    expect(link).toHaveAttribute("href", "/patient?subject=SUB-042");
  });

  it("renders the research-use disclaimer last inside the panel", () => {
    const { container } = renderPanel();

    const disclaimer = screen.getByRole("note");
    expect(disclaimer).toBeInTheDocument();
    const root = container.firstElementChild as HTMLElement;
    expect(root.lastElementChild).toBe(disclaimer);
  });
});
