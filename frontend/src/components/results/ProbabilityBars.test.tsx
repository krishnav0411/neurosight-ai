/**
 * ProbabilityBars: two calibrated-probability bars with raw-probability
 * tooltips.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ProbabilityBars } from "./ProbabilityBars";
import { makeStudyResult } from "../../test/fixtures";

describe("ProbabilityBars", () => {
  it("renders calibrated percentage labels for both classes", () => {
    render(<ProbabilityBars result={makeStudyResult()} />);

    expect(screen.getByText("Non-Demented")).toBeInTheDocument();
    expect(screen.getByText("Demented")).toBeInTheDocument();
    expect(screen.getByText("13%")).toBeInTheDocument();
    expect(screen.getByText("87%")).toBeInTheDocument();
  });

  it("exposes raw probabilities via title attributes", () => {
    render(<ProbabilityBars result={makeStudyResult()} />);

    const demented = screen.getByText("87%").closest("[title]") as HTMLElement;
    expect(demented).toHaveProperty(
      "title",
      expect.stringContaining("0.8702"),
    );
    const nonDemented = screen
      .getByText("13%")
      .closest("[title]") as HTMLElement;
    expect(nonDemented).toHaveProperty(
      "title",
      expect.stringContaining("0.1298"),
    );
  });

  it("sizes the demented fill to its calibrated percentage width", () => {
    render(<ProbabilityBars result={makeStudyResult()} />);
    const fill = screen.getByText("87%").closest("[title]")!.querySelector("[data-fill]") as HTMLElement;
    expect(fill.style.width).toBe("87%");
  });
});
