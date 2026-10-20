/**
 * LongitudinalPanel: wrapper logic around the recharts line chart.
 * Per the task spec, we test the wrapper (trend pill, caption, gating),
 * not the SVG itself.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LongitudinalPanel } from "./LongitudinalPanel";
import { makeStudyResult } from "../../test/fixtures";

describe("LongitudinalPanel", () => {
  it("shows the trend pill and caption for a multi-visit result", () => {
    render(<LongitudinalPanel longitudinal={makeStudyResult().longitudinal} />);
    expect(screen.getByText("Declining")).toBeInTheDocument();
    expect(screen.getByText("Declining")).toHaveClass("pill-warning");
    expect(
      screen.getByText("How the prediction evolves as visits accumulate"),
    ).toBeInTheDocument();
  });

  it("renders nothing (or no chart) when there are 1 or 0 data points", () => {
    const { container } = render(
      <LongitudinalPanel
        longitudinal={{ trend: "Stable", per_visit_probability: [0.3] }}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});
