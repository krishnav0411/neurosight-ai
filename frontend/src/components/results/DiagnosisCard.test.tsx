/**
 * DiagnosisCard: headline diagnosis, calibrated confidence, risk pill,
 * telemetry chips, and the low-confidence alert.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DiagnosisCard } from "./DiagnosisCard";
import { makeStudyResult } from "../../test/fixtures";

describe("DiagnosisCard", () => {
  it("renders the diagnosis badge, calibrated confidence, risk pill and probability bars", () => {
    render(<DiagnosisCard result={makeStudyResult()} />);

    // "Demented" appears twice: the diagnosis badge and the probability row.
    expect(screen.getAllByText("Demented").length).toBeGreaterThan(0);
    expect(screen.getByText("87.0%")).toBeInTheDocument();
    expect(screen.getByText("Calibrated")).toBeInTheDocument();
    expect(screen.getByText("High")).toBeInTheDocument();
    // The two slim probability rows render inside the card.
    expect(screen.getByText("Non-Demented")).toBeInTheDocument();
  });

  it("risk pill tone follows risk_level (High danger / Moderate warning / Low success)", () => {
    const { rerender } = render(
      <DiagnosisCard result={makeStudyResult({ risk_level: "High" })} />,
    );
    expect(screen.getByText("High").parentElement).toHaveClass("pill-danger");

    rerender(
      <DiagnosisCard result={makeStudyResult({ risk_level: "Moderate" })} />,
    );
    expect(screen.getByText("Moderate").parentElement).toHaveClass(
      "pill-warning",
    );

    rerender(<DiagnosisCard result={makeStudyResult({ risk_level: "Low" })} />);
    expect(screen.getByText("Low").parentElement).toHaveClass("pill-success");
  });

  it("shows the inconclusive alert (role=alert) above the card when confidence_band is low", () => {
    render(
      <DiagnosisCard result={makeStudyResult({ confidence_band: "low" })} />,
    );
    const alert = screen.getByRole("alert");
    expect(alert).toHaveTextContent(
      "Low confidence — treat this result as inconclusive.",
    );
    expect(alert).toHaveClass("bg-amber-50");
  });

  it("shows no alert when confidence_band is adequate", () => {
    render(<DiagnosisCard result={makeStudyResult()} />);
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
