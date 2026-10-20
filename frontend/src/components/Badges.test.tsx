/**
 * Shared badge components: StatusPill, TelemetryChip, DiagnosisBadge.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DiagnosisBadge, StatusPill, TelemetryChip } from "./Badges";

describe("StatusPill", () => {
  it("renders the label with pill classes and a rounded-full dot indicator", () => {
    render(<StatusPill label="Model ready" tone="success" />);
    const label = screen.getByText("Model ready");
    const pill = label.parentElement;
    expect(pill).not.toBeNull();
    expect(pill).toHaveClass("pill", "pill-success");
    const dot = label.previousElementSibling;
    expect(dot).not.toBeNull();
    expect(dot).toHaveProperty("tagName", "SPAN");
    expect(dot).toHaveClass("rounded-full");
  });

  it("supports all four tones", () => {
    const tones = ["success", "warning", "danger", "neutral"] as const;
    for (const tone of tones) {
      const { unmount } = render(<StatusPill label={`x-${tone}`} tone={tone} />);
      expect(screen.getByText(`x-${tone}`).parentElement).toHaveClass(
        `pill-${tone}`,
      );
      unmount();
    }
  });
});

describe("TelemetryChip", () => {
  it("renders children inside the chip class", () => {
    render(<TelemetryChip>v0.1.0</TelemetryChip>);
    const chip = screen.getByText("v0.1.0").closest("span");
    expect(chip).toHaveClass("chip");
  });
});

describe("DiagnosisBadge", () => {
  it("maps predicted class 0 to Non-demented (pill-success)", () => {
    render(<DiagnosisBadge predictedClass={0} />);
    const badge = screen.getByText("Non-demented").parentElement;
    expect(badge).toHaveClass("pill", "pill-success");
  });

  it("maps predicted class 1 to Demented (pill-danger)", () => {
    render(<DiagnosisBadge predictedClass={1} />);
    const badge = screen.getByText("Demented").parentElement;
    expect(badge).toHaveClass("pill", "pill-danger");
  });
});
