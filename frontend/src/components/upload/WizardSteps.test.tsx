/**
 * WizardSteps: header stepper for the analyze wizard.
 */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { WizardSteps } from "./WizardSteps";

describe("WizardSteps", () => {
  it("renders the three step labels with their index numbers", () => {
    render(<WizardSteps current={1} />);
    expect(screen.getByText("Upload")).toBeInTheDocument();
    expect(screen.getByText("Processing")).toBeInTheDocument();
    expect(screen.getByText("Results")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("marks the active step with text-primary and the completed ones with text-ink2", () => {
    const { rerender } = render(<WizardSteps current={2} />);
    const upload = screen.getByText("Upload").closest("[aria-current]");
    const processing = screen.getByText("Processing").closest("[aria-current]");
    const results = screen.getByText("Results").closest("[aria-current]");

    expect(upload).toHaveClass("text-ink2");
    expect(processing).toHaveClass("text-primary");
    // Future-step labels are meaningful text: ink2, not the low-contrast ink3.
    expect(results).toHaveClass("text-ink2");
    expect(results).not.toHaveClass("text-ink3");
    expect(upload).toHaveAttribute("aria-current", "false");
    expect(processing).toHaveAttribute("aria-current", "step");
    expect(results).toHaveAttribute("aria-current", "false");

    rerender(<WizardSteps current={3} />);
    expect(screen.getByText("Results").closest("[aria-current]")).toHaveClass(
      "text-primary",
    );
  });

  it("completed steps show a check mark", () => {
    render(<WizardSteps current={3} />);
    // Two completed steps -> two check glyphs before the two first labels.
    const upload = screen.getByText("Upload").closest("[aria-current]")!;
    expect(upload.querySelector("[data-check]")).not.toBeNull();
  });

  it("fires onStepClick only for earlier steps", async () => {
    const user = userEvent.setup();
    const onStepClick = vi.fn();
    render(<WizardSteps current={2} onStepClick={onStepClick} />);

    await user.click(screen.getByText("Upload")); // backwards: allowed
    expect(onStepClick).toHaveBeenCalledWith(1);

    onStepClick.mockClear();
    await user.click(screen.getByText("Results")); // forwards: ignored
    expect(onStepClick).not.toHaveBeenCalled();
  });
});
