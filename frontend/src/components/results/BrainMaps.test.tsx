/**
 * BrainMaps: visit tabs expose proper tablist/tab/tabpanel semantics and
 * ArrowLeft/ArrowRight move the selection with focus following. The panel
 * renders a single truthful Grad-CAM overlay viewport (the backend sends one
 * composite per slice) plus a clamped slice counter.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { GradcamVisit } from "../../api/types";
import { BrainMaps } from "./BrainMaps";

function makeVisit(visit: string): GradcamVisit {
  return {
    visit,
    slices: ["30"],
    images: { "30": "aGVsbG8=" }, // bare base64 payload
  };
}

function makeVisitSlices(visit: string, slices: string[]): GradcamVisit {
  const images: Record<string, string> = {};
  for (const slice of slices) images[slice] = "aGVsbG8=";
  return { visit, slices, images };
}

describe("BrainMaps", () => {
  it("wires tabs to the active panel with aria-controls and a tabpanel", () => {
    render(<BrainMaps gradcam={[makeVisit("MR1"), makeVisit("MR2")]} />);

    const tabs = screen.getAllByRole("tab");
    expect(tabs).toHaveLength(2);
    for (const tab of tabs) {
      expect(tab).toHaveAttribute("aria-controls", "brainmaps-active-panel");
    }
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(tabs[1]).toHaveAttribute("aria-selected", "false");

    const panel = screen.getByRole("tabpanel");
    expect(panel).toHaveAttribute("id", "brainmaps-active-panel");
    expect(panel).toHaveAttribute("aria-labelledby", "brainmaps-tab-0");
  });

  it("moves selection with ArrowRight/ArrowLeft and moves focus along", async () => {
    const user = userEvent.setup();
    render(<BrainMaps gradcam={[makeVisit("MR1"), makeVisit("MR2")]} />);
    const tabs = screen.getAllByRole("tab");

    await user.click(tabs[0]);
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{ArrowRight}");
    expect(tabs[1]).toHaveAttribute("aria-selected", "true");
    expect(tabs[1]).toHaveFocus();
    expect(screen.getByRole("tabpanel")).toHaveAttribute(
      "aria-labelledby",
      "brainmaps-tab-1",
    );

    await user.keyboard("{ArrowLeft}");
    expect(tabs[0]).toHaveAttribute("aria-selected", "true");
    expect(tabs[0]).toHaveFocus();
  });

  it("renders a single truthful Grad-CAM overlay viewport with the legend", () => {
    render(<BrainMaps gradcam={[makeVisit("MR1")]} />);

    // One viewport only: the payload carries one composite per slice.
    expect(screen.getAllByRole("img")).toHaveLength(1);
    expect(screen.getByRole("img")).toHaveAttribute(
      "alt",
      "Grad-CAM overlay — MR1, slice 30",
    );
    expect(screen.getByText("Grad-CAM overlay")).toBeInTheDocument();
    expect(screen.getByText("Low importance")).toBeInTheDocument();
    expect(screen.getByText("High importance")).toBeInTheDocument();

    // No phantom panels labelled with content the payload does not carry.
    expect(screen.queryByText("Original T1 MRI")).not.toBeInTheDocument();
    expect(screen.queryByText("Fusion Overlay")).not.toBeInTheDocument();
  });

  it("counts slices by clamped index and keeps the raw label in parentheses", () => {
    const three = makeVisitSlices("MR1", ["10", "20", "30"]);
    const two = makeVisitSlices("MR2", ["30", "40"]);
    const { rerender } = render(<BrainMaps gradcam={[three]} />);

    const counter = () => screen.getByText(/Slice \d+ \/ \d+/);
    expect(counter()).toHaveTextContent("Slice 1 / 3 (10)");

    const slider = screen.getByRole("slider", { name: "Slice (10)" });
    fireEvent.change(slider, { target: { value: "2" } });
    expect(counter()).toHaveTextContent("Slice 3 / 3 (30)");

    // A re-render whose visit has fewer slices clamps the index instead of
    // mixing the slice label into the count ("Slice 30 / 2").
    rerender(<BrainMaps gradcam={[two]} />);
    expect(counter()).toHaveTextContent("Slice 2 / 2 (40)");
  });
});
