/**
 * DisclaimerBanner renders the exact mandated sentence, verbatim.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DisclaimerBanner } from "./DisclaimerBanner";

const EXACT_TEXT =
  "Research use only. Not a medical device. This tool does not provide a diagnosis.";

describe("DisclaimerBanner", () => {
  it("renders the exact disclaimer sentence", () => {
    render(<DisclaimerBanner />);
    expect(screen.getByText(EXACT_TEXT)).toBeInTheDocument();
  });

  it("appends the className prop for spacing", () => {
    render(<DisclaimerBanner className="mt-8" />);
    const banner = screen.getByText(EXACT_TEXT);
    expect(banner).toHaveClass("mt-8");
  });
});
