import { describe, expect, it } from "vitest";
import { deriveMode, hasAnyDemographics } from "./mode";

describe("deriveMode", () => {
  it("mri_only when no demographics and no experimental toggle", () => {
    expect(deriveMode(false, false)).toBe("mri_only");
  });

  it("mri_demo when any demographics field is filled (no experimental)", () => {
    expect(deriveMode(true, false)).toBe("mri_demo");
  });

  it("mri_clinical when the experimental clinical-scores toggle is on, even without demographics", () => {
    expect(deriveMode(false, true)).toBe("mri_clinical");
  });

  it("mri_clinical wins when both demographics and experimental are on", () => {
    expect(deriveMode(true, true)).toBe("mri_clinical");
  });
});

describe("hasAnyDemographics", () => {
  it("is false for an empty payload or whitespace-only strings", () => {
    expect(hasAnyDemographics({})).toBe(false);
    expect(hasAnyDemographics({ age: "  ", sex: undefined })).toBe(false);
  });

  it("is true once any field carries a value", () => {
    expect(hasAnyDemographics({ mmse: 27 })).toBe(true);
    expect(hasAnyDemographics({ sex: "F" })).toBe(true);
  });
});
