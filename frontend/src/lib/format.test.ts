import { describe, expect, it } from "vitest";
import { deltaTiles, pct, pct1, riskColor, trendBadge } from "./format";

describe("pct", () => {
  it("formats a probability as a whole percent with no decimals", () => {
    expect(pct(0.871)).toBe("87%");
    expect(pct(0.084)).toBe("8%");
    expect(pct(0)).toBe("0%");
    expect(pct(1)).toBe("100%");
  });

  it("rounds half away from zero", () => {
    expect(pct(0.875)).toBe("88%");
    expect(pct(0.125)).toBe("13%");
  });
});

describe("riskColor", () => {
  it("maps the DEMENTED probability to severity colors", () => {
    expect(riskColor(0.8)).toBe("danger");
    expect(riskColor(0.95)).toBe("danger");
    expect(riskColor(0.5)).toBe("warning");
    expect(riskColor(0.79)).toBe("warning");
    expect(riskColor(0.49)).toBe("success");
    expect(riskColor(0.1)).toBe("success");
  });
});

describe("pct1", () => {
  it("formats a probability as a percent with exactly one decimal", () => {
    expect(pct1(0.87)).toBe("87.0%");
    expect(pct1(0.877)).toBe("87.7%");
    expect(pct1(0)).toBe("0.0%");
    expect(pct1(1)).toBe("100.0%");
  });

  it("rounds half away from zero at the first decimal", () => {
    expect(pct1(0.8755)).toBe("87.6%");
    expect(pct1(0.8754)).toBe("87.5%");
  });
});

describe("deltaTiles", () => {
  it("rising demented probability: positive delta, danger tone", () => {
    expect(deltaTiles([0.21, 0.55, 0.87])).toEqual({
      latest: "87%",
      delta: "+32%",
      deltaTone: "danger",
    });
  });

  it("falling demented probability: negative delta, success tone", () => {
    expect(deltaTiles([0.87, 0.55])).toEqual({
      latest: "55%",
      delta: "-32%",
      deltaTone: "success",
    });
  });

  it("flat series: +0% with success tone", () => {
    expect(deltaTiles([0.5, 0.5])).toEqual({
      latest: "50%",
      delta: "+0%",
      deltaTone: "success",
    });
  });

  it("a single point has no previous visit: +0% and the point itself", () => {
    expect(deltaTiles([0.3])).toEqual({
      latest: "30%",
      delta: "+0%",
      deltaTone: "success",
    });
  });

  it("an empty series renders honest placeholders", () => {
    expect(deltaTiles([])).toEqual({
      latest: "—",
      delta: "—",
      deltaTone: "success",
    });
  });
});

describe("trendBadge", () => {
  it("maps known trends to pill classes", () => {
    expect(trendBadge("Stable")).toEqual({ label: "Stable", cls: "pill-success" });
    expect(trendBadge("Declining")).toEqual({ label: "Declining", cls: "pill-warning" });
    expect(trendBadge("Rapidly Declining")).toEqual({
      label: "Rapidly Declining",
      cls: "pill-warning",
    });
  });

  it("falls back to neutral for unknown trends", () => {
    expect(trendBadge("Improving")).toEqual({ label: "Improving", cls: "pill-neutral" });
    expect(trendBadge("")).toEqual({ label: "", cls: "pill-neutral" });
  });
});
