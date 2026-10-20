/**
 * Pure patient-history helpers (TDD): progression alerting, timeline point
 * mapping and CSV export. Written against the AnalysisRecord contract.
 */
import { describe, expect, it } from "vitest";
import type { AnalysisRecord } from "../api/types";
import { historyToCsv, progressionAlert, toTimelinePoints } from "./patient";

function makeItem(overrides: Partial<AnalysisRecord> = {}): AnalysisRecord {
  return {
    analysis_id: "aaaaaaaa-0000-0000-0000-000000000000",
    status: "succeeded",
    created_at: "2026-01-01T00:00:00Z",
    ...overrides,
  };
}

function withDementedProb(prob: number): AnalysisRecord {
  return makeItem({
    result: {
      predicted_class: 1,
      class_labels: ["Non-Demented", "Demented"],
      probabilities: { "Non-Demented": 1 - prob, Demented: prob },
      raw_probabilities: { "Non-Demented": 1 - prob, Demented: prob },
      confidence: prob,
      confidence_band: "adequate",
      risk_level: "Low",
      input_mode: "mri_only",
      model_version: "v0.4.0-eval",
      visit_count: 1,
      visit_labels: ["MR1"],
      longitudinal: { trend: "Stable", per_visit_probability: [prob] },
      conversion_risk: { available: false },
      caveats: [],
    },
  });
}

describe("progressionAlert", () => {
  it("warns when the last rise exceeds 0.15", () => {
    const alert = progressionAlert([withDementedProb(0.3), withDementedProb(0.5)]);
    expect(alert).toEqual({
      level: "warning",
      message: "Increasing risk detected across analyses — consider follow-up",
    });
  });

  it("reports success for a small rise", () => {
    const alert = progressionAlert([withDementedProb(0.3), withDementedProb(0.35)]);
    expect(alert).toEqual({
      level: "success",
      message: "No significant progression detected",
    });
  });

  it("treats a rise just below the threshold as success (strictly greater than 0.15)", () => {
    const alert = progressionAlert([withDementedProb(0.2), withDementedProb(0.35)]);
    // 0.35 - 0.2 = 0.14999... in float, below the threshold.
    expect(alert?.level).toBe("success");
  });

  it("treats a mathematically exact 0.15 rise as success in both operand orders", () => {
    // 0.5 - 0.35 === 0.15000000000000002 in IEEE-754: float noise must not
    // turn an exact-threshold rise into a warning.
    expect(
      progressionAlert([withDementedProb(0.35), withDementedProb(0.5)])?.level,
    ).toBe("success");
    // Same noise on the mirrored operands: 0.45 - 0.3 === 0.15000000000000002.
    expect(
      progressionAlert([withDementedProb(0.3), withDementedProb(0.45)])?.level,
    ).toBe("success");
  });

  it("returns null with fewer than 2 usable values", () => {
    expect(progressionAlert([withDementedProb(0.4)])).toBeNull();
    expect(progressionAlert([])).toBeNull();
  });

  it("skips items without a Demented probability", () => {
    const history = [
      withDementedProb(0.2),
      makeItem({ analysis_id: "bbbbbbbb-0000-0000-0000-000000000000" }), // no result
      makeItem({
        analysis_id: "cccccccc-0000-0000-0000-000000000000",
        result: {
          ...withDementedProb(0.42).result!,
          probabilities: { "Non-Demented": 1 }, // no Demented key
        },
      }),
      withDementedProb(0.5),
    ];
    // Usable values: [0.2, 0.5] -> rise 0.3 -> warning.
    expect(progressionAlert(history)?.level).toBe("warning");
  });

  it("uses array order (chronological sorting is the caller's job)", () => {
    const alert = progressionAlert([withDementedProb(0.5), withDementedProb(0.2)]);
    // 0.2 - 0.5 is negative -> success, not warning.
    expect(alert?.level).toBe("success");
  });
});

describe("toTimelinePoints", () => {
  it("maps items with a Demented probability, labeling from completed_at", () => {
    const history = [
      makeItem({
        ...withDementedProb(0.2),
        analysis_id: "id-aaaaaaa-0001",
        completed_at: "2026-03-05T10:00:00Z",
      }),
      makeItem({
        ...withDementedProb(0.4),
        analysis_id: "id-bbbbbbb-0002",
      }),
    ];
    const points = toTimelinePoints(history);
    expect(points).toHaveLength(2);
    expect(points[0]).toEqual({
      ref: "id-aaaaaaa-0001",
      label: new Date("2026-03-05T10:00:00Z").toLocaleDateString(),
      prob: 0.2,
    });
    // No completed_at -> analysis_id.slice(0, 8) fallback.
    expect(points[1]).toEqual({
      ref: "id-bbbbbbb-0002",
      label: "id-bbbbb",
      prob: 0.4,
    });
  });

  it("omits items without a Demented probability", () => {
    const history = [
      makeItem({ analysis_id: "id-aaaaaaa-0001" }),
      withDementedProb(0.5),
    ];
    const points = toTimelinePoints(history);
    expect(points).toHaveLength(1);
    expect(points[0].prob).toBe(0.5);
    expect(points[0].ref).toBe("aaaaaaaa-0000-0000-0000-000000000000");
  });
});

describe("historyToCsv", () => {
  it("writes the exact header and one row per item", () => {
    const csv = historyToCsv([
      makeItem({ analysis_id: "id-1" }),
      makeItem({ analysis_id: "id-2", status: "failed" }),
    ]);
    const lines = csv.split("\n");
    expect(lines[0]).toBe(
      "analysis_id,created_at,status,predicted_class,confidence,risk_level,demented_prob",
    );
    expect(lines).toHaveLength(3);
    expect(lines[1]).toBe("id-1,2026-01-01T00:00:00Z,succeeded,,,,");
    expect(lines[2]).toBe("id-2,2026-01-01T00:00:00Z,failed,,,,");
  });

  it("resolves the predicted class label and probability fields", () => {
    const csv = historyToCsv([withDementedProb(0.87)]);
    expect(csv.split("\n")[1]).toBe(
      "aaaaaaaa-0000-0000-0000-000000000000,2026-01-01T00:00:00Z,succeeded,Demented,0.87,Low,0.87",
    );
  });

  it("quotes fields containing commas and doubles embedded quotes", () => {
    const item = makeItem({
      analysis_id: "id-1",
      result: {
        ...withDementedProb(0.5).result!,
        class_labels: ["Non-Demented", 'He said "yes", demented'],
        predicted_class: 1,
        risk_level: 'High" risk',
      },
    });
    const csv = historyToCsv([item]);
    const row = csv.split("\n")[1];
    expect(row).toContain('"He said ""yes"", demented"');
    expect(row).toContain('"High"" risk"');
  });

  it("prefixes formula-dangerous leading characters with an apostrophe", () => {
    for (const prefix of ["=", "+", "-", "@"]) {
      const csv = historyToCsv([makeItem({ analysis_id: `${prefix}cmd` })]);
      const row = csv.split("\n")[1];
      expect(row).toContain(`'${prefix}cmd`);
      expect(row).not.toMatch(new RegExp(`,\\${prefix}cmd`));
    }
  });
});
