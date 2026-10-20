import { describe, expect, it } from "vitest";
import { groupFiles, toVisitLabels } from "./visits";

describe("groupFiles", () => {
  it("splits two OASIS-named files with different MR ids into two groups, sorted by label", () => {
    const groups = groupFiles([
      { name: "OAS2_0002_MR1_mpr-1_nifti.img" },
      { name: "OAS2_0001_MR1_mpr-1_nifti.img" },
      { name: "OAS2_0001_MR1_mpr-1_nifti.hdr" },
    ]);

    expect(groups.map((g) => g.label)).toEqual(["OAS2_0001_MR1", "OAS2_0002_MR1"]);
    expect(groups[0]).toEqual({
      label: "OAS2_0001_MR1",
      files: 2,
      pairs: 1,
      orphans: [],
      complete: true,
    });
    expect(groups[1]).toEqual({
      label: "OAS2_0002_MR1",
      files: 1,
      pairs: 0,
      orphans: ["OAS2_0002_MR1_mpr-1_nifti.img"],
      complete: false,
    });
  });

  it("counts mpr-1.nifti.img + mpr-1.nifti.hdr as one complete pair via relativePath", () => {
    const groups = groupFiles([
      { name: "mpr-1.nifti.img", relativePath: "OAS2_0005_MR1/mpr-1.nifti.img" },
      { name: "mpr-1.nifti.hdr", relativePath: "OAS2_0005_MR1/mpr-1.nifti.hdr" },
    ]);

    expect(groups).toEqual([
      { label: "OAS2_0005_MR1", files: 2, pairs: 1, orphans: [], complete: true },
    ]);
  });

  it("marks a lone .img as an orphan with complete=false", () => {
    const groups = groupFiles([
      { name: "mpr-1.nifti.img", relativePath: "OAS2_0005_MR1/mpr-1.nifti.img" },
    ]);

    expect(groups[0].pairs).toBe(0);
    expect(groups[0].orphans).toEqual(["mpr-1.nifti.img"]);
    expect(groups[0].complete).toBe(false);
  });

  it("counts multiple pairs and self-contained .nii files within one visit", () => {
    const groups = groupFiles([
      { name: "mpr-1.nifti.img", relativePath: "OAS2_0010_MR1/mpr-1.nifti.img" },
      { name: "mpr-1.nifti.hdr", relativePath: "OAS2_0010_MR1/mpr-1.nifti.hdr" },
      { name: "mpr-2.nifti.img", relativePath: "OAS2_0010_MR1/mpr-2.nifti.img" },
      { name: "mpr-2.nifti.hdr", relativePath: "OAS2_0010_MR1/mpr-2.nifti.hdr" },
      { name: "t2.nii", relativePath: "OAS2_0010_MR1/t2.nii" },
    ]);

    expect(groups[0].files).toBe(5);
    expect(groups[0].pairs).toBe(3);
    expect(groups[0].orphans).toEqual([]);
    expect(groups[0].complete).toBe(true);
  });

  it("treats a .nii as self-contained", () => {
    const groups = groupFiles([{ name: "OAS2_0006_MR1_mpr-1.nii" }]);

    expect(groups).toEqual([
      { label: "OAS2_0006_MR1", files: 1, pairs: 1, orphans: [], complete: true },
    ]);
  });

  it("recognizes synthetic session identifiers (SYNTH_XXXX_MRn and BIDS)", () => {
    const groups = groupFiles([
      { name: "SYNTH_0001_MR1_mpr-1.nii" },
      { name: "SYNTH_0001_MR2_mpr-1.nii" },
    ]);

    expect(groups.map((g) => g.label)).toEqual(["SYNTH_0001_MR1", "SYNTH_0001_MR2"]);
    expect(groups[0].complete).toBe(true);
    expect(groups[1].complete).toBe(true);
  });

  it("lands files with no OAS2 pattern in SINGLE_VISIT", () => {
    const groups = groupFiles([{ name: "scan.img" }, { name: "scan.hdr" }]);

    expect(groups).toEqual([
      { label: "SINGLE_VISIT", files: 2, pairs: 1, orphans: [], complete: true },
    ]);
  });

  it("prefers relativePath over name when both contain different ids", () => {
    const groups = groupFiles([
      { name: "OAS2_0001_MR1_a.img", relativePath: "OAS2_0009_MR2/a.img" },
    ]);

    expect(groups.map((g) => g.label)).toEqual(["OAS2_0009_MR2"]);
  });

  it("pairs .img/.hdr case-insensitively on extension and base name", () => {
    const groups = groupFiles([
      { name: "MPR-1.Nifti.IMG", relativePath: "OAS2_0008_MR1/MPR-1.Nifti.IMG" },
      { name: "mpr-1.nifti.hdr", relativePath: "OAS2_0008_MR1/mpr-1.nifti.hdr" },
    ]);

    expect(groups[0].pairs).toBe(1);
    expect(groups[0].orphans).toEqual([]);
    expect(groups[0].complete).toBe(true);
  });

  it("ignores files with unsupported extensions entirely", () => {
    const groups = groupFiles([
      { name: "notes.txt", relativePath: "OAS2_0001_MR1/notes.txt" },
      { name: "scan.img", relativePath: "OAS2_0001_MR1/scan.img" },
    ]);

    expect(groups).toEqual([
      { label: "OAS2_0001_MR1", files: 1, pairs: 0, orphans: ["scan.img"], complete: false },
    ]);
    expect(groupFiles([{ name: "notes.txt" }])).toEqual([]);
  });
});

describe("toVisitLabels", () => {
  it("returns one label per input file, in order", () => {
    const files = [
      { name: "a.img", relativePath: "OAS2_0001_MR1/a.img" },
      { name: "OAS2_0003_MR1_b.nii" },
      { name: "c.txt" },
    ];

    expect(toVisitLabels(files)).toEqual(["OAS2_0001_MR1", "OAS2_0003_MR1", "SINGLE_VISIT"]);
  });

  it("uses relativePath when present, falling back to SINGLE_VISIT", () => {
    const files = [
      { name: "OAS2_0001_MR1_x.img", relativePath: "OAS2_0007_MR1/x.img" },
      { name: "unmatched.hdr" },
    ];

    expect(toVisitLabels(files)).toEqual(["OAS2_0007_MR1", "SINGLE_VISIT"]);
  });
});
