/**
 * FileDropZone: drag-and-drop + picker for MRI scan files with client-side
 * visit chips (OASIS pair pre-check).
 */
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { FileDropZone } from "./FileDropZone";

function makeFile(name: string, relativePath?: string): File {
  const file = new File(["dummy"], name);
  if (relativePath) {
    Object.defineProperty(file, "webkitRelativePath", { value: relativePath });
  }
  return file;
}

/** Controlled harness mirroring how AnalyzePage uses the zone. */
function Harness({ onFiles }: { onFiles?: (files: File[]) => void }) {
  const [files, setFiles] = useState<File[]>([]);
  return (
    <FileDropZone
      files={files}
      onFiles={(f) => {
        setFiles(f);
        onFiles?.(f);
      }}
    />
  );
}

describe("FileDropZone", () => {
  it("has a zone with role button and aria-label 'Upload scan files'", () => {
    render(<Harness />);
    expect(screen.getByRole("button", { name: "Upload scan files" })).toBeInTheDocument();
  });

  it("shows 'No visits detected yet' when no files are present", () => {
    render(<Harness />);
    expect(screen.getByText("No visits detected yet")).toBeInTheDocument();
    expect(screen.queryByText(/Detected \d+ MRI sessions/)).toBeNull();
  });

  it("selecting two complete OASIS visits shows 'Detected 2 MRI sessions' and two pair chips", async () => {
    const user = userEvent.setup();
    const onFiles = vi.fn();
    render(<Harness onFiles={onFiles} />);

    const input = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    await user.upload(input, [
      makeFile("mpr-1.nifti.img", "OAS2_0001_MR1/mpr-1.nifti.img"),
      makeFile("mpr-1.nifti.hdr", "OAS2_0001_MR1/mpr-1.nifti.hdr"),
      makeFile("mpr-1.nifti.img", "OAS2_0002_MR1/mpr-1.nifti.img"),
      makeFile("mpr-1.nifti.hdr", "OAS2_0002_MR1/mpr-1.nifti.hdr"),
    ]);

    expect(onFiles).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Detected 2 MRI sessions")).toBeInTheDocument();
    const chips = screen.getAllByText(/— 1 pair ✓/);
    expect(chips).toHaveLength(2);
    for (const chip of chips) {
      // Visit chips are StatusPill-like: dot + success pill for complete pairs.
      expect(chip.parentElement).toHaveClass("pill", "pill-success");
    }
    expect(screen.getByText(/OAS2_0001_MR1/)).toBeInTheDocument();
    expect(screen.getByText(/OAS2_0002_MR1/)).toBeInTheDocument();
  });

  it("dropping files via dataTransfer also populates the visit chips", () => {
    const onFiles = vi.fn();
    render(<Harness onFiles={onFiles} />);
    const zone = screen.getByRole("button", { name: "Upload scan files" });

    fireEvent.drop(zone, {
      dataTransfer: {
        files: [
          makeFile("a.nii", "OAS2_0003_MR1/a.nii"),
          makeFile("b.nii", "OAS2_0004_MR1/b.nii"),
        ],
      },
    });

    expect(onFiles).toHaveBeenCalledTimes(1);
    expect(screen.getByText("Detected 2 MRI sessions")).toBeInTheDocument();
    expect(screen.getAllByText(/— 1 pair ✓/)).toHaveLength(2);
  });

  it("a lone .img file shows the unpaired warning chip", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const input = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    await user.upload(input, [
      makeFile("mpr-1.nifti.img", "OAS2_0005_MR1/mpr-1.nifti.img"),
    ]);

    const warning = screen.getByText(/OAS2_0005_MR1 — unpaired ⚠/);
    expect(warning.parentElement).toHaveClass("pill", "pill-warning");
    // "Detected 1 MRI session" (singular) still announces the visit.
    expect(screen.getByText("Detected 1 MRI session")).toBeInTheDocument();
  });

  it("a dropped directory shows the folder warning and never calls onFiles", () => {
    const onFiles = vi.fn();
    render(<Harness onFiles={onFiles} />);
    const zone = screen.getByRole("button", { name: "Upload scan files" });

    fireEvent.drop(zone, {
      dataTransfer: {
        files: [],
        items: [
          {
            webkitGetAsEntry: () => ({ isDirectory: true }),
          },
        ],
      },
    });

    expect(onFiles).not.toHaveBeenCalled();
    const warning = screen.getByRole("alert");
    expect(warning).toHaveTextContent(
      "Folder upload is not supported — select files instead.",
    );
  });

  it("a drop with items but no files is treated as a folder drop", () => {
    const onFiles = vi.fn();
    render(<Harness onFiles={onFiles} />);
    const zone = screen.getByRole("button", { name: "Upload scan files" });

    fireEvent.drop(zone, {
      dataTransfer: {
        files: [],
        items: [{ webkitGetAsEntry: () => ({ isDirectory: false }) }],
      },
    });

    expect(onFiles).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toBeInTheDocument();
  });

  it("the folder warning is cleared by the next successful selection", async () => {
    const user = userEvent.setup();
    const onFiles = vi.fn();
    render(<Harness onFiles={onFiles} />);
    const zone = screen.getByRole("button", { name: "Upload scan files" });

    fireEvent.drop(zone, {
      dataTransfer: {
        files: [],
        items: [{ webkitGetAsEntry: () => ({ isDirectory: true }) }],
      },
    });
    expect(screen.getByRole("alert")).toBeInTheDocument();

    const input = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    await user.upload(input, [makeFile("a.nii", "OAS2_0006_MR1/a.nii")]);

    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByText("Detected 1 MRI session")).toBeInTheDocument();
  });

  it("Enter on the focused zone triggers the hidden file input", async () => {
    const user = userEvent.setup();
    render(<Harness />);
    const input = document.querySelector(
      'input[type="file"]',
    ) as HTMLInputElement;
    const clickSpy = vi.spyOn(input, "click");

    const zone = screen.getByRole("button", { name: "Upload scan files" });
    zone.focus();
    await user.keyboard("{Enter}");
    expect(clickSpy).toHaveBeenCalled();

    clickSpy.mockClear();
    await user.keyboard(" ");
    expect(clickSpy).toHaveBeenCalled();
  });
});
