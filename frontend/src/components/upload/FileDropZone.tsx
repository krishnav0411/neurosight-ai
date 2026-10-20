/**
 * Drag-and-drop / picker for MRI scan files (Analyze 7.5 + NIfTI).
 *
 * Controlled: the parent owns the File[]; every selection or drop hands the
 * full new list to onFiles (no client-side rejection — the server validates;
 * the visit chips below are only a client-side pre-check).
 *
 * The hidden <input type="file"
          tabIndex={-1}> is a SIBLING of the drop zone (not a child),
 * so the zone's role="button" does not contain a focusable input; the zone
 * merely triggers input.click() via a ref.
 *
 * Folder drops are not supported: a dataTransfer entry that resolves to a
 * directory (or a drop carrying items but no files) shows an inline warning
 * instead of a bogus empty selection.
 *
 * Chips: one per detected OASIS visit — "label — 1 pair ✓" / "label — N pairs ✓"
 * when every half-pair is matched, "label — unpaired ⚠" otherwise.
 */
import { useRef, useState } from "react";
import { StatusPill } from "../Badges";
import { groupFiles } from "../../lib/visits";

/** True when the drop carries a directory (or items but no files). */
function isFolderDrop(e: React.DragEvent): boolean {
  const items = e.dataTransfer?.items;
  if (items == null || items.length === 0) return false;
  for (const item of Array.from(items)) {
    const entry = (
      item as DataTransferItem & {
        webkitGetAsEntry?: () => { isDirectory: boolean } | null;
      }
    ).webkitGetAsEntry?.();
    if (entry?.isDirectory) return true;
  }
  return e.dataTransfer.files.length === 0;
}

export function FileDropZone({
  files,
  onFiles,
}: {
  files: File[];
  onFiles(files: File[]): void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [folderDropped, setFolderDropped] = useState(false);

  const openPicker = () => inputRef.current?.click();

  const visits = groupFiles(
    files.map((f) => ({
      name: f.name,
      relativePath:
        (f as File & { webkitRelativePath?: string }).webkitRelativePath ||
        undefined,
    })),
  );

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openPicker();
    }
  };

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload scan files"
        onClick={openPicker}
        onKeyDown={onKeyDown}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (isFolderDrop(e)) {
            setFolderDropped(true);
            return;
          }
          setFolderDropped(false);
          onFiles(Array.from(e.dataTransfer.files));
        }}
        className="cursor-pointer rounded-card border-2 border-dashed border-hairline bg-white px-6 py-10 text-center transition-colors hover:border-primary hover:bg-blue-50/50 focus-visible:border-primary"
      >
        {/* Inline brain glyph (stroke icon, primary tone). */}
        <svg
          aria-hidden="true"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.8"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mx-auto mb-2 h-6 w-6 text-primary"
        >
          <path d="M12 4.5a2.5 2.5 0 0 0-4.96-.46 2.5 2.5 0 0 0-1.98 3.86 2.5 2.5 0 0 0-1.02 4.3 2.5 2.5 0 0 0 1.6 4.16A2.5 2.5 0 0 0 12 19.5Z" />
          <path d="M12 4.5a2.5 2.5 0 0 1 4.96-.46 2.5 2.5 0 0 1 1.98 3.86 2.5 2.5 0 0 1 1.02 4.3 2.5 2.5 0 0 1-1.6 4.16A2.5 2.5 0 0 1 12 19.5Z" />
          <path d="M12 4.5v15" />
        </svg>
        <p className="text-[14px] font-medium text-ink">
          Drop MRI scan files here
        </p>
        <p className="mt-1 text-[13px] text-ink3">
          Analyze 7.5 (.img/.hdr pairs) or NIfTI (.nii) — click to browse
        </p>
      </div>

      {/* Sibling of the zone (not nested inside role="button"). */}
      <input
        ref={inputRef}
        type="file"
        tabIndex={-1}
        multiple
        accept=".img,.hdr,.nii"
        className="sr-only"
        aria-label="Choose scan files"
        onChange={(e) => {
          setFolderDropped(false); // cleared on the next selection
          if (e.target.files) onFiles(Array.from(e.target.files));
          e.target.value = "";
        }}
      />

      {folderDropped ? (
        <p
          role="alert"
          className="mt-3 text-[13px] font-medium text-warning"
        >
          Folder upload is not supported — select files instead.
        </p>
      ) : null}

      {visits.length === 0 ? (
        <p className="mt-3 text-[13px] text-ink3">No visits detected yet</p>
      ) : null}

      {visits.length > 0 ? (
        <div className="mt-3">
          <p className="text-[13px] font-medium text-ink2">
            Detected {visits.length} MRI session
            {visits.length === 1 ? "" : "s"}
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {visits.map((v) =>
              v.complete ? (
                <StatusPill
                  key={v.label}
                  label={`${v.label} — ${v.pairs} pair${v.pairs === 1 ? "" : "s"} ✓`}
                  tone="success"
                />
              ) : (
                <StatusPill
                  key={v.label}
                  label={`${v.label} — unpaired ⚠`}
                  tone="warning"
                />
              ),
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
