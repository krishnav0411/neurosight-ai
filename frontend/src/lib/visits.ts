/**
 * Visit grouping for OASIS-style MRI file listings.
 *
 * A "visit" is identified by an explicit OASIS session id of the form
 * OAS2_XXXX_MRn found anywhere in the file's relative path (or bare name).
 * Files without such a key are lumped into the pseudo-visit "SINGLE_VISIT".
 *
 * Within a visit, Analyze 7.5 half-pairs are matched: a `.img` completes
 * against the `.hdr` sharing its base name (filename minus final extension,
 * compared case-insensitively). A `.nii` is self-contained. Anything else is
 * ignored entirely.
 */

export interface VisitFile {
  name: string;
  relativePath?: string;
}

export interface VisitGroup {
  label: string;
  files: number;
  pairs: number;
  orphans: string[];
  complete: boolean;
}

export const SINGLE_VISIT = "SINGLE_VISIT";

const VISIT_ID_RE = /(?:OAS2_\d{4}_MR\d+|SYNTH_\d{4}_MR\d+|sub-[a-zA-Z0-9]+_ses-\d+)/i;
const SUPPORTED_EXT_RE = /\.(img|hdr|nii)$/i;

/** The visit label for a single file (SINGLE_VISIT when no OASIS id matches). */
export function visitKeyOf(file: VisitFile): string {
  const source = file.relativePath ?? file.name;
  const match = source.match(VISIT_ID_RE);
  return match ? match[0] : SINGLE_VISIT;
}

function baseNameOf(name: string): string {
  return name.replace(/\.[^.]+$/, "");
}

/** Group files into visits with pair/orphan bookkeeping, sorted by label. */
export function groupFiles(files: VisitFile[]): VisitGroup[] {
  const supported = files.filter((f) => SUPPORTED_EXT_RE.test(f.name));

  const buckets = new Map<string, VisitFile[]>();
  for (const file of supported) {
    const label = visitKeyOf(file);
    const bucket = buckets.get(label);
    if (bucket) bucket.push(file);
    else buckets.set(label, [file]);
  }

  const groups: VisitGroup[] = [];
  for (const [label, bucket] of buckets) {
    const imgs = new Map<string, string[]>(); // lowercase base name -> file names
    const hdrs = new Map<string, string[]>();
    let niiCount = 0;

    for (const file of bucket) {
      const lower = file.name.toLowerCase();
      if (lower.endsWith(".nii")) {
        niiCount += 1;
      } else if (lower.endsWith(".img")) {
        const base = baseNameOf(lower);
        const list = imgs.get(base);
        if (list) list.push(file.name);
        else imgs.set(base, [file.name]);
      } else {
        const base = baseNameOf(lower);
        const list = hdrs.get(base);
        if (list) list.push(file.name);
        else hdrs.set(base, [file.name]);
      }
    }

    let pairs = niiCount; // .nii files are self-contained
    const orphans: string[] = [];

    for (const [base, imgNames] of imgs) {
      const hdrCount = hdrs.get(base)?.length ?? 0;
      pairs += Math.min(imgNames.length, hdrCount);
      if (imgNames.length > hdrCount) orphans.push(...imgNames.slice(hdrCount));
    }
    for (const [base, hdrNames] of hdrs) {
      const imgCount = imgs.get(base)?.length ?? 0;
      if (hdrNames.length > imgCount) orphans.push(...hdrNames.slice(imgCount));
    }

    groups.push({
      label,
      files: bucket.length,
      pairs,
      orphans,
      complete: orphans.length === 0 && pairs > 0,
    });
  }

  groups.sort((a, b) => (a.label < b.label ? -1 : a.label > b.label ? 1 : 0));
  return groups;
}

/** One visit label per input file, parallel to `files` (for visit_labels_json). */
export function toVisitLabels(files: VisitFile[]): string[] {
  return files.map(visitKeyOf);
}
