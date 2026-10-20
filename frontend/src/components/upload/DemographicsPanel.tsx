/**
 * Collapsible "Optional demographics" panel for the analyze wizard.
 *
 * Fields mirror the OASIS demographic columns the backend accepts. Every
 * input carries an aria-label and a title tooltip. The panel shows the
 * active input mode (derived in the parent) as a chip, and hosts the
 * experimental clinical-scores toggle: when enabled, an always-visible
 * amber warning explains that clinical scores dominate the prediction.
 */
import { useState } from "react";
import type { Demographics } from "../../api/types";
import type { InputMode } from "../../lib/mode";

interface FieldSpec {
  key: keyof Demographics;
  label: string;
  title: string;
  step?: string;
}

const NUMERIC_FIELDS: FieldSpec[] = [
  { key: "age", label: "Age", title: "Age at scan (years)" },
  {
    key: "educ",
    label: "Education (years)",
    title: "Years of education",
  },
  {
    key: "ses",
    label: "SES",
    title: "Socioeconomic status (Hollingshead index)",
  },
  {
    key: "mmse",
    label: "MMSE",
    title: "Mini-Mental State Examination (0–30)",
  },
  {
    key: "cdr",
    label: "CDR",
    title: "Clinical Dementia Rating (0, 0.5, 1, 2)",
    step: "0.5",
  },
  {
    key: "etiv",
    label: "eTIV",
    title: "Estimated total intracranial volume (cm³)",
  },
  {
    key: "nwbv",
    label: "nWBV",
    title: "Normalized whole-brain volume",
    step: "0.001",
  },
  {
    key: "asf",
    label: "ASF",
    title: "Atlas scaling factor",
    step: "0.01",
  },
];

const INPUT_CLASS =
  "w-full rounded-control border border-hairline bg-white px-2.5 py-1.5 text-[13px] text-ink focus:border-primary focus:outline-none";

export function DemographicsPanel({
  values,
  onChange,
  mode,
  experimental,
  onExperimentalChange,
}: {
  values: Demographics;
  onChange(next: Demographics): void;
  /** Input mode derived in the parent from values + experimental. */
  mode: InputMode;
  experimental: boolean;
  onExperimentalChange(next: boolean): void;
}) {
  const [open, setOpen] = useState(false);

  const setField = (key: keyof Demographics, raw: string, numeric: boolean) => {
    const next: Demographics = { ...values };
    if (raw === "") {
      delete next[key];
    } else if (!numeric) {
      (next as Record<string, unknown>)[key] = raw;
    } else {
      // Numeric inputs can still hold junk (e.g. "1e", "--"): only store a
      // finite number, otherwise leave the field unset.
      const v = Number(raw);
      (next as Record<string, unknown>)[key] =
        Number.isFinite(v) && v !== null ? v : undefined;
    }
    onChange(next);
  };

  const modeChip =
    mode === "mri_only" ? (
      <span className="chip">Input mode: MRI-only (demographics ignored)</span>
    ) : (
      <span className="chip">Input mode: MRI + demographics</span>
    );

  return (
    <div className="rounded-card border border-hairline bg-white">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-[13px] font-medium text-ink2 hover:text-ink"
      >
        <span aria-hidden="true" className="text-ink3">
          {open ? "▾" : "▸"}
        </span>
        Optional demographics
      </button>

      {open ? (
        <div className="border-t border-hairline px-4 pb-4 pt-3">
          {modeChip}

          <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <div>
              <label
                htmlFor="demo-sex"
                className="mb-1 block text-[12px] text-ink2"
                title="Sex of the subject"
              >
                Sex
              </label>
              <select
                id="demo-sex"
                aria-label="Sex"
                title="Sex of the subject (M/F)"
                value={values.sex ?? ""}
                onChange={(e) => setField("sex", e.target.value, false)}
                className={INPUT_CLASS}
              >
                <option value="">—</option>
                <option value="M">M</option>
                <option value="F">F</option>
              </select>
            </div>

            {NUMERIC_FIELDS.map((f) => (
              <div key={f.key}>
                <label
                  htmlFor={`demo-${f.key}`}
                  className="mb-1 block text-[12px] text-ink2"
                  title={f.title}
                >
                  {f.label}
                </label>
                <input
                  id={`demo-${f.key}`}
                  type="number"
                  inputMode="decimal"
                  step={f.step}
                  min="0"
                  aria-label={f.label}
                  title={f.title}
                  value={values[f.key] ?? ""}
                  onChange={(e) => setField(f.key, e.target.value, true)}
                  className={INPUT_CLASS}
                />
              </div>
            ))}
          </div>

          <label className="mt-4 flex items-start gap-2 text-[13px] text-ink2">
            <input
              type="checkbox"
              checked={experimental}
              onChange={(e) => onExperimentalChange(e.target.checked)}
              className="mt-0.5"
              aria-label="Include clinical scores (MMSE/CDR) — experimental"
            />
            <span>
              Include clinical scores (MMSE/CDR) — experimental
              {experimental ? (
                <span className="pill pill-warning ml-2 align-middle">
                  Clinical scores strongly determine the diagnosis — do not
                  compare with MRI-only results.
                </span>
              ) : null}
            </span>
          </label>
        </div>
      ) : null}
    </div>
  );
}
