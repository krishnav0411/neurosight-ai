/**
 * API key modal.
 *
 * Opens by itself in two situations:
 *  1. on mount when no key is stored in localStorage (first run), and
 *  2. whenever the API client dispatches the `neurosight:unauthorized`
 *     window event (a 401 response invalidated the stored key).
 *
 * Save persists the key via setApiKey() and closes. Escape closes without
 * saving. Focus moves to the password input while the dialog is open.
 */
import { useEffect, useState } from "react";
import { getApiKey, setApiKey, UNAUTHORIZED_EVENT } from "../api/client";
import { useModalA11y } from "./dialogs/useModalA11y";

export function ApiKeyModal() {
  const [open, setOpen] = useState(() => getApiKey() === "");
  const [value, setValue] = useState("");
  const dialogRef = useModalA11y(open, () => setOpen(false));

  // First run without a stored key, or a 401 from the backend.
  useEffect(() => {
    const onUnauthorized = () => setOpen(true);
    window.addEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, onUnauthorized);
  }, []);

  if (!open) return null;

  const save = () => {
    if (value.trim() === "") return; // same guard as the disabled Save button
    setApiKey(value.trim());
    setValue("");
    setOpen(false);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="API key required"
        tabIndex={-1}
        className="w-full max-w-md rounded-card border border-hairline bg-white p-6 shadow-modal"
      >
        <h2 className="text-[15px] font-semibold text-ink">API key required</h2>
        <p className="mt-2 text-[13px] leading-5 text-ink2">
          This demo requires an API key. Ask the study coordinator for the
          shared demo key, then paste it below to continue.
        </p>

        <label
          htmlFor="api-key-input"
          className="mt-4 block text-[13px] font-medium text-ink2"
        >
          API key
        </label>
        <input
          id="api-key-input"
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") save();
          }}
          autoComplete="off"
          className="mt-1.5 w-full rounded-control border border-hairline bg-white px-3 py-2 text-[13px] text-ink placeholder:text-ink3 focus:border-primary focus:outline-none"
          placeholder="Paste the demo API key"
        />

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={() => setOpen(false)}
            className="rounded-control border border-hairline bg-white px-3 py-1.5 text-[13px] font-medium text-ink2 hover:bg-slate-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={save}
            disabled={value.trim() === ""}
            className="rounded-control bg-primary px-3 py-1.5 text-[13px] font-medium text-white hover:bg-primary-dark disabled:cursor-not-allowed disabled:opacity-50"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
