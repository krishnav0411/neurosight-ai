/**
 * Accessibility plumbing for modal dialogs.
 *
 * Given an `open` flag and an `onClose` callback, returns a ref to attach to
 * the dialog container element. While open the hook:
 *  - focuses the first focusable element inside the container on open
 *    (falling back to the container itself, which should carry tabIndex={-1}),
 *  - traps Tab / Shift+Tab within the container,
 *  - closes the dialog on Escape,
 *  - restores focus to the previously focused element on close.
 *
 * `onClose` may be an inline closure: it is kept in a ref so the effect only
 * depends on `open` and re-running renders never steal focus.
 */
import { useEffect, useRef } from "react";

const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(", ");

export function useModalA11y(
  open: boolean,
  onClose: () => void,
): React.RefObject<HTMLDivElement> {
  const ref = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  const restoreFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    onCloseRef.current = onClose;
  });

  useEffect(() => {
    if (!open) return;
    const container = ref.current;
    restoreFocusRef.current =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    const focusables = () =>
      container == null
        ? []
        : Array.from(
            container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR),
          );

    // (a) initial focus
    const first = focusables()[0];
    if (first) {
      first.focus();
    } else {
      container?.focus();
    }

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onCloseRef.current();
        return;
      }
      if (e.key !== "Tab" || container == null) return;

      // (b) trap Tab within the container
      const list = focusables();
      if (list.length === 0) {
        e.preventDefault();
        return;
      }
      const firstEl = list[0];
      const lastEl = list[list.length - 1];
      const active = document.activeElement;
      const inside = active != null && container.contains(active);
      if (e.shiftKey) {
        if (!inside || active === firstEl) {
          e.preventDefault();
          lastEl.focus();
        }
      } else if (!inside || active === lastEl) {
        e.preventDefault();
        firstEl.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("keydown", onKeyDown, true);
      // (d) restore focus when the dialog closes
      restoreFocusRef.current?.focus();
      restoreFocusRef.current = null;
    };
  }, [open]);

  return ref;
}
