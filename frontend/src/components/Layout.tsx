/**
 * App shell: responsive sidebar (fixed 240px on desktop, collapsible menu on mobile),
 * top bar with telemetry chip and backend health dot, and centered content column.
 * Hosts the route outlet and the API key modal.
 */
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { useHealth } from "../api/hooks";
import { TelemetryChip } from "./Badges";
import { ApiKeyModal } from "./ApiKeyModal";

const NAV_ITEMS = [
  { to: "/analyze", label: "Analyze" },
  { to: "/patient", label: "Patient history" },
  { to: "/about", label: "About" },
] as const;

function navLinkClass({ isActive }: { isActive: boolean }) {
  const base =
    "block rounded-control px-3 py-2 text-[13px] font-medium transition-colors";
  return isActive
    ? `${base} bg-blue-50 text-primary`
    : `${base} text-ink2 hover:bg-slate-50`;
}

export function Layout() {
  const health = useHealth();
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const dotClass = health.isError
    ? "bg-danger"
    : health.isSuccess
      ? "bg-success"
      : "bg-ink3";

  const healthText = health.isError
    ? "Backend unreachable"
    : health.isSuccess
      ? "Backend ok"
      : "Checking backend";

  return (
    <div className="min-h-screen">
      {/* Desktop sidebar */}
      <aside className="hidden md:block fixed inset-y-0 left-0 w-60 border-r border-hairline bg-white z-30">
        <div className="px-5 pb-4 pt-5">
          <h1 className="text-[16px] font-semibold leading-5 text-ink">
            NeuroSight AI
          </h1>
          <p className="mt-0.5 text-[11px] leading-4 text-ink2">
            Longitudinal Brain MRI Analysis
          </p>
        </div>
        <nav aria-label="Primary" className="mt-2 px-3">
          <ul className="space-y-1">
            {NAV_ITEMS.map((item) => (
              <li key={item.to}>
                <NavLink to={item.to} className={navLinkClass}>
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </aside>

      {/* Main content column */}
      <div className="md:ml-60 min-h-screen flex flex-col">
        <header className="flex h-14 items-center justify-between md:justify-end gap-3 border-b border-hairline bg-white px-4 sm:px-6 md:px-8">
          {/* Mobile hamburger & brand */}
          <div className="flex items-center gap-2.5 md:hidden">
            <button
              type="button"
              onClick={() => setMobileNavOpen((o) => !o)}
              aria-label={mobileNavOpen ? "Close navigation" : "Open navigation"}
              aria-expanded={mobileNavOpen}
              className="rounded-control p-1.5 text-ink2 hover:bg-slate-100 hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2"
            >
              <svg
                className="h-5 w-5"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                aria-hidden="true"
              >
                {mobileNavOpen ? (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M6 18L18 6M6 6l12 12"
                  />
                ) : (
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                )}
              </svg>
            </button>
            <span className="text-[14px] font-semibold text-ink">NeuroSight</span>
          </div>

          <div className="flex items-center gap-3">
            {health.data?.model_version ? (
              <TelemetryChip>{health.data.model_version}</TelemetryChip>
            ) : null}
            <span
              role="status"
              aria-label="Backend status"
              className={`h-2 w-2 rounded-full ${dotClass}`}
            >
              <span className="sr-only">{healthText}</span>
            </span>
          </div>
        </header>

        {/* Mobile slide-down navigation */}
        {mobileNavOpen ? (
          <div className="border-b border-hairline bg-white px-4 py-3 md:hidden shadow-sm">
            <p className="px-3 pb-2 text-[11px] font-medium uppercase tracking-wider text-ink3">
              Navigation
            </p>
            <nav aria-label="Mobile Navigation" className="space-y-1">
              {NAV_ITEMS.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  onClick={() => setMobileNavOpen(false)}
                  className={navLinkClass}
                >
                  {item.label}
                </NavLink>
              ))}
            </nav>
          </div>
        ) : null}

        {health.isError ? (
          <div
            role="status"
            className="border-b border-amber-200 bg-amber-50/80 px-4 py-2 text-center text-[12px] text-amber-900"
          >
            <span className="font-medium">Backend offline:</span> Live analysis requires a running NeuroSight API instance. You can still inspect interface structure, view options, and documentation.
          </div>
        ) : null}

        <main className="flex-1 p-4 sm:p-6 md:p-8">
          <div className="mx-auto max-w-[1200px]">
            <Outlet />
          </div>
        </main>
      </div>

      <ApiKeyModal />
    </div>
  );
}
