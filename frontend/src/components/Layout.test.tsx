/**
 * Layout shell: sidebar nav, top bar (telemetry chip + health dot), and the
 * ApiKeyModal gating (shown on first run without a stored key, hidden when a
 * key exists).
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UseQueryResult } from "@tanstack/react-query";
import type { Health } from "../api/types";
import { Layout } from "./Layout";

vi.mock("../api/hooks", () => ({
  useHealth: vi.fn(),
}));

import { useHealth } from "../api/hooks";
const useHealthMock = vi.mocked(useHealth);

function healthResult(
  overrides: Partial<UseQueryResult<Health, Error>> = {},
): UseQueryResult<Health, Error> {
  const base = {
    data: { status: "ok", model_version: "v0.1.0", uptime_s: 12, device: "cpu" },
    isLoading: false,
    isPending: false,
    isSuccess: true,
    isError: false,
  };
  return { ...base, ...overrides } as unknown as UseQueryResult<
    Health,
    Error
  >;
}

function renderLayout() {
  return render(
    <MemoryRouter initialEntries={["/analyze"]}>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/analyze" element={<p>analyze page content</p>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  localStorage.clear();
  useHealthMock.mockReturnValue(healthResult());
});

describe("Layout", () => {
  it("renders the sidebar wordmark, tagline and 3 nav links", () => {
    renderLayout();
    expect(screen.getByText("NeuroSight AI")).toBeInTheDocument();
    expect(
      screen.getByText("Longitudinal Brain MRI Analysis"),
    ).toBeInTheDocument();
    const nav = screen.getByRole("navigation", { name: "Primary" });
    const links = Array.from(nav.querySelectorAll("a")).map(
      (a) => a.textContent,
    );
    expect(links).toEqual(["Analyze", "Patient history", "About"]);
  });

  it("renders the active nav link with active styling", () => {
    renderLayout();
    const analyze = screen.getByRole("link", { name: "Analyze" });
    expect(analyze).toHaveClass("text-primary", "bg-blue-50");
    const about = screen.getByRole("link", { name: "About" });
    expect(about).toHaveClass("text-ink2");
    expect(about).not.toHaveClass("bg-blue-50");
  });

  it("shows the telemetry chip with model version and the health dot", () => {
    renderLayout();
    expect(screen.getByText("v0.1.0")).toHaveClass("chip");
    const dot = screen.getByLabelText("Backend status");
    expect(dot).toHaveClass("rounded-full", "bg-success");
  });

  it("health dot turns danger on query error and ink3 while loading", () => {
    const { rerender } = renderLayout();

    useHealthMock.mockReturnValue(
      healthResult({ isError: true, isSuccess: false, data: undefined }),
    );
    rerender(
      <MemoryRouter initialEntries={["/analyze"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/analyze" element={<p>analyze page content</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByLabelText("Backend status")).toHaveClass("bg-danger");

    useHealthMock.mockReturnValue(
      healthResult({
        isLoading: true,
        isPending: true,
        isSuccess: false,
        data: undefined,
      }),
    );
    rerender(
      <MemoryRouter initialEntries={["/analyze"]}>
        <Routes>
          <Route element={<Layout />}>
            <Route path="/analyze" element={<p>analyze page content</p>} />
          </Route>
        </Routes>
      </MemoryRouter>,
    );
    expect(screen.getByLabelText("Backend status")).toHaveClass("bg-ink3");
  });

  it("renders page content through the Outlet", () => {
    renderLayout();
    expect(screen.getByText("analyze page content")).toBeInTheDocument();
  });

  it("shows the ApiKeyModal when localStorage has no key", async () => {
    renderLayout();
    await waitFor(() =>
      expect(screen.getByRole("dialog")).toBeInTheDocument(),
    );
  });

  it("does NOT show the ApiKeyModal when a key exists", async () => {
    localStorage.setItem("neurosight.apiKey", "stored-key");
    renderLayout();
    // Give effects a tick to run; the dialog must never appear.
    await waitFor(() => expect(screen.getByRole("navigation")).toBeInTheDocument());
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("toggles the mobile navigation drawer and carries accessible focus styles", async () => {
    const user = userEvent.setup();
    renderLayout();
    const toggle = screen.getByRole("button", { name: "Open navigation" });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(toggle).toHaveClass("focus-visible:ring-2", "focus-visible:ring-primary");

    await user.click(toggle);
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(screen.getByRole("navigation", { name: "Mobile Navigation" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Close navigation" }));
    expect(screen.queryByRole("navigation", { name: "Mobile Navigation" })).toBeNull();
  });

  it("displays a truthful offline banner when backend health query errors", () => {
    useHealthMock.mockReturnValue(
      healthResult({ isError: true, isSuccess: false, data: undefined }),
    );
    renderLayout();
    const banner = screen.getByText(/Backend offline:/);
    expect(banner).toBeInTheDocument();
    expect(
      screen.getByText(/You can still inspect interface structure, view options, and documentation\./),
    ).toBeInTheDocument();
  });
});
