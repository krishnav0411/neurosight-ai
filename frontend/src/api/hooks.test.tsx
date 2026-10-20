/**
 * useAnalysis polling contract: GET the analysis every 2s while the status is
 * non-terminal (queued/running); once a terminal status (succeeded/failed/
 * cancelled) arrives, polling stops for good.
 *
 * `getAnalysis` is mocked via vi.mock("../api/client"); fake timers drive the
 * 2s refetch intervals.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAnalysis } from "./hooks";

const m = vi.hoisted(() => ({
  getAnalysis: vi.fn(),
  getHealth: vi.fn(),
  getModelCard: vi.fn(),
  getPatients: vi.fn(),
  getPatientHistory: vi.fn(),
  createAnalysis: vi.fn(),
  deleteAnalysis: vi.fn(),
}));

vi.mock("./client", () => m);

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

beforeEach(() => {
  vi.useFakeTimers();
  m.getAnalysis.mockReset();
});

afterEach(() => {
  vi.useRealTimers();
});

/** Advance fake timers in small steps so react-query's promise chain flushes. */
async function advance(ms: number) {
  await act(async () => {
    const steps = Math.ceil(ms / 100);
    for (let i = 0; i < steps; i++) {
      await vi.advanceTimersByTimeAsync(Math.min(100, ms - i * 100));
    }
  });
}

describe("useAnalysis polling", () => {
  it("polls every 2s until a terminal status, then stops", async () => {
    m.getAnalysis
      .mockResolvedValueOnce({ analysis_id: "a1", status: "running" })
      .mockResolvedValueOnce({ analysis_id: "a1", status: "succeeded" });

    const { result } = renderHook(() => useAnalysis("a1"), {
      wrapper: makeWrapper(),
    });

    // Initial fetch -> running.
    await advance(50);
    expect(m.getAnalysis).toHaveBeenCalledTimes(1);
    expect(result.current.data?.status).toBe("running");

    // One 2s interval -> the terminal call lands.
    await advance(2000);
    expect(m.getAnalysis).toHaveBeenCalledTimes(2);
    expect(result.current.data?.status).toBe("succeeded");

    // Terminal state reached: the call count must stabilize.
    await advance(2000);
    await advance(2000);
    expect(m.getAnalysis).toHaveBeenCalledTimes(2);
  });

  it("does not poll while disabled (id is null)", async () => {
    const { result } = renderHook(() => useAnalysis(null), {
      wrapper: makeWrapper(),
    });

    await advance(10_000);
    expect(result.current.fetchStatus).toBe("idle");
    expect(m.getAnalysis).not.toHaveBeenCalled();
  });
});
