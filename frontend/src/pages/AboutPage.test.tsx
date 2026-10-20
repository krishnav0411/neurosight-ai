/**
 * AboutPage: model card content, metrics digest table, loading skeleton and
 * error banner with retry. Hooks module is mocked for determinism.
 */
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { UseQueryResult } from "@tanstack/react-query";
import type { ModelCard } from "../api/types";
import { AboutPage } from "./AboutPage";

vi.mock("../api/hooks", () => ({
  useModelCard: vi.fn(),
}));

import { useModelCard } from "../api/hooks";
const useModelCardMock = vi.mocked(useModelCard);

const CARD: ModelCard = {
  content: "MODEL CARD TEXT",
  metrics_digest: {
    test_acc: 0.591,
    test_acc_ci: [0.36363636363636365, 0.7727272727272727],
    backbone: "resnet18",
  },
};

function result(
  overrides: Partial<UseQueryResult<ModelCard, Error>> = {},
): UseQueryResult<ModelCard, Error> {
  const base = {
    data: CARD,
    isLoading: false,
    isPending: false,
    isSuccess: true,
    isError: false,
    refetch: vi.fn(),
  };
  return { ...base, ...overrides } as unknown as UseQueryResult<
    ModelCard,
    Error
  >;
}

beforeEach(() => {
  useModelCardMock.mockReset();
});

describe("AboutPage", () => {
  it("renders the model card content as pre-wrapped text", () => {
    useModelCardMock.mockReturnValue(result());
    render(<AboutPage />);
    const text = screen.getByText("MODEL CARD TEXT");
    expect(text).toBeInTheDocument();
    expect(text).toHaveClass("whitespace-pre-wrap");
  });

  it("renders the metrics digest as a two-column table with formatted values and tnum", () => {
    useModelCardMock.mockReturnValue(result());
    render(<AboutPage />);

    const acc = screen.getByText("59.1%");
    expect(acc.tagName).toBe("TD");
    expect(acc).toHaveClass("tnum");

    // Key column shows the metric name, value column sits beside it.
    expect(acc.parentElement).toHaveTextContent("test_acc");

    // Array CI is sensibly formatted into percentage bounds
    const ci = screen.getByText("[36.4%, 77.3%]");
    expect(ci.tagName).toBe("TD");
    expect(ci).toHaveClass("tnum");

    // Non-numeric values render without forcing tnum formatting.
    expect(screen.getByText("resnet18")).toBeInTheDocument();
  });

  it("renders markdown tables in model card content as semantic HTML tables", () => {
    const tableMarkdown = [
      "## Evaluation Table",
      "| Metric | NeuroSight-E5 | Champion v1 |",
      "|---|---|---|",
      "| Accuracy | **59.1%** | 59.1% |",
      "| AUC-ROC | 0.777 | 0.785 |",
    ].join("\n");

    useModelCardMock.mockReturnValue(
      result({
        data: {
          content: tableMarkdown,
          metrics_digest: null,
        },
      }),
    );
    render(<AboutPage />);

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Metric" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "NeuroSight-E5" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "Accuracy" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "AUC-ROC" })).toBeInTheDocument();
  });

  it("renders a pulse skeleton while loading", () => {
    useModelCardMock.mockReturnValue(
      result({
        isLoading: true,
        isPending: true,
        isSuccess: false,
        data: undefined,
      }),
    );
    render(<AboutPage />);
    const pulse = document.querySelector(".animate-pulse");
    expect(pulse).not.toBeNull();
    expect(screen.queryByText("MODEL CARD TEXT")).not.toBeInTheDocument();
  });

  it("renders an error banner with a working retry button", async () => {
    const user = userEvent.setup();
    const refetch = vi.fn();
    useModelCardMock.mockReturnValue(
      result({
        isError: true,
        isSuccess: false,
        data: undefined,
        refetch,
      }),
    );
    render(<AboutPage />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(refetch).toHaveBeenCalledTimes(1);
  });

  it("keeps the disclaimer as a permanent resident", () => {
    useModelCardMock.mockReturnValue(result());
    render(<AboutPage />);
    expect(
      screen.getByText(
        "Research use only. Not a medical device. This tool does not provide a diagnosis.",
      ),
    ).toBeInTheDocument();
  });

  it("skeleton resolves into content (smoke: success after loading rerender)", async () => {
    useModelCardMock.mockReturnValueOnce(
      result({
        isLoading: true,
        isPending: true,
        isSuccess: false,
        data: undefined,
      }),
    );
    const { rerender } = render(<AboutPage />);
    useModelCardMock.mockReturnValue(result());
    rerender(<AboutPage />);
    await waitFor(() =>
      expect(screen.getByText("MODEL CARD TEXT")).toBeInTheDocument(),
    );
  });
});
