/**
 * About page: renders the backend model card (formatted markdown)
 * plus the metrics digest as a two-column table.
 * Loading shows a pulse skeleton; failure shows a retryable banner.
 * The disclaimer stays pinned at the bottom.
 */
import React from "react";
import { useModelCard } from "../api/hooks";
import { DisclaimerBanner } from "../components/DisclaimerBanner";

function renderMarkdown(content: string): React.ReactNode {
  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let inList = false;
  let listItems: React.ReactNode[] = [];

  const flushList = (key: string) => {
    if (inList && listItems.length > 0) {
      elements.push(
        <ul key={key} className="my-2 list-disc space-y-1 pl-5">
          {listItems}
        </ul>,
      );
      listItems = [];
      inList = false;
    }
  };

  const formatInline = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = [];
    const regex = /(\*\*.*?\*\*|`.*?`)/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;
    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      const token = match[0];
      if (token.startsWith("**") && token.endsWith("**")) {
        parts.push(
          <strong key={match.index} className="font-semibold text-ink">
            {token.slice(2, -2)}
          </strong>,
        );
      } else if (token.startsWith("`") && token.endsWith("`")) {
        parts.push(
          <code
            key={match.index}
            className="rounded bg-slate-100 px-1 py-0.5 font-mono text-xs text-ink"
          >
            {token.slice(1, -1)}
          </code>,
        );
      }
      lastIndex = regex.lastIndex;
    }
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    return parts.length > 0 ? parts : text;
  };

  lines.forEach((line, i) => {
    const trimmed = line.trim();
    if (trimmed.startsWith("### ")) {
      flushList(`flush-${i}`);
      elements.push(
        <h4 key={i} className="mb-1 mt-4 text-[13px] font-semibold text-ink">
          {formatInline(trimmed.slice(4))}
        </h4>,
      );
    } else if (trimmed.startsWith("## ")) {
      flushList(`flush-${i}`);
      elements.push(
        <h3
          key={i}
          className="mb-2 mt-5 border-b border-hairline pb-1 text-[14px] font-semibold text-ink"
        >
          {formatInline(trimmed.slice(3))}
        </h3>,
      );
    } else if (trimmed.startsWith("# ")) {
      flushList(`flush-${i}`);
      elements.push(
        <h2 key={i} className="mb-2 mt-3 text-[16px] font-bold text-ink">
          {formatInline(trimmed.slice(2))}
        </h2>,
      );
    } else if (trimmed === "---") {
      flushList(`flush-${i}`);
      elements.push(<hr key={i} className="my-4 border-hairline" />);
    } else if (trimmed.startsWith("- ")) {
      inList = true;
      listItems.push(
        <li key={i} className="text-[13px] text-ink2">
          {formatInline(trimmed.slice(2))}
        </li>,
      );
    } else if (trimmed.length === 0) {
      flushList(`flush-${i}`);
    } else {
      flushList(`flush-${i}`);
      elements.push(
        <p
          key={i}
          className="my-1 whitespace-pre-wrap text-[13px] leading-5 text-ink2"
        >
          {formatInline(line)}
        </p>,
      );
    }
  });
  flushList("flush-end");
  return elements;
}

function DigestTable({ digest }: { digest: Record<string, unknown> }) {
  const entries = Object.entries(digest);
  if (entries.length === 0) return null;

  return (
    <div className="mt-6 overflow-hidden rounded-control border border-hairline">
      <table className="w-full text-left text-[13px]">
        <thead>
          <tr className="border-b border-hairline bg-slate-50 text-ink2">
            <th scope="col" className="px-3 py-2 font-medium">
              Metric
            </th>
            <th scope="col" className="px-3 py-2 font-medium">
              Value
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([key, value]) => {
            const isNumber = typeof value === "number" && Number.isFinite(value);
            return (
              <tr key={key} className="border-b border-hairline last:border-b-0">
                <td className="px-3 py-2 text-ink2">{key}</td>
                <td className={`px-3 py-2 text-ink ${isNumber ? "tnum" : ""}`}>
                  {String(value)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function CardSkeleton() {
  return (
    <div aria-hidden="true" className="animate-pulse">
      <div className="rounded-card border border-hairline bg-white p-6">
        <div className="h-4 w-40 rounded bg-slate-200" />
        <div className="mt-4 space-y-2">
          <div className="h-3 w-full rounded bg-slate-100" />
          <div className="h-3 w-5/6 rounded bg-slate-100" />
          <div className="h-3 w-2/3 rounded bg-slate-100" />
        </div>
        <div className="mt-6 h-24 w-full rounded-control bg-slate-100" />
      </div>
    </div>
  );
}

export function AboutPage() {
  const { data, isLoading, isError, refetch } = useModelCard();

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-ink">About</h1>

      {isLoading ? (
        <CardSkeleton />
      ) : isError ? (
        <div
          role="alert"
          className="flex items-center justify-between rounded-card border border-red-200 bg-red-50 px-4 py-3 text-[13px] text-danger"
        >
          <span>Model card unavailable. Check the backend connection.</span>
          <button
            type="button"
            onClick={() => void refetch()}
            className="rounded-control border border-red-200 bg-white px-3 py-1.5 font-medium text-danger hover:bg-red-50"
          >
            Retry
          </button>
        </div>
      ) : data ? (
        <>
          <section className="rounded-card border border-hairline bg-white p-6 shadow-card">
            <h2 className="text-base font-semibold text-ink">Model card</h2>
            <div className="mt-3 text-[13px] leading-5 text-ink2">
              {renderMarkdown(data.content)}
            </div>
            {data.metrics_digest ? (
              <DigestTable digest={data.metrics_digest} />
            ) : null}
          </section>
        </>
      ) : null}

      <DisclaimerBanner />
    </div>
  );
}
