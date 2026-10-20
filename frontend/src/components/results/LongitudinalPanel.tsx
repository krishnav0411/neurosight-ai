/**
 * Longitudinal view ("Longitudinal progression"): recharts line chart of the
 * demented-class probability as visits accumulate, with a trend badge and
 * labelled dots, over a row of three stat tiles (Latest / Δ latest visit /
 * Visits). Renders nothing below 2 data points (single-visit results show a
 * "single-visit mode" chip instead).
 */
import {
  CartesianGrid,
  LabelList,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Longitudinal } from "../../api/types";
import { deltaTiles, pct, trendBadge } from "../../lib/format";

export function LongitudinalPanel({
  longitudinal,
  visitLabels,
  visitCount,
}: {
  longitudinal: Longitudinal;
  /** X-axis labels, parallel to per_visit_probability (optional). */
  visitLabels?: string[];
  /** Number of visits in the study (defaults to the series length). */
  visitCount?: number;
}) {
  const points = longitudinal.per_visit_probability;
  if (points.length <= 1) return null;

  // Fall back to V1..Vn when labels are missing or shorter than the series.
  const data = points.map((p, i) => ({
    visit: visitLabels?.[i] ?? `V${i + 1}`,
    p,
  }));

  const trend = trendBadge(longitudinal.trend);
  const tiles = deltaTiles(points);

  return (
    <div className="rounded-card border border-hairline bg-white p-5 shadow-card">
      <div className="flex items-center justify-between gap-3">
        <h3 className="text-base font-semibold text-ink">
          Longitudinal progression
        </h3>
        <span className={`pill ${trend.cls}`}>{trend.label}</span>
      </div>

      <p className="mt-1 text-[12px] text-ink3">
        How the prediction evolves as visits accumulate
      </p>

      <div className="mt-3 overflow-x-auto">
        <LineChart width={520} height={180} data={data} margin={{ top: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="visit" tick={{ fontSize: 11, fill: "#475569" }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#475569" }} />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="p"
            stroke="#dc2626"
            strokeWidth={2}
            dot
            isAnimationActive={false}
          >
            <LabelList
              dataKey="p"
              position="top"
              formatter={(value: unknown) => pct(Number(value))}
              style={{ fontSize: 11, fill: "#475569" }}
            />
          </Line>
        </LineChart>
      </div>

      {/* Stat tiles: latest probability, visit-over-visit delta, visit count. */}
      <div className="mt-3 grid grid-cols-3 gap-2">
        <div className="rounded-control bg-canvas p-3">
          <p className="text-[11px] text-ink3">Latest</p>
          <p className="tnum mt-0.5 text-[15px] font-semibold text-ink">
            {tiles.latest}
          </p>
        </div>
        <div className="rounded-control bg-canvas p-3">
          <p className="text-[11px] text-ink3">Δ latest visit</p>
          <p
            className={`tnum mt-0.5 text-[15px] font-semibold ${
              tiles.deltaTone === "danger" ? "text-danger" : "text-success"
            }`}
          >
            {tiles.delta}
          </p>
        </div>
        <div className="rounded-control bg-canvas p-3">
          <p className="text-[11px] text-ink3">Visits</p>
          <p className="tnum mt-0.5 text-[15px] font-semibold text-ink">
            {visitCount ?? points.length}
          </p>
        </div>
      </div>

      {/* Text alternative to the chart (same pattern as Timeline). */}
      <details className="mt-3">
        <summary className="cursor-pointer text-[12px] text-ink2 hover:text-ink">
          View as table
        </summary>
        <table className="mt-2 w-full text-[12px]">
          <caption className="sr-only">
            Demented probability per visit (text alternative to the chart)
          </caption>
          <thead>
            <tr className="text-left text-ink2">
              <th scope="col" className="py-1 font-medium">
                Visit
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Demented probability
              </th>
            </tr>
          </thead>
          <tbody>
            {data.map((d) => (
              <tr key={d.visit} className="border-t border-hairline text-ink2">
                <td className="py-1">{d.visit}</td>
                <td className="tnum py-1 text-right">{pct(d.p)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
