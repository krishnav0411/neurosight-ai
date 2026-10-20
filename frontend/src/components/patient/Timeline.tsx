/**
 * Timeline of the Demented probability across a patient's analyses.
 *
 * Recharts line chart (fixed 560x220, like the longitudinal panel) with the
 * line in danger red and dots colored by risk (>= 0.5 red, else green).
 * Per the design doc ("charts have text alternatives"), a collapsed-by-default
 * <details> below the chart renders the same data as a text table.
 */
import {
  CartesianGrid,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { pct } from "../../lib/format";
import type { TimelinePoint } from "../../lib/patient";

const LINE_COLOR = "#dc2626";
const HIGH_RISK_DOT = "#dc2626";
const LOW_RISK_DOT = "#16a34a";

export function Timeline({ points }: { points: TimelinePoint[] }) {
  if (points.length === 0) return null;

  return (
    <div className="rounded-card border border-hairline bg-white p-5 shadow-card">
      <h3 className="text-base font-semibold text-ink">Risk over time</h3>

      <div className="mt-3 overflow-x-auto">
        <LineChart width={560} height={220} data={points}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: "#475569" }} />
          <YAxis
            domain={[0, 1]}
            tick={{ fontSize: 11, fill: "#475569" }}
            tickFormatter={(v: number) => pct(v)}
          />
          <Tooltip
            // recharts types the value as ValueType (number | string | array);
            // this Line carries a numeric `prob`, so coerce for pct().
            formatter={(value) => [pct(Number(value)), "Demented probability"]}
          />
          <Line
            type="monotone"
            dataKey="prob"
            stroke={LINE_COLOR}
            strokeWidth={2}
            isAnimationActive={false}
            dot={(props: {
              cx?: number | string;
              cy?: number | string;
              payload?: TimelinePoint;
            }) => {
              const { cx, cy, payload } = props;
              if (cx == null || cy == null || payload == null) return null;
              return (
                <circle
                  key={payload.ref}
                  cx={cx}
                  cy={cy}
                  r={4}
                  fill={payload.prob >= 0.5 ? HIGH_RISK_DOT : LOW_RISK_DOT}
                  stroke="#ffffff"
                  strokeWidth={1}
                />
              );
            }}
          />
        </LineChart>
      </div>

      <p className="mt-2 text-[12px] text-ink3">
        Demented-class probability per analysis, oldest to newest
      </p>

      <details className="mt-3">
        <summary className="cursor-pointer text-[12px] text-ink2 hover:text-ink">
          View as table
        </summary>
        <table className="mt-2 w-full text-[12px]">
          <caption className="sr-only">
            Demented probability per analysis (text alternative to the chart)
          </caption>
          <thead>
            <tr className="text-left text-ink3">
              <th scope="col" className="py-1 font-medium">
                Analysis
              </th>
              <th scope="col" className="py-1 text-right font-medium">
                Demented probability
              </th>
            </tr>
          </thead>
          <tbody>
            {points.map((p) => (
              <tr key={p.ref} className="border-t border-hairline text-ink2">
                <td className="py-1">{p.label}</td>
                <td className="tnum py-1 text-right">{pct(p.prob)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
