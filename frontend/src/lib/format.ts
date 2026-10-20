/**
 * Small pure formatting helpers shared across dashboard views.
 */

/**
 * Format a probability as a whole-number percent string, rounding half away
 * from zero. Example: 0.871 -> "87%", 0.875 -> "88%".
 */
export function pct(x: number): string {
  const rounded =
    x >= 0 ? Math.floor(x * 100 + 0.5) : Math.ceil(x * 100 - 0.5);
  return `${rounded}%`;
}

/**
 * Severity color for the DEMENTED probability: a high demented probability is
 * bad, so it maps to danger red.
 */
export function riskColor(
  prob: number,
): "success" | "warning" | "danger" {
  if (prob >= 0.8) return "danger";
  if (prob >= 0.5) return "warning";
  return "success";
}

/**
 * Format a probability as a percent string with exactly one decimal place,
 * rounding half away from zero. Example: 0.87 -> "87.0%", 0.877 -> "87.7%".
 * Used for the headline confidence figure on the results dashboard.
 */
export function pct1(x: number): string {
  const rounded = Math.round(x * 1000) / 10;
  return `${rounded.toFixed(1)}%`;
}

/**
 * Stat tiles for the longitudinal card.
 *
 * - `latest`: the most recent per-visit (demented) probability, whole percent.
 * - `delta`: signed percent change of the latest visit versus the previous
 *   one ("+32%" / "-32%"; "+0%" when flat or when there is no previous visit).
 * - `deltaTone`: a RISING demented probability is worsening (danger red); a
 *   falling or flat one is steady/improving (success green).
 *
 * Empty series render honest placeholders ("—") — callers normally gate on
 * having at least one visit before rendering the tiles.
 */
export function deltaTiles(perVisit: number[]): {
  latest: string;
  delta: string;
  deltaTone: "success" | "danger";
} {
  if (perVisit.length === 0) {
    return { latest: "—", delta: "—", deltaTone: "success" };
  }

  const latest = pct(perVisit[perVisit.length - 1]);
  if (perVisit.length < 2) {
    return { latest, delta: "+0%", deltaTone: "success" };
  }

  const diff = perVisit[perVisit.length - 1] - perVisit[perVisit.length - 2];
  const rounded = Math.round(diff * 100); // whole-percent delta
  if (rounded === 0) {
    return { latest, delta: "+0%", deltaTone: "success" };
  }
  if (rounded > 0) {
    return { latest, delta: `+${rounded}%`, deltaTone: "danger" };
  }
  return { latest, delta: `-${Math.abs(rounded)}%`, deltaTone: "success" };
}

/**
 * Map a longitudinal trend string to a badge label + pill class.
 */
export function trendBadge(trend: string): { label: string; cls: string } {
  if (trend === "Stable") return { label: "Stable", cls: "pill-success" };
  if (trend === "Declining" || trend === "Rapidly Declining") {
    return { label: trend, cls: "pill-warning" };
  }
  return { label: trend, cls: "pill-neutral" };
}
