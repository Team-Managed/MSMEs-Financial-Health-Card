"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { AnalysisResponse } from "@/lib/types";

const SCENARIO_LABELS: Record<string, string> = {
  receivable_delay_60d: "Receivable Delay 60d",
  revenue_drop_20pct: "Revenue Drop 20%",
  buyer_loss: "Buyer Loss",
  rate_hike: "Rate Hike +15%",
};

interface Props {
  data: AnalysisResponse;
}

export default function StressPanel({ data }: Props) {
  const cfcrData = data.cfcr_by_scenario
    .filter((r) => r.scenario !== "baseline")
    .map((r) => ({
      name: SCENARIO_LABELS[r.scenario] ?? r.scenario,
      cfcr: r.cfcr,
      passFail: r.pass_fail,
    }));

  const scoreData = data.stress_results.map((r) => ({
    name: SCENARIO_LABELS[r.scenario] ?? r.scenario,
    delta: r.delta,
    stressed_score: r.stressed_score,
    key_driver: r.key_drivers[0] ?? "",
  }));

  return (
    <div className="space-y-5">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-4 text-sm font-semibold text-slate-700">
          CFCR Under Stress Scenarios
        </h3>
        <p className="mb-4 text-xs text-slate-400">
          Red = CFCR drops below 1.0 (liquidity failure). Baseline:{" "}
          {data.cfcr_baseline.toFixed(2)}
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={cfcrData}
            margin={{ top: 4, right: 16, left: 0, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis
              domain={[0, Math.max(data.cfcr_baseline * 1.2, 1.5)]}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(v) =>
                typeof v === "number" ? v.toFixed(3) : String(v ?? "")
              }
            />
            <ReferenceLine
              y={1.0}
              stroke="#ef4444"
              strokeDasharray="4 2"
              label={{ value: "1.0 threshold", fontSize: 11, fill: "#ef4444" }}
            />
            <Bar dataKey="cfcr" radius={[4, 4, 0, 0]}>
              {cfcrData.map((entry, i) => (
                <Cell key={i} fill={entry.passFail ? "#3b82f6" : "#ef4444"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-4 text-sm font-semibold text-slate-700">
          Health Score Impact
        </h3>
        <div className="space-y-3">
          {scoreData.map((r) => (
            <div key={r.name} className="rounded-md bg-slate-50 p-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-slate-700">
                  {r.name}
                </span>
                <span
                  className={`text-sm font-bold tabular-nums ${r.delta < 0 ? "text-red-600" : "text-emerald-600"}`}
                >
                  {r.delta > 0 ? "+" : ""}
                  {r.delta.toFixed(1)} pts → {r.stressed_score.toFixed(0)}/100
                </span>
              </div>
              {r.key_driver && (
                <p className="mt-1 text-xs text-slate-400">{r.key_driver}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
