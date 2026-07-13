"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CircleAlert, ShieldCheck } from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";

const LABELS: Record<string, string> = {
  receivable_delay_60d: "60d receivable delay",
  revenue_drop_20pct: "20% revenue drop",
  buyer_loss: "Primary buyer loss",
  rate_hike: "15% rate hike",
};

export default function StressPanel({ data }: { data: AnalysisResponse }) {
  const scoreByScenario = new Map(
    data.stress_results.map((item) => [item.scenario, item]),
  );
  const scenarios = data.cfcr_by_scenario
    .filter((item) => item.scenario !== "baseline")
    .map((item) => ({
      ...item,
      label: LABELS[item.scenario] ?? item.scenario,
      score: scoreByScenario.get(item.scenario),
    }))
    .sort((left, right) => left.cfcr - right.cfcr);
  const tailRisk = data.tail_risk;

  return (
    <section className="analysis-panel stress-panel">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Downside analysis</p>
          <h2>Stress test matrix</h2>
        </div>
        <span>Baseline CFCR {data.cfcr_baseline.toFixed(2)}</span>
      </div>
      <div className="stress-layout">
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart
              data={scenarios}
              margin={{ top: 10, right: 10, left: -20, bottom: 18 }}
            >
              <CartesianGrid vertical={false} stroke="#dde2df" />
              <XAxis
                dataKey="label"
                tick={{ fontSize: 10, fill: "#5f6864" }}
                interval={0}
                angle={-8}
                textAnchor="end"
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#7b837f" }}
                domain={[0, "auto"]}
              />
              <Tooltip
                cursor={{ fill: "#f1f3ef" }}
                formatter={(value) => [Number(value).toFixed(2), "CFCR"]}
              />
              <ReferenceLine
                y={1}
                stroke="#b4413a"
                strokeDasharray="5 4"
                label={{ value: "minimum", fill: "#9c3732", fontSize: 10 }}
              />
              <Bar dataKey="cfcr" radius={[3, 3, 0, 0]} maxBarSize={44}>
                {scenarios.map((item) => (
                  <Cell
                    key={item.scenario}
                    fill={item.pass_fail ? "#315c4b" : "#b4413a"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="scenario-list">
          {scenarios.map((item) => (
            <article key={item.scenario}>
              <span className={item.pass_fail ? "status-pass" : "status-fail"}>
                {item.pass_fail ? (
                  <ShieldCheck size={15} />
                ) : (
                  <CircleAlert size={15} />
                )}
              </span>
              <div>
                <strong>{item.label}</strong>
                <small>
                  {item.score?.key_drivers[0] ?? "Scenario applied"}
                </small>
              </div>
              <div className="scenario-metrics">
                <strong>{item.cfcr.toFixed(2)}</strong>
                <small>{item.score?.delta.toFixed(1) ?? "0.0"} pts</small>
              </div>
            </article>
          ))}
        </div>
      </div>
      {tailRisk ? (
        <div className="tail-risk-summary">
          <div className="tail-risk-intro">
            <p className="eyebrow">Sensitivity distribution</p>
            <h3>Borrower cash-flow tail</h3>
            <p>
              Seeded resampling complements the named scenarios. It is not an
              approval or calibrated probability of default.
            </p>
          </div>
          <dl className="tail-risk-metrics">
            <div>
              <dt>CFCR below 1.00</dt>
              <dd>{(tailRisk.probability_cfcr_below_one * 100).toFixed(1)}%</dd>
            </div>
            <div>
              <dt>5th percentile CFCR</dt>
              <dd>{tailRisk.cfcr_p05.toFixed(2)}x</dd>
            </div>
            <div>
              <dt>Average gap when below 1.00</dt>
              <dd>{tailRisk.expected_shortfall.toFixed(2)}x</dd>
            </div>
          </dl>
          <details className="tail-risk-assumptions">
            <summary>
              Review {tailRisk.simulations.toLocaleString()} simulation
              assumptions
            </summary>
            <ul>
              {tailRisk.assumptions.map((assumption) => (
                <li key={assumption}>{assumption}</li>
              ))}
            </ul>
          </details>
        </div>
      ) : null}
    </section>
  );
}
