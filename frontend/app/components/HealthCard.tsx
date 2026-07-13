import {
  ArrowDownRight,
  CheckCircle2,
  CircleAlert,
  Gauge,
  TrendingUp,
} from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";

export default function HealthCard({ data }: { data: AnalysisResponse }) {
  const passes = data.cfcr_baseline >= 1;
  const scoreLabel =
    data.baseline_score >= 75
      ? "Strong"
      : data.baseline_score >= 60
        ? "Monitor"
        : "Elevated risk";
  const worst = data.cfcr_by_scenario
    .filter((item) => item.scenario !== "baseline")
    .reduce((current, item) => (item.cfcr < current.cfcr ? item : current));

  return (
    <section className="health-summary">
      <div className={`cfcr-block ${passes ? "is-pass" : "is-fail"}`}>
        <div className="metric-heading">
          <span>
            <Gauge size={17} /> Cash Flow Coverage Ratio
          </span>
          <small>30-day resilience proxy</small>
        </div>
        <div className="cfcr-value">
          <strong>{data.cfcr_baseline.toFixed(2)}</strong>
          <span>
            {passes ? <CheckCircle2 size={18} /> : <CircleAlert size={18} />}
            {passes ? "Liquidity covered" : "Liquidity exposed"}
          </span>
        </div>
        <div className="threshold-track">
          <i
            style={{
              width: `${Math.min((data.cfcr_baseline / 2.5) * 100, 100)}%`,
            }}
          />
          <b style={{ left: "40%" }} />
        </div>
        <div className="threshold-labels">
          <span>0</span>
          <span>Minimum 1.00</span>
          <span>2.5+</span>
        </div>
      </div>

      <div className="score-block">
        <div className="metric-heading">
          <span>
            <TrendingUp size={17} /> Financial health indicator
          </span>
          <small>Weighted composite</small>
        </div>
        <div className="score-value">
          <strong>{data.baseline_score.toFixed(0)}</strong>
          <span>/ 100</span>
        </div>
        <div className="score-track">
          <i style={{ width: `${data.baseline_score}%` }} />
        </div>
        <p className="score-label">{scoreLabel}</p>
      </div>

      <div className="risk-callout">
        <ArrowDownRight size={20} />
        <span>
          <small>Lowest stressed CFCR</small>
          <strong>{worst.cfcr.toFixed(2)}</strong>
          <p>{worst.scenario.replaceAll("_", " ")}</p>
          <b>
            {worst.pass_fail ? "Remains above minimum" : "Falls below minimum"}
          </b>
        </span>
      </div>
    </section>
  );
}
