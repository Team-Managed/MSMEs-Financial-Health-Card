import {
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  FileCheck2,
} from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";

interface Props {
  data: AnalysisResponse;
}

export default function GroundingTrace({ data }: Props) {
  if (data.grounding_trace.length === 0) {
    return (
      <section className="analysis-panel audit-panel">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Evidence ledger</p>
            <h2>Grounding trace</h2>
          </div>
        </div>
        <p>No claims were emitted for verification.</p>
      </section>
    );
  }

  const passCount = data.grounding_trace.filter(
    (c) => c.status === "pass",
  ).length;
  const failCount = data.grounding_trace.filter(
    (c) => c.status === "fail",
  ).length;

  return (
    <details className="evidence-disclosure audit-panel">
      <summary>
        <span>
          <FileCheck2 size={16} /> Grounding trace
        </span>
        <span className="audit-summary-meta">
          {passCount} verified
          {failCount > 0 ? <b>{failCount} failed</b> : null}
          <ChevronDown size={16} />
        </span>
      </summary>
      <div className="evidence-body audit-body">
        <div className="section-heading audit-heading">
          <div>
            <p className="eyebrow">Evidence ledger</p>
            <h2>Claim verification</h2>
          </div>
        </div>
        <p className="audit-description">
          Every quantitative claim resolves to a risk-engine field; every
          citation resolves to a retrieved source.
        </p>
        <div className="audit-table-wrap">
          <table className="audit-table">
            <thead>
              <tr>
                <th>Status</th>
                <th>Type</th>
                <th>Claim</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {data.grounding_trace.map((check, i) => (
                <tr key={`${check.source}-${i}`}>
                  <td>
                    <span
                      className={
                        check.status === "pass" ? "audit-pass" : "audit-fail"
                      }
                    >
                      {check.status === "pass" ? (
                        <CheckCircle2 size={14} />
                      ) : (
                        <CircleAlert size={14} />
                      )}
                      {check.status}
                    </span>
                  </td>
                  <td>
                    <span className="type-chip">{check.type}</span>
                  </td>
                  <td>{check.claim}</td>
                  <td>
                    <code>{check.source}</code>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </details>
  );
}
