import { ChevronDown, Database } from "lucide-react";
import type { AnalysisResponse } from "@/lib/types";

interface Props {
  data: AnalysisResponse;
}

const DIMENSION_LABELS: Record<string, string> = {
  gst: "GST",
  upi: "UPI Cash Flow",
  aa: "AA Bank Data",
  epfo: "EPFO Payroll",
};

export default function WeightRationale({ data }: Props) {
  return (
    <details className="evidence-disclosure">
      <summary>
        <span>
          <Database size={16} /> Weight rationale
        </span>
        <ChevronDown size={16} />
      </summary>
      <div className="evidence-body">
        <p>
          Weights are locked before stress testing and remain unchanged across
          scenarios.
        </p>
        <div className="weight-grid">
          {(["gst", "upi", "aa", "epfo"] as const).map((dim) => (
            <div key={dim}>
              <span>{DIMENSION_LABELS[dim]}</span>
              <strong>{(data.weights_used[dim] * 100).toFixed(0)}%</strong>
            </div>
          ))}
        </div>
        {data.weight_rationale.length > 0 && (
          <div className="rationale-list">
            {data.weight_rationale.map((item, i) => (
              <div key={`${item.dimension}-${i}`}>
                <strong>
                  {DIMENSION_LABELS[item.dimension] ?? item.dimension}
                </strong>
                <p>{item.reasoning}</p>
                {item.cited_chunk_id && item.cited_chunk_id !== "default" && (
                  <code>[{item.cited_chunk_id}]</code>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </details>
  );
}
