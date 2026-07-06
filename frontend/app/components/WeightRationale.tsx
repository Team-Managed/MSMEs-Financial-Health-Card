"use client";
import { useState } from "react";
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
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between p-5 text-left"
      >
        <span className="text-sm font-semibold text-slate-700">
          RAG-Grounded Weight Rationale
        </span>
        <span className="text-slate-400">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="border-t border-slate-100 p-5">
          <p className="mb-4 text-xs text-slate-400">
            Weights are locked once per profile before stress testing — they do
            not change per scenario.
          </p>
          <div className="grid grid-cols-4 gap-3 text-center">
            {(["gst", "upi", "aa", "epfo"] as const).map((dim) => (
              <div key={dim} className="rounded-md bg-slate-50 p-3">
                <p className="text-xs text-slate-500">
                  {DIMENSION_LABELS[dim]}
                </p>
                <p className="mt-1 text-xl font-bold text-slate-800">
                  {(data.weights_used[dim] * 100).toFixed(0)}%
                </p>
              </div>
            ))}
          </div>

          {data.weight_rationale.length > 0 && (
            <div className="mt-4 space-y-2">
              {data.weight_rationale.map((item, i) => (
                <div
                  key={i}
                  className="rounded-md border border-slate-100 p-3 text-xs"
                >
                  <span className="font-semibold text-slate-700">
                    {DIMENSION_LABELS[item.dimension] ?? item.dimension}:
                  </span>{" "}
                  <span className="text-slate-600">{item.reasoning}</span>
                  {item.cited_chunk_id && item.cited_chunk_id !== "default" && (
                    <span className="ml-1 font-mono text-slate-400">
                      [{item.cited_chunk_id}]
                    </span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
