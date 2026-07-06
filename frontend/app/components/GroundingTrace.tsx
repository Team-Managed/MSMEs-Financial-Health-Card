import type { AnalysisResponse } from "@/lib/types";

interface Props {
  data: AnalysisResponse;
}

export default function GroundingTrace({ data }: Props) {
  if (data.grounding_trace.length === 0) {
    return (
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-700">
          Grounding Trace
        </h3>
        <p className="mt-2 text-xs text-slate-400">No claims to verify.</p>
      </div>
    );
  }

  const passCount = data.grounding_trace.filter(
    (c) => c.status === "pass",
  ).length;
  const failCount = data.grounding_trace.filter(
    (c) => c.status === "fail",
  ).length;

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-sm font-semibold text-slate-700">
          Grounding Trace
        </h3>
        <div className="flex gap-2 text-xs">
          <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-emerald-700">
            {passCount} pass
          </span>
          {failCount > 0 && (
            <span className="rounded-full bg-red-100 px-2 py-0.5 text-red-700">
              {failCount} fail
            </span>
          )}
        </div>
      </div>
      <p className="mb-3 text-xs text-slate-400">
        Checks that every number traces to Risk Engine output and every citation
        traces to a retrieved document.
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-slate-100 text-left text-slate-500">
              <th className="pb-2 pr-3">Status</th>
              <th className="pb-2 pr-3">Type</th>
              <th className="pb-2 pr-3">Claim</th>
              <th className="pb-2">Source</th>
            </tr>
          </thead>
          <tbody>
            {data.grounding_trace.map((check, i) => (
              <tr key={i} className="border-b border-slate-50 last:border-0">
                <td className="py-2 pr-3">
                  <span
                    className={`font-medium ${check.status === "pass" ? "text-emerald-600" : "text-red-600"}`}
                  >
                    {check.status === "pass" ? "✓" : "✗"} {check.status}
                  </span>
                </td>
                <td className="py-2 pr-3 text-slate-400">{check.type}</td>
                <td className="max-w-xs truncate py-2 pr-3 text-slate-600">
                  {check.claim}
                </td>
                <td className="py-2 font-mono text-slate-400">
                  {check.source}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
