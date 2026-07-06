import type { AnalysisResponse } from "@/lib/types";

interface Props {
  data: AnalysisResponse;
}

function CFCRGauge({ value, passFail }: { value: number; passFail: boolean }) {
  return (
    <div
      className={`rounded-xl border-2 p-6 text-center ${
        passFail
          ? "border-emerald-400 bg-emerald-50"
          : "border-red-400 bg-red-50"
      }`}
    >
      <p className="text-xs font-semibold uppercase tracking-widest text-slate-500">
        Cash Flow Coverage Ratio
      </p>
      <p className="mt-2 text-xs text-slate-400">
        MSME equivalent of a bank&apos;s Basel III LCR
      </p>
      <p
        className={`mt-4 text-6xl font-bold tabular-nums ${
          passFail ? "text-emerald-700" : "text-red-700"
        }`}
      >
        {value.toFixed(2)}
      </p>
      <p
        className={`mt-2 text-lg font-semibold ${
          passFail ? "text-emerald-600" : "text-red-600"
        }`}
      >
        {passFail ? "✓ PASSES — absorbs shock" : "✗ FAILS — liquidity at risk"}
      </p>
      <p className="mt-1 text-xs text-slate-500">Threshold: ≥ 1.00</p>
    </div>
  );
}

export default function HealthCard({ data }: Props) {
  const summary = data.profile_summary as Record<string, string>;

  return (
    <div className="space-y-5">
      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="mb-1 text-base font-semibold text-slate-800">
          {summary.business_name}
        </h2>
        <p className="text-xs text-slate-500">
          {summary.sector} · {summary.years_operating} years operating
        </p>
      </div>

      <CFCRGauge
        value={data.cfcr_baseline}
        passFail={data.cfcr_baseline >= 1.0}
      />

      <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="mb-3 text-sm font-semibold text-slate-700">
          Financial Health Score
        </h3>
        <div className="flex items-end gap-3">
          <span className="text-4xl font-bold text-slate-800">
            {data.baseline_score.toFixed(0)}
          </span>
          <span className="mb-1 text-slate-400">/&nbsp;100</span>
        </div>
        <div className="mt-3 h-2.5 w-full rounded-full bg-slate-100">
          <div
            className="h-2.5 rounded-full bg-blue-500 transition-all"
            style={{ width: `${data.baseline_score}%` }}
          />
        </div>
        <p className="mt-2 text-xs text-slate-400">
          Weights: GST {(data.weights_used.gst * 100).toFixed(0)}% · UPI{" "}
          {(data.weights_used.upi * 100).toFixed(0)}% · AA{" "}
          {(data.weights_used.aa * 100).toFixed(0)}% · EPFO{" "}
          {(data.weights_used.epfo * 100).toFixed(0)}%
        </p>
      </div>

      {data.narrative && (
        <div className="rounded-lg border border-slate-200 bg-white p-5 shadow-sm">
          <h3 className="mb-3 text-sm font-semibold text-slate-700">
            Assessment Narrative
          </h3>
          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-600">
            {data.narrative}
          </p>
        </div>
      )}
    </div>
  );
}
