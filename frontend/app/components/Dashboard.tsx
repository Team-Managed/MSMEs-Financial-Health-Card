"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import type { AnalysisResponse } from "@/lib/types";
import { analyzeCustom } from "@/lib/api";
import ProfileForm from "./ProfileForm";
import type { AnalysisParams } from "./ProfileForm";
import HealthCard from "./HealthCard";
import WeightRationale from "./WeightRationale";

// Lazy-load recharts-heavy component — only needed after a user triggers analysis
const StressPanel = dynamic(() => import("./StressPanel"), { ssr: false });

export default function Dashboard() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (params: AnalysisParams) => {
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const data = await analyzeCustom(params);
      setResult(data);
    } catch (e) {
      setError(
        `Analysis failed: ${e instanceof Error ? e.message : String(e)}`,
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-50">
      <header className="border-b border-slate-200 bg-white px-6 py-4">
        <h1 className="text-xl font-bold text-slate-900">
          MSME Financial Health Card
        </h1>
        <p className="text-xs text-slate-400">
          Stress-tested credit scoring · IDBI Innovate Track 03
        </p>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-8">
        {error !== null ? (
          <div className="mb-6 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        ) : null}

        {/* Phase 1: no result yet — centered form */}
        {result === null && !loading ? (
          <div className="mx-auto max-w-lg rounded-xl border border-slate-200 bg-white p-8 shadow-sm">
            <ProfileForm onSubmit={handleSubmit} loading={loading} />
          </div>
        ) : null}

        {/* Loading state */}
        {loading ? (
          <div className="flex flex-col items-center justify-center py-32 text-slate-400">
            <span className="mb-2 animate-pulse text-sm">
              Running pipeline…
            </span>
            <span className="text-xs text-slate-300">
              RAG retrieval → weight-setting → stress scenarios → narrative
            </span>
          </div>
        ) : null}

        {/* Phase 2: result available — sidebar + results */}
        {result !== null && !loading ? (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
            <div className="space-y-4">
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                <ProfileForm onSubmit={handleSubmit} loading={loading} />
              </div>
              <WeightRationale data={result} />
            </div>
            <div className="space-y-6">
              <HealthCard data={result} />
              <StressPanel data={result} />
            </div>
          </div>
        ) : null}
      </div>
    </main>
  );
}
