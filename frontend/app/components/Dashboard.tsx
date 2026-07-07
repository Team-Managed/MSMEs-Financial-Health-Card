"use client";
import { useState } from "react";
import dynamic from "next/dynamic";
import type { Persona, AnalysisResponse } from "@/lib/types";
import { analyzePersona } from "@/lib/api";
import PersonaSelector from "./PersonaSelector";
import HealthCard from "./HealthCard";
import GroundingTrace from "./GroundingTrace";
import WeightRationale from "./WeightRationale";

// Lazy-load recharts-heavy component — only needed after a user triggers analysis
const StressPanel = dynamic(() => import("./StressPanel"), { ssr: false });

interface Props {
  initialPersonas: Persona[];
  initialError: string | null;
}

export default function Dashboard({ initialPersonas, initialError }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(initialError);

  const handleSelect = async (id: string) => {
    setSelectedId(id);
    setResult(null);
    setError(null);
    setLoading(true);
    try {
      const data = await analyzePersona(id);
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
          Stress-tested credit scoring - IDBI Innovate Track 03
        </p>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-6">
        {error !== null ? (
          <div className="mb-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        ) : null}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
          <div className="space-y-4">
            <PersonaSelector
              personas={initialPersonas}
              selected={selectedId}
              onSelect={handleSelect}
              loading={loading}
            />
            {result !== null ? <WeightRationale data={result} /> : null}
          </div>

          {loading ? (
            <div className="flex items-center justify-center py-24 text-slate-400">
              <span className="animate-pulse text-sm">Running pipeline...</span>
            </div>
          ) : null}

          {result !== null && !loading ? (
            <div className="space-y-6">
              <HealthCard data={result} />
              <StressPanel data={result} />
              <GroundingTrace data={result} />
            </div>
          ) : null}
        </div>
      </div>
    </main>
  );
}
