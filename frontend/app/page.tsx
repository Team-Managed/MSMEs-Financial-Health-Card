"use client";
import { useEffect, useState } from "react";
import type { Persona, AnalysisResponse } from "@/lib/types";
import { fetchPersonas, analyzePersona } from "@/lib/api";
import PersonaSelector from "./components/PersonaSelector";
import HealthCard from "./components/HealthCard";
import StressPanel from "./components/StressPanel";
import GroundingTrace from "./components/GroundingTrace";
import WeightRationale from "./components/WeightRationale";

export default function Home() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPersonas()
      .then(setPersonas)
      .catch(() =>
        setError("Could not load personas — is the backend running?"),
      );
  }, []);

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
          Stress-tested credit scoring · IDBI Innovate Track 03
        </p>
      </header>

      <div className="mx-auto max-w-6xl px-4 py-6">
        {error && (
          <div className="mb-4 rounded-md bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[340px_1fr]">
          <div className="space-y-4">
            <PersonaSelector
              personas={personas}
              selected={selectedId}
              onSelect={handleSelect}
              loading={loading}
            />
            {result && <WeightRationale data={result} />}
          </div>

          {loading && (
            <div className="flex items-center justify-center py-24 text-slate-400">
              <span className="animate-pulse text-sm">Running pipeline...</span>
            </div>
          )}

          {result && !loading && (
            <div className="space-y-6">
              <HealthCard data={result} />
              <StressPanel data={result} />
              <GroundingTrace data={result} />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
