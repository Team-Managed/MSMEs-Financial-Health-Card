import type { Persona, AnalysisResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchPersonas(): Promise<Persona[]> {
  const res = await fetch(`${BASE_URL}/api/personas`);
  if (!res.ok) throw new Error(`fetchPersonas failed: ${res.status}`);
  return res.json();
}

export async function analyzePersona(id: string): Promise<AnalysisResponse> {
  const res = await fetch(
    `${BASE_URL}/api/msme/${encodeURIComponent(id)}/analyze`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    },
  );
  if (!res.ok) throw new Error(`analyzePersona failed: ${res.status}`);
  return res.json();
}
