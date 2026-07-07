import type { Persona, AnalysisResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchPersonas(): Promise<Persona[]> {
  const res = await fetch(`${BASE_URL}/api/personas`);
  if (!res.ok) throw new Error(`fetchPersonas failed: ${res.status}`);
  return res.json();
}

export async function analyzeCustom(
  sector: string,
  yearsOperating: number,
  profileType: string,
): Promise<AnalysisResponse> {
  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sector,
      years_operating: yearsOperating,
      profile_type: profileType,
    }),
  });
  if (!res.ok) throw new Error(`analyzeCustom failed: ${res.status}`);
  return res.json();
}
