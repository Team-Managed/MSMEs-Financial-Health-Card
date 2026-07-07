import type { Persona, AnalysisResponse } from "./types";
import type { AnalysisParams } from "@/app/components/ProfileForm";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchPersonas(): Promise<Persona[]> {
  const res = await fetch(`${BASE_URL}/api/personas`);
  if (!res.ok) throw new Error(`fetchPersonas failed: ${res.status}`);
  return res.json();
}

export async function analyzeCustom(
  params: AnalysisParams,
): Promise<AnalysisResponse> {
  const res = await fetch(`${BASE_URL}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sector: params.sector,
      years_operating: params.yearsOperating,
      profile_type: params.profileType,
      msme_tier: params.msmeTier,
      gst_registered: params.gstRegistered,
      employee_tier: params.employeeTier,
    }),
  });
  if (!res.ok) throw new Error(`analyzeCustom failed: ${res.status}`);
  return res.json();
}
