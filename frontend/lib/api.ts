import type { Persona, AnalysisResponse } from "./types";
import type { AnalysisParams } from "@/app/components/ProfileForm";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function fetchPersonas(): Promise<Persona[]> {
  const res = await fetch(`${BASE_URL}/api/personas`);
  if (!res.ok) throw new Error(`fetchPersonas failed: ${res.status}`);
  return res.json();
}

export async function analyzePersona(
  personaId: string,
): Promise<AnalysisResponse> {
  const res = await fetch(
    `${BASE_URL}/api/msme/${encodeURIComponent(personaId)}/analyze`,
    { method: "POST" },
  );
  if (!res.ok) throw new Error(`analyzePersona failed: ${res.status}`);
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
      requested_amount_lakh: params.requestedAmountLakh,
      annual_interest_rate_pct: params.annualInterestRatePct,
      expected_utilization_pct: params.expectedUtilizationPct,
      annual_turnover_lakh: params.annualTurnoverLakh,
      avg_monthly_inflow_lakh: params.avgMonthlyInflowLakh,
      avg_monthly_operating_outflow_lakh: params.avgMonthlyOperatingOutflowLakh,
      avg_bank_balance_lakh: params.avgBankBalanceLakh,
      existing_monthly_emi_lakh: params.existingMonthlyEmiLakh,
      top_buyer_share_pct: params.topBuyerSharePct,
      bounced_payments_12mo: params.bouncedPayments12mo,
      gst_filing_consistency_pct: params.gstFilingConsistencyPct,
      yoy_growth_pct: params.yoyGrowthPct,
    }),
  });
  if (!res.ok) throw new Error(`analyzeCustom failed: ${res.status}`);
  return res.json();
}
