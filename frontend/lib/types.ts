export interface Persona {
  id: string;
  business_name: string;
  sector: string;
}

export interface WeightVector {
  gst: number;
  upi: number;
  aa: number;
  epfo: number;
}

export interface WeightRationaleItem {
  dimension: "gst" | "upi" | "aa" | "epfo";
  reasoning: string;
  cited_chunk_id: string;
}

export interface CFCRResult {
  scenario: string;
  cfcr: number;
  pass_fail: boolean;
}

export interface StressResult {
  scenario: string;
  stressed_score: number;
  delta: number;
  key_drivers: string[];
}

export interface GroundingCheck {
  claim: string;
  type: "numeric" | "citation";
  source: string;
  status: "pass" | "fail";
}

export interface AnalysisResponse {
  profile_summary: Record<string, unknown>;
  cfcr_baseline: number;
  cfcr_by_scenario: CFCRResult[];
  weights_used: WeightVector;
  weight_rationale: WeightRationaleItem[];
  baseline_score: number;
  stress_results: StressResult[];
  narrative: string;
  grounding_trace: GroundingCheck[];
}
