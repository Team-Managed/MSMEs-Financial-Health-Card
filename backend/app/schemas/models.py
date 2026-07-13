from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class GSTData(BaseModel):
    monthly_turnover_series: list[float]   # 12 months
    filing_consistency_score: float        # 0–1
    yoy_growth_rate: float


class UPIData(BaseModel):
    monthly_inflow_series: list[float]     # 12 months
    monthly_outflow_series: list[float]    # 12 months
    transaction_frequency: int
    top_counterparty_share: float          # 0–1


class AABankData(BaseModel):
    existing_loan_count: int
    existing_loan_emi_total: float
    overdraft_utilization_rate: float      # 0–1
    avg_account_balance: float
    bounced_payment_count_12mo: int
    estimated_monthly_operating_outflow: float


class EPFOData(BaseModel):
    employee_count_series: list[int]       # 12 months
    payroll_consistency_score: float       # 0–1


class MSMEProfile(BaseModel):
    msme_id: str
    business_name: str
    sector: str
    years_operating: int
    gst: GSTData
    upi: UPIData
    aa_bank_data: AABankData
    epfo: EPFOData


class WeightVector(BaseModel):
    gst: float
    upi: float
    aa: float
    epfo: float

    @field_validator("gst", "upi", "aa", "epfo")
    @classmethod
    def must_be_finite_and_between_0_and_1(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("weight must be a finite number")
        if not 0.0 <= v <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "WeightVector":
        total = self.gst + self.upi + self.aa + self.epfo
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total:.8f}")
        return self


class WeightRationaleItem(BaseModel):
    dimension: Literal["gst", "upi", "aa", "epfo"]
    reasoning: str
    cited_chunk_id: str


class CFCRResult(BaseModel):
    scenario: str
    cfcr: float
    pass_fail: bool


class StressResult(BaseModel):
    scenario: str
    stressed_score: float
    delta: float
    key_drivers: list[str]


class TailRiskResult(BaseModel):
    probability_cfcr_below_one: float
    cfcr_p05: float
    expected_shortfall: float
    simulations: int
    model_version: str
    assumptions: list[str]


class GroundingCheck(BaseModel):
    claim: str
    type: Literal["numeric", "citation"]
    source: str
    status: Literal["pass", "fail"]


class CustomAnalyzeRequest(BaseModel):
    sector: str
    years_operating: int = Field(ge=1, le=99)
    profile_type: Literal["healthy", "ntc", "buyer_concentrated", "seasonal"]
    msme_tier: Literal["micro", "small", "medium"] = "micro"
    gst_registered: bool = True
    employee_tier: Literal["micro", "small", "medium"] = "micro"
    requested_amount_lakh: float = Field(default=10.0, ge=1.0, le=25.0)
    annual_interest_rate_pct: float = Field(default=12.0, gt=0.0, le=40.0)
    expected_utilization_pct: float = Field(default=75.0, ge=0.0, le=100.0)
    annual_turnover_lakh: float = Field(default=24.0, gt=0.0)
    avg_monthly_inflow_lakh: float = Field(default=1.6, gt=0.0)
    avg_monthly_operating_outflow_lakh: float = Field(default=1.2, ge=0.0)
    avg_bank_balance_lakh: float = Field(default=1.5, ge=0.0)
    existing_monthly_emi_lakh: float = Field(default=0.2, ge=0.0)
    top_buyer_share_pct: float = Field(default=20.0, ge=0.0, le=100.0)
    bounced_payments_12mo: int = Field(default=0, ge=0, le=100)
    gst_filing_consistency_pct: float = Field(default=95.0, ge=0.0, le=100.0)
    yoy_growth_pct: float = Field(default=12.0, ge=-100.0, le=500.0)


class AnalysisResponse(BaseModel):
    profile_summary: dict
    cfcr_baseline: float
    cfcr_by_scenario: list[CFCRResult]
    weights_used: WeightVector
    weight_rationale: list[WeightRationaleItem]
    baseline_score: float
    stress_results: list[StressResult]
    tail_risk: TailRiskResult
    narrative: str
    grounding_trace: list[GroundingCheck]
