from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, field_validator


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
    def must_be_between_0_and_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        return v


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


class GroundingCheck(BaseModel):
    claim: str
    type: Literal["numeric", "citation"]
    source: str
    status: Literal["pass", "fail"]


class CustomAnalyzeRequest(BaseModel):
    sector: str
    years_operating: int
    profile_type: Literal["healthy", "ntc", "buyer_concentrated", "seasonal"]


class AnalysisResponse(BaseModel):
    profile_summary: dict
    cfcr_baseline: float
    cfcr_by_scenario: list[CFCRResult]
    weights_used: WeightVector
    weight_rationale: list[WeightRationaleItem]
    baseline_score: float
    stress_results: list[StressResult]
    narrative: str
    grounding_trace: list[GroundingCheck]
