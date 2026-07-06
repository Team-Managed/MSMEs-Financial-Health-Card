import pytest
from pydantic import ValidationError
from backend.app.schemas.models import (
    MSMEProfile, WeightVector, CFCRResult, AnalysisResponse
)

def test_msme_profile_valid():
    profile = MSMEProfile(
        msme_id="p001",
        business_name="Test Co",
        sector="manufacturing",
        years_operating=5,
        gst={"monthly_turnover_series": [100000.0] * 12,
             "filing_consistency_score": 0.9,
             "yoy_growth_rate": 0.12},
        upi={"monthly_inflow_series": [80000.0] * 12,
             "monthly_outflow_series": [60000.0] * 12,
             "transaction_frequency": 240,
             "top_counterparty_share": 0.25},
        aa_bank_data={"existing_loan_count": 1,
                      "existing_loan_emi_total": 15000.0,
                      "overdraft_utilization_rate": 0.2,
                      "avg_account_balance": 120000.0,
                      "bounced_payment_count_12mo": 0,
                      "estimated_monthly_operating_outflow": 55000.0},
        epfo={"employee_count_series": [10] * 12,
              "payroll_consistency_score": 0.95}
    )
    assert profile.msme_id == "p001"

def test_weight_vector_must_be_floats():
    wv = WeightVector(gst=0.3, upi=0.3, aa=0.25, epfo=0.15)
    assert abs(wv.gst + wv.upi + wv.aa + wv.epfo - 1.0) < 0.01

def test_cfcr_pass_when_gte_one():
    r = CFCRResult(scenario="baseline", cfcr=1.25, pass_fail=True)
    assert r.pass_fail is True

def test_analysis_response_shape():
    import json
    resp = AnalysisResponse(
        profile_summary={"msme_id": "p001", "sector": "manufacturing"},
        cfcr_baseline=1.3,
        cfcr_by_scenario=[CFCRResult(scenario="baseline", cfcr=1.3, pass_fail=True)],
        weights_used=WeightVector(gst=0.3, upi=0.3, aa=0.25, epfo=0.15),
        weight_rationale=[],
        baseline_score=72.0,
        stress_results=[],
        narrative="Test narrative",
        grounding_trace=[]
    )
    assert resp.cfcr_baseline == 1.3
