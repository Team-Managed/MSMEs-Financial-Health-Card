import math
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
    assert abs(wv.gst + wv.upi + wv.aa + wv.epfo - 1.0) < 1e-6

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


# ── WeightVector integrity tests ──────────────────────────────────────────────

def test_weight_vector_rejects_nan():
    with pytest.raises(ValidationError):
        WeightVector(gst=float("nan"), upi=0.30, aa=0.25, epfo=0.15)


def test_weight_vector_rejects_positive_infinity():
    with pytest.raises(ValidationError):
        WeightVector(gst=float("inf"), upi=0.0, aa=0.0, epfo=0.0)


def test_weight_vector_rejects_negative_infinity():
    with pytest.raises(ValidationError):
        WeightVector(gst=float("-inf"), upi=0.5, aa=0.3, epfo=0.2)


def test_weight_vector_rejects_sum_too_high():
    # 0.40 + 0.40 + 0.10 + 0.20 = 1.10
    with pytest.raises(ValidationError):
        WeightVector(gst=0.40, upi=0.40, aa=0.10, epfo=0.20)


def test_weight_vector_rejects_sum_too_low():
    # 0.20 * 4 = 0.80
    with pytest.raises(ValidationError):
        WeightVector(gst=0.20, upi=0.20, aa=0.20, epfo=0.20)


def test_weight_vector_accepts_sum_exactly_one():
    wv = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    assert abs(wv.gst + wv.upi + wv.aa + wv.epfo - 1.0) < 1e-6
