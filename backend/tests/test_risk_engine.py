import pytest
from backend.app.schemas.models import WeightVector
from backend.app.data.personas import PERSONAS
from backend.app.graph.risk_engine import compute_cfcr, compute_health_score, compute_risk


def test_cfcr_baseline_above_one_for_healthy():
    profile = PERSONAS["healthy"]
    cfcr = compute_cfcr(
        avg_balance=profile.aa_bank_data.avg_account_balance,
        upi_inflows=profile.upi.monthly_inflow_series,
        emi_total=profile.aa_bank_data.existing_loan_emi_total,
        operating_outflow=profile.aa_bank_data.estimated_monthly_operating_outflow,
    )
    assert cfcr >= 1.0, f"Expected healthy CFCR >= 1.0, got {cfcr}"


def test_cfcr_drops_under_buyer_loss():
    profile = PERSONAS["buyer_concentrated"]
    inflows_stressed = [v * (1 - profile.upi.top_counterparty_share)
                        for v in profile.upi.monthly_inflow_series]
    cfcr_baseline = compute_cfcr(
        avg_balance=profile.aa_bank_data.avg_account_balance,
        upi_inflows=profile.upi.monthly_inflow_series,
        emi_total=profile.aa_bank_data.existing_loan_emi_total,
        operating_outflow=profile.aa_bank_data.estimated_monthly_operating_outflow,
    )
    cfcr_stressed = compute_cfcr(
        avg_balance=profile.aa_bank_data.avg_account_balance,
        upi_inflows=inflows_stressed,
        emi_total=profile.aa_bank_data.existing_loan_emi_total,
        operating_outflow=profile.aa_bank_data.estimated_monthly_operating_outflow,
    )
    assert cfcr_stressed < cfcr_baseline


def test_health_score_bounded_0_100():
    profile = PERSONAS["healthy"]
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    score = compute_health_score(profile, weights)
    assert 0 <= score <= 100


def test_buyer_concentration_flagged():
    from backend.app.graph.risk_engine import compute_risk
    profile = PERSONAS["buyer_concentrated"]
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    result = compute_risk(profile, weights)
    assert result["buyer_concentration_flag"] is True


def test_ntc_persona_emi_zero_does_not_divide_by_zero():
    profile = PERSONAS["ntc"]
    cfcr = compute_cfcr(
        avg_balance=profile.aa_bank_data.avg_account_balance,
        upi_inflows=profile.upi.monthly_inflow_series,
        emi_total=profile.aa_bank_data.existing_loan_emi_total,
        operating_outflow=profile.aa_bank_data.estimated_monthly_operating_outflow,
    )
    assert cfcr > 0
