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


# ── New stress-scenario invariant tests ──────────────────────────────────────

def test_revenue_drop_lowers_score_all_personas():
    """revenue_drop_20pct must reduce health score for every standard persona."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        result = compute_risk(profile, weights, scenarios=["revenue_drop_20pct"])
        delta = result["stress_results"][0].delta
        assert delta < 0, (
            f"revenue_drop_20pct should lower score for persona '{name}', got delta={delta}"
        )


def test_buyer_loss_lowers_score_material_share():
    """buyer_loss must reduce health score for personas with material counterparty share."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        if profile.upi.top_counterparty_share >= 0.10:
            result = compute_risk(profile, weights, scenarios=["buyer_loss"])
            delta = result["stress_results"][0].delta
            assert delta < 0, (
                f"buyer_loss should lower score for persona '{name}' "
                f"(share={profile.upi.top_counterparty_share:.2f}), got delta={delta}"
            )


def test_rate_hike_lowers_score_emi_positive():
    """rate_hike must reduce health score for profiles with existing EMI > 0."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        if profile.aa_bank_data.existing_loan_emi_total > 0:
            result = compute_risk(profile, weights, scenarios=["rate_hike"])
            delta = result["stress_results"][0].delta
            assert delta < 0, (
                f"rate_hike should lower score for persona '{name}' "
                f"(emi={profile.aa_bank_data.existing_loan_emi_total}), got delta={delta}"
            )


def test_stress_does_not_mutate_original_profile():
    """compute_risk must not mutate the original profile object for any persona."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        orig_turnover = list(profile.gst.monthly_turnover_series)
        orig_overdraft = profile.aa_bank_data.overdraft_utilization_rate
        compute_risk(profile, weights)
        assert list(profile.gst.monthly_turnover_series) == orig_turnover, (
            f"gst.monthly_turnover_series mutated for persona '{name}'"
        )
        assert profile.aa_bank_data.overdraft_utilization_rate == orig_overdraft, (
            f"aa_bank_data.overdraft_utilization_rate mutated for persona '{name}'"
        )


def test_rate_hike_no_effect_ntc_no_emi():
    """rate_hike must not change score for NTC profile with zero EMI."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    profile = PERSONAS["ntc"]
    assert profile.aa_bank_data.existing_loan_emi_total == 0.0
    result = compute_risk(profile, weights, scenarios=["rate_hike"])
    delta = result["stress_results"][0].delta
    assert delta == 0.0, f"rate_hike should not affect NTC (no EMI), got delta={delta}"


def test_receivable_delay_lowers_cfcr_all_personas():
    """receivable_delay_60d must reduce CFCR for every standard persona."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        result = compute_risk(profile, weights, scenarios=["receivable_delay_60d"])
        baseline_cfcr = result["cfcr_baseline"]
        stressed_cfcr = next(
            r.cfcr for r in result["cfcr_by_scenario"]
            if r.scenario == "receivable_delay_60d"
        )
        assert stressed_cfcr < baseline_cfcr, (
            f"receivable_delay_60d should lower CFCR for persona '{name}', "
            f"got baseline={baseline_cfcr}, stressed={stressed_cfcr}"
        )


def test_stress_scores_bounded_0_100():
    """All stressed scores must remain within [0, 100]."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        result = compute_risk(profile, weights)
        assert 0 <= result["baseline_score"] <= 100, f"baseline_score out of bounds for {name}"
        for sr in result["stress_results"]:
            assert 0 <= sr.stressed_score <= 100, (
                f"stressed_score out of bounds for {name}/{sr.scenario}: {sr.stressed_score}"
            )


def test_original_profile_unmodified_after_stress():
    """compute_risk must not mutate the input profile."""
    weights = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
    for name, profile in PERSONAS.items():
        orig_inflows = list(profile.upi.monthly_inflow_series)
        orig_emi = profile.aa_bank_data.existing_loan_emi_total
        orig_gst_growth = profile.gst.yoy_growth_rate
        compute_risk(profile, weights)
        assert list(profile.upi.monthly_inflow_series) == orig_inflows, \
            f"UPI inflows mutated for {name}"
        assert profile.aa_bank_data.existing_loan_emi_total == orig_emi, \
            f"EMI mutated for {name}"
        assert profile.gst.yoy_growth_rate == orig_gst_growth, \
            f"GST growth rate mutated for {name}"
