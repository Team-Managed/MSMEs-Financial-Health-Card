import pytest
from backend.app.data.mock_generators import generate_profile
from backend.app.data.personas import PERSONAS
from backend.app.schemas.models import MSMEProfile


def test_generate_profile_returns_valid_msme():
    profile = generate_profile(seed=42, sector="manufacturing", profile_type="healthy")
    assert isinstance(profile, MSMEProfile)
    assert len(profile.gst.monthly_turnover_series) == 12
    assert len(profile.upi.monthly_inflow_series) == 12
    assert len(profile.epfo.employee_count_series) == 12
    assert 0 <= profile.upi.top_counterparty_share <= 1


def test_generate_profile_reproducible():
    p1 = generate_profile(seed=7, sector="services", profile_type="healthy")
    p2 = generate_profile(seed=7, sector="services", profile_type="healthy")
    assert p1.gst.monthly_turnover_series == p2.gst.monthly_turnover_series


def test_unregistered_gst_reduces_gst_evidence_score():
    registered = generate_profile(
        seed=7, sector="services", profile_type="healthy", gst_registered=True,
    )
    unregistered = generate_profile(
        seed=7, sector="services", profile_type="healthy", gst_registered=False,
    )
    assert unregistered.gst.filing_consistency_score < registered.gst.filing_consistency_score


def test_personas_has_four_keys():
    assert set(PERSONAS.keys()) == {"healthy", "ntc", "buyer_concentrated", "seasonal"}


def test_buyer_concentrated_persona_has_high_share():
    p = PERSONAS["buyer_concentrated"]
    assert p.upi.top_counterparty_share >= 0.6


def test_ntc_persona_has_no_existing_loans():
    p = PERSONAS["ntc"]
    assert p.aa_bank_data.existing_loan_count == 0
