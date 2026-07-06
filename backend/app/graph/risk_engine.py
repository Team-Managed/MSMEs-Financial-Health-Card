"""
Deterministic Risk Engine — no LLM calls.
Implements CFCR and the Financial Health Score.
"""
from __future__ import annotations

import numpy as np
from backend.app.schemas.models import (
    MSMEProfile, WeightVector, CFCRResult, StressResult
)

# ── CFCR ──────────────────────────────────────────────────────────────────────

def _near_term_receivables(upi_inflows: list[float]) -> float:
    """
    Estimate near-term realizable receivables as 80% of the mean of the
    top-3 monthly UPI inflows. Mirrors the Basel numerator's HQLA concept
    but uses the only liquid-buffer proxy available in the data.
    """
    if not upi_inflows:
        return 0.0
    top3 = sorted(upi_inflows, reverse=True)[:3]
    return float(np.mean(top3) * 0.8)


def compute_cfcr(
    avg_balance: float,
    upi_inflows: list[float],
    emi_total: float,
    operating_outflow: float,
) -> float:
    """
    CFCR = Liquid Buffer / Projected Net Cash Outflow

    Liquid Buffer = avg_account_balance + near_term_receivables
    Outflow       = existing_loan_emi_total + operating_outflow_estimate

    Denominator floor of 1.0 prevents division-by-zero for NTC profiles
    with no existing loans and zero declared operating outflow.
    """
    liquid_buffer = avg_balance + _near_term_receivables(upi_inflows)
    net_outflow = max(emi_total + operating_outflow, 1.0)
    return round(liquid_buffer / net_outflow, 4)


# ── Financial Health Score ────────────────────────────────────────────────────

def _gst_score(gst) -> float:
    """Score 0–100 from GST data."""
    trend = min(max((gst.yoy_growth_rate + 0.2) / 0.4, 0), 1)   # -20%..+20% → 0..1
    consistency = gst.filing_consistency_score                    # already 0–1
    return round((trend * 0.5 + consistency * 0.5) * 100, 2)


def _upi_score(upi) -> float:
    """Score 0–100 from UPI data."""
    net_flows = [i - o for i, o in zip(upi.monthly_inflow_series, upi.monthly_outflow_series)]
    mean_net = float(np.mean(net_flows))
    if mean_net <= 0:
        return 0.0
    cv = float(np.std(net_flows) / mean_net) if mean_net > 0 else 1.0
    stability = min(max(1 - cv, 0), 1)
    concentration_penalty = max(0, upi.top_counterparty_share - 0.4)  # penalise >40%
    raw = stability - concentration_penalty
    return round(max(0, raw) * 100, 2)


def _aa_score(aa) -> float:
    """Score 0–100 from AA bank data."""
    bounce_penalty = min(aa.bounced_payment_count_12mo * 0.1, 0.5)
    overdraft_penalty = aa.overdraft_utilization_rate * 0.3
    base = 1.0 - bounce_penalty - overdraft_penalty
    return round(max(0, base) * 100, 2)


def _epfo_score(epfo) -> float:
    """Score 0–100 from EPFO data."""
    employees = epfo.employee_count_series
    if not employees or np.mean(employees) == 0:
        return 50.0
    cv = float(np.std(employees) / np.mean(employees))
    stability = min(max(1 - cv, 0), 1)
    payroll = epfo.payroll_consistency_score
    return round((stability * 0.4 + payroll * 0.6) * 100, 2)


def compute_health_score(profile: MSMEProfile, weights: WeightVector) -> float:
    """Weighted composite health score 0–100."""
    gst = _gst_score(profile.gst)
    upi = _upi_score(profile.upi)
    aa = _aa_score(profile.aa_bank_data)
    epfo = _epfo_score(profile.epfo)
    score = (gst * weights.gst + upi * weights.upi +
             aa * weights.aa + epfo * weights.epfo)
    return round(min(max(score, 0), 100), 2)


# ── Stress Scenarios ──────────────────────────────────────────────────────────

def _apply_stress(profile: MSMEProfile, scenario: str) -> tuple[MSMEProfile, str]:
    """
    Returns a perturbed copy of the profile and a human-readable driver string.
    Does not mutate the original.
    """
    import copy
    p = copy.deepcopy(profile)

    if scenario == "receivable_delay_60d":
        # Delay inflows: reduce last 2 months of inflows by 40%, spike overdraft
        p.upi.monthly_inflow_series[-1] *= 0.6
        p.upi.monthly_inflow_series[-2] *= 0.6
        p.aa_bank_data.overdraft_utilization_rate = min(
            p.aa_bank_data.overdraft_utilization_rate + 0.20, 1.0
        )
        driver = "UPI inflows delayed 60d (−40% last 2 months); overdraft +20pp"

    elif scenario == "revenue_drop_20pct":
        p.gst.monthly_turnover_series = [v * 0.80 for v in p.gst.monthly_turnover_series]
        p.upi.monthly_inflow_series = [v * 0.80 for v in p.upi.monthly_inflow_series]
        driver = "GST turnover and UPI inflows cut 20%"

    elif scenario == "buyer_loss":
        share = p.upi.top_counterparty_share
        p.upi.monthly_inflow_series = [
            v * (1 - share) for v in p.upi.monthly_inflow_series
        ]
        driver = f"Top counterparty ({share*100:.0f}% share) lost — UPI inflows zeroed for that portion"

    elif scenario == "rate_hike":
        p.aa_bank_data.existing_loan_emi_total *= 1.15   # +15% EMI
        driver = "Floating-rate repricing: EMI +15%"

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return p, driver


# ── Main entry point ──────────────────────────────────────────────────────────

SCENARIOS = ["receivable_delay_60d", "revenue_drop_20pct", "buyer_loss", "rate_hike"]


def compute_risk(profile: MSMEProfile, weights: WeightVector, scenarios: list[str] | None = None) -> dict:
    """
    Runs the full risk computation.
    `scenarios` defaults to all 4 standard scenarios if not provided.
    Returns a dict matching AnalysisResponse field shapes (before Pydantic construction).
    """
    active_scenarios = scenarios if scenarios is not None else SCENARIOS
    cfcr_baseline = compute_cfcr(
        avg_balance=profile.aa_bank_data.avg_account_balance,
        upi_inflows=profile.upi.monthly_inflow_series,
        emi_total=profile.aa_bank_data.existing_loan_emi_total,
        operating_outflow=profile.aa_bank_data.estimated_monthly_operating_outflow,
    )
    baseline_score = compute_health_score(profile, weights)

    # Cash-flow volatility (coefficient of variation on net UPI flows)
    net_flows = [i - o for i, o in zip(
        profile.upi.monthly_inflow_series, profile.upi.monthly_outflow_series
    )]
    mean_net = float(np.mean(net_flows))
    cv = float(np.std(net_flows) / mean_net) if mean_net > 0 else 0.0

    # Buyer concentration flag
    buyer_flag = profile.upi.top_counterparty_share >= 0.40

    # Per-scenario stress
    cfcr_by_scenario: list[CFCRResult] = [
        CFCRResult(scenario="baseline", cfcr=cfcr_baseline, pass_fail=cfcr_baseline >= 1.0)
    ]
    stress_results: list[StressResult] = []

    for scenario in active_scenarios:
        stressed_profile, driver = _apply_stress(profile, scenario)
        stressed_cfcr = compute_cfcr(
            avg_balance=stressed_profile.aa_bank_data.avg_account_balance,
            upi_inflows=stressed_profile.upi.monthly_inflow_series,
            emi_total=stressed_profile.aa_bank_data.existing_loan_emi_total,
            operating_outflow=stressed_profile.aa_bank_data.estimated_monthly_operating_outflow,
        )
        stressed_score = compute_health_score(stressed_profile, weights)
        cfcr_by_scenario.append(
            CFCRResult(scenario=scenario, cfcr=stressed_cfcr, pass_fail=stressed_cfcr >= 1.0)
        )
        stress_results.append(StressResult(
            scenario=scenario,
            stressed_score=stressed_score,
            delta=round(stressed_score - baseline_score, 2),
            key_drivers=[driver],
        ))

    return {
        "cfcr_baseline": cfcr_baseline,
        "cfcr_by_scenario": cfcr_by_scenario,
        "baseline_score": baseline_score,
        "weights_used": weights,
        "stress_results": stress_results,
        "cash_flow_volatility": round(cv, 4),
        "buyer_concentration_flag": buyer_flag,
    }
