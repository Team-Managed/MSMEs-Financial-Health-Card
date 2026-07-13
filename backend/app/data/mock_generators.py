import numpy as np
from backend.app.schemas.models import (
    MSMEProfile, GSTData, UPIData, AABankData, EPFOData
)


def _cv(arr: list[float]) -> float:
    a = np.array(arr)
    return float(np.std(a) / np.mean(a)) if np.mean(a) > 0 else 0.0


def generate_profile(
    seed: int,
    sector: str,
    profile_type: str,
    msme_id: str | None = None,
    business_name: str | None = None,
    years_operating: int = 5,
    msme_tier: str = "micro",
    employee_tier: str = "micro",
    annual_turnover: float | None = None,
    avg_monthly_inflow: float | None = None,
    avg_monthly_operating_outflow: float | None = None,
    avg_account_balance: float | None = None,
    existing_monthly_emi: float | None = None,
    top_counterparty_share: float | None = None,
    bounced_payments_12mo: int | None = None,
    filing_consistency_score: float | None = None,
    yoy_growth_rate: float | None = None,
) -> MSMEProfile:
    rng = np.random.default_rng(seed)

    # Scale all monetary values to reflect realistic INR amounts per MSME tier
    # micro ~₹1–5 Cr/yr → base ×1 | small ~₹5–50 Cr/yr → ×15 | medium ~₹50–250 Cr/yr → ×80
    TIER_SCALE = {"micro": 1, "small": 15, "medium": 80}
    EMPLOYEE_BASE = {"micro": 6, "small": 25, "medium": 100}
    scale = TIER_SCALE.get(msme_tier, 1)
    emp_base = EMPLOYEE_BASE.get(employee_tier, 6)

    if profile_type == "healthy":
        base_turnover = 200_000.0
        growth = 0.12
        filing_score = 0.95
        inflow_base = 160_000.0
        bounce_count = 0
        overdraft_rate = 0.15
        avg_balance = 150_000.0
        emi_total = 20_000.0
        employee_base = 15
        payroll_score = 0.95
        top_share = 0.20
        variance_pct = 0.08

    elif profile_type == "ntc":
        base_turnover = 120_000.0
        growth = 0.08
        filing_score = 0.88
        inflow_base = 100_000.0
        bounce_count = 0
        overdraft_rate = 0.05
        avg_balance = 80_000.0
        emi_total = 0.0
        employee_base = 8
        payroll_score = 0.90
        top_share = 0.30
        variance_pct = 0.10

    elif profile_type == "buyer_concentrated":
        base_turnover = 180_000.0
        growth = 0.09
        filing_score = 0.92
        inflow_base = 150_000.0
        bounce_count = 1
        overdraft_rate = 0.25
        avg_balance = 100_000.0
        emi_total = 18_000.0
        employee_base = 12
        payroll_score = 0.88
        top_share = 0.72
        variance_pct = 0.12

    elif profile_type == "seasonal":
        base_turnover = 150_000.0
        growth = 0.05
        filing_score = 0.80
        inflow_base = 120_000.0
        bounce_count = 2
        overdraft_rate = 0.30
        avg_balance = 70_000.0
        emi_total = 12_000.0
        employee_base = 10
        payroll_score = 0.75
        top_share = 0.35
        variance_pct = 0.35

    else:
        raise ValueError(f"Unknown profile_type: {profile_type}")

    # Apply MSME tier scale to all monetary values
    base_turnover *= scale
    inflow_base *= scale
    avg_balance *= scale
    emi_total *= scale

    if annual_turnover is not None:
        base_turnover = annual_turnover / 12
    if avg_monthly_inflow is not None:
        inflow_base = avg_monthly_inflow
    if avg_account_balance is not None:
        avg_balance = avg_account_balance
    if existing_monthly_emi is not None:
        emi_total = existing_monthly_emi
    if top_counterparty_share is not None:
        top_share = top_counterparty_share
    if bounced_payments_12mo is not None:
        bounce_count = bounced_payments_12mo
    if filing_consistency_score is not None:
        filing_score = filing_consistency_score
    if yoy_growth_rate is not None:
        growth = yoy_growth_rate

    # Override employee base from employee_tier if provided
    employee_base = emp_base

    # Build 12-month series with noise
    months = np.arange(12)
    noise = 1 + rng.normal(0, variance_pct, 12)
    turnover = [round(float(base_turnover * (1 + growth * m / 12) * noise[m]), 2)
                for m in range(12)]
    inflows = [round(float(inflow_base * noise[m]), 2) for m in range(12)]
    outflow_base = (
        avg_monthly_operating_outflow
        if avg_monthly_operating_outflow is not None
        else inflow_base * 0.75
    )
    outflows = [round(float(outflow_base * noise[m]), 2) for m in range(12)]
    employees = [max(1, int(employee_base + rng.integers(-2, 3))) for _ in range(12)]
    operating_outflow = round(float(np.mean(outflows)), 2)

    return MSMEProfile(
        msme_id=msme_id or f"{profile_type}_{seed}",
        business_name=business_name or f"{sector.title()} Co #{seed}",
        sector=sector,
        years_operating=years_operating,
        gst=GSTData(
            monthly_turnover_series=turnover,
            filing_consistency_score=filing_score,
            yoy_growth_rate=growth,
        ),
        upi=UPIData(
            monthly_inflow_series=inflows,
            monthly_outflow_series=outflows,
            transaction_frequency=int(rng.integers(180, 360)),
            top_counterparty_share=top_share,
        ),
        aa_bank_data=AABankData(
            existing_loan_count=0 if emi_total == 0 else int(rng.integers(1, 3)),
            existing_loan_emi_total=emi_total,
            overdraft_utilization_rate=overdraft_rate,
            avg_account_balance=avg_balance,
            bounced_payment_count_12mo=bounce_count,
            estimated_monthly_operating_outflow=operating_outflow,
        ),
        epfo=EPFOData(
            employee_count_series=employees,
            payroll_consistency_score=payroll_score,
        ),
    )
