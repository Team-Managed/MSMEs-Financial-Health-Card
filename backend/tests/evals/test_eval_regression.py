"""
Regression-tracking eval: guards against formula drift and budget bloat.

Layer structure:
  R1 — Deterministic (no API key): Risk Engine CFCR and health-score outputs
       must stay within the bounds in golden_dataset.json. Catches accidental
       changes to compute_cfcr(), compute_health_score(), or persona generators.

  R2 — Deterministic (no API key): stress scenario CFCR ordering invariants
       (e.g. buyer_loss must drop CFCR for buyer_concentrated persona).

  R3 — LLM cost/latency budget (requires GOOGLE_API_KEY): runs one full
       pipeline and asserts per-node token counts and estimated cost stay within
       golden_dataset["cost_budgets"]. Reports a formatted metrics table.
       Skipped when GOOGLE_API_KEY is not set.

golden_dataset.json is the single source of truth for all bounds and budgets.
To update baselines after an intentional change: recompute values, update the
JSON, and commit both together so the intent is clear in git history.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from backend.app.data.personas import PERSONAS
from backend.app.schemas.models import WeightVector
from backend.app.graph.risk_engine import compute_cfcr, compute_health_score, compute_risk
from backend.app.graph.metrics import compute_cost

GOLDEN = json.loads(
    (Path(__file__).parent / "golden_dataset.json").read_text()
)
BUDGETS = GOLDEN["cost_budgets"]
PERSONA_IDS = list(GOLDEN["personas"].keys())
_DEFAULT_WEIGHTS = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)


# ── R1: CFCR and health-score regression ─────────────────────────────────────

@pytest.mark.parametrize("persona_id", PERSONA_IDS)
def test_cfcr_no_regression(persona_id):
    """Baseline CFCR must stay >= expected_cfcr_min in golden_dataset.json."""
    profile = PERSONAS[persona_id]
    cfcr = compute_cfcr(
        avg_balance=profile.aa_bank_data.avg_account_balance,
        upi_inflows=profile.upi.monthly_inflow_series,
        emi_total=profile.aa_bank_data.existing_loan_emi_total,
        operating_outflow=profile.aa_bank_data.estimated_monthly_operating_outflow,
    )
    bound = GOLDEN["personas"][persona_id]["expected_cfcr_min"]
    assert cfcr >= bound, (
        f"[{persona_id}] CFCR REGRESSION: {cfcr:.4f} < golden min {bound}. "
        f"Update golden_dataset.json if this is an intentional change."
    )


@pytest.mark.parametrize("persona_id", PERSONA_IDS)
def test_health_score_no_regression(persona_id):
    """Baseline health score must stay >= expected_score_min in golden_dataset.json."""
    profile = PERSONAS[persona_id]
    score = compute_health_score(profile, _DEFAULT_WEIGHTS)
    bound = GOLDEN["personas"][persona_id]["expected_score_min"]
    assert score >= bound, (
        f"[{persona_id}] HEALTH SCORE REGRESSION: {score:.2f} < golden min {bound}. "
        f"Update golden_dataset.json if this is an intentional change."
    )


# ── R2: Stress ordering invariants ────────────────────────────────────────────

@pytest.mark.parametrize("persona_id", PERSONA_IDS)
def test_stress_scenario_ordering_invariants(persona_id):
    """buyer_loss must reduce CFCR for the buyer_concentrated persona."""
    bounds = GOLDEN["personas"][persona_id]
    if not bounds.get("buyer_loss_cfcr_must_drop"):
        pytest.skip(f"buyer_loss_cfcr_must_drop not set for {persona_id}")

    profile = PERSONAS[persona_id]
    risk = compute_risk(profile, _DEFAULT_WEIGHTS)

    baseline = risk["cfcr_baseline"]
    buyer_loss_cfcr = next(
        r.cfcr for r in risk["cfcr_by_scenario"] if r.scenario == "buyer_loss"
    )
    assert buyer_loss_cfcr < baseline, (
        f"[{persona_id}] buyer_loss CFCR {buyer_loss_cfcr:.4f} must be "
        f"< baseline {baseline:.4f}"
    )


# ── R3: LLM cost + latency budget (requires GOOGLE_API_KEY) ──────────────────

@pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping R3 LLM cost regression",
)
def test_llm_node_cost_within_budget(tmp_path):
    """
    Runs the full pipeline for the 'healthy' persona and asserts per-node token
    counts and per-run estimated cost stay within golden_dataset cost_budgets.
    Prints a formatted metrics table on completion.
    """
    from backend.app.graph.pipeline import run_pipeline
    from backend.app.rag.retriever import Retriever

    retriever = Retriever(chroma_dir=str(tmp_path / "empty"))
    result = run_pipeline("healthy", retriever=retriever)

    node_metrics: dict = result.get("_metrics", {})
    total_cost = sum(
        m.get("estimated_cost_usd", 0.0) for m in node_metrics.values()
    )
    total_input_tokens = sum(m.get("input_tokens", 0) for m in node_metrics.values())
    total_output_tokens = sum(m.get("output_tokens", 0) for m in node_metrics.values())

    # Print metrics table for CI/CD log visibility
    print("\n── LLM Node Metrics ─────────────────────────────────────────")
    print(f"{'node':<20} {'latency_ms':>10} {'input_tok':>10} "
          f"{'output_tok':>11} {'cost_usd':>12}")
    print("-" * 66)
    for node_name, m in sorted(node_metrics.items()):
        print(
            f"{node_name:<20} {m.get('latency_ms', 0):>10.1f} "
            f"{m.get('input_tokens', 0):>10} {m.get('output_tokens', 0):>11} "
            f"${m.get('estimated_cost_usd', 0):>11.7f}"
        )
    print("-" * 66)
    print(f"{'TOTAL':<20} {'':>10} {total_input_tokens:>10} "
          f"{total_output_tokens:>11} ${total_cost:>11.7f}")
    print(f"\nBudget: ${BUDGETS['max_estimated_cost_per_run_usd']}")

    # Per-node token assertions
    for node_name, m in node_metrics.items():
        in_tok = m.get("input_tokens", 0)
        out_tok = m.get("output_tokens", 0)
        if in_tok > 0:
            assert in_tok <= BUDGETS["max_input_tokens_per_llm_node"], (
                f"[{node_name}] Input tokens {in_tok} exceeds budget "
                f"{BUDGETS['max_input_tokens_per_llm_node']}. "
                f"Check for prompt bloat."
            )
        if out_tok > 0:
            assert out_tok <= BUDGETS["max_output_tokens_per_llm_node"], (
                f"[{node_name}] Output tokens {out_tok} exceeds budget "
                f"{BUDGETS['max_output_tokens_per_llm_node']}."
            )

    # Total cost assertion
    assert total_cost <= BUDGETS["max_estimated_cost_per_run_usd"], (
        f"Total estimated cost ${total_cost:.7f} exceeds budget "
        f"${BUDGETS['max_estimated_cost_per_run_usd']}. "
        f"Check for prompt inflation or unexpected LLM calls."
    )
