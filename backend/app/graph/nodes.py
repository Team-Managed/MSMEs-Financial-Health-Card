"""
LangGraph node functions for the MSME pipeline.
Each function accepts a state dict and returns a partial state dict.
"""
from __future__ import annotations

import json
import logging
import os
import re

import google.generativeai as genai

from backend.app.data.personas import PERSONAS
from backend.app.graph.risk_engine import compute_risk
from backend.app.rag.retriever import Retriever
from backend.app.schemas.models import WeightVector, WeightRationaleItem

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
_DEFAULT_RATIONALE = [
    WeightRationaleItem(dimension="gst", reasoning="Default equal weighting — no RAG context available.", cited_chunk_id="default"),
    WeightRationaleItem(dimension="upi", reasoning="Default equal weighting — no RAG context available.", cited_chunk_id="default"),
    WeightRationaleItem(dimension="aa", reasoning="Default equal weighting — no RAG context available.", cited_chunk_id="default"),
    WeightRationaleItem(dimension="epfo", reasoning="Default equal weighting — no RAG context available.", cited_chunk_id="default"),
]


def _get_gemini_model() -> genai.GenerativeModel:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Node 1: Data Aggregator ───────────────────────────────────────────────────

def node_aggregator(state: dict) -> dict:
    persona_id = state["persona_id"]
    profile = PERSONAS.get(persona_id)
    if profile is None:
        raise ValueError(f"Unknown persona_id: {persona_id!r}. Valid: {list(PERSONAS)}")
    return {"profile": profile}


# ── Node 1.5a: Sector Context Retriever ──────────────────────────────────────

def node_sector_retriever(state: dict) -> dict:
    profile = state["profile"]
    retriever: Retriever = state.get("retriever") or Retriever()
    query = (
        f"MSME credit risk assessment {profile.sector} "
        f"alternate data scoring weight "
        f"{'thin file' if profile.aa_bank_data.existing_loan_count == 0 else 'credit history'}"
    )
    chunks = retriever.query(query, n_results=5)
    return {"retrieved_chunks": chunks}


# ── Node 1.5b: Weight-Setter (LLM, RAG-grounded) ─────────────────────────────

_WEIGHT_SETTER_PROMPT = """You are a credit risk analyst setting data-source weights for an MSME credit scoring model.

MSME Profile:
- Sector: {sector}
- Years operating: {years_operating}
- New-to-credit (no prior loans): {ntc}
- Top UPI counterparty share: {top_share:.0%}

Relevant retrieved guidance:
{chunks}

Set weights for these 4 data dimensions. Weights must sum to 1.0. Justify each weight
by citing ONLY the retrieved guidance above (use chunk_id). If no chunk supports a
particular weight, set it to a reasonable default and say "no retrieved guidance — default used."

Respond with ONLY valid JSON:
{{
  "weights": {{"gst": <float>, "upi": <float>, "aa": <float>, "epfo": <float>}},
  "rationale": [
    {{"dimension": "gst", "reasoning": "<why>", "cited_chunk_id": "<id or empty>"}},
    {{"dimension": "upi", "reasoning": "<why>", "cited_chunk_id": "<id or empty>"}},
    {{"dimension": "aa",  "reasoning": "<why>", "cited_chunk_id": "<id or empty>"}},
    {{"dimension": "epfo","reasoning": "<why>", "cited_chunk_id": "<id or empty>"}}
  ]
}}"""


def node_weight_setter(state: dict) -> dict:
    profile = state["profile"]
    chunks: list[dict] = state.get("retrieved_chunks", [])

    if not chunks:
        logger.warning("No RAG chunks available — using default weights")
        return {"weights": _DEFAULT_WEIGHTS, "weight_rationale": _DEFAULT_RATIONALE}

    chunks_text = "\n".join(
        f"[{c['chunk_id']}] ({c['source']}, {c['section']}): {c['text'][:300]}"
        for c in chunks
    )
    prompt = _WEIGHT_SETTER_PROMPT.format(
        sector=profile.sector,
        years_operating=profile.years_operating,
        ntc=profile.aa_bank_data.existing_loan_count == 0,
        top_share=profile.upi.top_counterparty_share,
        chunks=chunks_text,
    )

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        weights = WeightVector(**data["weights"])
        rationale = [WeightRationaleItem(**r) for r in data["rationale"]]
        return {"weights": weights, "weight_rationale": rationale}
    except Exception as exc:
        logger.warning("Weight-setter LLM call failed (%s) — using defaults", exc)
        return {"weights": _DEFAULT_WEIGHTS, "weight_rationale": _DEFAULT_RATIONALE}


# ── Node 2: Stress Scenario Generator ────────────────────────────────────────

def node_stress_generator(state: dict) -> dict:
    return {"scenarios": ["receivable_delay_60d", "revenue_drop_20pct", "buyer_loss", "rate_hike"]}


# ── Node 3: Risk Engine ───────────────────────────────────────────────────────

def node_risk_engine(state: dict) -> dict:
    profile = state["profile"]
    weights: WeightVector = state.get("weights", _DEFAULT_WEIGHTS)
    scenarios: list[str] | None = state.get("scenarios", None)  # None → compute_risk uses its default
    risk_output = compute_risk(profile, weights, scenarios=scenarios)
    return {"risk_output": risk_output}


# ── Node 4: Explainer (LLM + RAG) ────────────────────────────────────────────

_EXPLAINER_PROMPT = """You are a financial analyst writing a credit health summary for a loan officer.

MSME: {business_name} ({sector}, {years_operating} years operating)

Risk Engine output:
- Baseline CFCR: {cfcr_baseline} (≥1.0 = survives shock; <1.0 = liquidity failure)
- Baseline Financial Health Score: {baseline_score}/100
- Stress scenarios:
{stress_table}
- Buyer concentration flag: {buyer_flag}
- Cash-flow volatility (CV): {cv}

Retrieved guidance used:
{chunks}

Write the Financial Health Card narrative (3–5 short paragraphs):
1. Open with CFCR headline — state pass/fail and what it means in plain language
2. Summarise the 2 stress scenarios that caused the largest CFCR drop
3. Note strengths (cite retrieved guidance where relevant — use [chunk_id])
4. Note key risks (same citation rule)
5. Close with 1-sentence loan-officer recommendation

Rules:
- Every number you cite must appear exactly in the Risk Engine output above
- Every regulatory/guidance claim must reference a retrieved chunk by [chunk_id]
- If no chunk supports a claim, state the point without a citation
- Do not invent chunk IDs
"""


def node_explainer(state: dict) -> dict:
    profile = state["profile"]
    risk = state["risk_output"]
    chunks: list[dict] = state.get("retrieved_chunks", [])

    stress_table = "\n".join(
        f"  {r.scenario}: score {r.stressed_score}/100 (delta {r.delta:+.1f}), "
        f"CFCR {next((c.cfcr for c in risk['cfcr_by_scenario'] if c.scenario == r.scenario), 'N/A')}"
        for r in risk["stress_results"]
    )
    chunks_text = "\n".join(
        f"[{c['chunk_id']}] ({c['source']}): {c['text'][:300]}"
        for c in chunks
    ) or "(no retrieved guidance available)"

    prompt = _EXPLAINER_PROMPT.format(
        business_name=profile.business_name,
        sector=profile.sector,
        years_operating=profile.years_operating,
        cfcr_baseline=risk["cfcr_baseline"],
        baseline_score=risk["baseline_score"],
        stress_table=stress_table,
        buyer_flag=risk["buyer_concentration_flag"],
        cv=risk["cash_flow_volatility"],
        chunks=chunks_text,
    )

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        return {"narrative": response.text.strip()}
    except Exception as exc:
        logger.warning("Explainer LLM call failed (%s)", exc)
        return {
            "narrative": (
                f"CFCR baseline: {risk['cfcr_baseline']} "
                f"({'PASS' if risk['cfcr_baseline'] >= 1.0 else 'FAIL'}). "
                f"Financial Health Score: {risk['baseline_score']}/100. "
                f"(Narrative generation unavailable — check GOOGLE_API_KEY.)"
            )
        }


# ── Node 5: Grounding Validator ───────────────────────────────────────────────

def _extract_numbers_from_text(text: str) -> list[tuple[str, float]]:
    """
    Extract (context_snippet, value) pairs for numbers that look like
    financial figures — catches both decimals (1.25) and whole numbers (72, 15000).
    Minimum 2 digits for whole numbers to avoid false positives on single digits.
    """
    pattern = re.compile(r"(?<!\w)(\d{2,}(?:\.\d{1,4})?|\d+\.\d{1,4})(?!\w)")
    results = []
    for m in pattern.finditer(text):
        start = max(0, m.start() - 40)
        snippet = text[start: m.end() + 40].replace("\n", " ").strip()
        results.append((snippet, float(m.group(1))))
    return results


def _flatten_risk_numbers(risk_output: dict) -> set[float]:
    """Collect all numeric values from the risk_output structure. Skips booleans."""
    numbers: set[float] = set()

    def _walk(obj):
        if isinstance(obj, bool):     # bool subclasses int — must check first
            return
        if isinstance(obj, (int, float)):
            numbers.add(round(float(obj), 4))
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)
        elif hasattr(obj, "__dict__"):
            _walk(obj.__dict__)

    _walk(risk_output)
    return numbers


def node_grounding_validator(state: dict) -> dict:
    from backend.app.schemas.models import GroundingCheck

    narrative: str = state.get("narrative", "")
    risk_output: dict = state["risk_output"]
    retrieved_chunks: list[dict] = state.get("retrieved_chunks", [])
    valid_chunk_ids = {c["chunk_id"] for c in retrieved_chunks}

    grounding_trace: list[GroundingCheck] = []
    allowed_numbers = _flatten_risk_numbers(risk_output)

    # 1. Numeric grounding check
    for snippet, value in _extract_numbers_from_text(narrative):
        tolerance = max(abs(value) * 0.01, 0.01)
        matched = any(abs(value - n) <= tolerance for n in allowed_numbers)
        grounding_trace.append(GroundingCheck(
            claim=snippet[:120],
            type="numeric",
            source="risk_engine_output",
            status="pass" if matched else "fail",
        ))

    # 2. Citation grounding check — find [chunk_id] patterns in narrative
    cited = re.findall(r"\[([a-z0-9_\-]{4,32})\]", narrative)
    for chunk_id in cited:
        is_valid = chunk_id in valid_chunk_ids
        grounding_trace.append(GroundingCheck(
            claim=f"Citation [{chunk_id}]",
            type="citation",
            source=chunk_id,
            status="pass" if is_valid else "fail",
        ))

    # 3. LLM fallback — only if any check failed
    failed = [c for c in grounding_trace if c.status == "fail"]
    if failed:
        try:
            model = _get_gemini_model()
            fail_summary = "\n".join(
                f"- [{c.type}] {c.claim[:100]}" for c in failed
            )
            prompt = (
                f"The following claims in a credit narrative failed grounding checks "
                f"(numeric claims not found in source data, or citations not in retrieved docs):\n"
                f"{fail_summary}\n\n"
                f"For each failed claim, output one line: "
                f"CLAIM: <original> | ISSUE: <why it fails> | FIX: <corrected version or 'remove'>"
            )
            response = model.generate_content(prompt)
            grounding_trace.append(GroundingCheck(
                claim="LLM fallback diagnosis",
                type="numeric",
                source="llm_fallback",
                status="fail",
            ))
            state["grounding_llm_diagnosis"] = response.text.strip()
        except Exception as exc:
            logger.warning("Grounding LLM fallback failed (%s)", exc)

    return {"grounding_trace": grounding_trace}
