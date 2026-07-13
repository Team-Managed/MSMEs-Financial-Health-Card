"""
LangGraph node functions for the MSME pipeline.
Each function accepts a state dict and returns a partial state dict.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import google.generativeai as genai
from langsmith import traceable

from backend.app.data.personas import PERSONAS
from backend.app.graph.risk_engine import compute_risk
from backend.app.rag.retriever import Retriever
from backend.app.schemas.models import WeightVector, WeightRationaleItem

logger = logging.getLogger(__name__)

_DEFAULT_WEIGHTS = WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)
_DEFAULT_RATIONALE = [
    WeightRationaleItem(dimension="gst", reasoning="GST filing consistency and turnover trend weighted at 30% as the primary formal-economy signal for credit assessment.", cited_chunk_id="default"),
    WeightRationaleItem(dimension="upi", reasoning="UPI cash-flow patterns weighted at 30% as a real-time proxy for business liquidity and revenue stability.", cited_chunk_id="default"),
    WeightRationaleItem(dimension="aa", reasoning="Account Aggregator bank data weighted at 25% reflecting repayment behaviour and balance adequacy.", cited_chunk_id="default"),
    WeightRationaleItem(dimension="epfo", reasoning="EPFO payroll consistency weighted at 15% as an indicator of workforce stability and operational continuity.", cited_chunk_id="default"),
]

# Map for fast lookup during rationale validation
_DEFAULT_RATIONALE_MAP: dict[str, WeightRationaleItem] = {
    item.dimension: item for item in _DEFAULT_RATIONALE
}
_REQUIRED_DIMS: frozenset[str] = frozenset({"gst", "upi", "aa", "epfo"})
_NO_GUIDANCE_PHRASES = ("no retrieved guidance", "default used")
# Matches any hard-coded percentage literal (e.g. "30%", "25 %") in rationale text.
_CONTRADICTORY_PCT_RE = re.compile(r"\d+\s*%")
_SAFE_SUMMARY_SOURCE = "__safe_summary__"
_EXPLAINER_CHUNKS_SOURCE = "__explainer_chunks__"


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _to_plain_data(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "model_dump"):
        return _to_plain_data(value.model_dump())
    if hasattr(value, "__dict__"):
        return _to_plain_data(vars(value))
    return value


def _resolve_source_field(data: Any, source_field: str) -> Any:
    current = data
    for part in source_field.split("."):
        match = re.fullmatch(r"([a-zA-Z_][a-zA-Z0-9_]*)(?:\[(\d+)\])?", part)
        if not match:
            raise KeyError(f"Invalid source_field segment: {part!r}")
        key = match.group(1)
        idx = match.group(2)

        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Unknown source_field key: {key!r}")
        current = current[key]

        if idx is not None:
            if not isinstance(current, list):
                raise KeyError(f"source_field index used on non-list key: {key!r}")
            i = int(idx)
            if i < 0 or i >= len(current):
                raise KeyError(f"source_field index out of range: {source_field!r}")
            current = current[i]
    return current


def _numeric_tolerance(expected: float) -> float:
    return max(abs(expected) * 0.001, 0.01)


def _parse_explainer_output(raw_text: str) -> tuple[str, list[dict[str, Any]]]:
    raw = _strip_code_fences(raw_text)
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Explainer response must be a JSON object")

    narrative = data.get("narrative")
    claims = data.get("claims")
    if not isinstance(narrative, str) or not narrative.strip():
        raise ValueError("Explainer response missing non-empty narrative")
    if not isinstance(claims, list):
        raise ValueError("Explainer response missing claims list")

    normalized: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Each claim must be an object")
        claim_type = claim.get("type")
        source_field = claim.get("source_field")
        text = claim.get("text")
        value = claim.get("value")

        if claim_type not in {"numeric", "citation"}:
            raise ValueError("Claim type must be 'numeric' or 'citation'")
        if not isinstance(source_field, str) or not source_field.strip():
            raise ValueError("Claim source_field must be a non-empty string")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Claim text must be a non-empty string")

        if claim_type == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Numeric claim value must be a number")
            normalized.append({
                "type": "numeric",
                "source_field": source_field.strip(),
                "value": float(value),
                "text": text.strip(),
            })
        else:
            if source_field.strip() != _EXPLAINER_CHUNKS_SOURCE:
                raise ValueError(
                    "Citation claim source_field must be "
                    f"{_EXPLAINER_CHUNKS_SOURCE!r}"
                )
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Citation claim value must be a non-empty chunk_id string")
            normalized.append({
                "type": "citation",
                "source_field": source_field.strip(),
                "value": value.strip(),
                "text": text.strip(),
            })

    return narrative.strip(), normalized


def _validate_claims(
    claims: list[dict[str, Any]],
    risk_output: dict[str, Any],
    valid_chunk_ids: set[str],
) -> tuple[list[Any], list[dict[str, Any]]]:
    from backend.app.schemas.models import GroundingCheck

    trace: list[GroundingCheck] = []
    validated: list[dict[str, Any]] = []

    for claim in claims:
        claim_type = claim["type"]
        source_field = claim["source_field"]
        value = claim["value"]
        text = claim["text"]

        if claim_type == "numeric":
            try:
                resolved = _resolve_source_field(risk_output, source_field)
                if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
                    raise ValueError("resolved source is not numeric")
                expected = float(resolved)
                tolerance = _numeric_tolerance(expected)
                matched = abs(float(value) - expected) <= tolerance
            except Exception:
                matched = False

            trace.append(GroundingCheck(
                claim=text[:120],
                type="numeric",
                source=source_field,
                status="pass" if matched else "fail",
            ))
            if matched:
                validated.append(claim)
            continue

        is_valid_citation = value in valid_chunk_ids
        trace.append(GroundingCheck(
            claim=text[:120],
            type="citation",
            source=str(value),
            status="pass" if is_valid_citation else "fail",
        ))
        if is_valid_citation:
            validated.append(claim)

    return trace, validated


def _render_validated_claims(
    validated_claims: list[dict[str, Any]],
    risk_output: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Render claims from validated source data, never from LLM-provided prose."""
    rendered_claims: list[dict[str, Any]] = []
    narrative_lines: list[str] = []

    for claim in validated_claims:
        if claim["type"] == "numeric":
            try:
                resolved = _resolve_source_field(risk_output, claim["source_field"])
            except KeyError:
                continue
            if isinstance(resolved, bool) or not isinstance(resolved, (int, float)):
                continue

            value = float(resolved)
            text = f"Verified risk metric {claim['source_field']}: {value:.4f}."
            rendered_claims.append({
                "type": "numeric",
                "source_field": claim["source_field"],
                "value": value,
                "text": text,
            })
            narrative_lines.append(text)
            continue

        citation_id = claim["value"]
        text = f"Validated guidance reference [{citation_id}]."
        rendered_claims.append({
            "type": "citation",
            "source_field": _EXPLAINER_CHUNKS_SOURCE,
            "value": citation_id,
            "text": text,
        })
        narrative_lines.append(text)

    if not rendered_claims:
        return None
    return " ".join(narrative_lines), rendered_claims


def _build_safe_summary_from_risk(risk_output: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    cfcr_baseline = float(risk_output.get("cfcr_baseline", 0.0))
    baseline_score = float(risk_output.get("baseline_score", 0.0))
    cfcr_rows = risk_output.get("cfcr_by_scenario", [])

    worst_index = None
    worst_cfcr = cfcr_baseline
    worst_scenario = "baseline"
    for idx, row in enumerate(cfcr_rows):
        scenario = row.get("scenario") if isinstance(row, dict) else None
        cfcr = row.get("cfcr") if isinstance(row, dict) else None
        if scenario == "baseline" or not isinstance(cfcr, (int, float)):
            continue
        if worst_index is None or float(cfcr) < worst_cfcr:
            worst_index = idx
            worst_cfcr = float(cfcr)
            worst_scenario = str(scenario)

    if worst_index is None:
        worst_source = "cfcr_baseline"
    else:
        worst_source = f"cfcr_by_scenario[{worst_index}].cfcr"

    narrative = (
        "Limited-confidence deterministic summary: one or more generated claims failed grounding checks, "
        "so the narrative was replaced with direct risk-engine facts only. "
        f"Baseline CFCR is {cfcr_baseline:.4f}. "
        f"Baseline Financial Health Score is {baseline_score:.2f}/100. "
        f"Lowest stressed CFCR is {worst_cfcr:.4f} in {worst_scenario}."
    )

    claims = [
        {
            "type": "numeric",
            "source_field": "cfcr_baseline",
            "value": cfcr_baseline,
            "text": f"Baseline CFCR is {cfcr_baseline:.4f}.",
        },
        {
            "type": "numeric",
            "source_field": "baseline_score",
            "value": baseline_score,
            "text": f"Baseline Financial Health Score is {baseline_score:.2f}/100.",
        },
        {
            "type": "numeric",
            "source_field": worst_source,
            "value": worst_cfcr,
            "text": f"Lowest stressed CFCR is {worst_cfcr:.4f} in {worst_scenario}.",
        },
    ]
    return narrative, claims


def _validate_rationale(
    rationale: list[WeightRationaleItem],
    retrieved_chunk_ids: set[str],
) -> list[WeightRationaleItem] | None:
    """Validate LLM-produced rationale against the retrieved chunk set.

    Returns a validated list, or None to signal full fallback to documented
    defaults (fail-closed).  Rules:
    - Duplicate or missing dimensions → None
    - Non-empty, non-'default' cited_chunk_id absent from retrieved_chunk_ids → None
    - Empty cited_chunk_id without an explicit "no retrieved guidance / default used"
      phrase in reasoning → None (reject entire result; avoid rationale/weight mismatch)
    - Empty cited_chunk_id with the explicit phrase but also a hard-coded percentage
      literal → None (contradictory wording; reject to prevent misleading output)
    """
    dims = [item.dimension for item in rationale]
    if set(dims) != _REQUIRED_DIMS or len(dims) != len(_REQUIRED_DIMS):
        logger.warning("Rationale dimension mismatch %s — using defaults", dims)
        return None

    result: list[WeightRationaleItem] = []
    for item in rationale:
        cid = item.cited_chunk_id
        if cid and cid != "default":
            if cid not in retrieved_chunk_ids:
                logger.warning(
                    "Fabricated chunk ID %r not in retrieved set — using defaults", cid
                )
                return None
            result.append(item)
        elif not cid:
            lower = item.reasoning.lower()
            if any(phrase in lower for phrase in _NO_GUIDANCE_PHRASES):
                # Explicit no-guidance phrase is acceptable — but reject if the
                # wording also contains a hard-coded percentage that could
                # contradict the actual LLM-chosen weight.
                if _CONTRADICTORY_PCT_RE.search(item.reasoning):
                    logger.warning(
                        "No-guidance rationale for '%s' contains a hard-coded "
                        "percentage — rejecting LLM result to avoid misleading mismatch",
                        item.dimension,
                    )
                    return None
                result.append(item)
            else:
                logger.warning(
                    "Empty cited_chunk_id for '%s' without explicit default/no-guidance "
                    "phrase — rejecting LLM result to prevent rationale/weight mismatch",
                    item.dimension,
                )
                return None
        else:
            # cited_chunk_id == "default" — already canonical
            result.append(item)
    return result


def _get_gemini_model() -> genai.GenerativeModel:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


# ── Node 1: Data Aggregator ───────────────────────────────────────────────────

def node_aggregator(state: dict) -> dict:
    # If a profile was injected directly (custom analysis), skip persona lookup
    if state.get("profile") is not None:
        return {}
    persona_id = state["persona_id"]
    profile = PERSONAS.get(persona_id)
    if profile is None:
        raise ValueError(f"Unknown persona_id: {persona_id!r}. Valid: {list(PERSONAS)}")
    return {"profile": profile}


# ── Node 1.5a: Sector Context Retriever ──────────────────────────────────────

@traceable(name="sector-retriever", run_type="retriever")
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


@traceable(name="weight-setter", run_type="llm")
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
        retrieved_chunk_ids = {c["chunk_id"] for c in chunks}
        validated_rationale = _validate_rationale(rationale, retrieved_chunk_ids)
        if validated_rationale is None:
            return {"weights": _DEFAULT_WEIGHTS, "weight_rationale": _DEFAULT_RATIONALE}
        return {"weights": weights, "weight_rationale": validated_rationale}
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


# ── Node 3.5: Explainer Retriever ────────────────────────────────────────────

@traceable(name="explainer-retriever", run_type="retriever")
def node_explainer_retriever(state: dict) -> dict:
    profile = state["profile"]
    risk = state["risk_output"]
    retriever: Retriever = state.get("retriever") or Retriever()

    worst_score = sorted(risk["stress_results"], key=lambda s: s.delta)[:2]
    worst_cfcr = sorted(
        [item for item in risk["cfcr_by_scenario"] if item.scenario != "baseline"],
        key=lambda s: s.cfcr,
    )[:2]

    ntc = "new to credit" if profile.aa_bank_data.existing_loan_count == 0 else "existing credit history"
    buyer_flag = "buyer concentration flagged" if risk["buyer_concentration_flag"] else "buyer concentration not flagged"
    worst_score_text = ", ".join(f"{item.scenario}({item.delta:+.1f})" for item in worst_score) or "none"
    worst_cfcr_text = ", ".join(f"{item.scenario}({item.cfcr})" for item in worst_cfcr) or "none"

    query = (
        f"MSME {profile.sector} risk explainer guidance; "
        f"{ntc}; {buyer_flag}; "
        f"worst stress scenarios by score: {worst_score_text}; "
        f"worst CFCR scenarios: {worst_cfcr_text}; "
        f"material risk concepts for loan officer explanation"
    )

    chunks = retriever.query(query, n_results=5)
    return {"explainer_chunks": chunks}


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
- Every numeric claim must include a source_field that points to a key/path in Risk Engine output.
- Every citation claim must include the cited retrieved chunk_id as value.
- Do not invent chunk IDs.

Respond with STRICT JSON ONLY (no markdown, no extra text):
{{
    "narrative": "<3-5 short paragraphs>",
    "claims": [
        {{"source_field": "cfcr_baseline", "value": 1.2345, "text": "Baseline CFCR is 1.2345.", "type": "numeric"}},
        {{"source_field": "stress_results[0].delta", "value": -8.2, "text": "Receivable delay reduces score by 8.2 points.", "type": "numeric"}},
        {{"source_field": "__explainer_chunks__", "value": "chunk_abc123", "text": "Concentration risk is material.", "type": "citation"}}
    ]
}}
"""


@traceable(name="explainer", run_type="llm")
def node_explainer(state: dict) -> dict:
    profile = state["profile"]
    risk = state["risk_output"]
    chunks: list[dict] = state.get("explainer_chunks", [])

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
        narrative, claims = _parse_explainer_output(response.text)
        return {"narrative": narrative, "claims": claims}
    except Exception as exc:
        logger.warning("Explainer LLM call failed (%s)", exc)
        safe_narrative, safe_claims = _build_safe_summary_from_risk(_to_plain_data(risk))
        return {
            "narrative": safe_narrative,
            "claims": safe_claims,
        }


# ── Node 5: Grounding Validator ───────────────────────────────────────────────

@traceable(name="grounding-validator")
def node_grounding_validator(state: dict) -> dict:
    from backend.app.schemas.models import GroundingCheck

    risk_output: dict = _to_plain_data(state["risk_output"])
    retrieved_chunks: list[dict] = state.get("explainer_chunks", [])
    claims: list[dict[str, Any]] = state.get("claims", [])
    valid_chunk_ids = {c["chunk_id"] for c in retrieved_chunks}

    grounding_trace: list[GroundingCheck] = []
    if claims:
        initial_trace, validated_claims = _validate_claims(claims, risk_output, valid_chunk_ids)
        grounding_trace.extend(initial_trace)
    else:
        validated_claims = []
        grounding_trace.append(GroundingCheck(
            claim="Missing structured claims",
            type="numeric",
            source=_SAFE_SUMMARY_SOURCE,
            status="fail",
        ))

    rendered = _render_validated_claims(validated_claims, risk_output)
    if rendered is None:
        rendered_narrative, rendered_claims = _build_safe_summary_from_risk(risk_output)
    else:
        rendered_narrative, rendered_claims = rendered

    return {
        "narrative": rendered_narrative,
        "claims": rendered_claims,
        "grounding_trace": grounding_trace,
    }
