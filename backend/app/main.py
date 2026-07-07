from __future__ import annotations
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")  # always resolves to backend/.env

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app.data.personas import PERSONAS
from backend.app.data.mock_generators import generate_profile
from backend.app.graph.pipeline import run_pipeline, run_pipeline_with_profile
from backend.app.rag.retriever import Retriever
from backend.app.schemas.models import (
    AnalysisResponse, CustomAnalyzeRequest, WeightVector, WeightRationaleItem,
    CFCRResult, StressResult, GroundingCheck
)

logger = logging.getLogger(__name__)

app = FastAPI(title="MSME Financial Health Card API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_retriever = Retriever()   # singleton — loads Chroma on startup


@app.get("/health")
def health():
    import langsmith.utils as ls_utils
    return {
        "status": "ok",
        "tracing_enabled": ls_utils.tracing_is_enabled(),
        "LANGSMITH_TRACING": os.environ.get("LANGSMITH_TRACING"),
        "LANGCHAIN_TRACING_V2": os.environ.get("LANGCHAIN_TRACING_V2"),
        "LANGSMITH_API_KEY_set": bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")),
    }


@app.get("/api/personas")
def get_personas():
    return [
        {"id": pid, "business_name": p.business_name, "sector": p.sector}
        for pid, p in PERSONAS.items()
    ]


def _build_response(state: dict) -> AnalysisResponse:
    risk = state["risk_output"]
    profile = state["profile"]
    return AnalysisResponse(
        profile_summary={
            "msme_id": profile.msme_id,
            "business_name": profile.business_name,
            "sector": profile.sector,
            "years_operating": profile.years_operating,
        },
        cfcr_baseline=risk["cfcr_baseline"],
        cfcr_by_scenario=risk["cfcr_by_scenario"],
        weights_used=state.get("weights", WeightVector(gst=0.30, upi=0.30, aa=0.25, epfo=0.15)),
        weight_rationale=state.get("weight_rationale", []),
        baseline_score=risk["baseline_score"],
        stress_results=risk["stress_results"],
        narrative=state.get("narrative", ""),
        grounding_trace=state.get("grounding_trace", []),
    )


@app.post("/api/msme/{persona_id}/analyze", response_model=AnalysisResponse)
def analyze(persona_id: str):
    if persona_id not in PERSONAS:
        raise HTTPException(status_code=404, detail=f"Persona '{persona_id}' not found")
    try:
        state = run_pipeline(persona_id, retriever=_retriever)
    except Exception as exc:
        logger.exception("Pipeline failed for persona %s", persona_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _build_response(state)


@app.post("/api/analyze", response_model=AnalysisResponse)
def analyze_custom(req: CustomAnalyzeRequest):
    """Run the full pipeline for a user-provided profile instead of a hardcoded persona."""
    import random
    profile = generate_profile(
        seed=random.randint(1, 999_999),
        sector=req.sector,
        profile_type=req.profile_type,
        years_operating=req.years_operating,
        msme_tier=req.msme_tier,
        employee_tier=req.employee_tier,
    )
    try:
        state = run_pipeline_with_profile(profile, retriever=_retriever)
    except Exception as exc:
        logger.exception("Pipeline failed for custom profile (sector=%s type=%s)", req.sector, req.profile_type)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _build_response(state)
