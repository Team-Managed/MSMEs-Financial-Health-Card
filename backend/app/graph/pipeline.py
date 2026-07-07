"""
LangGraph pipeline definition.
Graph: aggregator → sector_retriever → weight_setter → stress_generator
       → risk_engine → explainer → grounding_validator
"""
from __future__ import annotations
import operator
from typing import Annotated
from langgraph.graph import StateGraph, END
from langsmith import traceable
from backend.app.graph.nodes import (
    node_aggregator,
    node_sector_retriever,
    node_weight_setter,
    node_stress_generator,
    node_risk_engine,
    node_explainer,
    node_grounding_validator,
)
from backend.app.rag.retriever import Retriever
from backend.app.schemas.models import MSMEProfile


def _build_graph() -> StateGraph:
    graph = StateGraph(Annotated[dict, operator.or_])

    graph.add_node("aggregator", node_aggregator)
    graph.add_node("sector_retriever", node_sector_retriever)
    graph.add_node("weight_setter", node_weight_setter)
    graph.add_node("stress_generator", node_stress_generator)
    graph.add_node("risk_engine", node_risk_engine)
    graph.add_node("explainer", node_explainer)
    graph.add_node("grounding_validator", node_grounding_validator)

    graph.set_entry_point("aggregator")
    graph.add_edge("aggregator", "sector_retriever")
    graph.add_edge("sector_retriever", "weight_setter")
    graph.add_edge("weight_setter", "stress_generator")
    graph.add_edge("stress_generator", "risk_engine")
    graph.add_edge("risk_engine", "explainer")
    graph.add_edge("explainer", "grounding_validator")
    graph.add_edge("grounding_validator", END)

    return graph.compile()


_COMPILED_GRAPH = None


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


@traceable(name="msme-pipeline", project_name="msme-financial-health-card")
def run_pipeline(persona_id: str, retriever: Retriever | None = None) -> dict:
    if retriever is None:
        retriever = Retriever()

    initial_state = {
        "persona_id": persona_id,
        "retriever": retriever,
    }
    graph = _get_graph()
    final_state = graph.invoke(initial_state)
    result = dict(final_state)
    if "risk_output" in result:
        result = {**result["risk_output"], **result}
    return result


@traceable(name="msme-pipeline", project_name="msme-financial-health-card")
def run_pipeline_with_profile(profile: MSMEProfile, retriever: Retriever | None = None) -> dict:
    """Run the pipeline with a pre-built profile, bypassing node_aggregator's persona lookup."""
    if retriever is None:
        retriever = Retriever()

    initial_state = {
        "profile": profile,
        "retriever": retriever,
    }
    graph = _get_graph()
    final_state = graph.invoke(initial_state)
    result = dict(final_state)
    if "risk_output" in result:
        result = {**result["risk_output"], **result}
    return result
