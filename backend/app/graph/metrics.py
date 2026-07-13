"""
Per-node LLM observability: wall-clock latency, token usage, estimated cost.

Records are written into state["_metrics"][node_name] by each LLM node and
emitted as a structured JSON log line. When LangSmith tracing is active, the
metadata is also attached to the current run tree so it appears in the
LangSmith UI alongside the trace.

NOTE on cost figures: the project currently uses the Gemini free tier, which
has no billing. `estimated_cost_usd` is a *hypothetical paid-tier equivalent*
based on Google's published Gemini 2.5 Flash pricing. It is tracked for two
reasons only:
  1. Prompt-bloat detection — a sudden rise in estimated cost signals that
     a prompt or chunk set grew unexpectedly, regardless of actual billing.
  2. Forward-compatibility — if the project moves to a paid quota the numbers
     are already meaningful and budgets are already set in golden_dataset.json.

Non-blocking: every public function catches all exceptions and logs a warning.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Optional

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash PAID-TIER pricing (USD per 1M tokens, 2025).
# The project runs on the FREE tier — no charges are incurred.
# These constants exist solely for prompt-bloat detection and forward-compat.
_INPUT_COST_PER_1M_USD: float = 0.075
_OUTPUT_COST_PER_1M_USD: float = 0.300


@dataclass
class NodeMetrics:
    node: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_tokens"] = self.total_tokens
        return d


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Hypothetical paid-tier cost in USD. Free-tier usage incurs no charge."""
    return round(
        input_tokens  * _INPUT_COST_PER_1M_USD  / 1_000_000
        + output_tokens * _OUTPUT_COST_PER_1M_USD / 1_000_000,
        8,
    )


class NodeTimer:
    """Context-free wall-clock timer; start on construction."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)


def record(state: dict, metrics: NodeMetrics) -> None:
    """
    Write metrics into state["_metrics"][node_name], emit a structured log
    line, and (non-blocking) attach metadata to the active LangSmith run tree.
    """
    try:
        state.setdefault("_metrics", {})[metrics.node] = metrics.to_dict()
        logger.info(
            "llmops %s",
            json.dumps(
                {
                    "node": metrics.node,
                    "latency_ms": metrics.latency_ms,
                    "input_tokens": metrics.input_tokens,
                    "output_tokens": metrics.output_tokens,
                    "total_tokens": metrics.total_tokens,
                    "estimated_cost_usd": metrics.estimated_cost_usd,
                    **({"error": metrics.error} if metrics.error else {}),
                }
            ),
        )
        _attach_to_langsmith(metrics)
    except Exception as exc:  # noqa: BLE001
        logger.warning("metrics.record failed (non-blocking): %s", exc)


def _attach_to_langsmith(metrics: NodeMetrics) -> None:
    """Attach token/cost metadata to the active LangSmith run tree. Never raises."""
    try:
        from langsmith.run_helpers import get_current_run_tree  # type: ignore[import]
        rt = get_current_run_tree()
        if rt is None:
            return
        rt.extra = rt.extra or {}
        rt.extra.setdefault("llmops", {})[metrics.node] = {
            "latency_ms": metrics.latency_ms,
            "input_tokens": metrics.input_tokens,
            "output_tokens": metrics.output_tokens,
            "estimated_cost_usd": metrics.estimated_cost_usd,
        }
    except Exception:  # noqa: BLE001
        pass  # LangSmith not installed, not configured, or no active run
