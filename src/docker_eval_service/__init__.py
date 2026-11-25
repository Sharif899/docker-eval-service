"""docker-eval-service: a Dockerised LLM evaluation API.

FastAPI + PostgreSQL. Deterministic rubric scoring (1–5, five
criteria), pairwise RLHF comparison, persistent storage and
aggregate metrics. Scoring is deterministic heuristics, not a
judge LLM.
"""

from .scorer import (
    CRITERIA,
    PASS_THRESHOLD,
    TIE_THRESHOLD,
    compare_responses,
    compare_results,
    score_response,
)

__all__ = [
    "CRITERIA",
    "PASS_THRESHOLD",
    "TIE_THRESHOLD",
    "compare_responses",
    "compare_results",
    "score_response",
]

__version__ = "0.1.0"
