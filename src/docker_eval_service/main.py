"""docker-eval-service — a Dockerised LLM evaluation API.

FastAPI service that scores model responses on a five-criterion
rubric, compares pairs of responses RLHF-style, and persists
everything in PostgreSQL.

Scoring is deterministic heuristics (see :mod:`scorer`), not a
judge LLM — the README states this plainly.

Startup is strict: without a reachable PostgreSQL (``DATABASE_URL``)
the app refuses to start. There is no fallback store.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse

from .db import get_database_url, init_database
from .models import (
    CompareRequest,
    ComparisonOut,
    CriterionComparisonOut,
    CriterionScoreOut,
    EvaluationListOut,
    EvaluationOut,
    HealthOut,
    MetricsSummaryOut,
    ScoreRequest,
)
from .repository import ComparisonRecord, EvaluationRecord, PostgresRepository, Repository
from .scorer import compare_results, compare_responses, score_response

MAX_LIMIT = 100
DEFAULT_LIMIT = 20


def _evaluation_out(record: EvaluationRecord) -> EvaluationOut:
    return EvaluationOut(
        id=record.id,
        model=record.model,
        task_type=record.task_type,
        prompt=record.prompt,
        response=record.response,
        reference=record.reference,
        mean_score=record.mean_score,
        total_score=record.total_score,
        passed=record.passed,
        criteria=[CriterionScoreOut(**c.to_dict()) for c in record.criteria],
        created_at=record.created_at,
    )


def _comparison_out(record: ComparisonRecord) -> ComparisonOut:
    return ComparisonOut(
        id=record.id,
        evaluation_a_id=record.evaluation_a_id,
        evaluation_b_id=record.evaluation_b_id,
        model_a=record.model_a,
        model_b=record.model_b,
        task_type=record.task_type,
        prompt=record.prompt,
        winner=record.winner,
        mean_a=record.mean_a,
        mean_b=record.mean_b,
        comparisons=[CriterionComparisonOut(**c.to_dict()) for c in record.comparisons],
        rationale=record.rationale,
        created_at=record.created_at,
    )


def create_app(repository: Repository | None = None) -> FastAPI:
    """Build the app.

    ``repository`` is for tests only: when it is given, no database
    connection is attempted at startup. In production it is always
    ``None`` and the app connects to PostgreSQL from ``DATABASE_URL``,
    failing loudly if it cannot.
    """
    state: dict[str, Repository | None] = {"repository": repository}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["repository"] is None:
            database_url = get_database_url()
            init_database(database_url)
            state["repository"] = PostgresRepository(database_url)
        yield

    app = FastAPI(
        title="docker-eval-service",
        description=(
            "Dockerised LLM evaluation API: deterministic rubric scoring "
            "(1–5, five criteria), pairwise RLHF comparison, PostgreSQL "
            "persistence and aggregate metrics. Scoring is deterministic "
            "heuristics, not a judge LLM."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    def repo() -> Repository:
        assert state["repository"] is not None
        return state["repository"]

    @app.get("/health", response_model=HealthOut, tags=["operations"])
    def health() -> HealthOut | JSONResponse:
        if repo().ping():
            return HealthOut(status="ok", database="ok")
        # 503, not 200: the service is not usable without its
        # database, and the container healthcheck must fail.
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "database": "unreachable"},
        )

    @app.post("/evaluations/score", response_model=EvaluationOut,
              status_code=status.HTTP_201_CREATED, tags=["evaluations"])
    def score_evaluation(payload: ScoreRequest) -> EvaluationOut:
        result = score_response(payload.prompt, payload.response, payload.reference)
        record = repo().create_evaluation(
            model=payload.model, task_type=payload.task_type,
            prompt=payload.prompt, response=payload.response,
            reference=payload.reference, result=result,
        )
        return _evaluation_out(record)

    @app.post("/evaluations/compare", response_model=ComparisonOut,
              status_code=status.HTTP_201_CREATED, tags=["evaluations"])
    def compare_evaluations(payload: CompareRequest) -> ComparisonOut:
        result_a = score_response(payload.prompt, payload.response_a, payload.reference)
        result_b = score_response(payload.prompt, payload.response_b, payload.reference)
        result = compare_results(
            result_a, result_b, payload.prompt,
            payload.response_a, payload.response_b,
        )
        record_a = repo().create_evaluation(
            model=payload.model_a, task_type=payload.task_type,
            prompt=payload.prompt, response=payload.response_a,
            reference=payload.reference, result=result_a,
        )
        record_b = repo().create_evaluation(
            model=payload.model_b, task_type=payload.task_type,
            prompt=payload.prompt, response=payload.response_b,
            reference=payload.reference, result=result_b,
        )
        record = repo().create_comparison(
            evaluation_a_id=record_a.id, evaluation_b_id=record_b.id,
            model_a=payload.model_a, model_b=payload.model_b,
            task_type=payload.task_type, prompt=payload.prompt, result=result,
        )
        return _comparison_out(record)

    @app.get("/evaluations/{evaluation_id}", response_model=EvaluationOut,
             tags=["evaluations"])
    def get_evaluation(evaluation_id: str) -> EvaluationOut:
        record = repo().get_evaluation(evaluation_id)
        if record is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return _evaluation_out(record)

    @app.get("/evaluations", response_model=EvaluationListOut, tags=["evaluations"])
    def list_evaluations(
        model: str | None = Query(None, description="Filter by model name."),
        task_type: str | None = Query(None, description="Filter by task type."),
        limit: int = Query(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
        offset: int = Query(0, ge=0),
    ) -> EvaluationListOut:
        items, total = repo().list_evaluations(
            model=model, task_type=task_type, limit=limit, offset=offset
        )
        return EvaluationListOut(
            items=[_evaluation_out(item) for item in items],
            total=total, limit=limit, offset=offset,
        )

    @app.get("/metrics/summary", response_model=MetricsSummaryOut, tags=["metrics"])
    def metrics_summary() -> MetricsSummaryOut:
        summary = repo().metrics_summary()
        return MetricsSummaryOut(**summary.to_dict())

    return app


# The production app: repository is built from DATABASE_URL at startup.
app = create_app()

if __name__ == "__main__":  # pragma: no cover - run via uvicorn
    import uvicorn

    uvicorn.run(
        "docker_eval_service.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
