"""Shared test fixtures.

The in-memory repository below implements the same
:class:`docker_eval_service.repository.Repository` protocol as the
PostgreSQL implementation, so API tests run without a database.
It lives in ``tests/`` on purpose: it is test-only and is never a
product fallback — the real service fails loudly at startup when
Postgres is unreachable.
"""

from __future__ import annotations

import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# Tests import the seed script as a module (``scripts.seed``).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from docker_eval_service.main import create_app
from docker_eval_service.repository import (
    ComparisonRecord,
    ComparisonWin,
    EvaluationRecord,
    MetricsSummary,
    ModelAggregate,
    TaskTypeAggregate,
)
from docker_eval_service.scorer import PairwiseResult, RubricResult


class InMemoryRepository:
    """Test-only repository: same interface, no database."""

    def __init__(self) -> None:
        self.evaluations: dict[str, EvaluationRecord] = {}
        self.comparisons: dict[str, ComparisonRecord] = {}
        self._order: list[str] = []
        self._ping_ok = True

    def set_ping(self, ok: bool) -> None:
        self._ping_ok = ok

    def ping(self) -> bool:
        return self._ping_ok

    def create_evaluation(
        self, model: str, task_type: str, prompt: str, response: str,
        reference: str | None, result: RubricResult,
    ) -> EvaluationRecord:
        record = EvaluationRecord(
            id=str(uuid.uuid4()), model=model, task_type=task_type,
            prompt=prompt, response=response, reference=reference,
            mean_score=result.mean, total_score=result.total,
            passed=result.passed, criteria=list(result.criteria),
            created_at=datetime.now(timezone.utc),
        )
        self.evaluations[record.id] = record
        self._order.append(record.id)
        return record

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
        return self.evaluations.get(evaluation_id)

    def list_evaluations(
        self, model: str | None = None, task_type: str | None = None,
        limit: int = 20, offset: int = 0,
    ) -> tuple[list[EvaluationRecord], int]:
        records = [self.evaluations[eid] for eid in reversed(self._order)]
        if model is not None:
            records = [r for r in records if r.model == model]
        if task_type is not None:
            records = [r for r in records if r.task_type == task_type]
        return records[offset:offset + limit], len(records)

    def create_comparison(
        self, evaluation_a_id: str, evaluation_b_id: str,
        model_a: str, model_b: str, task_type: str, prompt: str,
        result: PairwiseResult,
    ) -> ComparisonRecord:
        record = ComparisonRecord(
            id=str(uuid.uuid4()), evaluation_a_id=evaluation_a_id,
            evaluation_b_id=evaluation_b_id, model_a=model_a, model_b=model_b,
            task_type=task_type, prompt=prompt, winner=result.winner,
            mean_a=result.mean_a, mean_b=result.mean_b,
            comparisons=list(result.comparisons), rationale=result.rationale,
            created_at=datetime.now(timezone.utc),
        )
        self.comparisons[record.id] = record
        return record

    def get_comparison(self, comparison_id: str) -> ComparisonRecord | None:
        return self.comparisons.get(comparison_id)

    def metrics_summary(self) -> MetricsSummary:
        records = list(self.evaluations.values())
        total = len(records)
        mean = round(sum(r.mean_score for r in records) / total, 2) if total else 0.0
        passed = sum(1 for r in records if r.passed)
        pass_rate = round(passed / total, 3) if total else 0.0

        by_model: list[ModelAggregate] = []
        for model in sorted({r.model for r in records}):
            group = [r for r in records if r.model == model]
            by_model.append(ModelAggregate(
                model=model, count=len(group),
                mean_score=round(sum(r.mean_score for r in group) / len(group), 2),
                pass_rate=round(sum(1 for r in group if r.passed) / len(group), 3),
            ))
        by_task_type: list[TaskTypeAggregate] = []
        for task_type in sorted({r.task_type for r in records}):
            group = [r for r in records if r.task_type == task_type]
            by_task_type.append(TaskTypeAggregate(
                task_type=task_type, count=len(group),
                mean_score=round(sum(r.mean_score for r in group) / len(group), 2),
                pass_rate=round(sum(1 for r in group if r.passed) / len(group), 3),
            ))
        wins = Counter()
        for comparison in self.comparisons.values():
            if comparison.winner == "A":
                wins[comparison.model_a] += 1
            elif comparison.winner == "B":
                wins[comparison.model_b] += 1
        comparison_wins = [
            ComparisonWin(model=model, wins=count)
            for model, count in sorted(wins.items())
        ]
        return MetricsSummary(
            total_evaluations=total,
            total_comparisons=len(self.comparisons),
            mean_score=mean, pass_rate=pass_rate,
            by_model=by_model, by_task_type=by_task_type,
            comparison_wins=comparison_wins,
        )


@pytest.fixture()
def repo() -> InMemoryRepository:
    return InMemoryRepository()


@pytest.fixture()
def client(repo):
    from fastapi.testclient import TestClient

    with TestClient(create_app(repository=repo)) as test_client:
        yield test_client


@pytest.fixture()
def postgres_url() -> str:
    """Live Postgres URL, or skip the test when there is none.

    Integration tests need a real database; they skip loudly (not
    silently) when DATABASE_URL is unset or unreachable.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL is not set — integration tests need live Postgres")
    try:
        import psycopg

        with psycopg.connect(url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:
        pytest.skip(f"PostgreSQL at DATABASE_URL is unreachable: {exc}")
    return url


@pytest.fixture()
def pg_repo(postgres_url):
    """A Postgres-backed repository with migrations applied and
    tables emptied before each test."""
    from docker_eval_service.db import init_database
    from docker_eval_service.repository import PostgresRepository

    init_database(postgres_url)
    import psycopg

    with psycopg.connect(postgres_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE comparisons, scores, evaluations RESTART IDENTITY CASCADE")
    return PostgresRepository(postgres_url)
