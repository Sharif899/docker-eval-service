"""Persistence: repository protocol and the PostgreSQL implementation.

The product code talks to Postgres only through the :class:`Repository`
protocol. The API tests use an in-memory implementation of the same
protocol that lives in ``tests/`` — it is test-only and is never a
product fallback: if Postgres is unreachable the service fails
loudly at startup instead of silently storing data somewhere else.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Protocol

from .scorer import (
    CRITERIA,
    CriterionComparison,
    CriterionScore,
    PairwiseResult,
    RubricResult,
)

# The rubric's criterion order (factual_accuracy, reasoning_quality,
# instruction_adherence, helpfulness, safety). Scores are stored one
# row per criterion with no position column, so reads must restore
# this order explicitly — see _criteria_from_rows.
_CRITERION_ORDER = {name: i for i, name in enumerate(CRITERIA)}

try:
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - psycopg absent in minimal envs
    dict_row = None  # type: ignore


@dataclass
class EvaluationRecord:
    id: str
    model: str
    task_type: str
    prompt: str
    response: str
    reference: str | None
    mean_score: float
    total_score: int
    passed: bool
    criteria: list[CriterionScore] = field(default_factory=list)
    created_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model": self.model,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "response": self.response,
            "reference": self.reference,
            "mean_score": self.mean_score,
            "total_score": self.total_score,
            "passed": self.passed,
            "criteria": [c.to_dict() for c in self.criteria],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ComparisonRecord:
    id: str
    evaluation_a_id: str
    evaluation_b_id: str
    model_a: str
    model_b: str
    task_type: str
    prompt: str
    winner: str
    mean_a: float
    mean_b: float
    comparisons: list[CriterionComparison] = field(default_factory=list)
    rationale: str = ""
    created_at: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "evaluation_a_id": self.evaluation_a_id,
            "evaluation_b_id": self.evaluation_b_id,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "winner": self.winner,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "rationale": self.rationale,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ModelAggregate:
    model: str
    count: int
    mean_score: float
    pass_rate: float


@dataclass
class TaskTypeAggregate:
    task_type: str
    count: int
    mean_score: float
    pass_rate: float


@dataclass
class ComparisonWin:
    model: str
    wins: int


@dataclass
class MetricsSummary:
    total_evaluations: int
    total_comparisons: int
    mean_score: float
    pass_rate: float
    by_model: list[ModelAggregate] = field(default_factory=list)
    by_task_type: list[TaskTypeAggregate] = field(default_factory=list)
    comparison_wins: list[ComparisonWin] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_evaluations": self.total_evaluations,
            "total_comparisons": self.total_comparisons,
            "mean_score": self.mean_score,
            "pass_rate": self.pass_rate,
            "by_model": [vars(a) for a in self.by_model],
            "by_task_type": [vars(a) for a in self.by_task_type],
            "comparison_wins": [vars(w) for w in self.comparison_wins],
        }


class Repository(Protocol):
    """What the service needs from storage, nothing more."""

    def ping(self) -> bool:
        """True when the database answers, False otherwise."""
        ...

    def create_evaluation(
        self, model: str, task_type: str, prompt: str, response: str,
        reference: str | None, result: RubricResult,
    ) -> EvaluationRecord:
        ...

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
        ...

    def list_evaluations(
        self, model: str | None = None, task_type: str | None = None,
        limit: int = 20, offset: int = 0,
    ) -> tuple[list[EvaluationRecord], int]:
        """Returns (page of records, total matching count)."""
        ...

    def create_comparison(
        self, evaluation_a_id: str, evaluation_b_id: str,
        model_a: str, model_b: str, task_type: str, prompt: str,
        result: PairwiseResult,
    ) -> ComparisonRecord:
        ...

    def get_comparison(self, comparison_id: str) -> ComparisonRecord | None:
        ...

    def metrics_summary(self) -> MetricsSummary:
        ...


def _criteria_from_rows(rows: list[dict]) -> list[CriterionScore]:
    """Criteria in rubric order, not database order.

    PostgreSQL returns rows in no guaranteed order, and ``ORDER BY
    criterion`` is alphabetical, which is *not* the rubric's order:
    alphabetically ``helpfulness`` comes second and
    ``reasoning_quality`` fourth, the reverse of the rubric. A fetched
    evaluation must round-trip exactly (same criteria, same order),
    so sort by the rubric here. A criterion the rubric does not know
    sorts last, alphabetically, rather than breaking the read.
    """
    ordered = sorted(
        rows,
        key=lambda row: (
            _CRITERION_ORDER.get(row["criterion"], len(_CRITERION_ORDER)),
            row["criterion"],
        ),
    )
    return [
        CriterionScore(
            criterion=row["criterion"],
            score=row["score"],
            rationale=row["rationale"],
            evidence=row["evidence"] or "",
        )
        for row in ordered
    ]


def _comparisons_from_json(data: list[dict]) -> list[CriterionComparison]:
    return [
        CriterionComparison(
            criterion=item["criterion"],
            score_a=item["score_a"],
            score_b=item["score_b"],
            winner=item["winner"],
            rationale=item["rationale"],
        )
        for item in data
    ]


class PostgresRepository:
    """PostgreSQL implementation of :class:`Repository` (psycopg v3)."""

    def __init__(self, conninfo: str) -> None:
        import psycopg

        self._conninfo = conninfo
        self._psycopg = psycopg

    def _connect(self):
        return self._psycopg.connect(self._conninfo, autocommit=True)

    def ping(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone() is not None
        except Exception:
            return False

    def create_evaluation(
        self, model: str, task_type: str, prompt: str, response: str,
        reference: str | None, result: RubricResult,
    ) -> EvaluationRecord:
        evaluation_id = str(uuid.uuid4())
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO evaluations
                    (id, model, task_type, prompt, response, reference,
                     mean_score, total_score, passed)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING created_at
                """,
                (evaluation_id, model, task_type, prompt, response, reference,
                 result.mean, result.total, result.passed),
            )
            created_at = cur.fetchone()[0]
            for criterion in result.criteria:
                cur.execute(
                    """
                    INSERT INTO scores
                        (id, evaluation_id, criterion, score, rationale, evidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), evaluation_id, criterion.criterion,
                     criterion.score, criterion.rationale, criterion.evidence),
                )
        return EvaluationRecord(
            id=evaluation_id, model=model, task_type=task_type, prompt=prompt,
            response=response, reference=reference, mean_score=result.mean,
            total_score=result.total, passed=result.passed,
            criteria=list(result.criteria), created_at=created_at,
        )

    def get_evaluation(self, evaluation_id: str) -> EvaluationRecord | None:
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM evaluations WHERE id = %s", (evaluation_id,))
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                "SELECT criterion, score, rationale, evidence FROM scores "
                "WHERE evaluation_id = %s ORDER BY criterion",
                (evaluation_id,),
            )
            criteria = _criteria_from_rows(cur.fetchall())
        return _evaluation_from_row(row, criteria)

    def list_evaluations(
        self, model: str | None = None, task_type: str | None = None,
        limit: int = 20, offset: int = 0,
    ) -> tuple[list[EvaluationRecord], int]:
        conditions, params = [], []
        if model is not None:
            conditions.append("model = %s")
            params.append(model)
        if task_type is not None:
            conditions.append("task_type = %s")
            params.append(task_type)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(f"SELECT COUNT(*) AS n FROM evaluations {where}", params)
            total = cur.fetchone()["n"]
            cur.execute(
                f"SELECT * FROM evaluations {where} "
                "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                params + [limit, offset],
            )
            rows = cur.fetchall()
            records = []
            for row in rows:
                cur.execute(
                    "SELECT criterion, score, rationale, evidence FROM scores "
                    "WHERE evaluation_id = %s ORDER BY criterion",
                    (row["id"],),
                )
                records.append(_evaluation_from_row(row, _criteria_from_rows(cur.fetchall())))
        return records, total

    def create_comparison(
        self, evaluation_a_id: str, evaluation_b_id: str,
        model_a: str, model_b: str, task_type: str, prompt: str,
        result: PairwiseResult,
    ) -> ComparisonRecord:
        comparison_id = str(uuid.uuid4())
        criteria_json = json.dumps([c.to_dict() for c in result.comparisons])
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO comparisons
                    (id, evaluation_a_id, evaluation_b_id, model_a, model_b,
                     task_type, prompt, winner, mean_a, mean_b, criteria, rationale)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING created_at
                """,
                (comparison_id, evaluation_a_id, evaluation_b_id, model_a, model_b,
                 task_type, prompt, result.winner, result.mean_a, result.mean_b,
                 criteria_json, result.rationale),
            )
            created_at = cur.fetchone()[0]
        return ComparisonRecord(
            id=comparison_id, evaluation_a_id=evaluation_a_id,
            evaluation_b_id=evaluation_b_id, model_a=model_a, model_b=model_b,
            task_type=task_type, prompt=prompt, winner=result.winner,
            mean_a=result.mean_a, mean_b=result.mean_b,
            comparisons=list(result.comparisons), rationale=result.rationale,
            created_at=created_at,
        )

    def get_comparison(self, comparison_id: str) -> ComparisonRecord | None:
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM comparisons WHERE id = %s", (comparison_id,))
            row = cur.fetchone()
            if row is None:
                return None
        return _comparison_from_row(row)

    def metrics_summary(self) -> MetricsSummary:
        with self._connect() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT COUNT(*) AS n, COALESCE(AVG(mean_score), 0) AS mean, "
                "COALESCE(AVG(CASE WHEN passed THEN 1 ELSE 0 END), 0) AS pass_rate "
                "FROM evaluations"
            )
            overall = cur.fetchone()
            cur.execute(
                "SELECT model, COUNT(*) AS n, AVG(mean_score) AS mean, "
                "AVG(CASE WHEN passed THEN 1 ELSE 0 END) AS pass_rate "
                "FROM evaluations GROUP BY model ORDER BY model"
            )
            by_model = [
                ModelAggregate(
                    model=r["model"], count=r["n"],
                    mean_score=round(float(r["mean"]), 2),
                    pass_rate=round(float(r["pass_rate"]), 3),
                )
                for r in cur.fetchall()
            ]
            cur.execute(
                "SELECT task_type, COUNT(*) AS n, AVG(mean_score) AS mean, "
                "AVG(CASE WHEN passed THEN 1 ELSE 0 END) AS pass_rate "
                "FROM evaluations GROUP BY task_type ORDER BY task_type"
            )
            by_task_type = [
                TaskTypeAggregate(
                    task_type=r["task_type"], count=r["n"],
                    mean_score=round(float(r["mean"]), 2),
                    pass_rate=round(float(r["pass_rate"]), 3),
                )
                for r in cur.fetchall()
            ]
            cur.execute("SELECT COUNT(*) AS n FROM comparisons")
            total_comparisons = cur.fetchone()["n"]
            cur.execute(
                "SELECT winner_model, COUNT(*) AS n FROM ("
                "  SELECT CASE WHEN winner = 'A' THEN model_a "
                "              WHEN winner = 'B' THEN model_b END AS winner_model "
                "  FROM comparisons WHERE winner IN ('A', 'B')"
                ") w GROUP BY winner_model ORDER BY winner_model"
            )
            wins = [
                ComparisonWin(model=r["winner_model"], wins=r["n"])
                for r in cur.fetchall()
            ]
        return MetricsSummary(
            total_evaluations=overall["n"],
            total_comparisons=total_comparisons,
            mean_score=round(float(overall["mean"]), 2),
            pass_rate=round(float(overall["pass_rate"]), 3),
            by_model=by_model,
            by_task_type=by_task_type,
            comparison_wins=wins,
        )


def _evaluation_from_row(row: dict, criteria: list[CriterionScore]) -> EvaluationRecord:
    return EvaluationRecord(
        id=str(row["id"]),
        model=row["model"],
        task_type=row["task_type"],
        prompt=row["prompt"],
        response=row["response"],
        reference=row["reference"],
        mean_score=float(row["mean_score"]),
        total_score=row["total_score"],
        passed=bool(row["passed"]),
        criteria=criteria,
        created_at=row["created_at"],
    )


def _comparison_from_row(row: dict) -> ComparisonRecord:
    criteria = row["criteria"]
    if isinstance(criteria, str):
        criteria = json.loads(criteria)
    return ComparisonRecord(
        id=str(row["id"]),
        evaluation_a_id=str(row["evaluation_a_id"]),
        evaluation_b_id=str(row["evaluation_b_id"]),
        model_a=row["model_a"],
        model_b=row["model_b"],
        task_type=row["task_type"],
        prompt=row["prompt"],
        winner=row["winner"],
        mean_a=float(row["mean_a"]),
        mean_b=float(row["mean_b"]),
        comparisons=_comparisons_from_json(criteria),
        rationale=row["rationale"],
        created_at=row["created_at"],
    )
