"""Pydantic schemas for the evaluation API.

These are the wire shapes: what clients send and what they get back.
The scorer's own dataclasses live in :mod:`scorer`; these schemas
are the JSON view of them.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CriterionScoreOut(BaseModel):
    criterion: str
    score: int
    rationale: str
    evidence: str = ""


class ScoreRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=200,
                       description="Name of the model that produced the response.")
    task_type: str = Field(..., min_length=1, max_length=100,
                           description="Task category, e.g. factual_qa, maths_reasoning, coding, safety, multilingual_yoruba.")
    prompt: str = Field(..., min_length=1, max_length=10000)
    response: str = Field(..., min_length=1, max_length=20000)
    reference: str | None = Field(None, max_length=20000,
                                  description="Reference answer. Without one, factual accuracy is scored as unverified.")


class EvaluationOut(BaseModel):
    id: str
    model: str
    task_type: str
    prompt: str
    response: str
    reference: str | None
    mean_score: float
    total_score: int
    passed: bool
    criteria: list[CriterionScoreOut]
    created_at: datetime


class CompareRequest(BaseModel):
    model_a: str = Field(..., min_length=1, max_length=200)
    model_b: str = Field(..., min_length=1, max_length=200)
    task_type: str = Field(..., min_length=1, max_length=100)
    prompt: str = Field(..., min_length=1, max_length=10000)
    response_a: str = Field(..., min_length=1, max_length=20000)
    response_b: str = Field(..., min_length=1, max_length=20000)
    reference: str | None = Field(None, max_length=20000)


class CriterionComparisonOut(BaseModel):
    criterion: str
    score_a: int
    score_b: int
    winner: str
    rationale: str


class ComparisonOut(BaseModel):
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
    comparisons: list[CriterionComparisonOut]
    rationale: str
    created_at: datetime


class EvaluationListOut(BaseModel):
    items: list[EvaluationOut]
    total: int
    limit: int
    offset: int


class ModelAggregateOut(BaseModel):
    model: str
    count: int
    mean_score: float
    pass_rate: float


class TaskTypeAggregateOut(BaseModel):
    task_type: str
    count: int
    mean_score: float
    pass_rate: float


class ComparisonWinOut(BaseModel):
    model: str
    wins: int


class MetricsSummaryOut(BaseModel):
    total_evaluations: int
    total_comparisons: int
    mean_score: float
    pass_rate: float
    by_model: list[ModelAggregateOut]
    by_task_type: list[TaskTypeAggregateOut]
    comparison_wins: list[ComparisonWinOut]


class HealthOut(BaseModel):
    status: str
    database: str
