"""Integration tests against live PostgreSQL.

These run only when DATABASE_URL points at a reachable Postgres
(CI provides one via the service container). Without it they skip
loudly — the skip message says exactly why.
"""

import pytest

from docker_eval_service.scorer import compare_responses, score_response

PROMPT = "What is the capital of France?"
REFERENCE = "The capital of France is Paris. Paris has been the capital since the 10th century."
GOOD = "The capital of France is Paris. It has been the capital since the 10th century."
BAD = "The capital of France is Lyon. Lyon is a large city in eastern France."


def test_postgres_ping(pg_repo):
    assert pg_repo.ping() is True


def test_create_and_get_evaluation_round_trip(pg_repo):
    result = score_response(PROMPT, GOOD, REFERENCE)
    record = pg_repo.create_evaluation(
        model="model-a", task_type="factual_qa", prompt=PROMPT,
        response=GOOD, reference=REFERENCE, result=result,
    )
    assert record.id
    assert record.mean_score == result.mean
    assert record.passed == result.passed
    assert len(record.criteria) == 5

    fetched = pg_repo.get_evaluation(record.id)
    assert fetched is not None
    assert fetched.to_dict() == record.to_dict()


def test_get_unknown_evaluation_returns_none(pg_repo):
    assert pg_repo.get_evaluation("00000000-0000-0000-0000-000000000000") is None


def test_list_filters_and_pagination(pg_repo):
    for model, task_type in [("model-a", "factual_qa"), ("model-b", "factual_qa"),
                             ("model-a", "maths_reasoning")]:
        pg_repo.create_evaluation(
            model=model, task_type=task_type, prompt=PROMPT, response=GOOD,
            reference=REFERENCE, result=score_response(PROMPT, GOOD, REFERENCE),
        )
    items, total = pg_repo.list_evaluations()
    assert total == 3 and len(items) == 3
    _, total_a = pg_repo.list_evaluations(model="model-a")
    assert total_a == 2
    _, total_maths = pg_repo.list_evaluations(task_type="maths_reasoning")
    assert total_maths == 1
    page, _ = pg_repo.list_evaluations(limit=2, offset=2)
    assert len(page) == 1


def test_create_and_get_comparison(pg_repo):
    result_a = score_response(PROMPT, GOOD, REFERENCE)
    result_b = score_response(PROMPT, BAD, REFERENCE)
    eval_a = pg_repo.create_evaluation(
        model="model-a", task_type="factual_qa", prompt=PROMPT,
        response=GOOD, reference=REFERENCE, result=result_a,
    )
    eval_b = pg_repo.create_evaluation(
        model="model-b", task_type="factual_qa", prompt=PROMPT,
        response=BAD, reference=REFERENCE, result=result_b,
    )
    pairwise = compare_responses(PROMPT, GOOD, BAD, REFERENCE)
    comparison = pg_repo.create_comparison(
        evaluation_a_id=eval_a.id, evaluation_b_id=eval_b.id,
        model_a="model-a", model_b="model-b", task_type="factual_qa",
        prompt=PROMPT, result=pairwise,
    )
    assert comparison.winner == "A"
    fetched = pg_repo.get_comparison(comparison.id)
    assert fetched is not None
    assert fetched.to_dict() == comparison.to_dict()
    assert pg_repo.get_comparison("00000000-0000-0000-0000-000000000000") is None


def test_metrics_summary_aggregates(pg_repo):
    pg_repo.create_evaluation(
        model="model-a", task_type="factual_qa", prompt=PROMPT, response=GOOD,
        reference=REFERENCE, result=score_response(PROMPT, GOOD, REFERENCE),
    )
    pg_repo.create_evaluation(
        model="model-b", task_type="factual_qa", prompt=PROMPT, response=BAD,
        reference=REFERENCE, result=score_response(PROMPT, BAD, REFERENCE),
    )
    summary = pg_repo.metrics_summary()
    assert summary.total_evaluations == 2
    assert summary.total_comparisons == 0
    models = {row.model: row for row in summary.by_model}
    assert models["model-a"].count == 1
    assert models["model-b"].count == 1
    assert models["model-a"].mean_score > models["model-b"].mean_score
    assert summary.comparison_wins == []


def test_metrics_summary_empty_database(pg_repo):
    summary = pg_repo.metrics_summary()
    assert summary.total_evaluations == 0
    assert summary.mean_score == 0.0
    assert summary.pass_rate == 0.0
