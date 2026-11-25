"""Unit tests for criteria ordering on Postgres reads.

Regression test for CI run #1 of docker-eval-service: the Postgres
repository read scores back with ``ORDER BY criterion``
(alphabetical), so ``helpfulness`` and ``reasoning_quality`` swapped
places against the rubric's order and the round-trip test failed.
Reads must restore the rubric's order — these tests run without a
database, so the ordering logic is covered locally as well as in CI.
"""

from docker_eval_service.repository import _criteria_from_rows
from docker_eval_service.scorer import CRITERIA


def _row(criterion: str) -> dict:
    return {
        "criterion": criterion,
        "score": 4,
        "rationale": f"{criterion} rationale",
        "evidence": "",
    }


def test_criteria_from_rows_restores_rubric_order():
    # Rows in alphabetical order, as Postgres returns them with
    # ORDER BY criterion.
    rows = [_row(c) for c in sorted(CRITERIA)]
    assert [c.criterion for c in _criteria_from_rows(rows)] == list(CRITERIA)


def test_criteria_from_rows_keeps_rubric_order_when_already_ordered():
    rows = [_row(c) for c in CRITERIA]
    assert [c.criterion for c in _criteria_from_rows(rows)] == list(CRITERIA)


def test_criteria_from_rows_sorts_unknown_criterion_last():
    rows = [_row(c) for c in CRITERIA] + [_row("zzz_unknown")]
    result = _criteria_from_rows(rows)
    assert result[-1].criterion == "zzz_unknown"
    assert [c.criterion for c in result[:-1]] == list(CRITERIA)
