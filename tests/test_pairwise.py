"""Unit tests for pairwise comparison — no database needed."""

from docker_eval_service.scorer import TIE_THRESHOLD, compare_responses

PROMPT = "What is the capital of France?"
REFERENCE = "The capital of France is Paris. Paris has been the capital since the 10th century."
GOOD = "The capital of France is Paris. It has been the capital since the 10th century."
BAD = "The capital of France is Lyon. Lyon is a large city in eastern France."


def test_compare_clear_winner_a():
    result = compare_responses(PROMPT, GOOD, BAD, REFERENCE)
    assert result.winner == "A"
    assert result.mean_a > result.mean_b
    assert result.mean_a - result.mean_b >= TIE_THRESHOLD
    assert "A" in result.rationale


def test_compare_clear_winner_b():
    result = compare_responses(PROMPT, BAD, GOOD, REFERENCE)
    assert result.winner == "B"
    assert result.mean_b > result.mean_a
    assert "B" in result.rationale


def test_compare_identical_responses_is_a_tie():
    result = compare_responses(PROMPT, GOOD, GOOD, REFERENCE)
    assert result.winner == "tie"
    assert result.mean_a == result.mean_b
    assert all(c.winner == "tie" for c in result.comparisons)


def test_compare_small_gap_is_a_tie():
    # Two near-identical good answers differ by less than the tie
    # threshold, so neither is a reliable preference.
    almost_good = "The capital of France is Paris, and it has been the capital since the 10th century."
    result = compare_responses(PROMPT, GOOD, almost_good, REFERENCE)
    assert abs(result.mean_a - result.mean_b) < TIE_THRESHOLD
    assert result.winner == "tie"


def test_compare_returns_per_criterion_breakdown():
    result = compare_responses(PROMPT, GOOD, BAD, REFERENCE)
    assert len(result.comparisons) == 5
    for comparison in result.comparisons:
        assert comparison.winner in ("A", "B", "tie")
        assert comparison.rationale
    factual = next(c for c in result.comparisons if c.criterion == "factual_accuracy")
    assert factual.winner == "A"
    assert factual.score_a > factual.score_b


def test_compare_result_is_deterministic():
    first = compare_responses(PROMPT, GOOD, BAD, REFERENCE)
    second = compare_responses(PROMPT, GOOD, BAD, REFERENCE)
    assert first.to_dict() == second.to_dict()


def test_compare_without_reference_is_a_tie():
    # Without a reference the scorer cannot tell Lyon from Paris:
    # factual accuracy is unverified (3) for both, and the other
    # criteria tie too. The comparison honestly reports a tie
    # instead of inventing a preference — this is why the API
    # asks callers to supply a reference whenever they have one.
    result = compare_responses(PROMPT, GOOD, BAD)
    assert result.winner == "tie"
    assert result.mean_a == result.mean_b
