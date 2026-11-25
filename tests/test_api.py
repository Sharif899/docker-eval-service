"""API tests — FastAPI TestClient against the test-only in-memory
repository. No database is needed for these."""

import pytest

GOOD_PAYLOAD = {
    "model": "model-a",
    "task_type": "factual_qa",
    "prompt": "What is the capital of France?",
    "response": "The capital of France is Paris. It has been the capital since the 10th century.",
    "reference": "The capital of France is Paris. Paris has been the capital since the 10th century.",
}

COMPARE_PAYLOAD = {
    "model_a": "model-a",
    "model_b": "model-b",
    "task_type": "factual_qa",
    "prompt": "What is the capital of France?",
    "response_a": "The capital of France is Paris. It has been the capital since the 10th century.",
    "response_b": "The capital of France is Lyon. Lyon is a large city in eastern France.",
    "reference": "The capital of France is Paris. Paris has been the capital since the 10th century.",
}


def score_one(client, **overrides):
    payload = {**GOOD_PAYLOAD, **overrides}
    response = client.post("/evaluations/score", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


# --- health -------------------------------------------------------------

def test_health_ok_when_database_answers(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_degraded_when_database_is_down(client, repo):
    repo.set_ping(False)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unreachable"}


# --- POST /evaluations/score --------------------------------------------

def test_score_returns_full_record(client):
    body = score_one(client)
    assert body["model"] == "model-a"
    assert body["task_type"] == "factual_qa"
    assert len(body["criteria"]) == 5
    assert {c["criterion"] for c in body["criteria"]} == {
        "factual_accuracy", "reasoning_quality", "instruction_adherence",
        "helpfulness", "safety",
    }
    assert body["total_score"] == sum(c["score"] for c in body["criteria"])
    assert body["mean_score"] == round(body["total_score"] / 5, 2)
    assert body["passed"] == (body["mean_score"] >= 4.0)
    assert body["id"]
    assert body["created_at"]


def test_score_persists_and_can_be_read_back(client):
    created = score_one(client)
    fetched = client.get(f"/evaluations/{created['id']}")
    assert fetched.status_code == 200
    assert fetched.json() == created


def test_score_without_reference_is_accepted(client):
    body = score_one(client, reference=None)
    assert body["reference"] is None
    factual = next(c for c in body["criteria"] if c["criterion"] == "factual_accuracy")
    assert factual["score"] == 3  # unverified, not correct


def test_score_rejects_missing_model(client):
    payload = {k: v for k, v in GOOD_PAYLOAD.items() if k != "model"}
    assert client.post("/evaluations/score", json=payload).status_code == 422


def test_score_rejects_empty_response(client):
    assert client.post("/evaluations/score", json={**GOOD_PAYLOAD, "response": ""}).status_code == 422


def test_score_rejects_empty_prompt(client):
    assert client.post("/evaluations/score", json={**GOOD_PAYLOAD, "prompt": ""}).status_code == 422


# --- GET /evaluations/{id} ----------------------------------------------

def test_get_unknown_evaluation_is_404(client):
    response = client.get("/evaluations/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    assert response.json()["detail"] == "evaluation not found"


# --- GET /evaluations (list) --------------------------------------------

def test_list_empty_when_nothing_stored(client):
    body = client.get("/evaluations").json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_list_filters_by_model_and_task_type(client):
    score_one(client, model="model-a", task_type="factual_qa")
    score_one(client, model="model-b", task_type="factual_qa")
    score_one(client, model="model-a", task_type="maths_reasoning")

    assert client.get("/evaluations").json()["total"] == 3
    assert client.get("/evaluations", params={"model": "model-a"}).json()["total"] == 2
    assert client.get("/evaluations", params={"task_type": "maths_reasoning"}).json()["total"] == 1
    assert client.get("/evaluations", params={"model": "model-a", "task_type": "factual_qa"}).json()["total"] == 1
    assert client.get("/evaluations", params={"model": "no-such-model"}).json()["total"] == 0


def test_list_pagination(client):
    for _ in range(5):
        score_one(client)
    page1 = client.get("/evaluations", params={"limit": 2, "offset": 0}).json()
    page2 = client.get("/evaluations", params={"limit": 2, "offset": 2}).json()
    page3 = client.get("/evaluations", params={"limit": 2, "offset": 4}).json()
    assert page1["total"] == 5 and len(page1["items"]) == 2
    assert len(page2["items"]) == 2
    assert len(page3["items"]) == 1
    ids = [item["id"] for item in page1["items"] + page2["items"] + page3["items"]]
    assert len(set(ids)) == 5  # no duplicates across pages


def test_list_rejects_limit_above_max(client):
    assert client.get("/evaluations", params={"limit": 101}).status_code == 422
    assert client.get("/evaluations", params={"limit": 0}).status_code == 422
    assert client.get("/evaluations", params={"offset": -1}).status_code == 422


# --- POST /evaluations/compare ------------------------------------------

def test_compare_returns_winner_and_links_both_evaluations(client):
    response = client.post("/evaluations/compare", json=COMPARE_PAYLOAD)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["winner"] == "A"
    assert body["mean_a"] > body["mean_b"]
    assert len(body["comparisons"]) == 5
    assert body["rationale"]
    # Both responses were stored as evaluations and linked.
    assert client.get(f"/evaluations/{body['evaluation_a_id']}").status_code == 200
    assert client.get(f"/evaluations/{body['evaluation_b_id']}").status_code == 200
    assert body["evaluation_a_id"] != body["evaluation_b_id"]


def test_compare_tie_when_responses_are_identical(client):
    payload = {**COMPARE_PAYLOAD, "response_b": COMPARE_PAYLOAD["response_a"]}
    body = client.post("/evaluations/compare", json=payload).json()
    assert body["winner"] == "tie"
    assert body["mean_a"] == body["mean_b"]


def test_compare_rejects_missing_response_b(client):
    payload = {k: v for k, v in COMPARE_PAYLOAD.items() if k != "response_b"}
    assert client.post("/evaluations/compare", json=payload).status_code == 422


# --- GET /metrics/summary -------------------------------------------------

def test_metrics_empty_when_nothing_stored(client):
    body = client.get("/metrics/summary").json()
    assert body["total_evaluations"] == 0
    assert body["total_comparisons"] == 0
    assert body["mean_score"] == 0.0
    assert body["pass_rate"] == 0.0
    assert body["by_model"] == []
    assert body["by_task_type"] == []
    assert body["comparison_wins"] == []


def test_metrics_aggregates_match_stored_data(client):
    score_one(client, model="model-a", task_type="factual_qa")
    score_one(client, model="model-a", task_type="factual_qa")
    score_one(client, model="model-b", task_type="maths_reasoning")
    client.post("/evaluations/compare", json=COMPARE_PAYLOAD)

    body = client.get("/metrics/summary").json()
    # 3 scores + 2 evaluations created by the comparison.
    assert body["total_evaluations"] == 5
    assert body["total_comparisons"] == 1
    assert body["comparison_wins"] == [{"model": "model-a", "wins": 1}]
    models = {row["model"]: row for row in body["by_model"]}
    assert models["model-a"]["count"] == 3
    assert models["model-b"]["count"] == 2
    tasks = {row["task_type"]: row for row in body["by_task_type"]}
    assert tasks["factual_qa"]["count"] == 4
    assert tasks["maths_reasoning"]["count"] == 1
    for row in body["by_model"] + body["by_task_type"]:
        assert 1.0 <= row["mean_score"] <= 5.0
        assert 0.0 <= row["pass_rate"] <= 1.0
