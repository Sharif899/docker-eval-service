"""Tests for the database setup helpers and the seed script.

Migration tests that need a live database live in
``test_repository.py``; these cover what can be checked without one.
"""

import json

import pytest

from docker_eval_service import db


def test_get_database_url_requires_env(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="DATABASE_URL is not set"):
        db.get_database_url()


def test_get_database_url_returns_value(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://eval:evalpass@localhost:5432/evaldb")
    assert db.get_database_url() == "postgresql://eval:evalpass@localhost:5432/evaldb"


def test_list_migrations_finds_init_sql():
    migrations = db.list_migrations()
    assert [p.name for p in migrations] == ["001_init.sql"]
    sql = migrations[0].read_text(encoding="utf-8")
    for table in ("evaluations", "scores", "comparisons", "schema_migrations"):
        assert table in sql


def test_migrations_are_idempotent(pg_repo, postgres_url):
    # Applied once by the pg_repo fixture; applying again must be a
    # no-op, not an error.
    assert db.apply_migrations(postgres_url) == []


# --- seed script ----------------------------------------------------------

from scripts.seed import (  # noqa: E402
    build_compare_payload,
    build_score_payload,
    load_examples,
)


def test_load_examples_returns_ten_items():
    items = load_examples()
    assert len(items) == 10
    categories = {item["category"] for item in items}
    assert categories == {
        "factual_qa", "maths_reasoning", "coding",
        "instruction_following", "safety", "multilingual_yoruba",
    }
    for item in items:
        assert item["prompt"] and item["reference"]
        assert item["response_a"] and item["response_b"]
        assert item["response_a"] != item["response_b"]


def test_build_score_payload_maps_fields():
    items = load_examples()
    payload = build_score_payload(items[0], "response_a", "model-a")
    assert payload == {
        "model": "model-a",
        "task_type": items[0]["category"],
        "prompt": items[0]["prompt"],
        "response": items[0]["response_a"],
        "reference": items[0]["reference"],
    }
    with pytest.raises(ValueError, match="response_a.*response_b"):
        build_score_payload(items[0], "response_c", "model-a")


def test_build_compare_payload_maps_fields():
    items = load_examples()
    payload = build_compare_payload(items[0])
    assert payload["model_a"] == "model-a"
    assert payload["model_b"] == "model-b"
    assert payload["response_a"] == items[0]["response_a"]
    assert payload["response_b"] == items[0]["response_b"]
    assert payload["task_type"] == items[0]["category"]


def test_load_examples_rejects_invalid_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "ok"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_examples(bad)


def test_load_examples_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No example items"):
        load_examples(empty)
