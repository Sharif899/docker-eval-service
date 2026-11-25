"""Seed the running service with the synthetic example data.

Reads ``data/examples.jsonl`` and loads it through the API:

* items 1–5 are scored individually (``POST /evaluations/score``)
  for both responses — 10 evaluations;
* items 6–10 are compared (``POST /evaluations/compare``) — 5
  comparisons, each storing its two evaluations — 10 more evaluations.

Total seeded: 20 evaluations, 5 comparisons, no duplicated rows.
The data is synthetic and documented as such in the file itself;
it exists so the service can be explored immediately after
``docker compose up``.

Usage:
    EVAL_API_URL=http://localhost:8000 python scripts/seed.py

Only the standard library is used, so the script also runs inside
the API container without extra installs.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "http://localhost:8000"
MODEL_A = "model-a"
MODEL_B = "model-b"

EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "data" / "examples.jsonl"


def load_examples(path: Path = EXAMPLES_PATH) -> list[dict]:
    """Read the example items, one JSON object per line."""
    items = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    if not items:
        raise ValueError(f"No example items found in {path}")
    return items


def build_score_payload(item: dict, which: str, model: str) -> dict:
    """Payload for POST /evaluations/score for one response."""
    if which not in ("response_a", "response_b"):
        raise ValueError(f"which must be 'response_a' or 'response_b', got {which!r}")
    return {
        "model": model,
        "task_type": item["category"],
        "prompt": item["prompt"],
        "response": item[which],
        "reference": item.get("reference"),
    }


def build_compare_payload(item: dict) -> dict:
    """Payload for POST /evaluations/compare for one item."""
    return {
        "model_a": MODEL_A,
        "model_b": MODEL_B,
        "task_type": item["category"],
        "prompt": item["prompt"],
        "response_a": item["response_a"],
        "response_b": item["response_b"],
        "reference": item.get("reference"),
    }


def _post(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach the API at {base_url}: {exc}. "
            "Is the service running? Try: docker compose up --build"
        ) from exc


def _get(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def run(base_url: str = DEFAULT_BASE_URL, path: Path = EXAMPLES_PATH) -> dict:
    """Seed the service. Returns a summary of what was created."""
    items = load_examples(path)
    scored = compared = 0
    for index, item in enumerate(items):
        if index < 5:
            result_a = _post(base_url, "/evaluations/score",
                             build_score_payload(item, "response_a", MODEL_A))
            result_b = _post(base_url, "/evaluations/score",
                             build_score_payload(item, "response_b", MODEL_B))
            print(f"{item['id']}: {MODEL_A} mean {result_a['mean_score']}, "
                  f"{MODEL_B} mean {result_b['mean_score']}")
            scored += 2
        else:
            result = _post(base_url, "/evaluations/compare", build_compare_payload(item))
            print(f"{item['id']}: winner {result['winner']} "
                  f"({MODEL_A} {result['mean_a']} vs {MODEL_B} {result['mean_b']})")
            compared += 1
    summary = _get(base_url, "/metrics/summary")
    print(f"Seeded {scored} scored evaluations and {compared} comparisons "
          f"({summary['total_evaluations']} evaluations in total).")
    print(f"Overall mean score: {summary['mean_score']}, "
          f"pass rate: {summary['pass_rate']}.")
    return {
        "evaluations_scored": scored,
        "comparisons": compared,
        "total_evaluations": summary["total_evaluations"],
        "mean_score": summary["mean_score"],
        "pass_rate": summary["pass_rate"],
    }


def main() -> int:
    base_url = os.environ.get("EVAL_API_URL", DEFAULT_BASE_URL)
    try:
        run(base_url)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
