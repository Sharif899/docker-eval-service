"""Small, deterministic text helpers for the scorer.

Everything here is dependency-free and deterministic: the same input
always produces the same output. The rules are deliberately small and
readable — a reviewer should be able to trace exactly how a score
was produced.
"""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "and", "or", "but",
    "if", "then", "than", "so", "as", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "my", "your", "his",
    "her", "our", "their", "not", "no", "do", "does", "did", "have", "has",
    "had", "will", "would", "can", "could", "should", "may", "might",
    "from", "into", "over", "under", "about", "after", "before", "between",
    # Common Yoruba particles — Yoruba text must be scored, not ignored.
    "ti", "ni", "lo", "ninu", "je", "o", "a", "e", "si", "ko", "se",
}

_WORD_RE = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?(?![A-Za-z])")


def normalize(text: str) -> str:
    """Lower-case, strip accents for matching, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(normalize(text))


def content_tokens(text: str) -> set[str]:
    """Content words only — stopwords removed — for overlap checks.

    Single-digit numbers are kept: in evaluation data a number is
    often the whole answer (a year, a speed, a count), so dropping
    it would make a wrong number invisible to the scorer.
    """
    return {t for t in tokens(text) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())}


def sentences(text: str) -> list[str]:
    """Split text into sentences, keeping the punctuation."""
    text = (text or "").strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def numbers(text: str) -> list[str]:
    return _NUMBER_RE.findall(text or "")


def overlap(a: str, b: str) -> float:
    """Jaccard overlap of content tokens: |A ∩ B| / |A ∪ B|."""
    set_a, set_b = content_tokens(a), content_tokens(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def clamp_score(value: float, low: int = 1, high: int = 5) -> int:
    """Map a 0–1 value onto a 1–5 rubric score, clamped to the scale."""
    score = int(round(low + value * (high - low)))
    return max(low, min(high, score))


def snippet(text: str, max_len: int = 120) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_len else text[: max_len - 3].rstrip() + "..."
