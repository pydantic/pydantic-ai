"""BM25 ranking, its query bounds, and excerpt clipping over plain-text documents.

Shared by `ConversationSearch` and `WorkLog`. The ranking core is ported from VStorm's
`pydantic-deepagents` (`pydantic_deep/features/history_archive/toolset.py`). It is a different
BM25 from `smart_search`'s `Bm25`, which stems and drops stopwords.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from pydantic_ai import ModelRetry

TOKENIZE_RE = re.compile(r'\w+')
"""Tokenizer regex: word runs (unicode letters, digits, underscore).

Scripts written without spaces (Chinese, Japanese, Thai) form one token per unbroken run,
so a query matches only the whole run, not a word inside it.
"""


@dataclass(frozen=True)
class Bm25Params:
    """BM25 tuning supplied by the owning capability."""

    k1: float
    b: float


def tokenize(text: str) -> list[str]:
    return [match.group().lower() for match in TOKENIZE_RE.finditer(text)]


MAX_QUERY_CHARS = 1_000
MAX_QUERY_TERMS = 32
"""Query bounds, the same as `Memory`'s `search_memory`: ranking costs unique terms times scanned documents."""


def check_query(query: str) -> None:
    """Reject a query whose ranking would hold the event loop, so the model can retry with a shorter one."""
    if len(query) > MAX_QUERY_CHARS:
        raise ModelRetry(f'Search queries must be at most {MAX_QUERY_CHARS} characters.')
    if len(set(tokenize(query))) > MAX_QUERY_TERMS:
        raise ModelRetry(f'Search queries must contain at most {MAX_QUERY_TERMS} unique terms.')


def compute_idf(term: str, term_counts: list[Counter[str]]) -> float:
    """Inverse document frequency, standard BM25 form `ln((N - df + 0.5)/(df + 0.5) + 1)`."""
    n = len(term_counts)
    df = sum(1 for counts in term_counts if term in counts)
    if df == 0:
        return 0.0
    return math.log((n - df + 0.5) / (df + 0.5) + 1.0)


def bm25_score(
    query_terms: list[str],
    counts: Counter[str],
    dl: int,
    idf_scores: dict[str, float],
    avgdl: float,
    params: Bm25Params,
) -> float:
    if dl == 0 or avgdl == 0:
        return 0.0
    score = 0.0
    for term in query_terms:
        tf = counts[term]
        if tf == 0:
            continue
        numerator = tf * (params.k1 + 1.0)
        denominator = tf + params.k1 * (1.0 - params.b + params.b * dl / avgdl)
        score += idf_scores[term] * numerator / denominator
    return score


def bm25_rank(query: str, documents: list[str], params: Bm25Params) -> list[tuple[int, float]]:
    """Rank documents by BM25 relevance, descending; only scores greater than zero."""
    query_terms = list(dict.fromkeys(tokenize(query)))
    if not query_terms or not documents:
        return []

    # Tokenize each document once; df, tf, and length all read the same counts.
    term_counts = [Counter(tokenize(doc)) for doc in documents]
    lengths = [counts.total() for counts in term_counts]
    avgdl = sum(lengths) / len(lengths)
    idf_scores = {term: compute_idf(term, term_counts) for term in query_terms}

    results: list[tuple[int, float]] = []
    for index, (counts, dl) in enumerate(zip(term_counts, lengths)):
        score = bm25_score(query_terms, counts, dl, idf_scores, avgdl, params)
        if score > 0:
            results.append((index, score))

    results.sort(key=lambda item: item[1], reverse=True)
    return results


def clip(text: str, limit: int, terms: frozenset[str] | None) -> str:
    """Cut `text` to about `limit` characters for display; `terms=None` keeps it whole for the index.

    A head cut alone would report a match whose term the excerpt never shows, so when the
    first query term ends past `limit` the window is centred on that term instead. It shifts
    to keep a long term whole when it fits in `limit`, and ends with the text when the term
    is near its end, so the excerpt keeps the full `limit`.
    """
    if terms is None or len(text) <= limit:
        return text
    focus = next((match for match in TOKENIZE_RE.finditer(text) if match.group().lower() in terms), None)
    if focus is None or focus.end() <= limit:
        return text[:limit] + '...'
    # Keep the whole term when it fits (a term longer than `limit // 2` would lose its tail to
    # plain centring), and pull the window back near the end so a late term keeps the full `limit`.
    start = min(max(focus.start() - limit // 2, focus.end() - limit), len(text) - limit)
    end = start + limit
    return ('...' if start else '') + text[start:end] + ('...' if end < len(text) else '')
