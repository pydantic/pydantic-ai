"""Local BM25 shortlist: identifier-aware terms, path/symbol boost, synonyms.

Cheap lexical ranking decides which snippets are worth a relevance judgment. It is a shortlist, not a
verdict: code with unrelated wording can be missed, which is why coverage is reported back to the caller.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from pydantic_ai_harness.smart_grep._chunks import Chunk

_STOP = frozenset(
    'a an the is are was were to of in on for from with where how does do this '
    'that it and or code function files file find which'.split()
)

_SYNONYM_GROUPS = (
    'auth authentication authenticate login signin session',
    'retry retries backoff attempt',
    'cache cached caching memo memoize',
    'delete remove cleanup unlink',
    'permission permissions authorization authorize access allowed',
    'expire expired expires expiration expiry timeout ttl',
    'save persist persistence storage store write',
    'error errors failure failed exception throw catch',
    'payment payments billing subscription invoice charge',
    'cancel abort cancellation',
    'parallel concurrent concurrency simultaneous',
    'duplicate duplicates deduplicate dedup unique',
)
_VOCABULARY = {term: group.split() for group in _SYNONYM_GROUPS for term in group.split()}

_CAMEL = re.compile(r'([a-z\d])([A-Z])')
_WORD = re.compile(r'[^\W_]+')
_SYNONYM_WEIGHT = 0.3
_META_BOOST = 3
_K1, _B = 1.2, 0.75


def terms(text: str) -> list[str]:
    """Lower-cased identifier-aware tokens: `refreshToken` -> refresh, token."""
    words = _WORD.findall(_CAMEL.sub(r'\1 \2', text).lower())
    return [w for w in words if len(w) > 1 and w not in _STOP]


def _query_weights(query: str) -> dict[str, float]:
    weights = dict.fromkeys(terms(query), 1.0)
    for term in list(weights):
        for synonym in _VOCABULARY.get(term, ()):
            weights.setdefault(synonym, _SYNONYM_WEIGHT)
    return weights


def rank(query: str, chunks: Sequence[Chunk]) -> list[Chunk]:
    """Chunks ordered by BM25 relevance to `query` (stable on ties)."""
    weights = _query_weights(query)
    docs: list[tuple[Counter[str], int]] = []
    for chunk in chunks:
        body, meta = terms(chunk.text), terms(f'{chunk.path} {chunk.symbol or ""}')
        freq = Counter(body)
        for word in meta:
            freq[word] += _META_BOOST
        docs.append((freq, len(body) + len(meta)))

    n = len(docs)
    average = (sum(length for _, length in docs) / n if n else 0) or 1
    df = {t: sum(1 for freq, _ in docs if t in freq) for t in weights}

    def score(doc: tuple[Counter[str], int]) -> float:
        freq, length = doc
        total = 0.0
        for word, weight in weights.items():
            tf = freq.get(word, 0)
            if not tf:
                continue
            idf = math.log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
            norm = _K1 * (1 - _B + _B * length / average)
            total += weight * idf * tf * (_K1 + 1) / (tf + norm)
        return total

    scored = [(score(doc), chunk) for doc, chunk in zip(docs, chunks)]
    scored.sort(key=lambda sc: (-sc[0], sc[1].path, sc[1].line))
    return [chunk for _, chunk in scored]
