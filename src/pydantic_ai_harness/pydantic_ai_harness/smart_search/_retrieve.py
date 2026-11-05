"""Local BM25 shortlist: identifier-aware terms, path/symbol boost, synonyms, and an incremental index.

Cheap lexical ranking decides which snippets are worth a relevance judgment. It is a shortlist, not a
verdict: code with unrelated wording can be missed, which is why coverage is reported back to the caller.
"""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter

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


class Bm25:
    """An inverted BM25 index. Documents are added and retired, never edited.

    Each term keeps one flat array of `(document, term frequency)` pairs, so a query only reads the
    postings of its own terms. Retired documents stay in the postings, skipped while scoring, until
    `compact` drops them.
    """

    def __init__(self) -> None:
        self._postings: dict[str, array[int]] = {}
        self._lengths = array('I')
        self._alive = bytearray()
        self._live_length = 0
        self.size = 0
        """Documents added and not yet retired."""

    @property
    def retired(self) -> int:
        """Retired documents still taking space until `compact`."""
        return len(self._lengths) - self.size

    def add(self, text: str, meta: str) -> int:
        """Index a document's `text`, boosting the terms of its `meta` (path and symbol). Returns its id."""
        body, boosted = terms(text), terms(meta)
        freq = Counter(body)
        for word in boosted:
            freq[word] += _META_BOOST
        doc, length = len(self._lengths), len(body) + len(boosted)
        self._lengths.append(length)
        self._alive.append(1)
        self._live_length += length
        self.size += 1
        for word, tf in freq.items():
            self._postings.setdefault(word, array('I')).extend((doc, tf))
        return doc

    def retire(self, doc: int) -> None:
        """Stop matching `doc`."""
        self._alive[doc] = 0
        self._live_length -= self._lengths[doc]
        self.size -= 1

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score of every live document sharing a term with `query`, by id."""
        n = self.size
        average = (self._live_length / n if n else 0) or 1
        alive, lengths = self._alive, self._lengths
        out: dict[int, float] = {}
        for word, weight in _query_weights(query).items():
            postings = self._postings.get(word)
            if postings is None:
                continue
            pairs = [(postings[i], postings[i + 1]) for i in range(0, len(postings), 2) if alive[postings[i]]]
            idf = math.log(1 + (n - len(pairs) + 0.5) / (len(pairs) + 0.5))
            for doc, tf in pairs:
                norm = _K1 * (1 - _B + _B * lengths[doc] / average)
                out[doc] = out.get(doc, 0.0) + weight * idf * tf * (_K1 + 1) / (tf + norm)
        return out

    def compact(self) -> array[int]:
        """Drop retired documents and renumber the rest. Returns each old id's new id, or -1 if retired."""
        renumber = array('i', [-1]) * len(self._lengths)
        live = 0
        for doc, alive in enumerate(self._alive):
            if alive:
                renumber[doc] = live
                live += 1
        postings: dict[str, array[int]] = {}
        for word, old in self._postings.items():
            kept = array('I')
            for i in range(0, len(old), 2):
                if (doc := renumber[old[i]]) >= 0:
                    kept.extend((doc, old[i + 1]))
            if kept:
                postings[word] = kept
        self._postings = postings
        self._lengths = array('I', (length for length, alive in zip(self._lengths, self._alive) if alive))
        self._alive = bytearray(b'\x01') * live
        return renumber
