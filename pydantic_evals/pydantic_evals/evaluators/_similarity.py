"""Deterministic similarity metrics for the built-in similarity evaluators."""

from __future__ import annotations

from typing import Literal

StringSimilarityMetric = Literal['levenshtein', 'jaro_winkler']


def levenshtein_ratio(a: str, b: str) -> float:
    """Normalized Levenshtein similarity: `1 - distance / max(len)`, 1.0 for equal strings."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def jaro_winkler(a: str, b: str, prefix_scale: float = 0.1) -> float:
    """Jaro-Winkler similarity in `[0, 1]`, boosting strings that share a prefix."""
    if a == b:
        return 1.0
    len_a, len_b = len(a), len(b)
    if not a or not b:
        return 0.0
    match_distance = max(len_a, len_b) // 2 - 1

    a_matches = [False] * len_a
    b_matches = [False] * len_b
    matches = 0
    for i, char_a in enumerate(a):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, len_b)
        for j in range(start, end):
            if not b_matches[j] and char_a == b[j]:
                a_matches[i] = b_matches[j] = True
                matches += 1
                break
    if matches == 0:
        return 0.0

    transpositions = 0
    j = 0
    for i, is_match in enumerate(a_matches):
        if not is_match:
            continue
        while not b_matches[j]:
            j += 1
        transpositions += a[i] != b[j]
        j += 1
    transpositions //= 2

    jaro = (matches / len_a + matches / len_b + (matches - transpositions) / matches) / 3
    prefix = 0
    for char_a, char_b in zip(a, b):
        if char_a != char_b or prefix == 4:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1 - jaro)


def string_similarity(metric: StringSimilarityMetric, a: str, b: str) -> float:
    if metric == 'levenshtein':
        return levenshtein_ratio(a, b)
    return jaro_winkler(a, b)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors of equal length."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return dot / norm if norm else 0.0
