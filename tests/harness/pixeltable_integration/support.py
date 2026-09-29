"""Catalog helpers shared by the Pixeltable tests.

Pixeltable rejects UDFs defined in `__main__`, so the test embedding lives in an importable module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable

import numpy as np
import pixeltable as pxt

DIM = 8

# Pixeltable's `create_table` and `Table.insert` signatures carry partially unknown types
# (`PathLike[Unknown]` in their data-source unions); give the tests one typed entry point.
create_table: Callable[..., pxt.Table] = pxt.create_table  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def get_table(path: str) -> pxt.Table:
    table = pxt.get_table(path)
    assert table is not None  # if_not_exists='error' raises instead of returning None
    return table


def insert_rows(table: pxt.Table, rows: list[dict[str, object]]) -> None:
    table.insert(rows)  # pyright: ignore[reportUnknownMemberType]


@pxt.udf  # pyright: ignore[reportUnknownMemberType]
def tiny_embed(text: str) -> pxt.Array[(8,), pxt.Float]:
    """Hash embedding for the similarity tests. Not a semantic model."""
    digest = hashlib.sha256(text.encode()).digest()
    raw = [(digest[i % 32] / 127.5) - 1.0 for i in range(DIM)]
    norm = math.hypot(*raw)  # never zero: no byte maps to 0.0
    # Pixeltable's `Array` annotation is a column type, not the ndarray the UDF returns.
    return np.array([value / norm for value in raw], dtype=np.float32)  # pyright: ignore[reportReturnType]
