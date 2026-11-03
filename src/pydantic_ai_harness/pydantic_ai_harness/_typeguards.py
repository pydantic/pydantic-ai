"""`TypeGuard`s that narrow a container to `object` elements, so pyright doesn't infer `Unknown`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeGuard


def is_mapping(value: object) -> TypeGuard[Mapping[object, object]]:
    return isinstance(value, Mapping)


def is_object_dict(value: object) -> TypeGuard[dict[object, object]]:
    return isinstance(value, dict)
