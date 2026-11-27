"""Private helpers for walking a capability tree."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic_ai.capabilities import AbstractCapability, WrapperCapability


def wrapper_chain(capability: AbstractCapability[Any]) -> Iterator[AbstractCapability[Any]]:
    """`capability`, then each capability it wraps through a `WrapperCapability` chain."""
    candidate: AbstractCapability[Any] | None = capability
    while candidate is not None:
        yield candidate
        candidate = candidate.wrapped if isinstance(candidate, WrapperCapability) else None
