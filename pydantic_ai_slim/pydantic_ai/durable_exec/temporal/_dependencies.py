from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

DepsT = TypeVar('DepsT')
DepsRefT = TypeVar('DepsRefT')


@dataclass(frozen=True)
class TemporalDependencyResolver(Generic[DepsT, DepsRefT]):
    """Convert agent dependencies to a small durable reference and rehydrate them in activities."""

    reference_type: type[DepsRefT]
    """The concrete reference type Temporal should deserialize."""

    _: KW_ONLY
    to_reference: Callable[[DepsT], DepsRefT]
    """Return a deterministic reference without performing I/O."""

    from_reference: Callable[[DepsRefT], Awaitable[DepsT]]
    """Load dependencies from a reference inside a Temporal activity."""
