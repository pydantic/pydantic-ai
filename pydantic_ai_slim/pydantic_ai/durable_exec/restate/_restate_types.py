from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

T = TypeVar('T')

__all__ = ['T', 'Context', 'RunOptions', 'TerminalError', 'Serde']


if TYPE_CHECKING:
    class TerminalError(Exception):
        """Type-checking fallback for `restate.TerminalError`."""

    @dataclass
    class RunOptions(Generic[T]):
        """Type-checking fallback for `restate.RunOptions`."""

        serde: Any | None = None
        max_attempts: int | None = None

    class Context(Protocol):
        """Type-checking fallback for `restate.Context`."""

        async def run_typed(
            self,
            name: str,
            fn: Callable[..., Awaitable[T] | T],
            options: RunOptions[T] | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> T: ...

    class Serde(Protocol[T]):
        """Type-checking fallback for `restate.serde.Serde`."""

        def deserialize(self, buf: bytes) -> T | None: ...

        def serialize(self, obj: T | None) -> bytes: ...
else:  # pragma: no cover
    from restate import Context, RunOptions, TerminalError
    from restate.serde import Serde

