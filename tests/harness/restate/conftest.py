"""Test helpers for the Restate durability capability.

No network and no Docker: `FakeRestateContext` is an in-memory stand-in for the Restate context
that reproduces the two behaviors the capability depends on -- a positional journal (each
`run_typed` records its result by encounter order) and a replay that serves stored entries without
re-running the action. Tests activate it by setting the SDK's real context var via
`restate_context(...)`.

Every stored value round-trips through the real Restate serde the capability passes
(`restate.serde.JsonSerde`), so the journal behaves as it does in production. The capability
reduces each value to a JSON-able payload with a `TypeAdapter` before it reaches the journal, so a
value that cannot be serialized fails at that dump, before the serde, identically here and in
production.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import pytest

pytest.importorskip('restate')

from restate.context import RunOptions
from restate.serde import JsonSerde
from restate.server_context import _restate_context_var  # pyright: ignore[reportPrivateUsage]

R = TypeVar('R')

_STORED_SERDE: JsonSerde[Any] = JsonSerde()


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass
class Entry:
    """One committed journal entry: the step's label and its serialized bytes."""

    name: str
    data: bytes


class FakeRestateContext:
    """In-memory stand-in for a Restate invocation context.

    `run_typed` mirrors the real primitive: the journal slot is claimed by encounter order (the
    `name` is a label, not the identity), a stored entry is served without calling the action, and
    every stored value round-trips through the serde the capability passes so a payload that cannot
    be serialized fails here the same way it would in production. An action that raises is not
    journaled -- the run aborts as a crash would, and a replay re-runs that step. `invoked` records
    the step names whose action actually ran, so a test can assert a step was reached exactly once
    across a replay.
    """

    def __init__(self, journal: list[Entry] | None = None) -> None:
        self.journal: list[Entry] = journal or []
        self.recorded: list[Entry] = []
        self.invoked: list[str] = []
        self.max_attempts: list[int | None] = []
        self._cursor = 0

    async def run_typed(
        self,
        name: str,
        action: Callable[..., Awaitable[R]],
        options: RunOptions[R] = RunOptions(),
        /,
        *args: Any,
        **kwargs: Any,
    ) -> R:
        self.max_attempts.append(options.max_attempts)
        index = self._cursor
        self._cursor += 1
        if index < len(self.journal):
            entry = self.journal[index]
            self.recorded.append(entry)
            return options.serde.deserialize(entry.data)  # pyright: ignore[reportReturnType]
        self.invoked.append(name)
        value = await action(*args, **kwargs)
        data = options.serde.serialize(value)
        self.recorded.append(Entry(name, data))
        return options.serde.deserialize(data)  # pyright: ignore[reportReturnType]

    def replay(self) -> FakeRestateContext:
        """A fresh context hydrated from the committed journal, as Restate does on a retry.

        The encounter cursor resets (a new attempt reaches the steps from the top) while the
        committed entries carry over, so `run_typed` serves them without re-running the action.
        """
        return FakeRestateContext(journal=[Entry(e.name, e.data) for e in self.recorded])

    @property
    def step_names(self) -> list[str]:
        """The committed step names, in journal order."""
        return [entry.name for entry in self.recorded]

    def stored(self, name: str) -> Any:
        """The deserialized value journaled under `name` (the last, if the name recurs)."""
        entry = next(entry for entry in reversed(self.recorded) if entry.name == name)
        return _STORED_SERDE.deserialize(entry.data)


@contextmanager
def restate_context(ctx: FakeRestateContext) -> Generator[None]:
    """Activate `ctx` as the current Restate context for the duration of the block."""
    # The fake implements only the `run_typed` surface the capability uses, so it stands in for the
    # 24-method `Context` ABC here.
    token = _restate_context_var.set(ctx)  # pyright: ignore[reportArgumentType]
    try:
        yield
    finally:
        _restate_context_var.reset(token)
