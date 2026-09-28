"""Test helpers for the Absurd durability capability.

No Postgres and no Docker: `FakeAsyncTaskContext` is an in-memory stand-in for
`absurd_sdk.AsyncTaskContext` that reproduces the two behaviors the capability
depends on -- encounter-order disambiguation of repeated step names, and a
replay that serves stored checkpoints without re-invoking `fn`. Tests activate
it by setting the SDK's context var via `absurd_task_context(...)`.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from typing import TypeVar, cast

import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import (
    AsyncTaskContext,
    JsonValue,
    TaskContext,
    _current_task_context,  # pyright: ignore[reportPrivateUsage]
)

R = TypeVar('R')


class FakeAsyncTaskContext(AsyncTaskContext):
    """In-memory stand-in for an Absurd async task context.

    `step` mirrors `AsyncTaskContext.step`: the checkpoint name is disambiguated by
    encounter order (`name`, `name#2`, ...) via the inherited `_get_checkpoint_name`, a stored
    checkpoint is served without calling `fn`, and every stored value is JSON round-tripped just
    as Postgres storage would be, so a non-serializable payload fails here the same way it would
    in production. `invoked` records the checkpoint names whose `fn` actually ran, so a test can
    assert a step was reached exactly once across a replay.
    """

    def __init__(self, *, store: dict[str, JsonValue] | None = None) -> None:
        self.task_id = 'fake-task'
        self._store: dict[str, JsonValue] = {} if store is None else store
        self._step_name_counter: dict[str, int] = {}
        self.invoked: list[str] = []

    @property
    def stored(self) -> dict[str, JsonValue]:
        """The checkpoint store, keyed by disambiguated step name."""
        return self._store

    async def step(self, name: str, fn: Callable[[], Awaitable[R]]) -> R:
        checkpoint_name = self._get_checkpoint_name(name)
        if checkpoint_name in self._store:
            return cast(R, self._store[checkpoint_name])
        self.invoked.append(checkpoint_name)
        stored = cast(JsonValue, json.loads(json.dumps(await fn())))
        self._store[checkpoint_name] = stored
        return cast(R, stored)

    def replay(self) -> FakeAsyncTaskContext:
        """A fresh context hydrated from the stored checkpoints, as Absurd does on a retry.

        The encounter counter resets (a new attempt reaches the steps from the top), while the
        stored checkpoints carry over, so `step` serves them without re-running `fn`.
        """
        return FakeAsyncTaskContext(store=cast('dict[str, JsonValue]', json.loads(json.dumps(self._store))))


class FakeSyncTaskContext(TaskContext):
    """In-memory stand-in for a synchronous Absurd task context.

    Only used to prove the capability rejects a sync context: an agent run is async and cannot be
    awaited from one.
    """

    def __init__(self) -> None:
        self.task_id = 'fake-sync-task'


@contextmanager
def absurd_task_context(ctx: AsyncTaskContext | TaskContext) -> Generator[None]:
    """Activate `ctx` as the current Absurd task context for the duration of the block."""
    token = _current_task_context.set(ctx)
    try:
        yield
    finally:
        _current_task_context.reset(token)
