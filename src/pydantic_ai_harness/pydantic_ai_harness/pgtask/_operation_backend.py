from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import ValidationError

from pydantic_ai.durable_exec import (
    JSON_CODEC,
    DurabilityCodec,
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
)

from ._context import current_task_steps

# pgtask steps take no per-operation options.
_NO_CONFIG = RoleBasedOperationConfig[None](model=None, event=None, capability=None, tool=None)


class CheckpointDecodeError(RuntimeError):
    """A step's checkpoint does not decode as what the step returns now."""


class PGTaskCodec(DurabilityCodec):
    """JSON, failing the run when a checkpoint does not decode.

    A tool's checkpoint is decoded inside the tool call, where a `ValidationError` reads as bad model
    arguments: the model would be asked to retry, and the retry would run the tool again. Raising
    another error type keeps the side effect from repeating.
    """

    def dump(self, tp: Any, value: Any) -> Any:
        return JSON_CODEC.dump(tp, value)

    def load(self, tp: Any, payload: Any) -> Any:
        try:
            return JSON_CODEC.load(tp, payload)
        except ValidationError as exc:
            raise CheckpointDecodeError(f'A pgtask checkpoint could not be decoded: {exc}') from exc


class PGTaskOperationBackend(JournalCallableOperationBackend[None]):
    """Runs each durable operation as one `task.step(...)` checkpoint."""

    def __init__(self, *, agent_name: str, default_model_id: str | None) -> None:
        super().__init__(agent_name=agent_name, default_model_id=default_model_id, config=_NO_CONFIG)

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: None,
    ) -> object:
        del operation_id, cache_key, config
        steps = current_task_steps()
        assert steps is not None
        return await steps.step(name, body)
