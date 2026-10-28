from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING

import anyio

if TYPE_CHECKING:
    from . import Model


@dataclass(eq=False)
class ModelContext:
    model: Model
    task_id: int | None = field(default=None, init=False)

    async def __aenter__(self) -> Model:
        self.task_id = anyio.get_current_task().id
        token = _current_model_context.set(self)
        try:
            return await self.model.__aenter__()
        finally:
            self.task_id = None
            _current_model_context.reset(token)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        self.task_id = anyio.get_current_task().id
        token = _current_model_context.set(self)
        try:
            return await self.model.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            self.task_id = None
            _current_model_context.reset(token)


_current_model_context: ContextVar[ModelContext | None] = ContextVar('_current_model_context', default=None)


def model_context_key() -> object | None:
    task_id = anyio.get_current_task().id
    context = _current_model_context.get()
    # A task spawned by a lifecycle hook inherits its ContextVar, but owns its own model contexts.
    if context is not None and context.task_id == task_id:
        return context
    # Direct model contexts carry no scope identity: a task can close another task's exit stack.
    return None
