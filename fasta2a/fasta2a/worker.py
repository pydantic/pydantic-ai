from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

if TYPE_CHECKING:
    from .broker import Broker, TaskContext
    from .schema import Artifact, Message, TaskSendParams
    from .storage import Storage


@dataclass
class Worker(ABC):
    """A runner is responsible for executing tasks."""

    broker: Broker
    storage: Storage

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def run(self, task_ctx: TaskContext[TaskSendParams]) -> None: ...

    @abstractmethod
    async def cancel(self, task_ctx: TaskContext[TaskSendParams]) -> None: ...

    @abstractmethod
    def build_message_history(self, task_history: list[Message]) -> list[Any]: ...

    @abstractmethod
    def build_artifacts(self, result: Any) -> list[Artifact]: ...


@dataclass
class InMemoryWorker(Worker):
    """A worker that executes tasks in memory."""

    async def __aenter__(self) -> Self:
        return self
