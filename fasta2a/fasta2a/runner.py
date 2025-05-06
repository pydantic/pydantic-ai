from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .schema import Artifact, Message, TaskSendParams

if TYPE_CHECKING:
    from .worker import TaskContext


class Runner(ABC):
    """A runner is responsible for executing tasks."""

    @abstractmethod
    async def run(self, task_ctx: TaskContext[TaskSendParams]) -> None: ...

    @abstractmethod
    def build_message_history(self, task_history: list[Message]) -> list[Any]: ...

    @abstractmethod
    def build_artifacts(self, result: Any) -> list[Artifact]: ...
