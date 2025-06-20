from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from a2a.server.tasks import TaskUpdater

    from .schema import Artifact, Message, TaskIdParams, TaskSendParams
    from .storage import Storage


class Worker(ABC):
    """A worker is responsible for executing tasks."""

    storage: Storage

    @abstractmethod
    async def run_task(self, params: TaskSendParams, updater: TaskUpdater) -> None: ...

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams, updater: TaskUpdater) -> None: ...

    @abstractmethod
    def build_message_history(self, task_history: list[Message]) -> list[Any]: ...

    @abstractmethod
    def build_artifacts(self, result: Any) -> list[Artifact]: ...
