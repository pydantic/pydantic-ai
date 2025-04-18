"""This module defines the Scheduler interface, which is responsible for scheduling tasks to workers.

The scheduler can work with both local and remote workers, providing a consistent interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self

from .schema import TaskIdParams, TaskSendParams
from .worker import InMemoryWorker


class Scheduler(ABC):
    """Scheduler interface for dispatching tasks to workers.

    The scheduler is responsible for determining when to send tasks to workers
    and provides a consistent interface regardless of whether the worker is local or remote.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """Enter the context manager."""

    @abstractmethod
    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        """Exit the context manager."""

    @abstractmethod
    async def schedule_task(self, params: TaskSendParams) -> None:
        """Schedule a task to be executed by a worker.

        Args:
            params: Parameters for the task to be scheduled.
        """

    @abstractmethod
    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task that has been scheduled.

        Args:
            params: Parameters identifying the task to cancel.
        """


@dataclass
class InMemoryScheduler(Scheduler):
    """A scheduler that directly dispatches tasks to a local worker."""

    worker: InMemoryWorker

    async def __aenter__(self) -> Self:
        """Enter the context manager."""
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        """Exit the context manager."""
        pass

    async def schedule_task(self, params: TaskSendParams) -> None:
        """Schedule a task on the local worker.

        Args:
            params: Parameters for the task to be scheduled.
        """
        await self.worker.run_task(params)

    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task on the local worker.

        Args:
            params: Parameters identifying the task to cancel.
        """
        await self.worker.cancel_task(params)
