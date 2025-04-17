"""This module defines the Storage class, which is responsible for storing and retrieving tasks."""

from abc import ABC, abstractmethod

from .schema import Task


class Storage(ABC):
    """A storage to retrieve and save tasks."""

    @abstractmethod
    async def load_task(self, task_id: str, history_length: int | None = None) -> Task | None:
        """Load a task from storage.

        If the task is not found, return None.
        """

    @abstractmethod
    async def save_task(self, task: Task): ...


class InMemoryStorage(Storage):
    """A storage to retrieve and save tasks in memory."""

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    async def load_task(self, task_id: str, history_length: int | None = None) -> Task:
        """Load a task from memory.

        Args:
            task_id: The id of the task to load.
            history_length: The number of messages to return in the history.

        Returns:
            The task.
        """
        task = self.tasks[task_id]
        if history_length and 'history' in task:
            task['history'] = task['history'][-history_length:]
        return task

    async def save_task(self, task: Task):
        self.tasks[task['id']] = task
