"""This module defines the Storage class, which is responsible for storing and retrieving tasks."""

from __future__ import annotations as _annotations

from a2a.server.tasks import TaskStore
from a2a.types import Task


class InMemoryStorage(TaskStore):
    """A storage to retrieve and save tasks in memory."""

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    async def get(self, task_id: str) -> Task | None:
        """Load a task from memory.

        Args:
            task_id: The id of the task to load.

        Returns:
            The task.
        """
        return self.tasks.get(task_id)

    async def save(self, task: Task) -> None:
        """Saves or updates a task in the in-memory store."""
        self.tasks[task.id] = task

    async def delete(self, task_id: str) -> None:
        """Deletes a task from the in-memory store by ID."""
        if task_id in self.tasks:
            del self.tasks[task_id]
