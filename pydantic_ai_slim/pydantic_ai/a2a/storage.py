"""This module defines the Storage class, which is responsible for storing and retrieving tasks."""

from abc import ABC, abstractmethod
from datetime import datetime

from .schema import Message, Task, TaskStatus


class Storage(ABC):
    """A storage to retrieve and save tasks."""

    @abstractmethod
    async def load_task(self, task_id: str, history_length: int | None = None) -> Task | None:
        """Load a task from storage.

        If the task is not found, return None.
        """

    @abstractmethod
    async def submit_task(self, id: str, session_id: str) -> Task:
        """Submit a task to storage."""

    @abstractmethod
    async def complete_task(self, task_id: str, message: Message) -> Task:
        """Save the result of a task."""


class InMemoryStorage(Storage):
    """A storage to retrieve and save tasks in memory."""

    def __init__(self):
        self.tasks: dict[str, Task] = {}

    async def load_task(self, task_id: str, history_length: int | None = None) -> Task | None:
        """Load a task from memory.

        Args:
            task_id: The id of the task to load.
            history_length: The number of messages to return in the history.

        Returns:
            The task.
        """
        if task_id not in self.tasks:
            return None

        task = self.tasks[task_id]
        if history_length and 'history' in task:
            task['history'] = task['history'][-history_length:]
        return task

    async def submit_task(self, id: str, session_id: str) -> Task:
        if id in self.tasks:
            raise ValueError(f'Task {id} already exists')

        task_status = TaskStatus(state='submitted', timestamp=datetime.now().isoformat())
        task = Task(id=id, session_id=session_id, status=task_status)
        self.tasks[id] = task
        return task

    async def complete_task(self, task_id: str, message: Message) -> Task:
        """Save the result of a task."""
        task = self.tasks[task_id]
        if 'history' not in task:
            task['history'] = []
        task['history'].append(message)
        return task
