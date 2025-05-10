"""This module defines the Storage class, which is responsible for storing and retrieving tasks."""

from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from datetime import datetime

from .schema import Artifact, Message, Task, TaskState, TaskStatus


class Storage(ABC):
    """A storage to retrieve and save tasks.

    The storage is used to update the status of a task and to save the result of a task.
    """

    @abstractmethod
    async def load_task(self, task_id: str, history_length: int | None = None) -> Task | None:
        """Load a task from storage.

        If the task is not found, return None.
        """

    @abstractmethod
    async def submit_task(self, task_id: str, session_id: str, message: Message) -> Task:
        """Submit a task to storage."""

    @abstractmethod
    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        message: Message | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> Task:
        """Update the state of a task."""


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

    async def submit_task(self, task_id: str, session_id: str, message: Message) -> Task:
        """Submit a task to storage."""
        if task_id in self.tasks:
            raise ValueError(f'Task {task_id} already exists')

        task_status = TaskStatus(state='submitted', timestamp=datetime.now().isoformat())
        task = Task(id=task_id, session_id=session_id, status=task_status, history=[message])
        self.tasks[task_id] = task
        return task

    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        message: Message | None = None,
        artifacts: list[Artifact] | None = None,
    ) -> Task:
        """Save the task as "working"."""
        task = self.tasks[task_id]
        task['status'] = TaskStatus(state=state, timestamp=datetime.now().isoformat())
        if message:
            if 'history' not in task:
                task['history'] = []
            task['history'].append(message)
        if artifacts:
            if 'artifacts' not in task:
                task['artifacts'] = []
            task['artifacts'].extend(artifacts)
        return task
