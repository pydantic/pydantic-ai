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
    async def submit_task(self, task_id: str, context_id: str, message: Message) -> Task:
        """Submit a task to storage."""

    @abstractmethod
    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        artifacts: list[Artifact] | None = None,
    ) -> Task:
        """Update the state of a task."""

    @abstractmethod
    async def add_message(self, message: Message) -> None:
        """Add a message to the history for both its task and context.

        This should be called for messages created during task execution,
        not for the initial message (which is handled by submit_task).
        """

    @abstractmethod
    async def get_context_history(self, context_id: str, history_length: int | None = None) -> list[Message]:
        """Get all messages across tasks in a context."""


class InMemoryStorage(Storage):
    """A storage to retrieve and save tasks in memory."""

    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.context_messages: dict[str, list[Message]] = {}

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

    async def submit_task(self, task_id: str, context_id: str, message: Message) -> Task:
        """Submit a task to storage."""
        if task_id in self.tasks:
            raise ValueError(f'Task {task_id} already exists')

        # Add IDs to the message
        message['task_id'] = task_id
        message['context_id'] = context_id

        task_status = TaskStatus(state='submitted', timestamp=datetime.now().isoformat())
        task = Task(id=task_id, context_id=context_id, kind='task', status=task_status, history=[message])
        self.tasks[task_id] = task

        # Add message to context storage directly (not via add_message to avoid duplication)
        if context_id not in self.context_messages:
            self.context_messages[context_id] = []
        self.context_messages[context_id].append(message)

        return task

    async def update_task(
        self,
        task_id: str,
        state: TaskState,
        artifacts: list[Artifact] | None = None,
    ) -> Task:
        """Update the state of a task."""
        task = self.tasks[task_id]
        task['status'] = TaskStatus(state=state, timestamp=datetime.now().isoformat())
        if artifacts:
            if 'artifacts' not in task:
                task['artifacts'] = []
            task['artifacts'].extend(artifacts)
        return task

    async def add_message(self, message: Message) -> None:
        """Add a message to the history for both its task and context."""
        if 'task_id' in message and message['task_id']:
            task_id = message['task_id']
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if 'history' not in task:
                    task['history'] = []
                task['history'].append(message)

        if 'context_id' in message and message['context_id']:
            context_id = message['context_id']
            if context_id not in self.context_messages:
                self.context_messages[context_id] = []
            self.context_messages[context_id].append(message)

    async def get_context_history(self, context_id: str, history_length: int | None = None) -> list[Message]:
        """Get all messages across tasks in a context."""
        messages = self.context_messages.get(context_id, [])
        if history_length:
            return messages[-history_length:]
        return messages
