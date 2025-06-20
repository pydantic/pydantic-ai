"""This module defines the Storage class, which is responsible for storing and retrieving tasks."""

from __future__ import annotations as _annotations

import asyncio
from abc import ABC, abstractmethod

from a2a.types import Task


class Storage(ABC):
    """A storage to retrieve and save tasks."""

    @abstractmethod
    async def get(self, task_id: str) -> Task | None:
        """Retrieves a task from the store by its ID."""

    @abstractmethod
    async def save(self, task: Task) -> None:
        """Saves or updates a task in the store."""

    @abstractmethod
    async def delete(self, task_id: str) -> None:
        """Deletes a task from the store by its ID."""


class InMemoryStorage(Storage):
    """A storage to retrieve and save tasks in memory."""

    def __init__(self) -> None:
        self.tasks: dict[str, Task] = {}
        self.lock = asyncio.Lock()

    async def get(self, task_id: str) -> Task | None:
        """Retrieves a task from memory."""
        async with self.lock:
            return self.tasks.get(task_id)

    async def save(self, task: Task) -> None:
        """Saves or updates a task in memory."""
        async with self.lock:
            self.tasks[task.id] = task

    async def delete(self, task_id: str) -> None:
        """Deletes a task from memory."""
        async with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
