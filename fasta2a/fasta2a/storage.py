"""
This module provides an alias for the TaskStore abstraction from the `google-a2a` SDK.
A TaskStore is responsible for persisting and retrieving task state.
"""

from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.server.tasks.task_store import TaskStore

Storage = TaskStore
"""Alias for `a2a.server.tasks.task_store.TaskStore`."""

__all__ = ["Storage", "TaskStore", "InMemoryTaskStore"]
