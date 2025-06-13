from __future__ import annotations as _annotations

from a2a.server.agent_execution import AgentExecutor
from a2a.server.tasks import TaskStore

Worker = AgentExecutor
"""
The `Worker` is the core component where you implement your agent's logic.

It is an alias for the `a2a.server.agent_execution.AgentExecutor` class from the 
Google A2A SDK. You should create a class that inherits from `Worker` and
implement the `execute` and `cancel` methods.
"""

__all__ = ["Worker", "TaskStore"]
