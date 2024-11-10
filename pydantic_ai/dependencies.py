from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = 'AgentDeps', 'CallContext'

AgentDeps = TypeVar('AgentDeps')
"""Type variable for agent dependencies."""


@dataclass
class CallContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    """Dependencies for the agent."""
    retry: int
    """Number of retries so far."""
    tool_name: str | None
    """Name of the tool being called."""
