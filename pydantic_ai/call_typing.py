from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = 'AgentDeps', 'CallContext'

AgentDeps = TypeVar('AgentDeps')


@dataclass
class CallContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    retry: int
    tool_name: str | None
