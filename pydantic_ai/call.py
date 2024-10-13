from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

AgentDeps = TypeVar('AgentDeps')


@dataclass
class CallContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    retry: int


class Retry(Exception):
    """Exception raised when a retriever function should be retried."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
