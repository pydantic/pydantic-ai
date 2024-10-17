from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from . import messages

__all__ = 'AgentDeps', 'Retry', 'CallContext', 'AgentError', 'UserError', 'UnexpectedModelBehaviour'

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


class AgentError(RuntimeError):
    """Exception raised when an Agent run fails due to a problem with the LLM being used or.

    This exception should always have a cause which you can access to find out what went wrong, it exists so you
    can access the history of messages when the error occurred.
    """

    history: list[messages.Message]
    agent_name: str

    def __init__(self, history: list[messages.Message], agent_name: str):
        self.history = history
        self.agent_name = agent_name
        super().__init__(f'Model error while running {agent_name}')


class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnexpectedModelBehaviour(RuntimeError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code"""

    def __init__(self, message: str, body: str | None = None):
        self.message = message
        if body is None:
            self.body: str | None = None
        else:
            try:
                self.body = json.dumps(json.loads(body), indent=2)
            except ValueError:
                self.body = body
        super().__init__(message, body)
