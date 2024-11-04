from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import ValidationError

from . import messages

if TYPE_CHECKING:
    from .models import Model

__all__ = (
    'AgentDeps',
    'ResultData',
    'Cost',
    'RunResult',
    'ModelRetry',
    'CallContext',
    'AgentError',
    'UserError',
    'UnexpectedModelBehaviour',
)

AgentDeps = TypeVar('AgentDeps')
ResultData = TypeVar('ResultData')


@dataclass
class Cost:
    """Cost of a request or run."""

    request_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    details: dict[str, int] | None = None

    def __add__(self, other: Cost) -> Cost:
        counts: dict[str, int] = {}
        for field in 'request_tokens', 'response_tokens', 'total_tokens':
            self_value = getattr(self, field)
            other_value = getattr(other, field)
            if self_value is not None or other_value is not None:
                counts[field] = (self_value or 0) + (other_value or 0)

        details = self.details.copy() if self.details is not None else None
        if other.details is not None:
            details = details or {}
            for key, value in other.details.items():
                details[key] = details.get(key, 0) + value

        return Cost(**counts, details=details or None)


@dataclass
class RunResult(Generic[ResultData]):
    """Result of a run."""

    response: ResultData
    all_messages: list[messages.Message]
    new_message_index: int
    cost: Cost

    def all_messages_json(self) -> bytes:
        """Return the history of messages as JSON bytes."""
        return messages.MessagesTypeAdapter.dump_json(self.all_messages)

    def new_messages(self) -> list[messages.Message]:
        """Return new messages associated with this run.

        System prompts and any messages from older runs are excluded.
        """
        return self.all_messages[self.new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [new_messages][] as JSON bytes."""
        return messages.MessagesTypeAdapter.dump_json(self.new_messages())


@dataclass
class CallContext(Generic[AgentDeps]):
    """Information about the current call."""

    deps: AgentDeps
    retry: int
    tool_name: str | None


class ModelRetry(Exception):
    """Exception raised when a retriever function should be retried.

    The agent will return the message to the model and ask it to try calling the function/tool again.
    """

    message: str

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

    def __init__(self, history: list[messages.Message], model: Model):
        self.history = history
        self.model_name = model.name()
        super().__init__(f'Error while running model {self.model_name}')

    def cause(self) -> ValidationError | UnexpectedModelBehaviour:
        """This is really just typing super and improved find-ability for `Exception.__cause__`."""
        cause = self.__cause__
        if isinstance(cause, (ValidationError, UnexpectedModelBehaviour)):
            return cause
        else:
            raise TypeError(
                f'Unexpected cause type for AgentError: {type(cause)}, '
                f'expected ValidationError or UnexpectedModelBehaviour'
            )

    def __str__(self) -> str:
        msg = super().__str__()
        cause = self.__cause__
        if isinstance(cause, UnexpectedModelBehaviour):
            return f'{msg}\n  caused by unexpected model behavior: {cause.message}'
        elif isinstance(cause, ValidationError):
            summary = str(cause).split('\n', 1)[0]
            return f'{msg}\n  caused by: {summary}'
        else:
            return msg


class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    message: str

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UnexpectedModelBehaviour(RuntimeError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code."""

    message: str
    body: str | None

    def __init__(self, message: str, body: str | None = None):
        self.message = message
        if body is None:
            self.body: str | None = None
        else:
            try:
                self.body = json.dumps(json.loads(body), indent=2)
            except ValueError:
                self.body = body
        super().__init__(message)

    def __str__(self) -> str:
        if self.body:
            return f'{self.message}, body:\n{self.body}'
        else:
            return self.message
