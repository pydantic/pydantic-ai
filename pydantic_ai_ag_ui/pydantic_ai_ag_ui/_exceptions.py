"""Exceptions for the AI Agent UI module."""

from __future__ import annotations

from dataclasses import InitVar, dataclass

from pydantic import ValidationError as PydanticValidationError


@dataclass
class RunError(Exception):
    """Exception raised for errors during agent runs."""

    message: str
    code: str

    def __str__(self) -> str:
        return self.message


@dataclass(kw_only=True)
class UnexpectedToolCallError(RunError):
    """Exception raised when an unexpected tool call is encountered."""

    tool_name: InitVar[str]
    message: str = ''
    code: str = 'unexpected_tool_call'

    def __post_init__(self, tool_name: str) -> None:
        """Set the message for the unexpected tool call.

        Args:
            tool_name: The name of the tool that was unexpectedly called.
        """
        self.message = f'unexpected tool call name={tool_name}'  # pragma: no cover


@dataclass
class NoMessagesError(RunError):
    """Exception raised when no messages are found in the input."""

    message: str = 'no messages found in the input'
    code: str = 'no_messages'


@dataclass
class InvalidStateError(RunError, PydanticValidationError):
    """Exception raised when an invalid state is provided."""

    message: str = 'invalid state provided'
    code: str = 'invalid_state'
