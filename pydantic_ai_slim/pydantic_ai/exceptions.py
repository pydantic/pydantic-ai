from __future__ import annotations as _annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from pydantic_core import core_schema

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover

if TYPE_CHECKING:
    from .messages import RetryPromptPart

__all__ = (
    'ModelRetry',
    'CallDeferred',
    'ApprovalRequired',
    'UserError',
    'AgentRunError',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'ModelAPIError',
    'ModelHTTPError',
    'IncompleteToolCall',
    'FallbackExceptionGroup',
)


class ModelRetry(Exception):
    """Exception to raise when a tool function should be retried.

    The agent will return the message to the model and ask it to try calling the function/tool again.
    """

    message: str
    """The message to return to the model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and other.message == self.message

    def __hash__(self) -> int:
        return hash((self.__class__, self.message))

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
        """Pydantic core schema to allow `ModelRetry` to be (de)serialized."""
        schema = core_schema.typed_dict_schema(
            {
                'message': core_schema.typed_dict_field(core_schema.str_schema()),
                'kind': core_schema.typed_dict_field(core_schema.literal_schema(['model-retry'])),
            }
        )
        return core_schema.no_info_after_validator_function(
            lambda dct: ModelRetry(dct['message']),
            schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: {'message': x.message, 'kind': 'model-retry'},
                return_schema=schema,
            ),
        )


class CallDeferred(Exception):
    """Exception to raise when a tool call should be deferred.

    See [tools docs](../deferred-tools.md#deferred-tools) for more information.

    Args:
        metadata: Optional dictionary of metadata to attach to the deferred tool call.
            This metadata will be available in `DeferredToolRequests.metadata` keyed by `tool_call_id`.
    """

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata = metadata
        super().__init__()


class ApprovalRequired(Exception):
    """Exception to raise when a tool call requires human-in-the-loop approval.

    See [tools docs](../deferred-tools.md#human-in-the-loop-tool-approval) for more information.

    Args:
        metadata: Optional dictionary of metadata to attach to the deferred tool call.
            This metadata will be available in `DeferredToolRequests.metadata` keyed by `tool_call_id`.
    """

    def __init__(self, metadata: dict[str, Any] | None = None):
        self.metadata = metadata
        super().__init__()


class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class AgentRunError(RuntimeError):
    """Base class for errors occurring during an agent run."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


class UsageLimitExceeded(AgentRunError):
    """Error raised when a Model's usage exceeds the specified limits."""


class UnexpectedModelBehavior(AgentRunError):
    """Error caused by unexpected Model behavior, e.g. an unexpected response code."""

    message: str
    """Description of the unexpected behavior."""
    body: str | None
    """The body of the response, if available."""

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


class ModelAPIError(AgentRunError):
    """Raised when a model provider API request fails."""

    model_name: str
    """The name of the model associated with the error."""

    def __init__(self, model_name: str, message: str):
        self.model_name = model_name
        super().__init__(message)


class ModelHTTPError(ModelAPIError):
    """Raised when an model provider response has a status code of 4xx or 5xx."""

    status_code: int
    """The HTTP status code returned by the API."""

    body: object | None
    """The body of the response, if available."""

    def __init__(self, status_code: int, model_name: str, body: object | None = None):
        self.status_code = status_code
        self.body = body
        message = f'status_code: {status_code}, model_name: {model_name}, body: {body}'
        super().__init__(model_name=model_name, message=message)


class FallbackExceptionGroup(ExceptionGroup[Any]):
    """A group of exceptions that can be raised when all fallback models fail."""


class ToolRetryError(Exception):
    """Exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: RetryPromptPart):
        self.tool_retry = tool_retry
        super().__init__(_format_tool_retry_message(tool_retry))


def _format_tool_retry_message(tool_retry: RetryPromptPart) -> str:
    """Format a human-readable error message for ToolRetryError."""
    content = tool_retry.content
    if isinstance(content, str):
        return f"Tool '{tool_retry.tool_name}' failed: {content}"

    # Format list of ErrorDetails in a human-readable way
    error_count = len(content)
    error_word = 'error' if error_count == 1 else 'errors'
    lines = [f"Tool '{tool_retry.tool_name}' failed: {error_count} validation {error_word}"]
    for error in content:
        loc = '.'.join(str(loc) for loc in error['loc'])
        lines.append(f'  {loc}: {error["msg"]}')
    return '\n'.join(lines)


class IncompleteToolCall(UnexpectedModelBehavior):
    """Error raised when a model stops due to token limit while emitting a tool call."""
