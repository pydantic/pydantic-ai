from __future__ import annotations as _annotations

import json
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import pydantic_core
from pydantic_core import core_schema

from ._warnings import (
    CostCalculationFailedWarning as CostCalculationFailedWarning,
    CostNotFoundWarning as CostNotFoundWarning,
    PydanticAIDeprecationWarning as PydanticAIDeprecationWarning,
)

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup as ExceptionGroup  # pragma: lax no cover
else:
    ExceptionGroup = ExceptionGroup  # pragma: lax no cover


if TYPE_CHECKING:
    from .messages import ModelMessage, ModelResponse, RetryPromptPart, ToolReturnPart
    from .usage import RunUsage

__all__ = (
    'ModelRetry',
    'CallDeferred',
    'ApprovalRequired',
    'SkipModelRequest',
    'SkipToolValidation',
    'SkipToolExecution',
    'UserError',
    'UndrainedPendingMessagesError',
    'AgentRunError',
    'RunCancelled',
    'SuspendedResponseExpired',
    'UnexpectedModelBehavior',
    'UsageLimitExceeded',
    'ConcurrencyLimitExceeded',
    'ModelAPIError',
    'ModelHTTPError',
    'ContentFilterError',
    'IncompleteToolCall',
    'MessageHistoryMutatedWarning',
    'CostCalculationFailedWarning',
    'CostNotFoundWarning',
    'PydanticAIDeprecationWarning',
    'FallbackExceptionGroup',
    'ToolFailed',
)


class ModelRetry(Exception):
    """Exception to raise to request a model retry.

    Can be raised from tool functions, output validators, and capability hooks
    (such as `after_model_request`, `after_tool_execute`, etc.) to send
    a retry prompt back to the model asking it to try again.

    For a terminal failure the model should see but not retry, raise
    [`ToolFailed`][pydantic_ai.exceptions.ToolFailed] instead.
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


class ToolFailed(Exception):
    """Exception to raise to report a terminal tool failure to the model.

    Raise this when a tool call is done and has failed — a missing resource, an unsupported
    operation, a definitive upstream error — and you want the model to see the failure
    and adapt rather than try the same call again. Can be raised from tool functions, args
    validators, and tool validation/execution hooks.

    Like [`ModelRetry`][pydantic_ai.exceptions.ModelRetry], this produces a failed tool result the
    model sees; unlike `ModelRetry` it does not prepend retry/correction instructions and does not
    consume the tool's retry budget. Bound repeated failures with
    [`UsageLimits`][pydantic_ai.usage.UsageLimits] at the run level instead.
    """

    message: str
    """The failure message to return to the model."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and other.message == self.message

    def __hash__(self) -> int:
        return hash((self.__class__, self.message))

    @classmethod
    def __get_pydantic_core_schema__(cls, _: Any, __: Any) -> core_schema.CoreSchema:
        """Pydantic core schema to allow `ToolFailed` to be (de)serialized."""
        serialized_schema = core_schema.typed_dict_schema(
            {
                'message': core_schema.typed_dict_field(core_schema.str_schema()),
                'kind': core_schema.typed_dict_field(core_schema.literal_schema(['tool-failed'])),
            }
        )
        deserialization_schema = core_schema.no_info_after_validator_function(
            lambda dct: cls(dct['message']),
            serialized_schema,
        )
        return core_schema.json_or_python_schema(
            json_schema=deserialization_schema,
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), deserialization_schema]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: {'message': x.message, 'kind': 'tool-failed'},
                return_schema=serialized_schema,
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

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.metadata,)


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

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.metadata,)


class SkipModelRequest(Exception):
    """Exception to raise in before/wrap model request hooks to skip the model call.

    The provided response will be used instead of calling the model.

    Note: when raised in `before_model_request`, any message history modifications
    made by earlier capabilities in that hook will not be persisted to the agent's
    message history, since the request preparation is aborted.
    """

    response: ModelResponse

    def __init__(self, response: ModelResponse):
        self.response = response
        super().__init__()


class SkipToolValidation(Exception):
    """Exception to raise in before/wrap tool validate hooks to skip validation.

    The provided args will be used as the validated arguments.
    """

    validated_args: dict[str, Any]

    def __init__(self, validated_args: dict[str, Any]):
        self.validated_args = validated_args
        super().__init__()


class SkipToolExecution(Exception):
    """Exception to raise in before/wrap tool execute hooks to skip execution.

    The provided result will be used as the tool result.
    """

    result: Any

    def __init__(self, result: Any):
        self.result = result
        super().__init__()


class UserError(RuntimeError):
    """Error caused by a usage mistake by the application developer — You!"""

    message: str
    """Description of the mistake."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class UndrainedPendingMessagesError(UserError):
    """Error that used to be raised when an agent run ended with messages still queued via `enqueue`.

    A bare `async for node in agent_run` loop used to skip the node hooks, so `'when_idle'`
    messages and end-of-run redirects (which drain in `after_node_run`) were stranded. Bare
    iteration now advances through [`AgentRun.next()`][pydantic_ai.run.AgentRun.next] like every
    other way of driving a run, so pending messages always drain and this error is no longer
    raised. It is kept so existing `except` clauses keep working.
    """


class AgentRunError(RuntimeError):
    """Base class for errors occurring during an agent run."""

    message: str
    """The error message."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        return self.message


_RUN_CANCELLED_ATTR = '_pydantic_ai_run_cancelled'


class RunCancelled(AgentRunError):
    """Raised when the agent run was cancelled by the application itself.

    Raised by [`AgentRun.cancel()`][pydantic_ai.run.AgentRun.cancel] and
    [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel].
    This is a normal, catchable application-level outcome: the run stopped because your own code
    asked it to. External cancellation of the task running the agent (`asyncio.Task.cancel()`,
    a timeout scope, workflow cancellation under durable execution) is infrastructure-level and
    keeps propagating as `asyncio.CancelledError` instead — it is never translated into this
    exception, and when both race, the external cancellation wins. (On Python 3.10, which lacks
    `Task.uncancel()`, the race cannot be disambiguated and a requested first-party cancellation
    wins instead.)

    Everything the run completed before the cancellation took effect — including the partial
    response of an interrupted stream and the results of tool calls that finished — is preserved
    in [`all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages]: pass it as
    `message_history` to a new run (with a new user prompt or not) to resume the conversation; any
    tool calls that never produced a result are automatically closed out with synthesized
    `outcome='interrupted'` returns before the history is sent to a model.

    Cancellation is terminal: capability hooks (`wrap_run`, `wrap_node_run`, `on_run_error`) may
    observe it and clean up, but cannot recover a cancelled run into a successful result.
    """

    def __init__(
        self,
        message: str,
        *,
        messages: Sequence[ModelMessage] = (),
        new_message_index: int = 0,
        usage: RunUsage | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ):
        if usage is None:
            from .usage import RunUsage

            usage = RunUsage()
        self._messages = list(messages)
        self._new_message_index = new_message_index
        self._usage = usage
        self._metadata = metadata
        self._run_id = run_id
        self._conversation_id = conversation_id
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple[str], dict[str, Any]]:
        return self.__class__, (self.message,), self.__dict__

    def _attach_to(self, exc: BaseException) -> None:
        setattr(exc, _RUN_CANCELLED_ATTR, self)

    @classmethod
    def from_cancellation(cls, exc: BaseException) -> RunCancelled | None:
        """Recover run state from a cancellation-related exception.

        External cancellation of a plain `agent.run()` keeps its standard asyncio semantics. Catch
        it with `except asyncio.CancelledError as exc`, then call
        `RunCancelled.from_cancellation(exc)` to access the partial run state attached by Pydantic
        AI. This also works with the `TimeoutError` raised by `asyncio.timeout()` or
        `asyncio.wait_for()`, whose exception chain contains the original `CancelledError`. An
        external `CancelledError` must keep propagating for timeouts and task groups to tear down
        correctly, so re-raise it after capturing the state rather than returning from the handler;
        only a first-party `RunCancelled` is yours to consume.

        Passing a `RunCancelled` directly returns the same instance, providing uniform handling for
        first-party and external cancellation paths.

        Python 3.11+ preserves the exception instance across an `await task` boundary. Python 3.10
        recreates the `CancelledError` there, but chains the original exception — and the attached
        run state — via `__context__`, which this method traverses; the chain is attached only to
        the first `await` of the cancelled task, so later awaits of the same task see an unchained
        exception. Use `capture_run_messages()` as the fallback when only message history is needed.
        """
        pending = [exc]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            if isinstance(current, cls):
                return current
            attached = getattr(current, _RUN_CANCELLED_ATTR, None)
            if isinstance(attached, cls):
                return attached

            if current.__cause__ is not None:
                pending.append(current.__cause__)
            if current.__context__ is not None:
                pending.append(current.__context__)
        return None

    def all_messages(self) -> list[ModelMessage]:
        """Return the complete resumable history of the cancelled run.

        This is a DETACHED snapshot of the run's message history at termination, ready to pass as
        `message_history` for a resumed run.

        Returns:
            List of messages.
        """
        return self._messages

    def all_messages_json(self) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.exceptions.RunCancelled.all_messages] as JSON bytes.

        Returns:
            JSON bytes representing the messages.
        """
        from .messages import ModelMessagesTypeAdapter

        return ModelMessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[ModelMessage]:
        """Return the messages produced during the cancelled run.

        Messages provided via `message_history` and messages from older runs are excluded.

        Returns:
            List of new messages.
        """
        return self._messages[self._new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.exceptions.RunCancelled.new_messages] as JSON bytes.

        Returns:
            JSON bytes representing the new messages.
        """
        from .messages import ModelMessagesTypeAdapter

        return ModelMessagesTypeAdapter.dump_json(self.new_messages())

    @property
    def response(self) -> ModelResponse:
        """Return the last response from the message history.

        Raises:
            ValueError: If the run was cancelled before receiving any model response.
        """
        from .messages import ModelResponse

        for message in reversed(self.all_messages()):
            if isinstance(message, ModelResponse):
                return message
        raise ValueError('No response found in the message history')

    @property
    def timestamp(self) -> datetime:
        """Return the timestamp of the last response.

        Raises:
            ValueError: If the run was cancelled before receiving any model response.
        """
        return self.response.timestamp

    @property
    def usage(self) -> RunUsage:
        """Return the usage of the cancelled run."""
        return self._usage

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        return self._metadata

    @property
    def run_id(self) -> str | None:
        """The unique identifier for the agent run, or `None` if it was cancelled before starting."""
        return self._run_id

    @property
    def conversation_id(self) -> str | None:
        """The conversation identifier, or `None` if the run was cancelled before starting."""
        return self._conversation_id


class SuspendedResponseExpired(AgentRunError):
    """Raised when resuming a suspended response whose server-side job is no longer available.

    Suspended/background jobs are only resumable within the provider's retention window (e.g. ~10
    minutes for OpenAI background mode). Resuming a persisted suspended response after that window
    raises this instead of an opaque provider HTTP error; start a new run from the preceding messages
    to retry from scratch.
    """


class UsageLimitExceeded(AgentRunError):
    """Error raised when a Model's usage exceeds the specified limits."""

    _HINT = (
        'Consider raising the limit, or see the docs on usage limits '
        'for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits'
    )

    def __init__(self, message: str):
        # Idempotent so reconstruction via `UsageLimitExceeded(*args)` (e.g. unpickling) doesn't re-append the hint.
        if self._HINT not in message:
            message = f'{message.removesuffix(".")}. {self._HINT}'
        super().__init__(message)


class ConcurrencyLimitExceeded(AgentRunError):
    """Error raised when the concurrency queue depth exceeds max_queued."""


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

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.message, self.body)

    def __str__(self) -> str:
        if self.body:
            return f'{self.message}, body:\n{self.body}'
        else:
            return self.message


class ContentFilterError(UnexpectedModelBehavior):
    """Raised when content filtering is triggered by the model provider."""


class ModelAPIError(AgentRunError):
    """Raised when a model provider API request fails."""

    model_name: str
    """The name of the model associated with the error."""

    def __init__(self, model_name: str, message: str):
        self.model_name = model_name
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.model_name, self.message)


class ModelHTTPError(ModelAPIError):
    """Raised when a model provider response has a status code of 4xx or 5xx."""

    status_code: int
    """The HTTP status code returned by the API."""

    body: object | None
    """The body of the response, if available."""

    headers: dict[str, str] | None
    """Response headers from the provider, with keys lowercased for consistent access.

    For example, use `exc.headers.get('retry-after')` to read the `Retry-After` header
    regardless of provider casing.  `None` when the provider does not supply headers
    (e.g. gRPC-based providers or synthesised errors).
    """

    suggested_model_id: str | None
    """A close known model identifier suggested from a provider-confirmed model-name error."""

    def __init__(
        self,
        status_code: int,
        model_name: str,
        body: object | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        suggested_model_id: str | None = None,
    ):
        self.status_code = status_code
        self.body = body
        self.headers = {k.lower(): v for k, v in headers.items()} if headers is not None else None
        self.suggested_model_id = suggested_model_id
        message = f'status_code: {status_code}, model_name: {model_name}, body: {body}'
        if suggested_model_id is not None:
            message += f'. Did you mean {suggested_model_id!r}?'
        super().__init__(model_name=model_name, message=message)

    def __reduce__(self) -> tuple[type, tuple[Any, ...], dict[str, Any]]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return (
            self.__class__,
            (self.status_code, self.model_name, self.body),
            {
                'headers': self.headers,
                'suggested_model_id': self.suggested_model_id,
            },
        )

    def __setstate__(self, state: dict[str, Any]) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        self.headers = state.get('headers')
        self.suggested_model_id = state.get('suggested_model_id')
        if self.suggested_model_id is not None:
            self.message += f'. Did you mean {self.suggested_model_id!r}?'
            self.args = (self.message,)

    @property
    def retry_after(self) -> float | None:
        """Seconds to wait before retrying, parsed from the `Retry-After` response header.

        Returns `None` when the header is absent or cannot be parsed. The header value
        is interpreted first as an integer number of seconds, then as an
        [HTTP-date](https://httpwg.org/specs/rfc9110.html#http.date) string.
        """
        if self.headers is None:
            return None
        raw = self.headers.get('retry-after')
        if raw is None:
            return None
        try:
            seconds = int(raw)
            if seconds < 0:
                return None
            return float(seconds)
        except (ValueError, OverflowError):
            pass
        try:
            retry_time = parsedate_to_datetime(raw)
            assert isinstance(retry_time, datetime)
            # asctime-date format (RFC 9110 §5.6.7) carries no timezone; treat as UTC.
            if retry_time.tzinfo is None:
                retry_time = retry_time.replace(tzinfo=timezone.utc)
            wait = (retry_time - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, wait)
        except (ValueError, TypeError, AssertionError):
            return None


class FallbackExceptionGroup(ExceptionGroup[Any]):
    """A group of exceptions that can be raised when all fallback models fail."""


class ToolRetryError(Exception):
    """Exception used to signal a `ToolRetry` message should be returned to the LLM."""

    def __init__(self, tool_retry: RetryPromptPart):
        self.tool_retry = tool_retry
        message = (
            tool_retry.content
            if isinstance(tool_retry.content, str)
            else self._format_error_details(tool_retry.content, tool_retry.tool_name)
        )
        super().__init__(message)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.tool_retry,)

    @staticmethod
    def _format_error_details(errors: list[pydantic_core.ErrorDetails], tool_name: str | None) -> str:
        """Format ErrorDetails as a human-readable message.

        We format manually rather than using ValidationError.from_exception_data because
        some error types (value_error, assertion_error, etc.) require an 'error' key in ctx,
        but when ErrorDetails are serialized, exception objects are stripped from ctx.
        The 'msg' field already contains the human-readable message, so we use that directly.
        """
        error_count = len(errors)
        lines = [
            f'{error_count} validation error{"" if error_count == 1 else "s"}{f" for {tool_name!r}" if tool_name else ""}'
        ]
        for e in errors:
            loc = '.'.join(str(x) for x in e['loc']) if e['loc'] else '__root__'
            lines.append(loc)
            lines.append(f'  {e["msg"]} [type={e["type"]}, input_value={e["input"]!r}]')
        return '\n'.join(lines)


class ToolFailedError(Exception):
    """Exception used to signal a failed `ToolReturnPart` should be returned to the LLM."""

    def __init__(self, tool_failed: ToolReturnPart):
        self.tool_failed = tool_failed
        # `content` may be non-`str` (a structured object or multimodal sequence), so stringify it
        # without the model-facing error wrapper in the human-readable exception message.
        super().__init__(tool_failed.model_response_str(wrap_if_error=False))

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.tool_failed,)


class IncompleteToolCall(UnexpectedModelBehavior):
    """Error raised when a model stops due to token limit while emitting a tool call."""


class MessageHistoryMutatedWarning(Warning):
    """Warning raised when in-place mutation of the message history is detected at the end of a run.

    Mutating messages that are already part of the run's history in place (e.g.
    `ctx.messages[0].parts[0].content = '...'` from a tool) is not supported: the per-request
    `gen_ai.input.messages` span attribute caches each message's serialized form, so spans recorded
    after the mutation may not match the messages actually sent to the model. The run-level
    `pydantic_ai.all_messages` attribute is always serialized fresh and does reflect the mutation.
    To transform history mid-run, build new message or part objects instead — e.g. with
    `dataclasses.replace`, passing the message a new `parts` list (replacing a message in the
    history and reassigning its `parts` list are both safe) — for instance in a history processor
    ([`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory]).

    The warning is best-effort: it's raised when a mutation is detected at the end of a successful
    run, which covers messages still present in the final history. Errored runs aren't checked —
    with warnings configured as errors, the warning would displace the run's own exception. Its
    absence does not guarantee that no stale span was recorded.
    """
