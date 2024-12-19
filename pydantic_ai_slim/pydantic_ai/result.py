from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, TypeVar, cast

import logfire_api

from . import _result, _utils, exceptions, messages as _messages, models
from .settings import UsageLimits
from .tools import AgentDeps, RunContext

__all__ = (
    'ResultData',
    'Usage',
    'RunResult',
    'StreamedRunResult',
)


ResultData = TypeVar('ResultData')
"""Type variable for the result data of a run."""

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')


@dataclass
class Usage:
    """LLM usage associated to a request or run.

    Responsibility for calculating usage is on the model; PydanticAI simply sums the usage information across requests.

    You'll need to look up the documentation of the model you're using to convert usage to monetary costs.
    """

    requests: int = 0
    """Number of requests made."""
    request_tokens: int | None = None
    """Tokens used in processing requests."""
    response_tokens: int | None = None
    """Tokens used in generating responses."""
    total_tokens: int | None = None
    """Total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`."""
    details: dict[str, int] | None = None
    """Any extra details returned by the model."""

    def __add__(self, other: Usage) -> Usage:
        """Add two Usages together.

        This is provided so it's trivial to sum usage information from multiple requests and runs.
        """
        counts: dict[str, int] = {}
        for f in 'requests', 'request_tokens', 'response_tokens', 'total_tokens':
            self_value = getattr(self, f)
            other_value = getattr(other, f)
            if self_value is not None or other_value is not None:
                counts[f] = (self_value or 0) + (other_value or 0)

        details = self.details.copy() if self.details is not None else None
        if other.details is not None:
            details = details or {}
            for key, value in other.details.items():
                details[key] = details.get(key, 0) + value

        return Usage(**counts, details=details or None)


@dataclass
class _BaseRunResult(ABC, Generic[ResultData]):
    """Base type for results.

    You should not import or use this type directly, instead use its subclasses `RunResult` and `StreamedRunResult`.
    """

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

    def all_messages(self) -> list[_messages.ModelMessage]:
        """Return the history of _messages."""
        # this is a method to be consistent with the other methods
        return self._all_messages

    def all_messages_json(self) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.result._BaseRunResult.all_messages] as JSON bytes."""
        return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        System prompts and any messages from older runs are excluded.
        """
        return self.all_messages()[self._new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.result._BaseRunResult.new_messages] as JSON bytes."""
        return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages())

    @abstractmethod
    def usage(self) -> Usage:
        raise NotImplementedError()


@dataclass
class RunResult(_BaseRunResult[ResultData]):
    """Result of a non-streamed run."""

    data: ResultData
    """Data from the final response in the run."""
    _usage: Usage

    def usage(self) -> Usage:
        """Return the usage of the whole run."""
        return self._usage


@dataclass
class StreamedRunResult(_BaseRunResult[ResultData], Generic[AgentDeps, ResultData]):
    """Result of a streamed run that returns structured data via a tool call."""

    usage_so_far: Usage
    """Usage of the run up until the last request."""
    _usage_limits: UsageLimits | None
    _stream_response: models.EitherStreamedResponse
    _result_schema: _result.ResultSchema[ResultData] | None
    _run_ctx: RunContext[AgentDeps]
    _result_validators: list[_result.ResultValidator[AgentDeps, ResultData]]
    _result_tool_name: str | None
    _on_complete: Callable[[], Awaitable[None]]
    is_complete: bool = field(default=False, init=False)
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream`][pydantic_ai.result.StreamedRunResult.stream],
    [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
    [`stream_structured`][pydantic_ai.result.StreamedRunResult.stream_structured] or
    [`get_data`][pydantic_ai.result.StreamedRunResult.get_data] completes.
    """

    async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[ResultData]:
        """Stream the response as an async iterable.

        The pydantic validator for structured data will be called in
        [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
        on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the response data.
        """
        if isinstance(self._stream_response, models.StreamTextResponse):
            async for text in self.stream_text(debounce_by=debounce_by):
                yield cast(ResultData, text)
        else:
            async for structured_message, is_last in self.stream_structured(debounce_by=debounce_by):
                yield await self.validate_structured_result(structured_message, allow_partial=not is_last)

    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
        """Stream the text result as an async iterable.

        !!! note
            This method will fail if the response is structured,
            e.g. if [`is_structured`][pydantic_ai.result.StreamedRunResult.is_structured] returns `True`.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        usage_checking_stream = _get_usage_checking_stream_response(
            self._stream_response, self._usage_limits, self.usage
        )

        with _logfire.span('response stream text') as lf_span:
            if isinstance(self._stream_response, models.StreamStructuredResponse):
                raise exceptions.UserError('stream_text() can only be used with text responses')
            if delta:
                async with _utils.group_by_temporal(usage_checking_stream, debounce_by) as group_iter:
                    async for _ in group_iter:
                        yield ''.join(self._stream_response.get())
                final_delta = ''.join(self._stream_response.get(final=True))
                if final_delta:
                    yield final_delta
            else:
                # a quick benchmark shows it's faster to build up a string with concat when we're
                # yielding at each step
                chunks: list[str] = []
                combined = ''
                async with _utils.group_by_temporal(usage_checking_stream, debounce_by) as group_iter:
                    async for _ in group_iter:
                        new = False
                        for chunk in self._stream_response.get():
                            chunks.append(chunk)
                            new = True
                        if new:
                            combined = await self._validate_text_result(''.join(chunks))
                            yield combined

                new = False
                for chunk in self._stream_response.get(final=True):
                    chunks.append(chunk)
                    new = True
                if new:
                    combined = await self._validate_text_result(''.join(chunks))
                    yield combined
                lf_span.set_attribute('combined_text', combined)
                await self._marked_completed(_messages.ModelResponse.from_text(combined))

    async def stream_structured(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        """Stream the response as an async iterable of Structured LLM Messages.

        !!! note
            This method will fail if the response is text,
            e.g. if [`is_structured`][pydantic_ai.result.StreamedRunResult.is_structured] returns `False`.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the structured response message and whether that is the last message.
        """
        usage_checking_stream = _get_usage_checking_stream_response(
            self._stream_response, self._usage_limits, self.usage
        )

        with _logfire.span('response stream structured') as lf_span:
            if isinstance(self._stream_response, models.StreamTextResponse):
                raise exceptions.UserError('stream_structured() can only be used with structured responses')
            else:
                # we should already have a message at this point, yield that first if it has any content
                msg = self._stream_response.get()
                for item in msg.parts:
                    if isinstance(item, _messages.ToolCallPart) and item.has_content():
                        yield msg, False
                        break
                async with _utils.group_by_temporal(usage_checking_stream, debounce_by) as group_iter:
                    async for _ in group_iter:
                        msg = self._stream_response.get()
                        for item in msg.parts:
                            if isinstance(item, _messages.ToolCallPart) and item.has_content():
                                yield msg, False
                                break
                msg = self._stream_response.get(final=True)
                yield msg, True
                lf_span.set_attribute('structured_response', msg)
                await self._marked_completed(msg)

    async def get_data(self) -> ResultData:
        """Stream the whole response, validate and return it."""
        usage_checking_stream = _get_usage_checking_stream_response(
            self._stream_response, self._usage_limits, self.usage
        )

        async for _ in usage_checking_stream:
            pass

        if isinstance(self._stream_response, models.StreamTextResponse):
            text = ''.join(self._stream_response.get(final=True))
            text = await self._validate_text_result(text)
            await self._marked_completed(_messages.ModelResponse.from_text(text))
            return cast(ResultData, text)
        else:
            message = self._stream_response.get(final=True)
            await self._marked_completed(message)
            return await self.validate_structured_result(message)

    @property
    def is_structured(self) -> bool:
        """Return whether the stream response contains structured data (as opposed to text)."""
        return isinstance(self._stream_response, models.StreamStructuredResponse)

    def usage(self) -> Usage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self.usage_so_far + self._stream_response.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._stream_response.timestamp()

    async def validate_structured_result(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> ResultData:
        """Validate a structured result message."""
        assert self._result_schema is not None, 'Expected _result_schema to not be None'
        assert self._result_tool_name is not None, 'Expected _result_tool_name to not be None'
        match = self._result_schema.find_named_tool(message.parts, self._result_tool_name)
        if match is None:
            raise exceptions.UnexpectedModelBehavior(
                f'Invalid message, unable to find tool: {self._result_schema.tool_names()}'
            )

        call, result_tool = match
        result_data = result_tool.validate(call, allow_partial=allow_partial, wrap_validation_errors=False)

        for validator in self._result_validators:
            result_data = await validator.validate(result_data, call, self._run_ctx)
        return result_data

    async def _validate_text_result(self, text: str) -> str:
        for validator in self._result_validators:
            text = await validator.validate(  # pyright: ignore[reportAssignmentType]
                text,  # pyright: ignore[reportArgumentType]
                None,
                self._run_ctx,
            )
        return text

    async def _marked_completed(self, message: _messages.ModelResponse) -> None:
        self.is_complete = True
        self._all_messages.append(message)
        await self._on_complete()


def _get_usage_checking_stream_response(
    stream_response: AsyncIterator[ResultData], limits: UsageLimits | None, get_usage: Callable[[], Usage]
) -> AsyncIterator[ResultData]:
    if limits is not None and limits.has_token_limits():

        async def _usage_checking_iterator():
            async for item in stream_response:
                limits.check_tokens(get_usage())
                yield item

        return _usage_checking_iterator()
    else:
        return stream_response
