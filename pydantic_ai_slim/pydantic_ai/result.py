from __future__ import annotations as _annotations

from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, Union, cast

from typing_extensions import TypeVar, assert_type

from . import _result, _utils, exceptions, messages as _messages, models
from .messages import AgentStreamEvent, FinalResultEvent
from .tools import AgentDepsT, RunContext
from .usage import Usage, UsageLimits

__all__ = 'ResultDataT', 'ResultDataT_inv', 'ResultValidatorFunc'


T = TypeVar('T')
"""An invariant TypeVar."""
ResultDataT_inv = TypeVar('ResultDataT_inv', default=str)
"""
An invariant type variable for the result data of a model.

We need to use an invariant typevar for `ResultValidator` and `ResultValidatorFunc` because the result data type is used
in both the input and output of a `ResultValidatorFunc`. This can theoretically lead to some issues assuming that types
possessing ResultValidator's are covariant in the result data type, but in practice this is rarely an issue, and
changing it would have negative consequences for the ergonomics of the library.

At some point, it may make sense to change the input to ResultValidatorFunc to be `Any` or `object` as doing that would
resolve these potential variance issues.
"""
ResultDataT = TypeVar('ResultDataT', default=str, covariant=True)
"""Covariant type variable for the result data type of a run."""

ResultValidatorFunc = Union[
    Callable[[RunContext[AgentDepsT], ResultDataT_inv], ResultDataT_inv],
    Callable[[RunContext[AgentDepsT], ResultDataT_inv], Awaitable[ResultDataT_inv]],
    Callable[[ResultDataT_inv], ResultDataT_inv],
    Callable[[ResultDataT_inv], Awaitable[ResultDataT_inv]],
]
"""
A function that always takes and returns the same type of data (which is the result type of an agent run), and:

* may or may not take [`RunContext`][pydantic_ai.tools.RunContext] as a first argument
* may or may not be async

Usage `ResultValidatorFunc[AgentDepsT, T]`.
"""


@dataclass
class AgentStream(Generic[AgentDepsT, ResultDataT]):
    _raw_stream_response: models.StreamedResponse
    _result_schema: _result.ResultSchema[ResultDataT] | None
    _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]]
    _run_ctx: RunContext[AgentDepsT]
    _usage_limits: UsageLimits | None

    _agent_stream_iterator: AsyncIterator[AgentStreamEvent] | None = field(default=None, init=False)
    _final_result_event: FinalResultEvent | None = field(default=None, init=False)
    _initial_run_ctx_usage: Usage = field(init=False)

    def __post_init__(self):
        self._initial_run_ctx_usage = copy(self._run_ctx.usage)

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[ResultDataT]:
        """Asynchronously stream the (validated) agent outputs."""
        async for response in self.stream_responses(debounce_by=debounce_by):
            if self._final_result_event is not None:
                yield await self._validate_response(response, self._final_result_event.tool_name, allow_partial=True)
        if self._final_result_event is not None:
            yield await self._validate_response(
                self._raw_stream_response.get(), self._final_result_event.tool_name, allow_partial=False
            )

    async def stream_responses(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[_messages.ModelResponse]:
        """Asynchronously stream the (unvalidated) model responses for the agent."""
        # if the message currently has any parts with content, yield before streaming
        msg = self._raw_stream_response.get()
        for part in msg.parts:
            if part.has_content():
                yield msg
                break

        async with _utils.group_by_temporal(self, debounce_by) as group_iter:
            async for _items in group_iter:
                yield self._raw_stream_response.get()  # current state of the response

    def usage(self) -> Usage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._initial_run_ctx_usage + self._raw_stream_response.usage()

    async def _validate_response(
        self, message: _messages.ModelResponse, result_tool_name: str | None, *, allow_partial: bool = False
    ) -> ResultDataT:
        """Validate a structured result message."""
        if self._result_schema is not None and result_tool_name is not None:
            match = self._result_schema.find_named_tool(message.parts, result_tool_name)
            if match is None:
                raise exceptions.UnexpectedModelBehavior(
                    f'Invalid response, unable to find tool: {self._result_schema.tool_names()}'
                )

            call, result_tool = match
            result_data = result_tool.validate(call, allow_partial=allow_partial, wrap_validation_errors=False)

            for validator in self._result_validators:
                result_data = await validator.validate(result_data, call, self._run_ctx)
            return result_data
        else:
            text = '\n\n'.join(x.content for x in message.parts if isinstance(x, _messages.TextPart))
            for validator in self._result_validators:
                text = await validator.validate(
                    text,
                    None,
                    self._run_ctx,
                )
            # Since there is no result tool, we can assume that str is compatible with ResultDataT
            return cast(ResultDataT, text)

    def __aiter__(self) -> AsyncIterator[AgentStreamEvent]:
        """Stream [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s.

        This proxies the _raw_stream_response and sends all events to the agent stream, while also checking for matches
        on the result schema and emitting a [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] if/when the
        first match is found.
        """
        if self._agent_stream_iterator is not None:
            return self._agent_stream_iterator

        async def aiter():
            result_schema = self._result_schema
            allow_text_result = result_schema is None or result_schema.allow_text_result

            def _get_final_result_event(e: _messages.ModelResponseStreamEvent) -> _messages.FinalResultEvent | None:
                """Return an appropriate FinalResultEvent if `e` corresponds to a part that will produce a final result."""
                if isinstance(e, _messages.PartStartEvent):
                    new_part = e.part
                    if isinstance(new_part, _messages.ToolCallPart):
                        if result_schema:
                            for call, _ in result_schema.find_tool([new_part]):
                                return _messages.FinalResultEvent(
                                    tool_name=call.tool_name, tool_call_id=call.tool_call_id
                                )
                    elif allow_text_result:
                        assert_type(e, _messages.PartStartEvent)
                        return _messages.FinalResultEvent(tool_name=None, tool_call_id=None)

            usage_checking_stream = _get_usage_checking_stream_response(
                self._raw_stream_response, self._usage_limits, self.usage
            )
            async for event in usage_checking_stream:
                yield event
                if (final_result_event := _get_final_result_event(event)) is not None:
                    self._final_result_event = final_result_event
                    yield final_result_event
                    break

            # If we broke out of the above loop, we need to yield the rest of the events
            # If we didn't, this will just be a no-op
            async for event in usage_checking_stream:
                yield event

        self._agent_stream_iterator = aiter()
        return self._agent_stream_iterator


@dataclass
class StreamedRunResult(Generic[AgentDepsT, ResultDataT]):
    """Result of a streamed run that returns structured data via a tool call."""

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

    _usage_limits: UsageLimits | None
    _stream_response: models.StreamedResponse
    _result_schema: _result.ResultSchema[ResultDataT] | None
    _run_ctx: RunContext[AgentDepsT]
    _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]]
    _result_tool_name: str | None
    _on_complete: Callable[[], Awaitable[None]]

    _initial_run_ctx_usage: Usage = field(init=False)
    is_complete: bool = field(default=False, init=False)
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream`][pydantic_ai.result.StreamedRunResult.stream],
    [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
    [`stream_structured`][pydantic_ai.result.StreamedRunResult.stream_structured] or
    [`get_data`][pydantic_ai.result.StreamedRunResult.get_data] completes.
    """

    def __post_init__(self):
        self._initial_run_ctx_usage = copy(self._run_ctx.usage)

    def all_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        # this is a method to be consistent with the other methods
        if result_tool_return_content is not None:
            raise NotImplementedError('Setting result tool return content is not supported for this result type.')
        return self._all_messages

    def all_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(result_tool_return_content=result_tool_return_content)
        )

    def new_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(result_tool_return_content=result_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes:
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.

        Args:
            result_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the result tool call if you want to continue
                the conversation and want to set the response to the result tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(result_tool_return_content=result_tool_return_content)
        )

    async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[ResultDataT]:
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
        async for structured_message, is_last in self.stream_structured(debounce_by=debounce_by):
            result = await self.validate_structured_result(structured_message, allow_partial=not is_last)
            yield result

    async def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> AsyncIterator[str]:
        """Stream the text result as an async iterable.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        if self._result_schema and not self._result_schema.allow_text_result:
            raise exceptions.UserError('stream_text() can only be used with text responses')

        if delta:
            async for text in self._stream_response_text(delta=delta, debounce_by=debounce_by):
                yield text
        else:
            async for text in self._stream_response_text(delta=delta, debounce_by=debounce_by):
                combined_validated_text = await self._validate_text_result(text)
                yield combined_validated_text
        await self._marked_completed(self._stream_response.get())

    async def stream_structured(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        """Stream the response as an async iterable of Structured LLM Messages.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the structured response message and whether that is the last message.
        """
        # if the message currently has any parts with content, yield before streaming
        msg = self._stream_response.get()
        for part in msg.parts:
            if part.has_content():
                yield msg, False
                break

        async for msg in self._stream_response_structured(debounce_by=debounce_by):
            yield msg, False

        msg = self._stream_response.get()
        yield msg, True

        await self._marked_completed(msg)

    async def get_data(self) -> ResultDataT:
        """Stream the whole response, validate and return it."""
        usage_checking_stream = _get_usage_checking_stream_response(
            self._stream_response, self._usage_limits, self.usage
        )

        async for _ in usage_checking_stream:
            pass
        message = self._stream_response.get()
        await self._marked_completed(message)
        return await self.validate_structured_result(message)

    def usage(self) -> Usage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._initial_run_ctx_usage + self._stream_response.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._stream_response.timestamp

    async def validate_structured_result(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> ResultDataT:
        """Validate a structured result message."""
        if self._result_schema is not None and self._result_tool_name is not None:
            match = self._result_schema.find_named_tool(message.parts, self._result_tool_name)
            if match is None:
                raise exceptions.UnexpectedModelBehavior(
                    f'Invalid response, unable to find tool: {self._result_schema.tool_names()}'
                )

            call, result_tool = match
            result_data = result_tool.validate(call, allow_partial=allow_partial, wrap_validation_errors=False)

            for validator in self._result_validators:
                result_data = await validator.validate(result_data, call, self._run_ctx)
            return result_data
        else:
            text = '\n\n'.join(x.content for x in message.parts if isinstance(x, _messages.TextPart))
            for validator in self._result_validators:
                text = await validator.validate(
                    text,
                    None,
                    self._run_ctx,
                )
            # Since there is no result tool, we can assume that str is compatible with ResultDataT
            return cast(ResultDataT, text)

    async def _validate_text_result(self, text: str) -> str:
        for validator in self._result_validators:
            text = await validator.validate(
                text,
                None,
                self._run_ctx,
            )
        return text

    async def _marked_completed(self, message: _messages.ModelResponse) -> None:
        self.is_complete = True
        self._all_messages.append(message)
        await self._on_complete()

    async def _stream_response_structured(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[_messages.ModelResponse]:
        async with _utils.group_by_temporal(self._stream_response, debounce_by) as group_iter:
            async for _items in group_iter:
                yield self._stream_response.get()

    async def _stream_response_text(
        self, *, delta: bool = False, debounce_by: float | None = 0.1
    ) -> AsyncIterator[str]:
        """Stream the response as an async iterable of text."""

        # Define a "merged" version of the iterator that will yield items that have already been retrieved
        # and items that we receive while streaming. We define a dedicated async iterator for this so we can
        # pass the combined stream to the group_by_temporal function within `_stream_text_deltas` below.
        async def _stream_text_deltas_ungrouped() -> AsyncIterator[tuple[str, int]]:
            # yields tuples of (text_content, part_index)
            # we don't currently make use of the part_index, but in principle this may be useful
            # so we retain it here for now to make possible future refactors simpler
            msg = self._stream_response.get()
            for i, part in enumerate(msg.parts):
                if isinstance(part, _messages.TextPart) and part.content:
                    yield part.content, i

            async for event in self._stream_response:
                if (
                    isinstance(event, _messages.PartStartEvent)
                    and isinstance(event.part, _messages.TextPart)
                    and event.part.content
                ):
                    yield event.part.content, event.index
                elif (
                    isinstance(event, _messages.PartDeltaEvent)
                    and isinstance(event.delta, _messages.TextPartDelta)
                    and event.delta.content_delta
                ):
                    yield event.delta.content_delta, event.index

        async def _stream_text_deltas() -> AsyncIterator[str]:
            async with _utils.group_by_temporal(_stream_text_deltas_ungrouped(), debounce_by) as group_iter:
                async for items in group_iter:
                    # Note: we are currently just dropping the part index on the group here
                    yield ''.join([content for content, _ in items])

        if delta:
            async for text in _stream_text_deltas():
                yield text
        else:
            # a quick benchmark shows it's faster to build up a string with concat when we're
            # yielding at each step
            deltas: list[str] = []
            async for text in _stream_text_deltas():
                deltas.append(text)
                yield ''.join(deltas)


@dataclass
class FinalResult(Generic[ResultDataT]):
    """Marker class storing the final result of an agent run and associated metadata."""

    data: ResultDataT
    """The final result data."""
    tool_name: str | None
    """Name of the final result tool; `None` if the result came from unstructured text content."""
    tool_call_id: str | None
    """ID of the tool call that produced the final result; `None` if the result came from unstructured text content."""


def _get_usage_checking_stream_response(
    stream_response: AsyncIterable[_messages.ModelResponseStreamEvent],
    limits: UsageLimits | None,
    get_usage: Callable[[], Usage],
) -> AsyncIterable[_messages.ModelResponseStreamEvent]:
    if limits is not None and limits.has_token_limits():

        async def _usage_checking_iterator():
            async for item in stream_response:
                limits.check_tokens(get_usage())
                yield item

        return _usage_checking_iterator()
    else:
        return stream_response
