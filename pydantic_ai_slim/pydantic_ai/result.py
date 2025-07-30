from __future__ import annotations as _annotations

import warnings
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from copy import copy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, cast

from pydantic import ValidationError
from typing_extensions import TypeVar, deprecated, overload

from pydantic_ai._tool_manager import ToolManager

from . import _utils, exceptions, messages as _messages, models
from ._output import (
    OutputDataT_inv,
    OutputSchema,
    OutputValidator,
    OutputValidatorFunc,
    PlainTextOutputSchema,
    TextOutputSchema,
    ToolOutputSchema,
)
from ._run_context import AgentDepsT, RunContext
from .messages import AgentStreamEvent, FinalResultEvent
from .output import (
    OutputDataT,
    ToolOutput,
)
from .usage import Usage, UsageLimits

__all__ = (
    'OutputDataT',
    'OutputDataT_inv',
    'ToolOutput',
    'OutputValidatorFunc',
)


T = TypeVar('T')
"""An invariant TypeVar."""


@dataclass
class AgentStream(Generic[AgentDepsT, OutputDataT]):
    _raw_stream_response: models.StreamedResponse
    _output_schema: OutputSchema[OutputDataT]
    _output_validators: list[OutputValidator[AgentDepsT, OutputDataT]]
    _run_ctx: RunContext[AgentDepsT]
    _usage_limits: UsageLimits | None
    _tool_manager: ToolManager[AgentDepsT]

    _agent_stream_iterator: AsyncIterator[AgentStreamEvent] | None = field(default=None, init=False)
    _final_result_event: FinalResultEvent | None = field(default=None, init=False)
    _initial_run_ctx_usage: Usage = field(init=False)

    def __post_init__(self):
        self._initial_run_ctx_usage = copy(self._run_ctx.usage)

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        """Asynchronously stream the (validated) agent outputs."""
        async for response in self.stream_responses(debounce_by=debounce_by):
            if self._final_result_event is not None:
                try:
                    yield await self._validate_response(response, allow_partial=True)
                except ValidationError:
                    pass
        if self._final_result_event is not None:  # pragma: no branch
            yield await self._validate_response(self._raw_stream_response.get())

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
        if not isinstance(self._output_schema, PlainTextOutputSchema):
            raise exceptions.UserError('stream_text() can only be used with text responses')

        if delta:
            async for text in self._stream_response_text(delta=True, debounce_by=debounce_by):
                yield text
        else:
            async for text in self._stream_response_text(delta=False, debounce_by=debounce_by):
                for validator in self._output_validators:
                    text = await validator.validate(text, self._run_ctx)  # pragma: no cover
                yield text

    def get(self) -> _messages.ModelResponse:
        """Get the current state of the response."""
        return self._raw_stream_response.get()

    def usage(self) -> Usage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._initial_run_ctx_usage + self._raw_stream_response.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._raw_stream_response.timestamp

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate the output and return it."""
        async for _ in self:
            pass

        return await self._validate_response(self._raw_stream_response.get())

    async def _validate_response(self, message: _messages.ModelResponse, *, allow_partial: bool = False) -> OutputDataT:
        """Validate a structured result message."""
        if self._final_result_event is None:
            raise exceptions.UnexpectedModelBehavior('Invalid response, unable to find output')  # pragma: no cover

        output_tool_name = self._final_result_event.tool_name

        if isinstance(self._output_schema, ToolOutputSchema) and output_tool_name is not None:
            tool_call = next(
                (
                    part
                    for part in message.parts
                    if isinstance(part, _messages.ToolCallPart) and part.tool_name == output_tool_name
                ),
                None,
            )
            if tool_call is None:
                raise exceptions.UnexpectedModelBehavior(  # pragma: no cover
                    f'Invalid response, unable to find tool call for {output_tool_name!r}'
                )
            return await self._tool_manager.handle_call(
                tool_call, allow_partial=allow_partial, wrap_validation_errors=False
            )
        elif deferred_tool_calls := self._tool_manager.get_deferred_tool_calls(message.parts):
            if not self._output_schema.allows_deferred_tool_calls:
                raise exceptions.UserError(
                    'A deferred tool call was present, but `DeferredToolCalls` is not among output types. To resolve this, add `DeferredToolCalls` to the list of output types for this agent.'
                )
            return cast(OutputDataT, deferred_tool_calls)
        elif isinstance(self._output_schema, TextOutputSchema):
            text = '\n\n'.join(x.content for x in message.parts if isinstance(x, _messages.TextPart))

            result_data = await self._output_schema.process(
                text, self._run_ctx, allow_partial=allow_partial, wrap_validation_errors=False
            )
            for validator in self._output_validators:
                result_data = await validator.validate(result_data, self._run_ctx)
            return result_data
        else:
            raise exceptions.UnexpectedModelBehavior(  # pragma: no cover
                'Invalid response, unable to process text output'
            )

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
            msg = self._raw_stream_response.get()
            for i, part in enumerate(msg.parts):
                if isinstance(part, _messages.TextPart) and part.content:
                    yield part.content, i

            async for event in self._raw_stream_response:
                if (
                    isinstance(event, _messages.PartStartEvent)
                    and isinstance(event.part, _messages.TextPart)
                    and event.part.content
                ):
                    yield event.part.content, event.index  # pragma: no cover
                elif (  # pragma: no branch
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

    def __aiter__(self) -> AsyncIterator[AgentStreamEvent]:
        """Stream [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent]s.

        This proxies the _raw_stream_response and sends all events to the agent stream, while also checking for matches
        on the result schema and emitting a [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] if/when the
        first match is found.
        """
        if self._agent_stream_iterator is not None:
            return self._agent_stream_iterator

        async def aiter():
            output_schema = self._output_schema

            def _get_final_result_event(e: _messages.ModelResponseStreamEvent) -> _messages.FinalResultEvent | None:
                """Return an appropriate FinalResultEvent if `e` corresponds to a part that will produce a final result."""
                if isinstance(e, _messages.PartStartEvent):
                    new_part = e.part
                    if isinstance(new_part, _messages.TextPart) and isinstance(
                        output_schema, TextOutputSchema
                    ):  # pragma: no branch
                        return _messages.FinalResultEvent(tool_name=None, tool_call_id=None)
                    elif isinstance(new_part, _messages.ToolCallPart) and (
                        tool_def := self._tool_manager.get_tool_def(new_part.tool_name)
                    ):
                        if tool_def.kind == 'output':
                            return _messages.FinalResultEvent(
                                tool_name=new_part.tool_name, tool_call_id=new_part.tool_call_id
                            )
                        elif tool_def.kind == 'deferred':
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
class StreamedRunResult(Generic[AgentDepsT, OutputDataT]):
    """Result of a streamed run that returns structured data via a tool call."""

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

    _stream_response: AgentStream[AgentDepsT, OutputDataT]
    _on_complete: Callable[[], Awaitable[None]]

    is_complete: bool = field(default=False, init=False)
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream`][pydantic_ai.result.StreamedRunResult.stream],
    [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
    [`stream_structured`][pydantic_ai.result.StreamedRunResult.stream_structured] or
    [`get_output`][pydantic_ai.result.StreamedRunResult.get_output] completes.
    """

    @overload
    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def all_messages(self, *, result_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    def all_messages(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: deprecated, use `output_tool_return_content` instead.

        Returns:
            List of messages.
        """
        # this is a method to be consistent with the other methods
        content = coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        if content is not None:
            raise NotImplementedError('Setting output tool return content is not supported for this result type.')
        return self._all_messages

    @overload
    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def all_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes: ...

    def all_messages_json(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> bytes:  # pragma: no cover
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: deprecated, use `output_tool_return_content` instead.

        Returns:
            JSON bytes representing the messages.
        """
        content = coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        return _messages.ModelMessagesTypeAdapter.dump_json(self.all_messages(output_tool_return_content=content))

    @overload
    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]: ...

    def new_messages(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> list[_messages.ModelMessage]:  # pragma: no cover
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: deprecated, use `output_tool_return_content` instead.

        Returns:
            List of new messages.
        """
        content = coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        return self.all_messages(output_tool_return_content=content)[self._new_message_index :]

    @overload
    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes: ...

    @overload
    @deprecated('`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.')
    def new_messages_json(self, *, result_tool_return_content: str | None = None) -> bytes: ...

    def new_messages_json(
        self, *, output_tool_return_content: str | None = None, result_tool_return_content: str | None = None
    ) -> bytes:  # pragma: no cover
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.
            result_tool_return_content: deprecated, use `output_tool_return_content` instead.

        Returns:
            JSON bytes representing the new messages.
        """
        content = coalesce_deprecated_return_content(output_tool_return_content, result_tool_return_content)
        return _messages.ModelMessagesTypeAdapter.dump_json(self.new_messages(output_tool_return_content=content))

    async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
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
        async for output in self._stream_response.stream_output(debounce_by=debounce_by):
            yield output
        await self._marked_completed(self._stream_response.get())

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
        async for text in self._stream_response.stream_text(delta=delta, debounce_by=debounce_by):
            yield text
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
        async for msg in self._stream_response.stream_responses(debounce_by=debounce_by):
            yield msg, False

        msg = self._stream_response.get()
        yield msg, True

        await self._marked_completed(msg)

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate and return it."""
        output = await self._stream_response.get_output()
        await self._marked_completed(self._stream_response.get())
        return output

    @deprecated('`get_data` is deprecated, use `get_output` instead.')
    async def get_data(self) -> OutputDataT:
        return await self.get_output()

    def usage(self) -> Usage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._stream_response.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._stream_response.timestamp()

    @deprecated('`validate_structured_result` is deprecated, use `validate_structured_output` instead.')
    async def validate_structured_result(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        return await self.validate_structured_output(message, allow_partial=allow_partial)

    async def validate_structured_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        """Validate a structured result message."""
        return await self._stream_response._validate_response(  # pyright: ignore[reportPrivateUsage]
            message, allow_partial=allow_partial
        )

    async def _marked_completed(self, message: _messages.ModelResponse) -> None:
        self.is_complete = True
        self._all_messages.append(message)
        await self._on_complete()


@dataclass(repr=False)
class FinalResult(Generic[OutputDataT]):
    """Marker class storing the final output of an agent run and associated metadata."""

    output: OutputDataT
    """The final result data."""
    tool_name: str | None = None
    """Name of the final output tool; `None` if the output came from unstructured text content."""
    tool_call_id: str | None = None
    """ID of the tool call that produced the final output; `None` if the output came from unstructured text content."""

    @property
    @deprecated('`data` is deprecated, use `output` instead.')
    def data(self) -> OutputDataT:
        return self.output

    __repr__ = _utils.dataclasses_no_defaults_repr


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


def coalesce_deprecated_return_content(
    output_tool_return_content: T | None, result_tool_return_content: T | None
) -> T | None:
    """Return the first non-None value."""
    if output_tool_return_content is None:
        if result_tool_return_content is not None:  # pragma: no cover
            warnings.warn(
                '`result_tool_return_content` is deprecated, use `output_tool_return_content` instead.',
                DeprecationWarning,
                stacklevel=3,
            )
        return result_tool_return_content
    return output_tool_return_content
