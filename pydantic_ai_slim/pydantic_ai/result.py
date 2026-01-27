from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, cast, overload

from pydantic import ValidationError
from typing_extensions import TypeVar, deprecated

from . import _utils, exceptions, messages as _messages, models
from ._output import (
    OutputDataT_inv,
    OutputSchema,
    OutputValidator,
    OutputValidatorFunc,
    TextOutputSchema,
)
from ._run_context import AgentDepsT, RunContext
from ._tool_manager import ToolManager
from .messages import ModelResponseStreamEvent
from .output import (
    DeferredToolRequests,
    OutputDataT,
    ToolOutput,
)
from .usage import RunUsage, UsageLimits

if TYPE_CHECKING:
    from .run import AgentRunResult

__all__ = (
    'OutputDataT',
    'OutputDataT_inv',
    'ToolOutput',
    'OutputValidatorFunc',
    'StreamedRunResultSync',
)


T = TypeVar('T')
"""An invariant TypeVar."""


@dataclass(kw_only=True)
class AgentStream(Generic[AgentDepsT, OutputDataT]):
    _raw_stream_response: models.StreamedResponse
    _output_schema: OutputSchema[OutputDataT]
    _model_request_parameters: models.ModelRequestParameters
    _output_validators: list[OutputValidator[AgentDepsT, OutputDataT]]
    _run_ctx: RunContext[AgentDepsT]
    _usage_limits: UsageLimits | None
    _tool_manager: ToolManager[AgentDepsT]
    _metadata_getter: Callable[[], dict[str, Any] | None] | None = field(default=None, repr=False)

    _agent_stream_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _initial_run_ctx_usage: RunUsage = field(init=False)
    _cached_output: OutputDataT | None = field(default=None, init=False)

    def __post_init__(self):
        self._initial_run_ctx_usage = deepcopy(self._run_ctx.usage)

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        """Asynchronously stream the (validated) agent outputs."""
        if self._cached_output is not None:
            yield deepcopy(self._cached_output)
            return

        last_response: _messages.ModelResponse | None = None
        async for response in self.stream_responses(debounce_by=debounce_by):
            if self._raw_stream_response.final_result_event is None or (
                last_response and response.parts == last_response.parts
            ):
                continue
            last_response = response

            try:
                yield await self.validate_response_output(response, allow_partial=True)
            except ValidationError:
                pass

        if self._raw_stream_response.final_result_event is not None:  # pragma: no branch
            response = self.response
            # Final validation with allow_partial=False (the default).
            # We always yield the final result even if the content matches the last partial yield, because:
            # 1. Output validators/functions receive partial_output=False only on this final call,
            #    and may behave differently based on that flag
            # 2. Users can rely on the last yielded item being the fully validated output
            self._cached_output = await self.validate_response_output(response)
            yield deepcopy(self._cached_output)

    async def stream_responses(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[_messages.ModelResponse]:
        """Asynchronously stream the (unvalidated) model responses for the agent."""
        # if the message currently has any parts with content, yield before streaming
        msg = self.response
        for part in msg.parts:
            if part.has_content():
                yield msg
                break

        async with _utils.group_by_temporal(self, debounce_by) as group_iter:
            async for _items in group_iter:
                yield self.response  # current state of the response

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
        if not isinstance(self._output_schema, TextOutputSchema):
            raise exceptions.UserError('stream_text() can only be used with text responses')

        # Yield cached output for both delta and non-delta modes
        # This is expected that the subsequent calls to `stream_text()`
        # yield full not delta output even for `delta=True`
        if isinstance(self._cached_output, str):
            yield self._cached_output
            return

        if delta:
            async for text in self._stream_response_text(delta=True, debounce_by=debounce_by):
                yield text
        else:
            async for text in self._stream_response_text(delta=False, debounce_by=debounce_by):
                for validator in self._output_validators:
                    text = await validator.validate(text, replace(self._run_ctx, partial_output=True))
                yield text

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        assert self._run_ctx.run_id is not None
        return self._run_ctx.run_id

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        if self._metadata_getter is not None:
            return self._metadata_getter()
        return self._run_ctx.metadata

    # TODO (v2): Drop in favor of `response` property
    def get(self) -> _messages.ModelResponse:
        """Get the current state of the response."""
        return self._raw_stream_response.get()

    @property
    def response(self) -> _messages.ModelResponse:
        """Get the current state of the response."""
        return self.get()

    # TODO (v2): Make this a property
    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._initial_run_ctx_usage + self._raw_stream_response.usage()

    # TODO (v2): Make this a property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._raw_stream_response.timestamp

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate the output and return it."""
        if self._cached_output is not None:
            return deepcopy(self._cached_output)

        # Iterate through any stream events
        async for _ in self:
            pass

        # Final validation with `allow_partial=False` (default)
        self._cached_output = await self.validate_response_output(self.response)
        return deepcopy(self._cached_output)

    async def validate_response_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        """Validate a structured result message."""
        final_result_event = self._raw_stream_response.final_result_event
        if final_result_event is None:
            raise exceptions.UnexpectedModelBehavior('Invalid response, unable to find output')  # pragma: no cover

        output_tool_name = final_result_event.tool_name

        if self._output_schema.toolset and output_tool_name is not None:
            tool_call = next(
                (part for part in message.tool_calls if part.tool_name == output_tool_name),
                None,
            )
            if tool_call is None:
                raise exceptions.UnexpectedModelBehavior(  # pragma: no cover
                    f'Invalid response, unable to find tool call for {output_tool_name!r}'
                )
            return await self._tool_manager.handle_call(
                tool_call, allow_partial=allow_partial, wrap_validation_errors=False
            )
        elif deferred_tool_requests := _get_deferred_tool_requests(message.tool_calls, self._tool_manager):
            if not self._output_schema.allows_deferred_tools:
                raise exceptions.UserError(
                    'A deferred tool call was present, but `DeferredToolRequests` is not among output types. To resolve this, add `DeferredToolRequests` to the list of output types for this agent.'
                )
            return cast(OutputDataT, deferred_tool_requests)
        elif self._output_schema.allows_image and message.images:
            return cast(OutputDataT, message.images[0])
        elif text_processor := self._output_schema.text_processor:
            text = ''
            for part in message.parts:
                if isinstance(part, _messages.TextPart):
                    text += part.content
                elif isinstance(part, _messages.BuiltinToolCallPart):
                    # Text parts before a built-in tool call are essentially thoughts,
                    # not part of the final result output, so we reset the accumulated text
                    text = ''

            result_data = await text_processor.process(
                text,
                run_context=replace(self._run_ctx, partial_output=allow_partial),
                allow_partial=allow_partial,
                wrap_validation_errors=False,
            )
            for validator in self._output_validators:
                result_data = await validator.validate(
                    result_data, replace(self._run_ctx, partial_output=allow_partial)
                )
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
            msg = self.response
            for i, part in enumerate(msg.parts):
                if isinstance(part, _messages.TextPart) and part.content:
                    yield part.content, i

            last_text_index: int | None = None
            async for event in self._raw_stream_response:
                if (
                    isinstance(event, _messages.PartStartEvent)
                    and isinstance(event.part, _messages.TextPart)
                    and event.part.content
                ):
                    last_text_index = event.index
                    yield event.part.content, event.index
                elif (
                    isinstance(event, _messages.PartDeltaEvent)
                    and isinstance(event.delta, _messages.TextPartDelta)
                    and event.delta.content_delta
                ):
                    last_text_index = event.index
                    yield event.delta.content_delta, event.index
                elif (
                    isinstance(event, _messages.PartStartEvent)
                    and isinstance(event.part, _messages.BuiltinToolCallPart)
                    and last_text_index is not None
                ):
                    # Text parts that are interrupted by a built-in tool call should not be joined together directly
                    yield '\n\n', event.index
                    last_text_index = None

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

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Stream [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s."""
        if self._agent_stream_iterator is None:
            self._agent_stream_iterator = _get_usage_checking_stream_response(
                self._raw_stream_response, self._usage_limits, self.usage
            )

        return self._agent_stream_iterator


@dataclass(init=False)
class StreamedRunResult(Generic[AgentDepsT, OutputDataT]):
    """Result of a streamed run that returns structured data via a tool call."""

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

    _stream_response: AgentStream[AgentDepsT, OutputDataT] | None = None
    _on_complete: Callable[[], Awaitable[None]] | None = None

    _run_result: AgentRunResult[OutputDataT] | None = None

    is_complete: bool = field(default=False, init=False)
    """Whether the stream has all been received.

    This is set to `True` when one of
    [`stream_output`][pydantic_ai.result.StreamedRunResult.stream_output],
    [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text],
    [`stream_responses`][pydantic_ai.result.StreamedRunResult.stream_responses] or
    [`get_output`][pydantic_ai.result.StreamedRunResult.get_output] completes.
    """

    @overload
    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        stream_response: AgentStream[AgentDepsT, OutputDataT] | None,
        on_complete: Callable[[], Awaitable[None]] | None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        *,
        run_result: AgentRunResult[OutputDataT],
    ) -> None: ...

    def __init__(
        self,
        all_messages: list[_messages.ModelMessage],
        new_message_index: int,
        stream_response: AgentStream[AgentDepsT, OutputDataT] | None = None,
        on_complete: Callable[[], Awaitable[None]] | None = None,
        run_result: AgentRunResult[OutputDataT] | None = None,
    ) -> None:
        self._all_messages = all_messages
        self._new_message_index = new_message_index

        self._stream_response = stream_response
        self._on_complete = on_complete
        self._run_result = run_result

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of _messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        # this is a method to be consistent with the other methods
        if output_tool_return_content is not None:
            raise NotImplementedError('Setting output tool return content is not supported for this result type.')
        return self._all_messages

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResult.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.all_messages(output_tool_return_content=output_tool_return_content)
        )

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self.all_messages(output_tool_return_content=output_tool_return_content)[self._new_message_index :]

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResult.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return _messages.ModelMessagesTypeAdapter.dump_json(
            self.new_messages(output_tool_return_content=output_tool_return_content)
        )

    @deprecated('`StreamedRunResult.stream` is deprecated, use `stream_output` instead.')
    async def stream(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        async for output in self.stream_output(debounce_by=debounce_by):
            yield output

    async def stream_output(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[OutputDataT]:
        """Stream the output as an async iterable.

        The pydantic validator for structured data will be called in
        [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
        on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured outputs to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An async iterable of the response data.
        """
        if self._run_result is not None:
            yield self._run_result.output
            await self._marked_completed()
        elif self._stream_response is not None:
            async for output in self._stream_response.stream_output(debounce_by=debounce_by):
                yield output
            await self._marked_completed(self.response)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

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
        if self._run_result is not None:  # pragma: no cover
            # We can't really get here, as `_run_result` is only set in `run_stream` when `CallToolsNode` produces `DeferredToolRequests` output
            # as a result of a tool function raising `CallDeferred` or `ApprovalRequired`.
            # That'll change if we ever support something like `raise EndRun(output: OutputT)` where `OutputT` could be `str`.
            if not isinstance(self._run_result.output, str):
                raise exceptions.UserError('stream_text() can only be used with text responses')
            yield self._run_result.output
            await self._marked_completed()
        elif self._stream_response is not None:
            async for text in self._stream_response.stream_text(delta=delta, debounce_by=debounce_by):
                yield text
            await self._marked_completed(self.response)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @deprecated('`StreamedRunResult.stream_structured` is deprecated, use `stream_responses` instead.')
    async def stream_structured(
        self, *, debounce_by: float | None = 0.1
    ) -> AsyncIterator[tuple[_messages.ModelResponse, bool]]:
        async for msg, last in self.stream_responses(debounce_by=debounce_by):
            yield msg, last

    async def stream_responses(
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
        if self._run_result is not None:
            yield self.response, True
            await self._marked_completed()
        elif self._stream_response is not None:
            # if the message currently has any parts with content, yield before streaming
            async for msg in self._stream_response.stream_responses(debounce_by=debounce_by):
                yield msg, False

            msg = self.response
            yield msg, True

            await self._marked_completed(msg)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def get_output(self) -> OutputDataT:
        """Stream the whole response, validate and return it."""
        if self._run_result is not None:
            output = self._run_result.output
            await self._marked_completed()
            return output
        elif self._stream_response is not None:
            output = await self._stream_response.get_output()
            await self._marked_completed(self.response)
            return output
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the current state of the response."""
        if self._run_result is not None:
            return self._run_result.response
        elif self._stream_response is not None:
            return self._stream_response.get()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        if self._run_result is not None:
            return self._run_result.metadata
        elif self._stream_response is not None:
            return self._stream_response.metadata
        else:
            return None

    # TODO (v2): Make this a property
    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        if self._run_result is not None:
            return self._run_result.usage()
        elif self._stream_response is not None:
            return self._stream_response.usage()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    # TODO (v2): Make this a property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        if self._run_result is not None:
            return self._run_result.timestamp()
        elif self._stream_response is not None:
            return self._stream_response.timestamp()
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        if self._run_result is not None:
            return self._run_result.run_id
        elif self._stream_response is not None:
            return self._stream_response.run_id
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    @deprecated('`validate_structured_output` is deprecated, use `validate_response_output` instead.')
    async def validate_structured_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        return await self.validate_response_output(message, allow_partial=allow_partial)

    async def validate_response_output(
        self, message: _messages.ModelResponse, *, allow_partial: bool = False
    ) -> OutputDataT:
        """Validate a structured result message."""
        if self._run_result is not None:
            return self._run_result.output
        elif self._stream_response is not None:
            return await self._stream_response.validate_response_output(message, allow_partial=allow_partial)
        else:
            raise ValueError('No stream response or run result provided')  # pragma: no cover

    async def _marked_completed(self, message: _messages.ModelResponse | None = None) -> None:
        if self.is_complete:
            return
        self.is_complete = True
        if message is not None:
            if self._stream_response:  # pragma: no branch
                message.run_id = self._stream_response.run_id
            self._all_messages.append(message)
        if self._on_complete is not None:
            await self._on_complete()


@dataclass(init=False)
class StreamedRunResultSync(Generic[AgentDepsT, OutputDataT]):
    """Synchronous wrapper for [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] that only exposes sync methods."""

    _streamed_run_result: StreamedRunResult[AgentDepsT, OutputDataT]

    def __init__(self, streamed_run_result: StreamedRunResult[AgentDepsT, OutputDataT]) -> None:
        self._streamed_run_result = streamed_run_result

    def all_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return the history of messages.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of messages.
        """
        return self._streamed_run_result.all_messages(output_tool_return_content=output_tool_return_content)

    def all_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return all messages from [`all_messages`][pydantic_ai.result.StreamedRunResultSync.all_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the messages.
        """
        return self._streamed_run_result.all_messages_json(output_tool_return_content=output_tool_return_content)

    def new_messages(self, *, output_tool_return_content: str | None = None) -> list[_messages.ModelMessage]:
        """Return new messages associated with this run.

        Messages from older runs are excluded.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            List of new messages.
        """
        return self._streamed_run_result.new_messages(output_tool_return_content=output_tool_return_content)

    def new_messages_json(self, *, output_tool_return_content: str | None = None) -> bytes:  # pragma: no cover
        """Return new messages from [`new_messages`][pydantic_ai.result.StreamedRunResultSync.new_messages] as JSON bytes.

        Args:
            output_tool_return_content: The return content of the tool call to set in the last message.
                This provides a convenient way to modify the content of the output tool call if you want to continue
                the conversation and want to set the response to the output tool call. If `None`, the last message will
                not be modified.

        Returns:
            JSON bytes representing the new messages.
        """
        return self._streamed_run_result.new_messages_json(output_tool_return_content=output_tool_return_content)

    def stream_output(self, *, debounce_by: float | None = 0.1) -> Iterator[OutputDataT]:
        """Stream the output as an iterable.

        The pydantic validator for structured data will be called in
        [partial mode](https://docs.pydantic.dev/dev/concepts/experimental/#partial-validation)
        on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the output chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured outputs to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An iterable of the response data.
        """
        return _utils.sync_async_iterator(self._streamed_run_result.stream_output(debounce_by=debounce_by))

    def stream_text(self, *, delta: bool = False, debounce_by: float | None = 0.1) -> Iterator[str]:
        """Stream the text result as an iterable.

        !!! note
            Result validators will NOT be called on the text result if `delta=True`.

        Args:
            delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.
        """
        return _utils.sync_async_iterator(self._streamed_run_result.stream_text(delta=delta, debounce_by=debounce_by))

    def stream_responses(self, *, debounce_by: float | None = 0.1) -> Iterator[tuple[_messages.ModelResponse, bool]]:
        """Stream the response as an iterable of Structured LLM Messages.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. `None` means no debouncing.
                Debouncing is particularly important for long structured responses to reduce the overhead of
                performing validation as each token is received.

        Returns:
            An iterable of the structured response message and whether that is the last message.
        """
        return _utils.sync_async_iterator(self._streamed_run_result.stream_responses(debounce_by=debounce_by))

    def get_output(self) -> OutputDataT:
        """Stream the whole response, validate and return it."""
        return _utils.get_event_loop().run_until_complete(self._streamed_run_result.get_output())

    @property
    def response(self) -> _messages.ModelResponse:
        """Return the current state of the response."""
        return self._streamed_run_result.response

    def usage(self) -> RunUsage:
        """Return the usage of the whole run.

        !!! note
            This won't return the full usage until the stream is finished.
        """
        return self._streamed_run_result.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._streamed_run_result.timestamp()

    @property
    def run_id(self) -> str:
        """The unique identifier for the agent run."""
        return self._streamed_run_result.run_id

    @property
    def metadata(self) -> dict[str, Any] | None:
        """Metadata associated with this agent run, if configured."""
        return self._streamed_run_result.metadata

    def validate_response_output(self, message: _messages.ModelResponse, *, allow_partial: bool = False) -> OutputDataT:
        """Validate a structured result message."""
        return _utils.get_event_loop().run_until_complete(
            self._streamed_run_result.validate_response_output(message, allow_partial=allow_partial)
        )

    @property
    def is_complete(self) -> bool:
        """Whether the stream has all been received.

        This is set to `True` when one of
        [`stream_output`][pydantic_ai.result.StreamedRunResultSync.stream_output],
        [`stream_text`][pydantic_ai.result.StreamedRunResultSync.stream_text],
        [`stream_responses`][pydantic_ai.result.StreamedRunResultSync.stream_responses] or
        [`get_output`][pydantic_ai.result.StreamedRunResultSync.get_output] completes.
        """
        return self._streamed_run_result.is_complete


@dataclass(repr=False)
class FinalResult(Generic[OutputDataT]):
    """Marker class storing the final output of an agent run and associated metadata."""

    output: OutputDataT
    """The final result data."""

    tool_name: str | None = None
    """Name of the final output tool; `None` if the output came from unstructured text content."""

    tool_call_id: str | None = None
    """ID of the tool call that produced the final output; `None` if the output came from unstructured text content."""

    __repr__ = _utils.dataclasses_no_defaults_repr


def _get_usage_checking_stream_response(
    stream_response: models.StreamedResponse,
    limits: UsageLimits | None,
    get_usage: Callable[[], RunUsage],
) -> AsyncIterator[ModelResponseStreamEvent]:
    if limits is not None and limits.has_token_limits():

        async def _usage_checking_iterator():
            async for item in stream_response:
                limits.check_tokens(get_usage())
                yield item

        return _usage_checking_iterator()
    else:
        return aiter(stream_response)


def _get_deferred_tool_requests(
    tool_calls: Iterable[_messages.ToolCallPart], tool_manager: ToolManager[AgentDepsT]
) -> DeferredToolRequests | None:
    """Get the deferred tool requests from the model response tool calls."""
    approvals: list[_messages.ToolCallPart] = []
    calls: list[_messages.ToolCallPart] = []

    for tool_call in tool_calls:
        tool_def = tool_manager.get_tool_def(tool_call.tool_name)
        if tool_def is not None:  # pragma: no branch
            if tool_def.kind == 'unapproved':
                approvals.append(tool_call)
            elif tool_def.kind == 'external':
                calls.append(tool_call)

    if not calls and not approvals:
        return None

    return DeferredToolRequests(calls=calls, approvals=approvals)
