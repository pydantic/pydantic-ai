from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, Union, cast

import logfire_api
from typing_extensions import TypeVar

from . import _result, _utils, exceptions, messages as _messages, models
from .tools import AgentDepsT, RunContext
from .usage import Usage, UsageLimits

__all__ = 'ResultDataT', 'ResultDataT_inv', 'ResultValidatorFunc', 'RunResult', 'StreamedRunResult'


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

Usage `ResultValidatorFunc[AgentDeps, T]`.
"""

_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')


@dataclass
class _BaseRunResult(ABC, Generic[ResultDataT]):
    """Base type for results.

    You should not import or use this type directly, instead use its subclasses `RunResult` and `StreamedRunResult`.
    """

    _all_messages: list[_messages.ModelMessage]
    _new_message_index: int

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
        """Return all messages from [`all_messages`][pydantic_ai.result._BaseRunResult.all_messages] as JSON bytes.

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
        """Return new messages from [`new_messages`][pydantic_ai.result._BaseRunResult.new_messages] as JSON bytes.

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

    @abstractmethod
    def usage(self) -> Usage:
        raise NotImplementedError()


@dataclass
class RunResult(_BaseRunResult[ResultDataT]):
    """Result of a non-streamed run."""

    data: ResultDataT
    """Data from the final response in the run."""
    _result_tool_name: str | None
    _usage: Usage

    def usage(self) -> Usage:
        """Return the usage of the whole run."""
        return self._usage

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
        if result_tool_return_content is not None:
            return self._set_result_tool_return(result_tool_return_content)
        else:
            return self._all_messages

    def _set_result_tool_return(self, return_content: str) -> list[_messages.ModelMessage]:
        """Set return content for the result tool.

        Useful if you want to continue the conversation and want to set the response to the result tool call.
        """
        if not self._result_tool_name:
            raise ValueError('Cannot set result tool return content when the return type is `str`.')
        messages = deepcopy(self._all_messages)
        last_message = messages[-1]
        for part in last_message.parts:
            if isinstance(part, _messages.ToolReturnPart) and part.tool_name == self._result_tool_name:
                part.content = return_content
                return messages
        raise LookupError(f'No tool call found with tool name {self._result_tool_name!r}.')


@dataclass
class StreamedRunResult(_BaseRunResult[ResultDataT], Generic[AgentDepsT, ResultDataT]):
    """Result of a streamed run that returns structured data via a tool call."""

    _usage_limits: UsageLimits | None
    _stream_response: models.StreamedResponse
    _result_schema: _result.ResultSchema[ResultDataT] | None
    _run_ctx: RunContext[AgentDepsT]
    _result_validators: list[_result.ResultValidator[AgentDepsT, ResultDataT]]
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

        usage_checking_stream = _get_usage_checking_stream_response(
            self._stream_response, self._usage_limits, self.usage
        )

        # Define a "merged" version of the iterator that will yield items that have already been retrieved
        # and items that we receive while streaming. We define a dedicated async iterator for this so we can
        # pass the combined stream to the group_by_temporal function within `_stream_text_deltas` below.
        async def _stream_text_deltas_ungrouped() -> AsyncIterator[tuple[str, int]]:
            # if the response currently has any parts with content, yield those before streaming
            msg = self._stream_response.get()
            for i, part in enumerate(msg.parts):
                if isinstance(part, _messages.TextPart) and part.content:
                    yield part.content, i

            async for event in usage_checking_stream:
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
                    yield ''.join([content for content, _ in items])

        with _logfire.span('response stream text') as lf_span:
            if delta:
                async for text in _stream_text_deltas():
                    yield text
            else:
                # a quick benchmark shows it's faster to build up a string with concat when we're
                # yielding at each step
                deltas: list[str] = []
                combined_validated_text = ''
                async for text in _stream_text_deltas():
                    deltas.append(text)
                    combined_text = ''.join(deltas)
                    combined_validated_text = await self._validate_text_result(combined_text)
                    yield combined_validated_text

                lf_span.set_attribute('combined_text', combined_validated_text)
                await self._marked_completed(
                    _messages.ModelResponse(
                        parts=[_messages.TextPart(combined_validated_text)],
                        model_name=self._stream_response.model_name(),
                    )
                )

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
        usage_checking_stream = _get_usage_checking_stream_response(
            self._stream_response, self._usage_limits, self.usage
        )

        with _logfire.span('response stream structured') as lf_span:
            # if the message currently has any parts with content, yield before streaming
            msg = self._stream_response.get()
            for part in msg.parts:
                if part.has_content():
                    yield msg, False
                    break

            async with _utils.group_by_temporal(usage_checking_stream, debounce_by) as group_iter:
                async for _events in group_iter:
                    msg = self._stream_response.get()
                    yield msg, False
                msg = self._stream_response.get()
                yield msg, True
                # TODO: Should this now be `final_response` instead of `structured_response`?
                lf_span.set_attribute('structured_response', msg)
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
        return self._run_ctx.usage + self._stream_response.usage()

    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._stream_response.timestamp()

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
