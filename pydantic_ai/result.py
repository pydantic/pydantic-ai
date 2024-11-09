from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import logfire_api

from . import _result, _utils, exceptions, messages, models
from .call_typing import AgentDeps

__all__ = (
    'ResultData',
    'Cost',
    'RunResult',
    'StreamedRunResult',
)


ResultData = TypeVar('ResultData')
_logfire = logfire_api.Logfire(otel_scope='pydantic-ai')


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
class _BaseRunResult(ABC, Generic[ResultData]):
    """Result of a run."""

    _all_messages: list[messages.Message]
    _new_message_index: int

    def all_messages(self) -> list[messages.Message]:
        """Return the history of messages."""
        # this is a method to be consistent with the other methods
        return self._all_messages

    def all_messages_json(self) -> bytes:
        """Return the history of messages as JSON bytes."""
        return messages.MessagesTypeAdapter.dump_json(self.all_messages())

    def new_messages(self) -> list[messages.Message]:
        """Return new messages associated with this run.

        System prompts and any messages from older runs are excluded.
        """
        return self.all_messages()[self._new_message_index :]

    def new_messages_json(self) -> bytes:
        """Return new messages from [new_messages][] as JSON bytes."""
        return messages.MessagesTypeAdapter.dump_json(self.new_messages())

    @abstractmethod
    def cost(self) -> Cost:
        """Return the cost of the whole run."""
        raise NotImplementedError()


@dataclass
class RunResult(_BaseRunResult[ResultData]):
    """Result of a run."""

    response: ResultData
    _cost: Cost

    def cost(self) -> Cost:
        return self._cost


@dataclass
class StreamedRunResult(_BaseRunResult[ResultData], Generic[AgentDeps, ResultData]):
    """Result of a streamed run that returns structured data via a tool call."""

    cost_so_far: Cost
    """Cost up until the last request."""
    _stream_response: models.StreamTextResponse | models.StreamToolCallResponse
    _result_schema: _result.ResultSchema[ResultData] | None
    _deps: AgentDeps
    _result_validators: list[_result.ResultValidator[AgentDeps, ResultData]]
    is_complete: bool = False
    """Whether the stream has all been received."""

    async def stream(
        self,
        *,
        text_delta: bool = False,
        debounce_by: float | None = 0.1,
    ) -> AsyncIterator[ResultData]:
        """Stream the response as an async iterable.

        Result validators are called on each iteration, if `text_delta=False` (the default) or for structured
        responses.

        !!!
            Note that the result validators will NOT be called on the final text result if `text_delta=True`.

        The pydantic validator for the tool type will be called in [partial mode](#) on each iteration.

        Args:
            text_delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. if `AUTO` (default),
                the response stream is debounced by 0.2 seconds unless `text_delta` is `True`, in which case it
                doesn't make sense to debounce. `None` means no debouncing. Debouncing is important particularly
                for long structured responses to reduce the overhead of performing validation as each token is received.

        Returns: An async iterable of the response data.
        """
        if isinstance(self._stream_response, models.StreamTextResponse):
            with _logfire.span('response stream text') as lf_span:
                if text_delta:
                    async with _utils.group_by_temporal(self._stream_response, debounce_by) as group_iter:
                        async for chunks in group_iter:
                            yield ''.join(chunks)  # pyright: ignore[reportReturnType]
                else:
                    # a quick benchmark shows it's faster to build up a string with concat when we're
                    # yielding at each step
                    combined = ''
                    async with _utils.group_by_temporal(self._stream_response, debounce_by) as group_iter:
                        async for chunks in group_iter:
                            combined += ''.join(chunks)
                            combined = await self._validate_text_result(combined)
                            yield cast(ResultData, combined)
                    lf_span.set_attribute('combined_text', combined)
                self.is_complete = True
        else:
            assert not text_delta, 'Cannot use `text_delta=True` for structured responses'
            async for tool_message in self.stream_structured(debounce_by=debounce_by):
                yield await self.validate_structured_result(tool_message, allow_partial=True)

    async def stream_structured(self, *, debounce_by: float | None = 0.1) -> AsyncIterator[messages.LLMToolCalls]:
        """Stream the response as an async iterable of Structured LLM Messages.

        !!! note
            This method is only available for structured responses, e.g. if `is_structured()` returns `True`.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. if `AUTO` (default),
                the response stream is debounced by 0.2 seconds unless `text_delta` is `True`, in which case it
                doesn't make sense to debounce. `None` means no debouncing. Debouncing is important particularly
                for long structured responses to reduce the overhead of performing validation as each token is received.
        """
        with _logfire.span('response stream structured') as lf_span:
            if isinstance(self._stream_response, models.StreamTextResponse):
                raise exceptions.UserError('stream_messages() can only be used with structured responses')
            else:
                # we should already have a message at this point, yield that first if it has any content
                msg = self._stream_response.get()
                if any(call.has_content() for call in msg.calls):
                    yield msg
                async with _utils.group_by_temporal(self._stream_response, debounce_by) as group_iter:
                    async for _ in group_iter:
                        msg = self._stream_response.get()
                        if any(call.has_content() for call in msg.calls):
                            yield msg
                lf_span.set_attribute('structured_response', msg)
            self.is_complete = True

    async def get_response(self) -> ResultData:
        """Stream the whole response, validate and return it."""
        if isinstance(self._stream_response, models.StreamTextResponse):
            text = ''.join([chunk async for chunk in self._stream_response])
            text = await self._validate_text_result(text)
            self.is_complete = True
            return cast(ResultData, text)
        else:
            async for _ in self._stream_response:
                pass
            tool_message = self._stream_response.get()
            self.is_complete = True
            return await self.validate_structured_result(tool_message)

    def is_structured(self) -> bool:
        """Return whether the stream response contains structured data (as opposed to text)."""
        return isinstance(self._stream_response, models.StreamToolCallResponse)

    def cost(self) -> Cost:
        """Return the cost of the whole run.

        NOTE: this won't return the full cost until the stream is finished.
        """
        return self.cost_so_far + self._stream_response.cost()

    async def validate_structured_result(
        self, message: messages.LLMToolCalls, *, allow_partial: bool = False
    ) -> ResultData:
        assert self._result_schema is not None, 'Expected _result_schema to not be None'
        match = self._result_schema.find_tool(message)
        if match is None:
            raise exceptions.UnexpectedModelBehaviour(
                f'Invalid message, unable to find tool: {self._result_schema.tool_names()}'
            )

        call, result_tool = match
        result_data = result_tool.validate(call, allow_partial=allow_partial, wrap_validation_errors=False)

        for validator in self._result_validators:
            result_data = await validator.validate(result_data, self._deps, 0, call)
        return result_data

    async def _validate_text_result(self, text: str) -> str:
        for validator in self._result_validators:
            text = await validator.validate(  # pyright: ignore[reportAssignmentType]
                text,  # pyright: ignore[reportArgumentType]
                self._deps,
                0,
                None,
            )
        return text
