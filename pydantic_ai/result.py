from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Generic, TypeVar, Union, cast

from . import _result, _utils, exceptions, messages, models
from .call_typing import AgentDeps

__all__ = (
    'ResultData',
    'Cost',
    'RunResult',
    'EitherStreamedRunResult',
    'StreamedTextRunResult',
    'StreamedToolCallRunResult',
)

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


class Auto:
    pass


AUTO = Auto()
DEFAULT_DEBOUNCE = 0.2


@dataclass
class StreamedTextRunResult(_BaseRunResult[str], Generic[AgentDeps]):
    """Text result of a streamed run."""

    cost_so_far: Cost
    """Cost up until the last request."""
    _stream_response: models.StreamTextResponse
    _deps: AgentDeps
    _result_validators: list[_result.ResultValidator[AgentDeps, str]]

    async def response_stream(
        self, text_delta: bool = False, debounce_by: float | None | Auto = AUTO
    ) -> AsyncIterator[str]:
        """Stream the response text as an async iterable.

        Result validators are called on each iteration, if `text_delta=False`.

        !!!
            Note this means that the result validators will NOT be called on the final result if `text_delta=True`.

        Args:
            text_delta: if `True`, yield each chunk of text as it is received, if `False` (default), yield the full text
                up to the current point.
            debounce_by: by how much (if at all) to debounce/group the response chunks by. if `AUTO` (default),
                the response stream is debounced by 0.2 seconds unless `text_delta` is `True`, in which case it
                doesn't make sense to debounce. `None` means no debouncing. Debouncing is important particularly
                for long structured responses to reduce the overhead of performing validation as each token is received.

        Returns: An async iterable of the response data.
        """
        if text_delta:
            assert debounce_by is AUTO or debounce_by is None, 'debounce_by must be None or AUTO if text_delta=True'
            async for chunk in self._stream_response:
                yield chunk
        else:
            # a quick benchmark shows it's faster build up a string with concat when we're yielding at each step
            combined = ''
            soft_max_interval = DEFAULT_DEBOUNCE if debounce_by is AUTO else cast(float, debounce_by)
            async for chunks in _utils.group_by_temporal(self._stream_response, soft_max_interval):
                combined += ''.join(chunks)
                combined = await self._validate_result(combined)
                yield combined

    async def get_response(self) -> str:
        """Stream the whole response, validate and return it."""
        text = ''.join([chunk async for chunk in self._stream_response])
        return await self._validate_result(text)

    def cost(self) -> Cost:
        """Return the cost of the whole run.

        NOTE: this won't return the full cost until the stream is finished.
        """
        return self.cost_so_far + self._stream_response.cost()

    async def _validate_result(self, text: str) -> str:
        for validator in self._result_validators:
            text = await validator.validate(text, self._deps, 0, None)
        return text


@dataclass
class StreamedToolCallRunResult(_BaseRunResult[ResultData], Generic[ResultData, AgentDeps]):
    """Result of a streamed run that returns structured data via a tool call."""

    cost_so_far: Cost
    """Cost up until the last request."""
    _stream_response: models.StreamToolCallResponse
    _result_schema: _result.ResultSchema[ResultData]
    _deps: AgentDeps
    _result_validators: list[_result.ResultValidator[AgentDeps, ResultData]]

    async def response_stream(self, debounce_by: float | None = DEFAULT_DEBOUNCE) -> AsyncIterator[ResultData]:
        """Stream the response structured data as an async iterable.

        The pydantic validator for the tool type will be called in [partial mode](#) on each iteration.

        Args:
            debounce_by: by how much (if at all) to debounce/group the response chunks by. if `AUTO` (default),
                the response stream is debounced by 0.2 seconds unless `text_delta` is `True`, in which case it
                doesn't make sense to debounce. `None` means no debouncing. Debouncing is important particularly
                for long structured responses to reduce the overhead of performing validation as each token is received.

        Returns: An async iterable of the response data.
        """
        async for _ in _utils.group_by_temporal(self._stream_response, debounce_by):
            tool_message = self._stream_response.get()
            yield await self._validate_result(tool_message)

    async def get_response(self) -> ResultData:
        """Stream the whole response, validate and return it."""
        async for _ in self._stream_response:
            pass
        tool_message = self._stream_response.get()
        return await self._validate_result(tool_message)

    def cost(self) -> Cost:
        """Return the cost of the whole run.

        NOTE: this won't return the full cost until the stream is finished.
        """
        return self.cost_so_far + self._stream_response.cost()

    async def _validate_result(self, message: messages.LLMToolCalls) -> ResultData:
        match = self._result_schema.find_tool(message)
        if match is None:
            raise exceptions.UnexpectedModelBehaviour(
                f'Invalid message, unable to find tool: {self._result_schema.tool_names()}'
            )

        call, result_tool = match
        result_data = result_tool.validate(call)

        for validator in self._result_validators:
            result_data = await validator.validate(result_data, self._deps, 0, call)
        return result_data


# Usage `EitherStreamedRunResult[ResultData, AgentDeps]`
EitherStreamedRunResult = Union[StreamedTextRunResult[AgentDeps], StreamedToolCallRunResult[ResultData, AgentDeps]]