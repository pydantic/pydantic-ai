from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Callable, Union, cast

from typing_extensions import TypeAlias, overload

from .. import _utils, result
from ..messages import LLMMessage, LLMToolCalls, Message, ToolCall
from . import (
    AbstractToolDefinition,
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamTextResponse,
    StreamToolCallResponse,
)


@dataclass(frozen=True)
class AgentInfo:
    """Information about an agent passed to a function."""

    retrievers: Mapping[str, AbstractToolDefinition]
    allow_text_result: bool
    result_tools: list[AbstractToolDefinition] | None


@dataclass
class DeltaToolCall:
    name: str | None = None
    args: str | None = None


DeltaToolCalls = dict[int, DeltaToolCall]

# TODO these should be coroutines
FunctionDef: TypeAlias = Callable[[list[Message], AgentInfo], LLMMessage]
StreamFunctionDef: TypeAlias = Callable[[list[Message], AgentInfo], Union[Iterable[str], Iterable[DeltaToolCalls]]]


@dataclass
class ToolDescription:
    name: str
    description: str
    json_schema: _utils.ObjectJsonSchema


@dataclass(init=False)
class FunctionModel(Model):
    # NOTE: Avoid test discovery by pytest.
    __test__ = False

    function: FunctionDef | None = None
    stream_function: StreamFunctionDef | None = None

    @overload
    def __init__(self, function: FunctionDef) -> None: ...

    @overload
    def __init__(self, *, stream_function: StreamFunctionDef) -> None: ...

    def __init__(self, function: FunctionDef | None = None, *, stream_function: StreamFunctionDef | None = None):
        if function is None and stream_function is None:
            raise TypeError('Either `function` or `stream_function` must be provided')
        self.function = function
        self.stream_function = stream_function

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        result_tools = list(result_tools) if result_tools is not None else None
        return FunctionAgentModel(
            self.function, self.stream_function, AgentInfo(retrievers, allow_text_result, result_tools)
        )

    def name(self) -> str:
        labels: list[str] = []
        if self.function is not None:
            labels.append(self.function.__name__)
        if self.stream_function is not None:
            labels.append(f'stream-{self.stream_function.__name__}')
        return f'function:{",".join(labels)}'


@dataclass
class FunctionAgentModel(AgentModel):
    function: FunctionDef | None
    stream_function: StreamFunctionDef | None
    agent_info: AgentInfo

    async def request(self, messages: list[Message]) -> tuple[LLMMessage, result.Cost]:
        assert self.function is not None, 'FunctionModel must receive a `function` to support non-streamed requests'
        return self.function(messages, self.agent_info), result.Cost()

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        assert (
            self.stream_function is not None
        ), 'FunctionModel must receive a `stream_function` to support streamed requests'
        response_data = iter(self.stream_function(messages, self.agent_info))
        try:
            first = next(response_data)
        except StopIteration as e:
            raise ValueError('Stream function must return at least one item') from e

        if isinstance(first, str):
            text_stream = cast(Iterable[str], response_data)
            yield FunctionStreamTextResponse(iter(chain([first], text_stream)))
        else:
            structured_stream = cast(Iterable[DeltaToolCalls], response_data)
            # noinspection PyTypeChecker
            yield FunctionStreamToolCallResponse(iter(chain([first], structured_stream)), {})


@dataclass
class FunctionStreamTextResponse(StreamTextResponse):
    _iter: Iterator[str]
    _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        self._buffer.append(_utils.sync_anext(self._iter))

    def get(self) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> result.Cost:
        return result.Cost()

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class FunctionStreamToolCallResponse(StreamToolCallResponse):
    _iter: Iterator[DeltaToolCalls]
    _delta_tool_calls: dict[int, DeltaToolCall]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    async def __anext__(self) -> None:
        tool_call = _utils.sync_anext(self._iter)

        for key, new in tool_call.items():
            if current := self._delta_tool_calls.get(key):
                current.name = _utils.add_optional(current.name, new.name)
                current.args = _utils.add_optional(current.args, new.args)
            else:
                self._delta_tool_calls[key] = new

    def get(self) -> LLMToolCalls:
        """Map tool call deltas to a `LLMToolCalls`."""
        calls: list[ToolCall] = []
        for c in self._delta_tool_calls.values():
            if c.name is not None and c.args is not None:
                calls.append(ToolCall.from_json(c.name, c.args))

        return LLMToolCalls(calls, timestamp=self._timestamp)

    def cost(self) -> result.Cost:
        return result.Cost()

    def timestamp(self) -> datetime:
        return self._timestamp
