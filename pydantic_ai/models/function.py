"""A model controlled by a local function.

[FunctionModel][pydantic_ai.models.function.FunctionModel] is similar to [TestModel][pydantic_ai.models.test.TestModel],
but allows greater control over the model's behavior.

It's primary use case for more advanced unit testing than is possible with `TestModel`.
"""

from __future__ import annotations as _annotations

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Callable, Union, cast

import pydantic_core
from typing_extensions import TypeAlias, assert_never, overload

from .. import _utils, result
from ..messages import ArgsJson, Message, ModelAnyResponse, ModelStructuredResponse, ToolCall
from . import (
    AbstractToolDefinition,
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamStructuredResponse,
    StreamTextResponse,
)


@dataclass(init=False)
class FunctionModel(Model):
    """A model controlled by a local function.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    function: FunctionDef | None = None
    stream_function: StreamFunctionDef | None = None

    @overload
    def __init__(self, function: FunctionDef) -> None: ...

    @overload
    def __init__(self, *, stream_function: StreamFunctionDef) -> None: ...

    @overload
    def __init__(self, function: FunctionDef, *, stream_function: StreamFunctionDef) -> None: ...

    def __init__(self, function: FunctionDef | None = None, *, stream_function: StreamFunctionDef | None = None):
        """Initialize a `FunctionModel`.

        Either `function` or `stream_function` must be provided, providing both is allowed.

        Args:
            function: The function to call for non-streamed requests.
            stream_function: The function to call for streamed requests.
        """
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


@dataclass(frozen=True)
class AgentInfo:
    """Information about an agent.

    This is passed as the second to functions.
    """

    retrievers: Mapping[str, AbstractToolDefinition]
    """The retrievers available on this agent."""
    allow_text_result: bool
    """Whether a plain text result is allowed."""
    result_tools: list[AbstractToolDefinition] | None
    """The tools that can called as the final result of the run."""


@dataclass
class DeltaToolCall:
    """Incremental change to a tool call.

    Used to describe a chunk when streaming structured responses.
    """

    name: str | None = None
    """Incremental change to the name of the tool."""
    json_args: str | None = None
    """Incremental change to the arguments as JSON"""


DeltaToolCalls: TypeAlias = dict[int, DeltaToolCall]
"""A mapping of tool call IDs to incremental changes."""

FunctionDef: TypeAlias = Callable[[list[Message], AgentInfo], Union[ModelAnyResponse, Awaitable[ModelAnyResponse]]]
"""A function used to generate a non-streamed response."""

StreamFunctionDef: TypeAlias = Callable[[list[Message], AgentInfo], AsyncIterator[Union[str, DeltaToolCalls]]]
"""A function used to generate a streamed response.

While this is defined as having return type of `AsyncIterator[Union[str, DeltaToolCalls]]`, it should
really be considered as `Union[AsyncIterator[str], AsyncIterator[DeltaToolCalls]`,

E.g. you need to yield all text or all `DeltaToolCalls`, not mix them.
"""


@dataclass
class FunctionAgentModel(AgentModel):
    """Implementation of `AgentModel` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""

    function: FunctionDef | None
    stream_function: StreamFunctionDef | None
    agent_info: AgentInfo

    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, result.Cost]:
        assert self.function is not None, 'FunctionModel must receive a `function` to support non-streamed requests'
        if inspect.iscoroutinefunction(self.function):
            response = await self.function(messages, self.agent_info)
        else:
            response_ = await _utils.run_in_executor(self.function, messages, self.agent_info)
            response = cast(ModelAnyResponse, response_)
        # TODO is `messages` right here? Should it just be new messages?
        return response, _estimate_cost(chain(messages, [response]))

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        assert (
            self.stream_function is not None
        ), 'FunctionModel must receive a `stream_function` to support streamed requests'
        response_stream = self.stream_function(messages, self.agent_info)
        try:
            first = await response_stream.__anext__()
        except StopAsyncIteration as e:
            raise ValueError('Stream function must return at least one item') from e

        if isinstance(first, str):
            text_stream = cast(AsyncIterator[str], response_stream)
            yield FunctionStreamTextResponse(first, text_stream)
        else:
            structured_stream = cast(AsyncIterator[DeltaToolCalls], response_stream)
            yield FunctionStreamStructuredResponse(first, structured_stream)


@dataclass
class FunctionStreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""

    _next: str | None
    _iter: AsyncIterator[str]
    _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        if self._next is not None:
            self._buffer.append(self._next)
            self._next = None
        else:
            self._buffer.append(await self._iter.__anext__())

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> result.Cost:
        return result.Cost()

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class FunctionStreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""

    _next: DeltaToolCalls | None
    _iter: AsyncIterator[DeltaToolCalls]
    _delta_tool_calls: dict[int, DeltaToolCall] = field(default_factory=dict)
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    async def __anext__(self) -> None:
        if self._next is not None:
            tool_call = self._next
            self._next = None
        else:
            tool_call = await self._iter.__anext__()

        for key, new in tool_call.items():
            if current := self._delta_tool_calls.get(key):
                current.name = _utils.add_optional(current.name, new.name)
                current.json_args = _utils.add_optional(current.json_args, new.json_args)
            else:
                self._delta_tool_calls[key] = new

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[ToolCall] = []
        for c in self._delta_tool_calls.values():
            if c.name is not None and c.json_args is not None:
                calls.append(ToolCall.from_json(c.name, c.json_args))

        return ModelStructuredResponse(calls, timestamp=self._timestamp)

    def cost(self) -> result.Cost:
        return result.Cost()

    def timestamp(self) -> datetime:
        return self._timestamp


def _estimate_cost(messages: Iterable[Message]) -> result.Cost:
    """Very rough guesstimate of the number of tokens associate with a series of messages.

    This is designed to be used solely to give plausible numbers for testing!
    """
    # there seem to be about 50 tokens of overhead for both Gemini and OpenAI calls, so add that here ¯\_(ツ)_/¯

    request_tokens = 50
    response_tokens = 0
    for message in messages:
        if message.role == 'system' or message.role == 'user':
            request_tokens += _string_cost(message.content)
        elif message.role == 'tool-return':
            request_tokens += _string_cost(message.model_response_str())
        elif message.role == 'retry-prompt':
            request_tokens += _string_cost(message.model_response())
        elif message.role == 'model-text-response':
            response_tokens += _string_cost(message.content)
        elif message.role == 'model-structured-response':
            for call in message.calls:
                if isinstance(call.args, ArgsJson):
                    args_str = call.args.args_json
                else:
                    args_str = pydantic_core.to_json(call.args.args_object).decode()

                response_tokens += 1 + _string_cost(args_str)
        else:
            assert_never(message)
    return result.Cost(
        request_tokens=request_tokens, response_tokens=response_tokens, total_tokens=request_tokens + response_tokens
    )


def _string_cost(content: str) -> int:
    return len(re.split(r'[\s",.:]+', content))
