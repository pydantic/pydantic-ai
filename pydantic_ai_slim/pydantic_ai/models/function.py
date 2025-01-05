from __future__ import annotations as _annotations

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import chain
from typing import Callable, Union

from typing_extensions import TypeAlias, assert_never, overload

from .. import _utils, result
from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import AgentModel, Model, StreamedResponse


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

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        return FunctionAgentModel(
            self.function, self.stream_function, AgentInfo(function_tools, allow_text_result, result_tools, None)
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

    This is passed as the second to functions used within [`FunctionModel`][pydantic_ai.models.function.FunctionModel].
    """

    function_tools: list[ToolDefinition]
    """The function tools available on this agent.

    These are the tools registered via the [`tool`][pydantic_ai.Agent.tool] and
    [`tool_plain`][pydantic_ai.Agent.tool_plain] decorators.
    """
    allow_text_result: bool
    """Whether a plain text result is allowed."""
    result_tools: list[ToolDefinition]
    """The tools that can called as the final result of the run."""
    model_settings: ModelSettings | None
    """The model settings passed to the run call."""


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

FunctionDef: TypeAlias = Callable[[list[ModelMessage], AgentInfo], Union[ModelResponse, Awaitable[ModelResponse]]]
"""A function used to generate a non-streamed response."""

StreamFunctionDef: TypeAlias = Callable[[list[ModelMessage], AgentInfo], AsyncIterator[Union[str, DeltaToolCalls]]]
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

    async def request(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> tuple[ModelResponse, result.Usage]:
        agent_info = replace(self.agent_info, model_settings=model_settings)

        assert self.function is not None, 'FunctionModel must receive a `function` to support non-streamed requests'
        if inspect.iscoroutinefunction(self.function):
            response = await self.function(messages, agent_info)
        else:
            response_ = await _utils.run_in_executor(self.function, messages, agent_info)
            assert isinstance(response_, ModelResponse), response_
            response = response_
        # TODO is `messages` right here? Should it just be new messages?
        return response, _estimate_usage(chain(messages, [response]))

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None
    ) -> AsyncIterator[StreamedResponse]:
        assert (
            self.stream_function is not None
        ), 'FunctionModel must receive a `stream_function` to support streamed requests'
        response_stream = self.stream_function(messages, self.agent_info)

        # Explicitly check that we get at least one value, so we can produce a nicer error message for misuse
        try:
            first = await response_stream.__anext__()
        except StopAsyncIteration as e:
            raise ValueError('Stream function must return at least one item') from e

        async def peeked_stream():
            yield first
            async for item in response_stream:
                yield item

        yield FunctionStreamedResponse(peeked_stream())


_CONTENT_PART_INDEX = -1


@dataclass
class FunctionStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""

    _iter: AsyncIterator[str | DeltaToolCalls]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    _parts: dict[int, ModelResponsePart] = field(default_factory=dict, init=False)
    _event_iterator: AsyncIterator[ModelResponseStreamEvent | None] | None = field(default=None, init=False)

    async def __anext__(self) -> ModelResponseStreamEvent | None:
        if self._event_iterator is None:
            self._event_iterator = self._get_event_iterator()
        return await self._event_iterator.__anext__()

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent | None]:
        async for item in self._iter:
            if isinstance(item, str):
                text = item
                content_part = self._parts.get(_CONTENT_PART_INDEX)
                if content_part is None:
                    content_part = TextPart(content=text)
                    self._parts[_CONTENT_PART_INDEX] = content_part
                    yield PartStartEvent(index=_CONTENT_PART_INDEX, part=content_part)
                else:
                    assert isinstance(content_part, TextPart), 'Cannot switch to text part mid-stream'
                    delta = TextPartDelta(content_delta=text)
                    self._parts[_CONTENT_PART_INDEX] = delta.apply(content_part)
                    yield PartDeltaEvent(index=_CONTENT_PART_INDEX, delta=delta)
            else:
                delta_tool_calls = item
                for index, delta_tool_call in delta_tool_calls.items():
                    existing_part = self._parts.get(index)
                    if existing_part is None:
                        assert (
                            delta_tool_call.name is not None
                        ), 'The first delta_tool_call with a given index must include a tool name'
                        part = ToolCallPart.from_raw_args(
                            tool_name=delta_tool_call.name, args=delta_tool_call.json_args or ''
                        )
                        self._parts[index] = part
                        yield PartStartEvent(index=index, part=part)
                    else:
                        assert isinstance(existing_part, ToolCallPart), 'Cannot switch to tool call part mid-stream'
                        if delta_tool_call.json_args is not None:
                            delta = ToolCallPartDelta(delta_tool_call.json_args)
                            self._parts[index] = delta.apply(existing_part)
                            yield PartDeltaEvent(index=index, delta=delta)

    def get(self, *, final: bool = False) -> ModelResponse:
        parts = [self._parts[index] for index in sorted(self._parts)]
        return ModelResponse(parts, timestamp=self._timestamp)

    def usage(self) -> result.Usage:
        return _estimate_usage([self.get()])

    def timestamp(self) -> datetime:
        return self._timestamp


def _estimate_usage(messages: Iterable[ModelMessage]) -> result.Usage:
    """Very rough guesstimate of the token usage associated with a series of messages.

    This is designed to be used solely to give plausible numbers for testing!
    """
    # there seem to be about 50 tokens of overhead for both Gemini and OpenAI calls, so add that here ¯\_(ツ)_/¯
    request_tokens = 50
    response_tokens = 0
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, (SystemPromptPart, UserPromptPart)):
                    request_tokens += _estimate_string_usage(part.content)
                elif isinstance(part, ToolReturnPart):
                    request_tokens += _estimate_string_usage(part.model_response_str())
                elif isinstance(part, RetryPromptPart):
                    request_tokens += _estimate_string_usage(part.model_response())
                else:
                    assert_never(part)
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    response_tokens += _estimate_string_usage(part.content)
                elif isinstance(part, ToolCallPart):
                    call = part
                    response_tokens += 1 + _estimate_string_usage(call.args_as_json_str())
                else:
                    assert_never(part)
        else:
            assert_never(message)
    return result.Usage(
        request_tokens=request_tokens, response_tokens=response_tokens, total_tokens=request_tokens + response_tokens
    )


def _estimate_string_usage(content: str) -> int:
    if not content:
        return 0
    return len(re.split(r'[\s",.:]+', content))
