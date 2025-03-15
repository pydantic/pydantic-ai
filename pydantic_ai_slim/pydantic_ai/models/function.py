from __future__ import annotations as _annotations

import inspect
import re
from collections.abc import AsyncIterator, Awaitable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import chain
from typing import Callable, Union

from typing_extensions import TypeAlias, assert_never, overload

from .. import _utils, usage
from .._utils import PeekableAsyncStream
from ..messages import (
    AudioUrl,
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import Model, ModelRequestParameters, StreamedResponse


@dataclass(init=False)
class FunctionModel(Model):
    """A model controlled by a local function.

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    function: FunctionDef | None = None
    stream_function: StreamFunctionDef | None = None

    _model_name: str = field(repr=False)
    _system: str = field(default='function', repr=False)

    @overload
    def __init__(self, function: FunctionDef, *, model_name: str | None = None) -> None: ...

    @overload
    def __init__(self, *, stream_function: StreamFunctionDef, model_name: str | None = None) -> None: ...

    @overload
    def __init__(
        self, function: FunctionDef, *, stream_function: StreamFunctionDef, model_name: str | None = None
    ) -> None: ...

    def __init__(
        self,
        function: FunctionDef | None = None,
        *,
        stream_function: StreamFunctionDef | None = None,
        model_name: str | None = None,
    ):
        """Initialize a `FunctionModel`.

        Either `function` or `stream_function` must be provided, providing both is allowed.

        Args:
            function: The function to call for non-streamed requests.
            stream_function: The function to call for streamed requests.
            model_name: The name of the model. If not provided, a name is generated from the function names.
        """
        if function is None and stream_function is None:
            raise TypeError('Either `function` or `stream_function` must be provided')
        self.function = function
        self.stream_function = stream_function

        function_name = self.function.__name__ if self.function is not None else ''
        stream_function_name = self.stream_function.__name__ if self.stream_function is not None else ''
        self._model_name = model_name or f'function:{function_name}:{stream_function_name}'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        agent_info = AgentInfo(
            model_request_parameters.function_tools,
            model_request_parameters.allow_text_result,
            model_request_parameters.result_tools,
            model_settings,
        )

        assert self.function is not None, 'FunctionModel must receive a `function` to support non-streamed requests'

        if inspect.iscoroutinefunction(self.function):
            response = await self.function(messages, agent_info)
        else:
            response_ = await _utils.run_in_executor(self.function, messages, agent_info)
            assert isinstance(response_, ModelResponse), response_
            response = response_
        response.model_name = self._model_name
        # TODO is `messages` right here? Should it just be new messages?
        return response, _estimate_usage(chain(messages, [response]))

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        agent_info = AgentInfo(
            model_request_parameters.function_tools,
            model_request_parameters.allow_text_result,
            model_request_parameters.result_tools,
            model_settings,
        )

        assert self.stream_function is not None, (
            'FunctionModel must receive a `stream_function` to support streamed requests'
        )

        response_stream = PeekableAsyncStream(self.stream_function(messages, agent_info))

        first = await response_stream.peek()
        if isinstance(first, _utils.Unset):
            raise ValueError('Stream function must return at least one item')

        yield FunctionStreamedResponse(_model_name=self._model_name, _iter=response_stream)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system


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
    tool_call_id: str | None = None
    """Incremental change to the tool call ID."""


DeltaToolCalls: TypeAlias = dict[int, DeltaToolCall]
"""A mapping of tool call IDs to incremental changes."""

# TODO: Change the signature to Callable[[list[ModelMessage], ModelSettings, ModelRequestParameters], ...]
FunctionDef: TypeAlias = Callable[[list[ModelMessage], AgentInfo], Union[ModelResponse, Awaitable[ModelResponse]]]
"""A function used to generate a non-streamed response."""

# TODO: Change signature as indicated above
StreamFunctionDef: TypeAlias = Callable[[list[ModelMessage], AgentInfo], AsyncIterator[Union[str, DeltaToolCalls]]]
"""A function used to generate a streamed response.

While this is defined as having return type of `AsyncIterator[Union[str, DeltaToolCalls]]`, it should
really be considered as `Union[AsyncIterator[str], AsyncIterator[DeltaToolCalls]`,

E.g. you need to yield all text or all `DeltaToolCalls`, not mix them.
"""


@dataclass
class FunctionStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for [FunctionModel][pydantic_ai.models.function.FunctionModel]."""

    _model_name: str
    _iter: AsyncIterator[str | DeltaToolCalls]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    def __post_init__(self):
        self._usage += _estimate_usage([])

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for item in self._iter:
            if isinstance(item, str):
                response_tokens = _estimate_string_tokens(item)
                self._usage += usage.Usage(response_tokens=response_tokens, total_tokens=response_tokens)
                yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=item)
            else:
                delta_tool_calls = item
                for dtc_index, delta_tool_call in delta_tool_calls.items():
                    if delta_tool_call.json_args:
                        response_tokens = _estimate_string_tokens(delta_tool_call.json_args)
                        self._usage += usage.Usage(response_tokens=response_tokens, total_tokens=response_tokens)
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=dtc_index,
                        tool_name=delta_tool_call.name,
                        args=delta_tool_call.json_args,
                        tool_call_id=delta_tool_call.tool_call_id,
                    )
                    if maybe_event is not None:
                        yield maybe_event

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _estimate_usage(messages: Iterable[ModelMessage]) -> usage.Usage:
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
                    request_tokens += _estimate_string_tokens(part.content)
                elif isinstance(part, ToolReturnPart):
                    request_tokens += _estimate_string_tokens(part.model_response_str())
                elif isinstance(part, RetryPromptPart):
                    request_tokens += _estimate_string_tokens(part.model_response())
                else:
                    assert_never(part)
        elif isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, TextPart):
                    response_tokens += _estimate_string_tokens(part.content)
                elif isinstance(part, ToolCallPart):
                    call = part
                    response_tokens += 1 + _estimate_string_tokens(call.args_as_json_str())
                else:
                    assert_never(part)
        else:
            assert_never(message)
    return usage.Usage(
        request_tokens=request_tokens, response_tokens=response_tokens, total_tokens=request_tokens + response_tokens
    )


def _estimate_string_tokens(content: str | Sequence[UserContent]) -> int:
    if not content:
        return 0
    if isinstance(content, str):
        return len(re.split(r'[\s",.:]+', content.strip()))
    else:  # pragma: no cover
        tokens = 0
        for part in content:
            if isinstance(part, str):
                tokens += len(re.split(r'[\s",.:]+', part.strip()))
            # TODO(Marcelo): We need to study how we can estimate the tokens for these types of content.
            if isinstance(part, (AudioUrl, ImageUrl)):
                tokens += 0
            elif isinstance(part, BinaryContent):
                tokens += len(part.data)
            else:
                tokens += 0
        return tokens
