from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

from httpx import AsyncClient as AsyncHTTPClient
from mistralai import CompletionChunk, FunctionCall, TextChunk
from mistralai.types.basemodel import Unset
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, result
from .._utils import now_utc as _now_utc
from ..messages import (
    ArgsJson,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolCall as PydanticToolCall,
    ToolReturn,
)
from . import (
    AbstractToolDefinition,
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamTextResponse,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from mistralai import Mistral, models
    from mistralai.models import (
        ChatCompletionResponse,
        CompletionEvent,
        Tool,
        ToolCall,
        ToolTypedDict,
    )
    from mistralai.models.assistantmessage import AssistantMessage
    from mistralai.models.function import Function
    from mistralai.models.toolmessage import ToolMessage
    from mistralai.models.usermessage import UserMessage
    from mistralai.types import UNSET
    from mistralai.utils.eventstreaming import EventStreamAsync
except ImportError as e:
    raise ImportError(
        "Please install `mistral` to use the Mistral model, "
        "you can use the `mistral` optional group — `pip install 'pydantic-ai[mistral]'`"
    ) from e

MistralModelName = Literal[
    'mistral-small-latest',
    'small-mistral',
    'mistral-large-latest',
    'codestral-latest',
]


@dataclass(init=False)
class MistralModel(Model):
    """A model that uses Mistral.

    Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.

    [API Documentation](https://docs.mistral.ai/)

    """

    model_name: MistralModelName | str
    client: Mistral = field(repr=False)

    def __init__(
        self,
        model_name: MistralModelName,
        *,
        api_key: str | Callable[[], str | None] | None = None,
        client: Mistral | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        self.model_name = model_name

        if client is not None:
            assert (
                http_client is None
            ), 'Cannot provide both `mistral_client` and `http_client`'
            self.client = client
        elif http_client is not None:
            self.client = Mistral(api_key=api_key, async_client=http_client)
        else:
            self.client = Mistral(
                api_key=api_key, async_client=cached_async_http_client()
            )

    async def agent_model(
        self,
        function_tools: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> AgentModel:
        check_allow_model_requests()
        tools: list[Tool | ToolTypedDict | None] = [self._map_tool_definition(r) for r in function_tools.values()]
        if result_tools is not None:
            tools += [self._map_tool_definition(r) for r in result_tools]
        return MistralAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            tools,
        )

    def name(self) -> str:
        return f'mistral:{self.model_name}'

    @staticmethod
    def _map_tool_definition(
        f: AbstractToolDefinition,
    ) -> Tool | ToolTypedDict:
        """Convert an `AbstractToolDefinition` to a `Tool` or `ToolTypedDict`.

        This is a utility function used to convert our internal representation of a tool to the
        representation expected by the Mistral API.
        """
        function = Function(
            name=f.name, parameters=f.json_schema, description=f.description
        )
        return Tool(function=function)


@dataclass
class MistralAgentModel(AgentModel):
    """Implementation of `AgentModel` for Mistral models."""

    client: Mistral
    model_name: str
    allow_text_result: bool
    tools: list[Tool] | list[ToolTypedDict]| None = None 

    async def request(
        self, messages: list[Message]
    ) -> tuple[ModelAnyResponse, result.Cost]:
        response = await self._completions_create(messages, False)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(
        self, messages: list[Message]
    ) -> AsyncIterator[EitherStreamedResponse]:
        response = await self._stream_create(messages, True)
        async with response:
            yield await self._process_streamed_response(response)

    async def _completions_create(
        self, messages: list[Message], stream: bool
    ) -> ChatCompletionResponse:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        mistral_messages = [self._map_message(m) for m in messages]
        response = await self.client.chat.complete_async(
            model=str(self.model_name),
            messages=mistral_messages,
            temperature=0.0,
            n=1,
            tools=self.tools or UNSET, 
            tool_choice=tool_choice or None,
            stream=stream,
        )
        assert response, 'TODO: fix this'  # TODO: see when None
        return response

    async def _stream_create(
        self, messages: list[Message], stream: bool
    ) -> EventStreamAsync[CompletionEvent]:
        # standalone function to make it easier to override
        if not self.tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
        elif not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'

        mistral_messages = [self._map_message(m) for m in messages]
        response = await self.client.chat.stream_async(
            model=str(self.model_name),
            messages=mistral_messages,
            temperature=0.0,
            n=1,
            tools=self.tools or UNSET,
            tool_choice=tool_choice or None,
            stream=stream,
        )
        assert response, 'TODO: fix this'  # TODO: see when None
        return response

    @staticmethod
    def _process_response(response: ChatCompletionResponse) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp: datetime
        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = _now_utc()

       
        assert response.choices # TODO: see how improve
        choice = response.choices[0]

        if (
            choice.message.tool_calls is not None
            and not isinstance(choice.message.tool_calls, Unset)
        ):
            tools_calls = choice.message.tool_calls
            tools = [
                (
                    PydanticToolCall.from_json(
                        tool_name=c.function.name,
                        args_json=c.function.arguments,
                        tool_id=c.id,
                    )
                    if isinstance(c.function.arguments, str)
                    else PydanticToolCall.from_dict(
                        tool_name=c.function.name,
                        args_dict=c.function.arguments,
                        tool_id=c.id,
                    )
                )
                for c in tools_calls
            ]
            return ModelStructuredResponse(
                tools,
                timestamp=timestamp,
            )
        else:
            content = choice.message.content
            assert content, f'Unexpected null content is assitant msg: {choice.message}'
            assert not isinstance(
                content, list
            ), f'Unexpected ContentChunk from stream, need to be response not stream: {content}'
            return ModelTextResponse(content, timestamp=timestamp)

    @staticmethod
    async def _process_streamed_response(
        response: EventStreamAsync[CompletionEvent],
    ) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        try:
            first_chunk = await response.__anext__()
        except StopAsyncIteration as e:  # pragma: no cover
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e
        timestamp = datetime.fromtimestamp(first_chunk.data.created or 0, tz=timezone.utc) # TODO: see 0 now or not
        choice = first_chunk.data.choices[0]
        delta = choice.delta
        start_cost = _map_cost(first_chunk.data)

        
        # the first chunk may only contain `role`, so we iterate until we get either `tool_calls` or `content`
        while delta.tool_calls is None or isinstance(delta.tool_calls, Unset) and delta.content is None or len(delta.content) == 0:
                try:
                    next_chunk = await response.__anext__()
                except StopAsyncIteration as e:
                    raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e
                delta = next_chunk.data.choices[0].delta
                start_cost += _map_cost(next_chunk.data)

        if delta.tool_calls is not None and isinstance(delta.tool_calls, Unset):
            return MistralStreamTextResponse(delta.content, response, timestamp, start_cost)
        else:
            assert False # TODO:
            assert delta.tool_calls is not None, f'Expected delta with tool_calls, got {delta}'
            return GroqStreamStructuredResponse(
                response,
                {c.index: c for c in delta.tool_calls},
                timestamp,
                start_cost,
            )


    @staticmethod
    def _map_message(message: Message) -> models.Messages:
        """Just maps a `pydantic_ai.Message` to a `Mistral.types.ChatCompletionMessageParam`."""
        if message.role == 'system':
            # SystemPrompt ->
            return AssistantMessage(content=message.content)
        elif message.role == 'user':
            # UserPrompt ->
            return UserMessage(content=message.content)
        elif message.role == 'tool-return':
            # ToolReturn ->
            return ToolMessage(
                tool_call_id=_guard_tool_id(message),
                content=message.model_response_str(),
            )
        elif message.role == 'retry-prompt':
            # RetryPrompt ->
            if message.tool_name is None:
                return UserMessage(content=message.model_response())
            else:
                return ToolMessage(
                    tool_call_id=_guard_tool_id(message),
                    content=message.model_response(),
                )
        elif message.role == 'model-text-response':
            # ModelTextResponse ->
            return AssistantMessage(content=message.content)
        elif message.role == 'model-structured-response':
            assert (
                message.role == 'model-structured-response'
            ), f'Expected role to be "llm-tool-calls", got {message.role}'
            # ModelStructuredResponse ->
            return AssistantMessage(
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


def _guard_tool_id(t: PydanticToolCall | RetryPrompt | ToolReturn) -> str:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    assert t.tool_id is not None, f'Mistral requires `tool_id` to be set: {t}'
    return t.tool_id


def _map_tool_call(t: PydanticToolCall) -> ToolCall:
    assert isinstance(t.args, ArgsJson), f'Expected ArgsDict, got {t.args}'
    return ToolCall(
        id=_guard_tool_id(t),
        type='function',
        function=FunctionCall(name=t.tool_name, arguments=t.args.args_json),
    )
    
def _map_cost(response: ChatCompletionResponse | CompletionChunk) -> result.Cost:
    usage = response.usage
    if usage is None:
        return result.Cost()
    else:
        return result.Cost(
            request_tokens=usage.prompt_tokens,
            response_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            details=None,
        )


@dataclass
class MistralStreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for Groq models."""

    _first: str | None
    _response: EventStreamAsync[CompletionEvent]
    _timestamp: datetime
    _cost: result.Cost
    _buffer: list[str] = field(default_factory=list, init=False)


    async def __anext__(self) -> None:
        if self._first is not None:
            self._buffer.append(self._first)
            self._first = None
            return None

        chunk = await self._response.generator.__anext__()
        self._cost = _map_cost(chunk.data)

        try:
            choice = chunk.data.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        # we don't raise StopAsyncIteration on the last chunk because usage comes after this
        if choice.finish_reason is None:
            assert choice.delta.content is not None, f'Expected delta with content, invalid chunk: {chunk!r}'
        if choice.delta.content is not None:
            if isinstance(choice.delta.content, str):
                self._buffer.append(choice.delta.content)
            elif isinstance(choice.delta.content, TextChunk):
                self._buffer.append(choice.delta.content.text)

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> result.Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp

# @dataclass
# class MistralStreamStructuredResponse(StreamStructuredResponse):
#     """Implementation of `StreamStructuredResponse` for Groq models."""

#     _response: AsyncStream[ChatCompletionChunk]
#     _delta_tool_calls: dict[int, ChoiceDeltaToolCall]
#     _timestamp: datetime
#     _cost: result.Cost

#     async def __anext__(self) -> None:
#         chunk = await self._response.__anext__()
#         self._cost = _map_cost(chunk)

#         try:
#             choice = chunk.choices[0]
#         except IndexError:
#             raise StopAsyncIteration()

#         if choice.finish_reason is not None:
#             raise StopAsyncIteration()

#         assert choice.delta.content is None, f'Expected tool calls, got content instead, invalid chunk: {chunk!r}'

#         for new in choice.delta.tool_calls or []:
#             if current := self._delta_tool_calls.get(new.index):
#                 if current.function is None:
#                     current.function = new.function
#                 elif new.function is not None:
#                     current.function.name = _utils.add_optional(current.function.name, new.function.name)
#                     current.function.arguments = _utils.add_optional(current.function.arguments, new.function.arguments)
#             else:
#                 self._delta_tool_calls[new.index] = new

#     def get(self, *, final: bool = False) -> ModelStructuredResponse:
#         calls: list[ToolCall] = []
#         for c in self._delta_tool_calls.values():
#             if f := c.function:
#                 if f.name is not None and f.arguments is not None:
#                     calls.append(ToolCall.from_json(f.name, f.arguments, c.id))

#         return ModelStructuredResponse(calls, timestamp=self._timestamp)

#     def cost(self) -> Cost:
#         return self._cost

#     def timestamp(self) -> datetime:
#         return self._timestamp
    
