from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from httpx import AsyncClient as AsyncHTTPClient
from typing_extensions import assert_never

from .. import UnexpectedModelBehavior
from .._utils import now_utc as _now_utc
from ..messages import (
    ArgsJson,
    Message,
    ModelAnyResponse,
    ModelStructuredResponse,
    ModelTextResponse,
    ToolCall as PydanticToolCall,
)
from ..result import Cost
from ..tools import ToolDefinition
from . import (
    AgentModel,
    EitherStreamedResponse,
    Model,
    StreamStructuredResponse,
    StreamTextResponse,
    cached_async_http_client,
)

try:
    from json_repair import repair_json
    from mistralai import (
        CompletionChunk as MistralCompletionChunk,
        FunctionCall as MistralFunctionCall,
        Mistral,
        TextChunk as MistralTextChunk,
    )
    from mistralai.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionEvent as MistralCompletionEvent,
        Messages as MistralMessages,
        Tool as MistralTool,
        ToolCall as MistralToolCall,
    )
    from mistralai.models.assistantmessage import AssistantMessage as MistralAssistantMessage
    from mistralai.models.function import Function as MistralFunction
    from mistralai.models.toolmessage import ToolMessage as MistralToolMessage
    from mistralai.models.usermessage import UserMessage as MistralUserMessage
    from mistralai.types.basemodel import Unset as MistralUnset
    from mistralai.utils.eventstreaming import EventStreamAsync as MistralEventStreamAsync
except ImportError as e:
    raise ImportError(
        'Please install `mistral` to use the Mistral model, '
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
        """Initialize a Mistral model.

        Args:
            model_name: The name of the model to use.
            api_key: The API key to use for authentication,
            client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
        """
        self.model_name = model_name

        if client is not None:
            assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
            self.client = client
        elif http_client is not None:
            self.client = Mistral(api_key=api_key, async_client=http_client)
        else:
            self.client = Mistral(api_key=api_key, async_client=cached_async_http_client())

    async def agent_model(
        self,
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        """Create an agent model, this is called for each step of an agent run.

        Args:
            function_tools: The tools available to the agent.
            allow_text_result: Whether a plain text final response/result is permitted.
            result_tools: Tool definitions for the final result tool(s), if any.

        Returns:
            An agent model.
        """
        return MistralAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            function_tools if function_tools else None,
            result_tools if result_tools else None,
        )

    def name(self) -> str:
        return f'mistral:{self.model_name}'


@dataclass
class MistralAgentModel(AgentModel):
    """Implementation of `AgentModel` for Mistral models."""

    client: Mistral
    model_name: str
    allow_text_result: bool
    function_tools: list[ToolDefinition] | None
    result_tools: list[ToolDefinition] | None

    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, Cost]:
        """Make a non-streaming request to the model.

        Args:
            messages: The list of messages to send to the model.

        Returns:
            A tuple of the model's response, and the estimated cost of the request.
        """
        response = await self._completions_create(messages)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        """Make a streaming request to the model.

        Args:
            messages: The list of messages to send to the model.

        Yields:
            A streaming response from the model.
        """
        response = await self._stream_completions_create(messages)
        async with response:
            yield await self._process_streamed_response(self.function_tools is not None, self.result_tools, response)

    async def _completions_create(
        self,
        messages: list[Message],
    ) -> MistralChatCompletionResponse:
        """Make a non-streaming request to the model.

        Args:
            messages: The list of messages to send to the model.

        Returns:
            A non-streaming response from the model.
        """
        mistral_messages = [self._map_message(m) for m in messages]
        # Note: Function calling Mode
        # "auto": default mode. Model decides if it uses the tool or not.
        # "any": forces tool use. WARN: If use 'any' it will be infinity loop with Model.
        # "none": prevents tool use.
        tool_choice: Literal['none', 'any', 'auto'] | None = None
        if not self.function_tools and not self.result_tools:
            tool_choice = 'none'
        else:
            tool_choice = 'auto'

        response = await self.client.chat.complete_async(
            model=str(self.model_name),
            messages=mistral_messages,
            n=1,
            tools=self._map_function_and_result_tools_definition(),
            tool_choice=tool_choice,
            stream=False,
        )
        assert response, 'A unexpected empty response.'
        return response

    async def _stream_completions_create(
        self,
        messages: list[Message],
    ) -> MistralEventStreamAsync[MistralCompletionEvent]:
        """Create a streaming completion request to the Mistral model.

        This method handles different modes of operation based on the presence of
        `function_tools` and `result_tools`. It determines the tool choice strategy
        and constructs the appropriate request to stream responses from the model.

        Args:
            messages: The list of messages to be processed by the model.

        Returns:
            An asynchronous event stream of `MistralCompletionEvent` from the model.

        Raises:
            AssertionError: If the response is unexpectedly empty.
        """
        response: MistralEventStreamAsync[MistralCompletionEvent] | None = None
        mistral_messages = [self._map_message(m) for m in messages]

        if self.result_tools and self.function_tools or self.function_tools:
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                n=1,
                tools=self._map_function_and_result_tools_definition(),
                tool_choice='auto',  # Note WARN: If use 'any' it will be infinity loop with Model.
            )

        elif self.result_tools:
            # JSON Mode
            schema: str | list[dict[str, Any]]
            if len(self.result_tools) == 1:
                schema = _generate_json_simple_schema(self.result_tools[0].parameters_json_schema)
            else:
                parameters_json_schemas = [tool.parameters_json_schema for tool in self.result_tools]
                schema = _generate_jsom_simple_schemas(parameters_json_schemas)

            mistral_messages.append(
                MistralUserMessage(
                    content=f"""Answer in JSON Object, respect this following format:\n```\n{schema}\n```\n"""
                )
            )
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                response_format={'type': 'json_object'},
            )

        else:
            # Stream Mode
            response = await self.client.chat.stream_async(
                model=str(self.model_name), messages=mistral_messages, stream=True, n=1
            )
        assert response
        return response

    def _map_function_and_result_tools_definition(self) -> list[MistralTool] | None:
        """Map function and result tools to MistralTool format.

        Returns None if both function_tools and result_tools are empty.
        """
        tools = []

        all_tools: list[ToolDefinition] = []
        if self.function_tools:
            all_tools.extend(self.function_tools)
        if self.result_tools:
            all_tools.extend(self.result_tools)

        tools = [
            MistralTool(
                function=MistralFunction(name=r.name, parameters=r.parameters_json_schema, description=r.description)
            )
            for r in all_tools
        ]
        return tools if tools else None

    @staticmethod
    def _process_response(response: MistralChatCompletionResponse) -> ModelAnyResponse:
        """Process a non-streamed response, and prepare a message to return."""
        timestamp: datetime
        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = _now_utc()

        assert response.choices, 'A unexpected empty response choice.'
        choice = response.choices[0]
        tools_calls = choice.message.tool_calls
        if tools_calls is not None and isinstance(tools_calls, list):
            tools = [
                (
                    PydanticToolCall.from_json(
                        tool_name=c.function.name,
                        args_json=c.function.arguments,
                        tool_call_id=c.id,
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
        is_function_tools: bool,
        result_tools: list[ToolDefinition] | None,
        response: MistralEventStreamAsync[MistralCompletionEvent],
    ) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        start_cost = Cost()

        # the first chunk may contain enough information so we iterate until we get either `tool_calls` or `content`
        while True:
            try:
                event = await response.__anext__()
                chunk = event.data
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e

            start_cost += _map_cost(chunk)

            timestamp: datetime
            if chunk.created:
                timestamp = datetime.fromtimestamp(chunk.created, tz=timezone.utc)
            else:
                timestamp = _now_utc()

            if chunk.choices:
                delta = chunk.choices[0].delta
                content: str | None = None

                if isinstance(delta.content, list) and isinstance(delta.content[0], MistralTextChunk):
                    content = delta.content[0].text
                elif isinstance(delta.content, str):
                    content = delta.content
                elif isinstance(delta.content, MistralUnset):
                    pass
                else:
                    assert False, f'Other type of instance, Will manage in the futur (Image, Reference), object:{delta}'

                if content and content == '':
                    content = None

                tool_calls: list[MistralToolCall] | None = None
                if isinstance(delta.tool_calls, list):
                    tool_calls = delta.tool_calls

                if content and result_tools:
                    return MistralStreamStructuredResponse(
                        is_function_tools,
                        result_tools,
                        response,
                        None,
                        content,
                        timestamp,
                        start_cost,
                    )
                elif content:
                    return MistralStreamTextResponse(content, response, timestamp, start_cost)
                elif tool_calls and not result_tools:
                    tool_calls_param = {c.id if c.id else 'null': c for c in tool_calls}

                    return MistralStreamStructuredResponse(
                        is_function_tools,
                        result_tools,
                        response,
                        tool_calls_param,
                        content,
                        timestamp,
                        start_cost,
                    )

    @staticmethod
    def _map_message(message: Message) -> MistralMessages:
        """Just maps a `pydantic_ai.Message` to a `Mistral Message`."""
        if message.role == 'system':
            # SystemPrompt ->
            return MistralAssistantMessage(content=message.content)
        elif message.role == 'user':
            # UserPrompt ->
            return MistralUserMessage(content=message.content)
        elif message.role == 'tool-return':
            # ToolReturn ->
            return MistralToolMessage(
                tool_call_id=message.tool_call_id,
                content=message.model_response_str(),
            )
        elif message.role == 'retry-prompt':
            # RetryPrompt ->
            if message.tool_name is None:
                return MistralUserMessage(content=message.model_response())
            else:
                return MistralToolMessage(
                    tool_call_id=message.tool_call_id,
                    content=message.model_response(),
                )
        elif message.role == 'model-text-response':
            # ModelTextResponse ->
            return MistralAssistantMessage(content=message.content)
        elif message.role == 'model-structured-response':
            # ModelStructuredResponse ->
            return MistralAssistantMessage(
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


@dataclass
class MistralStreamTextResponse(StreamTextResponse):
    """Implementation of `StreamTextResponse` for Mistral models."""

    _first: str | None
    _response: MistralEventStreamAsync[MistralCompletionEvent]
    _timestamp: datetime
    _cost: Cost
    _buffer: list[str] = field(default_factory=list, init=False)

    async def __anext__(self) -> None:
        if self._first is not None and len(self._first) > 0:
            self._buffer.append(self._first)
            self._first = None
            return None

        chunk = await self._response.__anext__()
        self._cost = _map_cost(chunk.data)

        try:
            choice = chunk.data.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is None:
            assert choice.delta.content is not None, f'Expected delta with content, invalid chunk: {chunk!r}'
        if isinstance(choice.delta.content, str):
            self._buffer.append(choice.delta.content)
        elif isinstance(choice.delta.content, MistralTextChunk):
            self._buffer.append(choice.delta.content.text)

    def get(self, *, final: bool = False) -> Iterable[str]:
        yield from self._buffer
        self._buffer.clear()

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


@dataclass
class MistralStreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for Mistral models."""

    _is_function_tools: bool
    _result_tools: list[ToolDefinition] | None
    _response: MistralEventStreamAsync[MistralCompletionEvent]
    _delta_tool_calls: dict[str, MistralToolCall] | None
    _delta_content: str | None
    _timestamp: datetime
    _cost: Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost = _map_cost(chunk.data)

        try:
            choice = chunk.data.choices[0]

        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is not None:
            raise StopAsyncIteration()

        delta_content = choice.delta.content
        content: str | None = None
        if isinstance(delta_content, list) and isinstance(delta_content[0], MistralTextChunk):
            content = delta_content[0].text
        elif isinstance(delta_content, str):
            content = delta_content
        elif isinstance(delta_content, MistralUnset):
            content = None
        else:
            assert False, f'Other type of instance, Will manage in the futur (Image, Reference), object:{delta_content}'

        if self._delta_tool_calls and self._result_tools or self._delta_tool_calls:
            for new in choice.delta.tool_calls or []:
                if current := self._delta_tool_calls.get(new.id or 'null'):
                    current.function = new.function
                else:
                    self._delta_tool_calls[new.id or 'null'] = new
        elif self._result_tools and content:
            if not self._delta_content:
                self._delta_content = content
            else:
                self._delta_content += content

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[PydanticToolCall] = []
        if self._delta_tool_calls and self._result_tools or self._delta_tool_calls:
            for c in self._delta_tool_calls.values():
                if f := c.function:
                    tool = (
                        PydanticToolCall.from_json(
                            tool_name=f.name,
                            args_json=f.arguments,
                            tool_call_id=c.id,
                        )
                        if isinstance(f.arguments, str)
                        else PydanticToolCall.from_dict(tool_name=f.name, args_dict=f.arguments, tool_id=c.id)
                    )
                    calls.append(tool)
        elif self._delta_content:
            # NOTE: Params set for the most efficient and fastest way.
            decoded_object = repair_json(self._delta_content, return_objects=True, skip_json_loads=True)
            assert isinstance(decoded_object, dict)

            if decoded_object:
                tool = PydanticToolCall.from_dict(
                    tool_name='final_result',  # TODO: Get tool_name
                    args_dict=decoded_object,
                )
                calls.append(tool)

        return ModelStructuredResponse(calls, timestamp=self._timestamp)

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


def _generate_json_simple_schema(schema: dict[str, Any]) -> Any:
    """Generates a JSON example from a JSON schema.

    :param schema: The JSON schema.
    :return: A JSON example based on the schema.
    """
    if schema.get('type') == 'object':
        example: dict[str, Any] = {}
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                example[key] = _generate_json_simple_schema(value)
        return example

    if schema.get('type') == 'array':
        if 'items' in schema:
            return [_generate_json_simple_schema(schema['items'])]

    if schema.get('type') == 'string':
        return 'String value'

    if schema.get('type') == 'number':
        return 'Number value'

    if schema.get('type') == 'integer':
        return 'integer value'

    if schema.get('type') == 'boolean':
        return 'Boolean value'

    if schema.get('type') == 'null':
        return 'null value'

    return None


def _generate_jsom_simple_schemas(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generates JSON examples from a list of JSON schemas.

    :param schemas: The list of JSON schemas.
    :return: A list of JSON examples based on the schemas.
    """
    examples: list[dict[str, Any]] = []
    for schema in schemas:
        example = _generate_json_simple_schema(schema)
        examples.append(example)
    return examples


def _map_tool_call(t: PydanticToolCall) -> MistralToolCall:
    if isinstance(t.args, ArgsJson):
        return MistralToolCall(
            id=t.tool_call_id,
            type='function',
            function=MistralFunctionCall(name=t.tool_name, arguments=t.args.args_json),
        )
    else:
        return MistralToolCall(
            id=t.tool_call_id,
            type='function',
            function=MistralFunctionCall(name=t.tool_name, arguments=t.args.args_dict),
        )


def _map_cost(response: MistralChatCompletionResponse | MistralCompletionChunk) -> Cost:
    if response.usage:
        return Cost(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            details=None,
        )
    else:
        return Cost()
