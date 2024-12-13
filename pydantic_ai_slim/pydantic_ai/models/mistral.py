from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Union

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
    ToolCall,
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
        Content as MistralContent,
        FunctionCall as MistralFunctionCall,
        Mistral,
        OptionalNullable as MistralOptionalNullable,
        TextChunk as MistralTextChunk,
        ToolChoiceEnum as MistralToolChoiceEnum,
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
    from mistralai.models.systemmessage import SystemMessage as MistralSystemMessage
    from mistralai.models.toolmessage import ToolMessage as MistralToolMessage
    from mistralai.models.usermessage import UserMessage as MistralUserMessage
    from mistralai.types.basemodel import Unset as MistralUnset
    from mistralai.utils.eventstreaming import EventStreamAsync as MistralEventStreamAsync
except ImportError as e:
    raise ImportError(
        'Please install `mistral` to use the Mistral model, '
        "you can use the `mistral` optional group — `pip install 'pydantic-ai-slim[mistral]'`"
    ) from e

LatestMistralModelName = Literal[
    'mistral-large-latest', 'mistral-small-latest', 'codestral-latest', 'mistral-moderation-latest'
]
"""Latest named Mistral models."""

MistralModelName = Union[str, LatestMistralModelName]
"""Possible Mistral model names.

Since Mistral supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
Since [the Mistral docs](https://docs.mistral.ai/getting-started/models/models_overview/) for a full list.
"""


@dataclass(init=False)
class MistralModel(Model):
    """A model that uses Mistral.

    Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.

    [API Documentation](https://docs.mistral.ai/)
    """

    model_name: MistralModelName
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
        """Create an agent model, this is called for each step of an agent run from Pydantic AI call."""
        return MistralAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            function_tools,
            result_tools,
        )

    def name(self) -> str:
        return f'mistral:{self.model_name}'


@dataclass
class MistralAgentModel(AgentModel):
    """Implementation of `AgentModel` for Mistral models."""

    client: Mistral
    model_name: str
    allow_text_result: bool
    function_tools: list[ToolDefinition]
    result_tools: list[ToolDefinition]

    async def request(self, messages: list[Message]) -> tuple[ModelAnyResponse, Cost]:
        """Make a non-streaming request to the model from Pydantic AI call."""
        response = await self._completions_create(messages)
        return self._process_response(response), _map_cost(response)

    @asynccontextmanager
    async def request_stream(self, messages: list[Message]) -> AsyncIterator[EitherStreamedResponse]:
        """Make a streaming request to the model from Pydantic AI call."""
        response = await self._stream_completions_create(messages)
        is_function_tool = True if self.function_tools else False  # TODO: see how improve
        async with response:
            yield await self._process_streamed_response(is_function_tool, self.result_tools, response)

    async def _completions_create(
        self,
        messages: list[Message],
    ) -> MistralChatCompletionResponse:
        """Make a non-streaming request to the model."""
        mistral_messages = [self._map_message(m) for m in messages]

        tool_choice = self._get_tool_choice()

        response = await self.client.chat.complete_async(
            model=str(self.model_name),
            messages=mistral_messages,
            n=1,
            tools=self._map_function_and_result_tools_definition(),
            tool_choice=tool_choice,
            stream=False,
        )
        assert response, 'A unexpected empty response from Mistral.'
        return response

    async def _stream_completions_create(
        self,
        messages: list[Message],
    ) -> MistralEventStreamAsync[MistralCompletionEvent]:
        """Create a streaming completion request to the Mistral model."""
        response: MistralEventStreamAsync[MistralCompletionEvent] | None = None
        mistral_messages = [self._map_message(m) for m in messages]

        tool_choice = self._get_tool_choice()

        if self.result_tools and self.function_tools or self.function_tools:
            # Function Calling Mode
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                n=1,
                tools=self._map_function_and_result_tools_definition(),
                tool_choice=tool_choice,
            )

        elif self.result_tools:
            # Json Mode
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
                stream=True,
            )

        else:
            # Stream Mode
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                stream=True,
            )
        assert response, 'A unexpected empty response from Mistral.'
        return response

    def _get_tool_choice(self) -> MistralToolChoiceEnum | None:
        """Get tool choice for the model.

        - "auto": Default mode. Model decides if it uses the tool or not.
        - "any": Select any tool.
        - "none": Prevents tool use.
        - "required": Forces tool use.
        """
        if not self.function_tools and not self.result_tools:
            return None
        elif not self.allow_text_result:
            return 'required'
        else:
            return 'auto'

    def _map_function_and_result_tools_definition(self) -> list[MistralTool] | None:
        """Map function and result tools to MistralTool format.

        Returns None if both function_tools and result_tools are empty.
        """
        all_tools: list[ToolDefinition] = self.function_tools + self.result_tools
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
        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = _now_utc()

        assert response.choices, 'A unexpected empty response choice.'
        choice = response.choices[0]

        tools_calls = choice.message.tool_calls
        if tools_calls is not None and isinstance(tools_calls, list):
            return ModelStructuredResponse(
                [_map_mistral_to_pydantic_tool_call(tool_call) for tool_call in tools_calls],
                timestamp=timestamp,
            )
        else:
            content = choice.message.content
            assert content, f'Unexpected null content in assistant msg: {choice.message}'
            assert not isinstance(
                content, list
            ), f'Unexpected ContentChunk from stream, got {type(content)}, expected list'
            return ModelTextResponse(content, timestamp=timestamp)

    @staticmethod
    async def _process_streamed_response(
        is_function_tools: bool,
        result_tools: list[ToolDefinition],
        response: MistralEventStreamAsync[MistralCompletionEvent],
    ) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        start_cost = Cost()

        # Iterate until we get either `tool_calls` or `content` from the first chunk.
        while True:
            try:
                event = await response.__anext__()
                chunk = event.data
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e

            start_cost += _map_cost(chunk)

            if chunk.created:
                timestamp = datetime.fromtimestamp(chunk.created, tz=timezone.utc)
            else:
                timestamp = _now_utc()

            if chunk.choices:
                delta = chunk.choices[0].delta
                content = _map_delta_content(delta.content)

                tool_calls: list[MistralToolCall] | None = None
                if isinstance(delta.tool_calls, list):
                    tool_calls = delta.tool_calls

                if content and result_tools:
                    return MistralStreamStructuredResponse(
                        is_function_tools,
                        {},
                        {c.name: c for c in result_tools},
                        response,
                        content,
                        timestamp,
                        start_cost,
                    )

                elif content:
                    return MistralStreamTextResponse(content, response, timestamp, start_cost)

                elif tool_calls and not result_tools:
                    return MistralStreamStructuredResponse(
                        is_function_tools,
                        {c.id if c.id else 'null': c for c in tool_calls},
                        {c.name: c for c in result_tools},
                        response,
                        None,
                        timestamp,
                        start_cost,
                    )

    @staticmethod
    def _map_message(message: Message) -> MistralMessages:
        """Just maps a `pydantic_ai.Message` to a `Mistral Message`."""
        if message.role == 'system':
            # SystemPrompt ->
            return MistralSystemMessage(content=message.content)
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
                tool_calls=[_map_pydantic_to_mistral_tool_call(t) for t in message.calls],
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
        self._cost += _map_cost(chunk.data)

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
    _function_tools: dict[str, MistralToolCall]
    _result_tools: dict[str, ToolDefinition]
    _response: MistralEventStreamAsync[MistralCompletionEvent]
    _delta_content: str | None
    _timestamp: datetime
    _cost: Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost += _map_cost(chunk.data)

        try:
            choice = chunk.data.choices[0]

        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is not None:
            raise StopAsyncIteration()

        delta = choice.delta
        content = _map_delta_content(delta.content)

        if self._function_tools and self._result_tools or self._function_tools:
            for new in delta.tool_calls or []:
                if current := self._function_tools.get(new.id or 'null'):
                    current.function = new.function
                else:
                    self._function_tools[new.id or 'null'] = new
        elif self._result_tools and content:
            if not self._delta_content:
                self._delta_content = content
            else:
                self._delta_content += content

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[ToolCall] = []

        if self._function_tools and self._result_tools or self._function_tools:
            for tool_call in self._function_tools.values():
                tool = _map_mistral_to_pydantic_tool_call(tool_call)
                calls.append(tool)
        elif self._delta_content and self._result_tools:
            # TODO: add test on tool_name !=
            # TODO add test when result_name not the ssame
            # NOTE: Params set for the most efficient and fastest way.
            output_json = repair_json(self._delta_content, return_objects=True, skip_json_loads=True)
            assert isinstance(
                output_json, dict
            ), f'Expected repair_json as type dict, invalid type: {type(output_json)}'
            if output_json:
                for result_tool in self._result_tools.values():
                    # NOTE: Additional verification to prevent JSON validation to crash in `result.py`
                    # Ensures required parameters in the JSON schema are respected, especially for stream-based return types.
                    # For example, `return_type=list[str]` expects a 'response' key with value type array of str.
                    # when `{"response":` then `repair_json` sets `{"response": ""}` (type not found default str)
                    # when `{"response": {` then `repair_json` sets `{"response": {}}` (type found)
                    # This ensures it's corrected to `{"response": {}}` and other required parameters and type.
                    if not _validate_required_json_shema(output_json, result_tool.parameters_json_schema):
                        continue

                    tool = ToolCall.from_dict(
                        tool_name=result_tool.name,
                        args_dict=output_json,
                    )
                    calls.append(tool)

        return ModelStructuredResponse(calls, timestamp=self._timestamp)

    def cost(self) -> Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp


TYPE_MAPPING = {
    'string': str,
    'integer': int,
    'number': float,
    'boolean': bool,
    'array': list,
    'object': dict,
    'null': type(None),
}


def _validate_required_json_shema(json_dict: dict[str, Any], json_schema: dict[str, Any]) -> bool:
    """Validate that all required parameters in the JSON schema are present in the JSON dictionary."""
    required_params = json_schema.get('required', [])
    properties = json_schema.get('properties', {})

    for param in required_params:
        if param not in json_dict:
            return False

        param_schema = properties.get(param, {})
        param_type = param_schema.get('type')
        param_items_type = param_schema.get('items', {}).get('type')

        if param_type == 'array' and param_items_type:
            if not isinstance(json_dict[param], list):
                return False
            for item in json_dict[param]:
                if not isinstance(item, TYPE_MAPPING[param_items_type]):
                    return False
        elif param_type and not isinstance(json_dict[param], TYPE_MAPPING[param_type]):
            return False

        if isinstance(json_dict[param], dict) and 'properties' in param_schema:
            nested_schema = param_schema
            if not _validate_required_json_shema(json_dict[param], nested_schema):
                return False

    return True


def _generate_json_simple_schema(schema: dict[str, Any]) -> Any:
    """Generates a JSON example from a JSON schema."""
    if schema.get('type') == 'object':
        example: dict[str, Any] = {}
        if properties := schema.get('properties'):
            for key, value in properties.items():
                example[key] = _generate_json_simple_schema(value)
        return example

    if schema.get('type') == 'array':
        if items := schema.get('items'):
            return [_generate_json_simple_schema(items)]

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
    """Generates JSON examples from a list of JSON schemas."""
    examples: list[dict[str, Any]] = []
    for schema in schemas:
        example = _generate_json_simple_schema(schema)
        examples.append(example)
    return examples


def _map_mistral_to_pydantic_tool_call(tool_call: MistralToolCall) -> ToolCall:
    """Maps a MistralToolCall to a ToolCall."""
    tool_call_id = tool_call.id or None
    func_call = tool_call.function

    if isinstance(func_call.arguments, str):
        return ToolCall.from_json(
            tool_name=func_call.name,
            args_json=func_call.arguments,
            tool_call_id=tool_call_id,
        )
    else:
        return ToolCall.from_dict(tool_name=func_call.name, args_dict=func_call.arguments, tool_call_id=tool_call_id)


def _map_pydantic_to_mistral_tool_call(t: ToolCall) -> MistralToolCall:
    """Maps a Pydantic ToolCall to a MistralToolCall."""
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
    """Maps a Mistral Completion Chunk or Chat Completion Response to a Cost."""
    if response.usage:
        return Cost(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            details=None,
        )
    else:
        return Cost()


def _map_delta_content(delta_content: MistralOptionalNullable[MistralContent]) -> str | None:
    """Maps the delta content from a Mistral Completion Chunk to a string or None."""
    content: str | None = None

    if isinstance(delta_content, list) and isinstance(delta_content[0], MistralTextChunk):
        content = delta_content[0].text
    elif isinstance(delta_content, str):
        content = delta_content
    elif isinstance(delta_content, MistralUnset) or delta_content is None:
        content = None
    else:
        assert False, f'Other data types like (Image, Reference) are not yet supported,  got {type(delta_content)}'

    if content and content == '':
        content = None
    return content
