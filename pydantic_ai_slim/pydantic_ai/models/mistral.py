from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from httpx import AsyncClient as AsyncHTTPClient
from mistralai import CompletionChunk, FunctionCall, TextChunk
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
    from mistralai.types.basemodel import Unset
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
        *,
        function_tools: list[ToolDefinition],
        allow_text_result: bool,
        result_tools: list[ToolDefinition],
    ) -> AgentModel:
        
        return MistralAgentModel(
            self.client,
            self.model_name,
            allow_text_result,
            function_tools if function_tools else None,
            result_tools if result_tools else None
        )

    def name(self) -> str:
        return f'mistral:{self.model_name}'



@dataclass
class MistralAgentModel(AgentModel):
    """Implementation of `AgentModel` for Mistral models."""

    client: Mistral
    model_name: str
    allow_text_result: bool
    #tools: list[Tool] | None = None 
    function_tools: list[ToolDefinition] | None = None 
    result_tools: list[ToolDefinition] | None = None 
   
    def _map_function_and_result_tools_definition(self) -> list[Tool] | None:
        if self.function_tools is None:
            return None
        
        tools: list[Tool] | None = [self._map_tool_definition(r) for r in self.function_tools]
        if self.result_tools:
            tools += [self._map_tool_definition(r) for r in self.result_tools]        
        if not tools:
            tools = None
        return tools  
    
    def _map_tools_definition(self) -> list[Tool] | None:
        if self.function_tools is None:
            return None
        
        tools: list[Tool] | None = [self._map_tool_definition(r) for r in self.function_tools]
        #if self.result_tools:
        #    tools += [self._map_tool_definition(r) for r in result_tools]        
        if not tools:
            tools = None
        return tools

    def _map_result_tools_definition(self) -> list[Tool] | None:
        if self.result_tools is None:
            return None
        
        tools: list[Tool] | None = [self._map_tool_definition(r) for r in self.result_tools]       
        if not tools:
            tools = None
        return tools
        
    @staticmethod
    def _map_tool_definition(
        f: ToolDefinition,
    ) -> Tool:
        """Convert an `AbstractToolDefinition` to a `Tool` or `ToolTypedDict`.

        This is a utility function used to convert our internal representation of a tool to the
        representation expected by the Mistral API.
        """
        function = Function(
            name=f.name, parameters=f.parameters_json_schema, description=f.description
        )
        return Tool(function=function)
        
    async def request(
        self, messages: list[Message]
    ) -> tuple[ModelAnyResponse, result.Cost]:
        response = await self._completions_create(messages)
        return self._process_response(response), _map_cost(response)


    @asynccontextmanager
    async def request_stream(
        self, messages: list[Message]
    ) -> AsyncIterator[EitherStreamedResponse]:

        response = await self._stream_completions_create(messages)
        async with response:
            yield await self._process_streamed_response(
                self.function_tools is not None, 
                self.result_tools,
                response)
                
    async def _completions_create(
        self, messages: list[Message], 
    ) ->  ChatCompletionResponse:
        
        mistral_messages = [self._map_message(m) for m in messages]
        tool_choice: Literal['none', 'required', 'auto'] | None = None
        if not self.allow_text_result:
            tool_choice = 'required'
        else:
            tool_choice = 'auto'
            
        response = await self.client.chat.complete_async(
            model=str(self.model_name),
            messages=mistral_messages,
            n=1,
            tools=self._map_function_and_result_tools_definition(),
            tool_choice=tool_choice,
            stream=False
            )
        assert response
        return response
        
    
    async def _stream_completions_create(
        self, messages: list[Message],
    ) -> EventStreamAsync[CompletionEvent]:
        
        response: Optional[EventStreamAsync[CompletionEvent]] = None
        mistral_messages = [self._map_message(m) for m in messages]
        
        if self.result_tools and self.function_tools:
            tool_choice: Literal['none', 'required', 'auto'] | None = None
            if not self.allow_text_result:
                tool_choice = 'required'
            else:
                tool_choice = 'auto'
                            
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                stream=True,
                n=1,
                tools=self._map_tools_definition(),
                tool_choice=tool_choice,
            )
            
        elif self.result_tools: 
            schema: str | List[Dict[str, Any]]
            if len(self.result_tools) == 1:
                schema = generate_example_from_schema(self.result_tools[0].parameters_json_schema)
            else:
                parameters_json_schemas = [tool.parameters_json_schema for tool in self.result_tools]
                schema = generate_examples_from_schemas(parameters_json_schemas)
            
            mistral_messages.append(UserMessage(content=f"""Answer in JSON Object format here the JSON Schema:\n{schema}"""))
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                stream=True,
                response_format = {'type': 'json_object'},
            )
            
        else:
            response = await self.client.chat.stream_async(
                model=str(self.model_name),
                messages=mistral_messages,
                stream=True,
                n=1
            )
        assert response
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
        is_function_tools: bool,
        result_tools: list[ToolDefinition] | None,
        response: EventStreamAsync[CompletionEvent],
    ) -> EitherStreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        timestamp: datetime | None = None
        start_cost = result.Cost()
        # the first chunk may contain enough information so we iterate until we get either `tool_calls` or `content`
        while True:
            try:
                chunk = await response.__anext__()
            except StopAsyncIteration as e:
                raise UnexpectedModelBehavior('Streamed response ended without content or tool calls') from e

            timestamp = timestamp or datetime.fromtimestamp(chunk.data.created, tz=timezone.utc)
            start_cost += _map_cost(chunk.data)

            if chunk.data.choices:
                delta = chunk.data.choices[0].delta
                if delta.content is not None and delta.content != '' and not isinstance(delta.content, Unset):
                    if not result_tools and not is_function_tools:
                        return MistralStreamTextResponse(delta.content, response, timestamp, start_cost)
                        
                    else:
                        return MistralStreamStructuredResponse(
                        is_function_tools,
                        result_tools,
                        response,
                        {c.id: c for c in delta.tool_calls},
                        delta.content,
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

            # ModelStructuredResponse ->
            return AssistantMessage(
                tool_calls=[_map_tool_call(t) for t in message.calls],
            )
        else:
            assert_never(message)


def _guard_tool_id(t: PydanticToolCall | RetryPrompt | ToolReturn) -> str | None:
    """Type guard that checks a `tool_id` is not None both for static typing and runtime."""
    #assert t.tool_id is not None, f'Mistral requires `tool_id` to be set: {t}'
    return t.tool_id


def _map_tool_call(t: PydanticToolCall) -> ToolCall:
    if isinstance(t.args, ArgsJson):
        return ToolCall(
            id=t.tool_id,
            type='function',
            function=FunctionCall(name=t.tool_name, arguments=t.args.args_json),
        )
    else:
        return ToolCall(
            id=t.tool_id,
            type='function',
            function=FunctionCall(name=t.tool_name, arguments=t.args.args_dict),
        )       
    
def _map_cost(response: ChatCompletionResponse | CompletionChunk) -> result.Cost:
    if response.usage is None:
        return result.Cost()
    else:
        usage = response.usage
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

        # we don't raise StopAsyncIteration on the last chunk because usage comes after this
        if choice.finish_reason is None:
            assert choice.delta.content is not None, f'Expected delta with content, invalid chunk: {chunk!r}'
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

@dataclass
class MistralStreamStructuredResponse(StreamStructuredResponse):
    """Implementation of `StreamStructuredResponse` for Groq models."""
    _is_function_tools: bool
    _result_tools: list[ToolDefinition] | None 
    _response: EventStreamAsync[CompletionEvent]
    _delta_tool_calls: dict[str, ToolCall]
    _delta_content: str | None 
    _timestamp: datetime
    _cost: result.Cost

    async def __anext__(self) -> None:
        chunk = await self._response.__anext__()
        self._cost = _map_cost(chunk.data)

        try:
            choice = chunk.data.choices[0]
        except IndexError:
            raise StopAsyncIteration()

        if choice.finish_reason is not None:
            raise StopAsyncIteration()

        #assert choice.delta.content is None, f'Expected tool calls, got content instead, invalid chunk: {chunk!r}'

        
        if self._is_function_tools and self._result_tools:
            for new in choice.delta.tool_calls or []:
                if current := self._delta_tool_calls.get(new.id):
                    current.function = new.function
                            
                else:
                    self._delta_tool_calls[new.id] = new
        elif self._result_tools:
            self._delta_content += choice.delta.content
            

    def get(self, *, final: bool = False) -> ModelStructuredResponse:
        calls: list[PydanticToolCall] = []
        if self._is_function_tools and self._result_tools:
            
            for c in self._delta_tool_calls.values():
                if f := c.function:
                    tool = PydanticToolCall.from_json(
                            tool_name=f.name,
                            args_json=f.arguments,
                            tool_id=c.id,
                        ) if isinstance(f.arguments, str) else PydanticToolCall.from_dict(
                            tool_name=f.name,
                            args_dict=f.arguments,
                            tool_id=c.id)        
                    calls.append(tool)
        else:
            decoded_object = repair_json(self._delta_content, return_objects=True)
            
            if isinstance(decoded_object, Dict):
                tool = PydanticToolCall.from_dict(
                                tool_name='final_result',
                                args_dict=decoded_object,
                            )       
                calls.append(tool)
                        

        return ModelStructuredResponse(calls, timestamp=self._timestamp)


    
    def cost(self) -> result.Cost:
        return self._cost

    def timestamp(self) -> datetime:
        return self._timestamp
    
    
def generate_example_from_schema(schema: Dict[str, Any]) -> Any:
    """Generates a JSON example from a JSON schema.

    :param schema: The JSON schema.
    :return: A JSON example based on the schema.
    """
    if schema.get('type') == 'object':
        example = {}
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                example[key] = generate_example_from_schema(value)
        return example

    if schema.get('type') == 'array':
        if 'items' in schema:
            return [generate_example_from_schema(schema['items'])]

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

def generate_examples_from_schemas(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generates JSON examples from a list of JSON schemas.

    :param schemas: The list of JSON schemas.
    :return: A list of JSON examples based on the schemas.
    """
    examples = []
    for schema in schemas:
        example = generate_example_from_schema(schema)
        examples.append(example)
    return examples