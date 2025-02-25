from __future__ import annotations as _annotations

import base64
import os
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import chain
from typing import Any, Callable, Literal, Union, cast

import pydantic_core
from httpx import AsyncClient as AsyncHTTPClient, Timeout
from typing_extensions import assert_never

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils
from .._utils import now_utc as _now_utc
from ..messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from ..result import Usage
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    cached_async_http_client,
    check_allow_model_requests,
)

try:
    from mistralai import (
        UNSET,
        CompletionChunk as MistralCompletionChunk,
        Content as MistralContent,
        ContentChunk as MistralContentChunk,
        FunctionCall as MistralFunctionCall,
        ImageURL as MistralImageURL,
        ImageURLChunk as MistralImageURLChunk,
        Mistral,
        OptionalNullable as MistralOptionalNullable,
        TextChunk as MistralTextChunk,
        ToolChoiceEnum as MistralToolChoiceEnum,
    )
    from mistralai.models import (
        ChatCompletionResponse as MistralChatCompletionResponse,
        CompletionEvent as MistralCompletionEvent,
        Messages as MistralMessages,
        SDKError,
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

LatestMistralModelNames = Literal[
    'mistral-large-latest', 'mistral-small-latest', 'codestral-latest', 'mistral-moderation-latest'
]
"""Latest  Mistral models."""

MistralModelName = Union[str, LatestMistralModelNames]
"""Possible Mistral model names.

Since Mistral supports a variety of date-stamped models, we explicitly list the most popular models but
allow any name in the type hints.
Since [the Mistral docs](https://docs.mistral.ai/getting-started/models/models_overview/) for a full list.
"""


class MistralModelSettings(ModelSettings):
    """Settings used for a Mistral model request."""

    # This class is a placeholder for any future mistral-specific settings


@dataclass(init=False)
class MistralModel(Model):
    """A model that uses Mistral.

    Internally, this uses the [Mistral Python client](https://github.com/mistralai/client-python) to interact with the API.

    [API Documentation](https://docs.mistral.ai/)
    """

    client: Mistral = field(repr=False)
    json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n"""

    _model_name: MistralModelName = field(repr=False)
    _system: str | None = field(default='mistral', repr=False)

    def __init__(
        self,
        model_name: MistralModelName,
        *,
        api_key: str | Callable[[], str | None] | None = None,
        client: Mistral | None = None,
        http_client: AsyncHTTPClient | None = None,
        json_mode_schema_prompt: str = """Answer in JSON Object, respect the format:\n```\n{schema}\n```\n""",
    ):
        """Initialize a Mistral model.

        Args:
            model_name: The name of the model to use.
            api_key: The API key to use for authentication, if unset uses `MISTRAL_API_KEY` environment variable.
            client: An existing `Mistral` client to use, if provided, `api_key` and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for making HTTP requests.
            json_mode_schema_prompt: The prompt to show when the model expects a JSON object as input.
        """
        self._model_name = model_name
        self.json_mode_schema_prompt = json_mode_schema_prompt

        if client is not None:
            assert http_client is None, 'Cannot provide both `mistral_client` and `http_client`'
            assert api_key is None, 'Cannot provide both `mistral_client` and `api_key`'
            self.client = client
        else:
            api_key = os.getenv('MISTRAL_API_KEY') if api_key is None else api_key
            self.client = Mistral(api_key=api_key, async_client=http_client or cached_async_http_client())

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, Usage]:
        """Make a non-streaming request to the model from Pydantic AI call."""
        check_allow_model_requests()
        response = await self._completions_create(
            messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
        )
        return self._process_response(response), _map_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        """Make a streaming request to the model from Pydantic AI call."""
        check_allow_model_requests()
        response = await self._stream_completions_create(
            messages, cast(MistralModelSettings, model_settings or {}), model_request_parameters
        )
        async with response:
            yield await self._process_streamed_response(model_request_parameters.result_tools, response)

    @property
    def model_name(self) -> MistralModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str | None:
        """The system / model provider."""
        return self._system

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        model_settings: MistralModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> MistralChatCompletionResponse:
        """Make a non-streaming request to the model."""
        try:
            response = await self.client.chat.complete_async(
                model=str(self._model_name),
                messages=list(chain(*(self._map_message(m) for m in messages))),
                n=1,
                tools=self._map_function_and_result_tools_definition(model_request_parameters) or UNSET,
                tool_choice=self._get_tool_choice(model_request_parameters),
                stream=False,
                max_tokens=model_settings.get('max_tokens', UNSET),
                temperature=model_settings.get('temperature', UNSET),
                top_p=model_settings.get('top_p', 1),
                timeout_ms=self._get_timeout_ms(model_settings.get('timeout')),
                random_seed=model_settings.get('seed', UNSET),
            )
        except SDKError as e:
            if (status_code := e.status_code) >= 400:
                raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=e.body) from e
            raise

        assert response, 'A unexpected empty response from Mistral.'
        return response

    async def _stream_completions_create(
        self,
        messages: list[ModelMessage],
        model_settings: MistralModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> MistralEventStreamAsync[MistralCompletionEvent]:
        """Create a streaming completion request to the Mistral model."""
        response: MistralEventStreamAsync[MistralCompletionEvent] | None
        mistral_messages = list(chain(*(self._map_message(m) for m in messages)))

        if (
            model_request_parameters.result_tools
            and model_request_parameters.function_tools
            or model_request_parameters.function_tools
        ):
            # Function Calling
            response = await self.client.chat.stream_async(
                model=str(self._model_name),
                messages=mistral_messages,
                n=1,
                tools=self._map_function_and_result_tools_definition(model_request_parameters) or UNSET,
                tool_choice=self._get_tool_choice(model_request_parameters),
                temperature=model_settings.get('temperature', UNSET),
                top_p=model_settings.get('top_p', 1),
                max_tokens=model_settings.get('max_tokens', UNSET),
                timeout_ms=self._get_timeout_ms(model_settings.get('timeout')),
                presence_penalty=model_settings.get('presence_penalty'),
                frequency_penalty=model_settings.get('frequency_penalty'),
            )

        elif model_request_parameters.result_tools:
            # Json Mode
            parameters_json_schemas = [tool.parameters_json_schema for tool in model_request_parameters.result_tools]
            user_output_format_message = self._generate_user_output_format(parameters_json_schemas)
            mistral_messages.append(user_output_format_message)

            response = await self.client.chat.stream_async(
                model=str(self._model_name),
                messages=mistral_messages,
                response_format={'type': 'json_object'},
                stream=True,
            )

        else:
            # Stream Mode
            response = await self.client.chat.stream_async(
                model=str(self._model_name),
                messages=mistral_messages,
                stream=True,
            )
        assert response, 'A unexpected empty response from Mistral.'
        return response

    def _get_tool_choice(self, model_request_parameters: ModelRequestParameters) -> MistralToolChoiceEnum | None:
        """Get tool choice for the model.

        - "auto": Default mode. Model decides if it uses the tool or not.
        - "any": Select any tool.
        - "none": Prevents tool use.
        - "required": Forces tool use.
        """
        if not model_request_parameters.function_tools and not model_request_parameters.result_tools:
            return None
        elif not model_request_parameters.allow_text_result:
            return 'required'
        else:
            return 'auto'

    def _map_function_and_result_tools_definition(
        self, model_request_parameters: ModelRequestParameters
    ) -> list[MistralTool] | None:
        """Map function and result tools to MistralTool format.

        Returns None if both function_tools and result_tools are empty.
        """
        all_tools: list[ToolDefinition] = (
            model_request_parameters.function_tools + model_request_parameters.result_tools
        )
        tools = [
            MistralTool(
                function=MistralFunction(name=r.name, parameters=r.parameters_json_schema, description=r.description)
            )
            for r in all_tools
        ]
        return tools if tools else None

    def _process_response(self, response: MistralChatCompletionResponse) -> ModelResponse:
        """Process a non-streamed response, and prepare a message to return."""
        assert response.choices, 'Unexpected empty response choice.'

        if response.created:
            timestamp = datetime.fromtimestamp(response.created, tz=timezone.utc)
        else:
            timestamp = _now_utc()

        choice = response.choices[0]
        content = choice.message.content
        tool_calls = choice.message.tool_calls

        parts: list[ModelResponsePart] = []
        if text := _map_content(content):
            parts.append(TextPart(content=text))

        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                tool = self._map_mistral_to_pydantic_tool_call(tool_call=tool_call)
                parts.append(tool)

        return ModelResponse(parts, model_name=response.model, timestamp=timestamp)

    async def _process_streamed_response(
        self,
        result_tools: list[ToolDefinition],
        response: MistralEventStreamAsync[MistralCompletionEvent],
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        if first_chunk.data.created:
            timestamp = datetime.fromtimestamp(first_chunk.data.created, tz=timezone.utc)
        else:
            timestamp = datetime.now(tz=timezone.utc)

        return MistralStreamedResponse(
            _response=peekable_response,
            _model_name=self._model_name,
            _timestamp=timestamp,
            _result_tools={c.name: c for c in result_tools},
        )

    @staticmethod
    def _map_mistral_to_pydantic_tool_call(tool_call: MistralToolCall) -> ToolCallPart:
        """Maps a MistralToolCall to a ToolCall."""
        tool_call_id = tool_call.id or None
        func_call = tool_call.function

        return ToolCallPart(func_call.name, func_call.arguments, tool_call_id)

    @staticmethod
    def _map_pydantic_to_mistral_tool_call(t: ToolCallPart) -> MistralToolCall:
        """Maps a pydantic-ai ToolCall to a MistralToolCall."""
        return MistralToolCall(
            id=t.tool_call_id,
            type='function',
            function=MistralFunctionCall(name=t.tool_name, arguments=t.args),
        )

    def _generate_user_output_format(self, schemas: list[dict[str, Any]]) -> MistralUserMessage:
        """Get a message with an example of the expected output format."""
        examples: list[dict[str, Any]] = []
        for schema in schemas:
            typed_dict_definition: dict[str, Any] = {}
            for key, value in schema.get('properties', {}).items():
                typed_dict_definition[key] = self._get_python_type(value)
            examples.append(typed_dict_definition)

        example_schema = examples[0] if len(examples) == 1 else examples
        return MistralUserMessage(content=self.json_mode_schema_prompt.format(schema=example_schema))

    @classmethod
    def _get_python_type(cls, value: dict[str, Any]) -> str:
        """Return a string representation of the Python type for a single JSON schema property.

        This function handles recursion for nested arrays/objects and `anyOf`.
        """
        # 1) Handle anyOf first, because it's a different schema structure
        if any_of := value.get('anyOf'):
            # Simplistic approach: pick the first option in anyOf
            # (In reality, you'd possibly want to merge or union types)
            return f'Optional[{cls._get_python_type(any_of[0])}]'

        # 2) If we have a top-level "type" field
        value_type = value.get('type')
        if not value_type:
            # No explicit type; fallback
            return 'Any'

        # 3) Direct simple type mapping (string, integer, float, bool, None)
        if value_type in SIMPLE_JSON_TYPE_MAPPING and value_type != 'array' and value_type != 'object':
            return SIMPLE_JSON_TYPE_MAPPING[value_type]

        # 4) Array: Recursively get the item type
        if value_type == 'array':
            items = value.get('items', {})
            return f'list[{cls._get_python_type(items)}]'

        # 5) Object: Check for additionalProperties
        if value_type == 'object':
            additional_properties = value.get('additionalProperties', {})
            additional_properties_type = additional_properties.get('type')
            if (
                additional_properties_type in SIMPLE_JSON_TYPE_MAPPING
                and additional_properties_type != 'array'
                and additional_properties_type != 'object'
            ):
                # dict[str, bool/int/float/etc...]
                return f'dict[str, {SIMPLE_JSON_TYPE_MAPPING[additional_properties_type]}]'
            elif additional_properties_type == 'array':
                array_items = additional_properties.get('items', {})
                return f'dict[str, list[{cls._get_python_type(array_items)}]]'
            elif additional_properties_type == 'object':
                # nested dictionary of unknown shape
                return 'dict[str, dict[str, Any]]'
            else:
                # If no additionalProperties type or something else, default to a generic dict
                return 'dict[str, Any]'

        # 6) Fallback
        return 'Any'

    @staticmethod
    def _get_timeout_ms(timeout: Timeout | float | None) -> int | None:
        """Convert a timeout to milliseconds."""
        if timeout is None:
            return None
        if isinstance(timeout, float):
            return int(1000 * timeout)
        raise NotImplementedError('Timeout object is not yet supported for MistralModel.')

    @classmethod
    def _map_user_message(cls, message: ModelRequest) -> Iterable[MistralMessages]:
        for part in message.parts:
            if isinstance(part, SystemPromptPart):
                yield MistralSystemMessage(content=part.content)
            elif isinstance(part, UserPromptPart):
                yield cls._map_user_prompt(part)
            elif isinstance(part, ToolReturnPart):
                yield MistralToolMessage(
                    tool_call_id=part.tool_call_id,
                    content=part.model_response_str(),
                )
            elif isinstance(part, RetryPromptPart):
                if part.tool_name is None:
                    yield MistralUserMessage(content=part.model_response())
                else:
                    yield MistralToolMessage(
                        tool_call_id=part.tool_call_id,
                        content=part.model_response(),
                    )
            else:
                assert_never(part)

    @classmethod
    def _map_message(cls, message: ModelMessage) -> Iterable[MistralMessages]:
        """Just maps a `pydantic_ai.Message` to a `MistralMessage`."""
        if isinstance(message, ModelRequest):
            yield from cls._map_user_message(message)
        elif isinstance(message, ModelResponse):
            content_chunks: list[MistralContentChunk] = []
            tool_calls: list[MistralToolCall] = []

            for part in message.parts:
                if isinstance(part, TextPart):
                    content_chunks.append(MistralTextChunk(text=part.content))
                elif isinstance(part, ToolCallPart):
                    tool_calls.append(cls._map_pydantic_to_mistral_tool_call(part))
                else:
                    assert_never(part)
            yield MistralAssistantMessage(content=content_chunks, tool_calls=tool_calls)
        else:
            assert_never(message)

    @staticmethod
    def _map_user_prompt(part: UserPromptPart) -> MistralUserMessage:
        content: str | list[MistralContentChunk]
        if isinstance(part.content, str):
            content = part.content
        else:
            content = []
            for item in part.content:
                if isinstance(item, str):
                    content.append(MistralTextChunk(text=item))
                elif isinstance(item, ImageUrl):
                    content.append(MistralImageURLChunk(image_url=MistralImageURL(url=item.url)))
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    if item.is_image:
                        image_url = MistralImageURL(url=f'data:{item.media_type};base64,{base64_encoded}')
                        content.append(MistralImageURLChunk(image_url=image_url, type='image_url'))
                    else:
                        raise RuntimeError('Only image binary content is supported for Mistral.')
                else:  # pragma: no cover
                    raise RuntimeError(f'Unsupported content type: {type(item)}')
        return MistralUserMessage(content=content)


MistralToolCallId = Union[str, None]


@dataclass
class MistralStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Mistral models."""

    _model_name: MistralModelName
    _response: AsyncIterable[MistralCompletionEvent]
    _timestamp: datetime
    _result_tools: dict[str, ToolDefinition]

    _delta_content: str = field(default='', init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        chunk: MistralCompletionEvent
        async for chunk in self._response:
            self._usage += _map_usage(chunk.data)

            try:
                choice = chunk.data.choices[0]
            except IndexError:
                continue

            # Handle the text part of the response
            content = choice.delta.content
            text = _map_content(content)
            if text:
                # Attempt to produce a result tool call from the received text
                if self._result_tools:
                    self._delta_content += text
                    maybe_tool_call_part = self._try_get_result_tool_from_text(self._delta_content, self._result_tools)
                    if maybe_tool_call_part:
                        yield self._parts_manager.handle_tool_call_part(
                            vendor_part_id='result',
                            tool_name=maybe_tool_call_part.tool_name,
                            args=maybe_tool_call_part.args_as_dict(),
                            tool_call_id=maybe_tool_call_part.tool_call_id,
                        )
                else:
                    yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=text)

            # Handle the explicit tool calls
            for index, dtc in enumerate(choice.delta.tool_calls or []):
                # It seems that mistral just sends full tool calls, so we just use them directly, rather than building
                yield self._parts_manager.handle_tool_call_part(
                    vendor_part_id=index, tool_name=dtc.function.name, args=dtc.function.arguments, tool_call_id=dtc.id
                )

    @property
    def model_name(self) -> MistralModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp

    @staticmethod
    def _try_get_result_tool_from_text(text: str, result_tools: dict[str, ToolDefinition]) -> ToolCallPart | None:
        output_json: dict[str, Any] | None = pydantic_core.from_json(text, allow_partial='trailing-strings')
        if output_json:
            for result_tool in result_tools.values():
                # NOTE: Additional verification to prevent JSON validation to crash in `_result.py`
                # Ensures required parameters in the JSON schema are respected, especially for stream-based return types.
                # Example with BaseModel and required fields.
                if not MistralStreamedResponse._validate_required_json_schema(
                    output_json, result_tool.parameters_json_schema
                ):
                    continue

                # The following part_id will be thrown away
                return ToolCallPart(tool_name=result_tool.name, args=output_json)

    @staticmethod
    def _validate_required_json_schema(json_dict: dict[str, Any], json_schema: dict[str, Any]) -> bool:
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
                    if not isinstance(item, VALID_JSON_TYPE_MAPPING[param_items_type]):
                        return False
            elif param_type and not isinstance(json_dict[param], VALID_JSON_TYPE_MAPPING[param_type]):
                return False

            if isinstance(json_dict[param], dict) and 'properties' in param_schema:
                nested_schema = param_schema
                if not MistralStreamedResponse._validate_required_json_schema(json_dict[param], nested_schema):
                    return False

        return True


VALID_JSON_TYPE_MAPPING: dict[str, Any] = {
    'string': str,
    'integer': int,
    'number': float,
    'boolean': bool,
    'array': list,
    'object': dict,
    'null': type(None),
}

SIMPLE_JSON_TYPE_MAPPING = {
    'string': 'str',
    'integer': 'int',
    'number': 'float',
    'boolean': 'bool',
    'array': 'list',
    'null': 'None',
}


def _map_usage(response: MistralChatCompletionResponse | MistralCompletionChunk) -> Usage:
    """Maps a Mistral Completion Chunk or Chat Completion Response to a Usage."""
    if response.usage:
        return Usage(
            request_tokens=response.usage.prompt_tokens,
            response_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            details=None,
        )
    else:
        return Usage()


def _map_content(content: MistralOptionalNullable[MistralContent]) -> str | None:
    """Maps the delta content from a Mistral Completion Chunk to a string or None."""
    result: str | None = None

    if isinstance(content, MistralUnset) or not content:
        result = None
    elif isinstance(content, list):
        for chunk in content:
            if isinstance(chunk, MistralTextChunk):
                result = result or '' + chunk.text
            else:
                assert False, f'Other data types like (Image, Reference) are not yet supported,  got {type(chunk)}'
    elif isinstance(content, str):
        result = content

    # Note: Check len to handle potential mismatch between function calls and responses from the API. (`msg: not the same number of function class and responses`)
    if result and len(result) == 0:
        result = None

    return result
