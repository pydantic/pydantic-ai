from __future__ import annotations

import functools
import typing
from collections.abc import AsyncIterator, Iterable, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from typing import TYPE_CHECKING, Any, Generic, Literal, Union, cast, overload

import anyio
import anyio.to_thread
from typing_extensions import ParamSpec, assert_never

from pydantic_ai import _utils, usage
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    DocumentUrl,
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
    VideoUrl,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse, cached_async_http_client
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

if TYPE_CHECKING:
    from botocore.client import BaseClient
    from botocore.eventstream import EventStream
    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient
    from mypy_boto3_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        ContentBlockUnionTypeDef,
        ConverseRequestTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamMetadataEventTypeDef,
        ConverseStreamOutputTypeDef,
        GuardrailConfigurationTypeDef,
        ImageBlockTypeDef,
        InferenceConfigurationTypeDef,
        MessageUnionTypeDef,
        PerformanceConfigurationTypeDef,
        PromptVariableValuesTypeDef,
        SystemContentBlockTypeDef,
        ToolChoiceTypeDef,
        ToolTypeDef,
        VideoBlockTypeDef,
    )


LatestBedrockModelNames = Literal[
    'amazon.titan-tg1-large',
    'amazon.titan-text-lite-v1',
    'amazon.titan-text-express-v1',
    'us.amazon.nova-pro-v1:0',
    'us.amazon.nova-lite-v1:0',
    'us.amazon.nova-micro-v1:0',
    'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    'anthropic.claude-3-5-haiku-20241022-v1:0',
    'us.anthropic.claude-3-5-haiku-20241022-v1:0',
    'anthropic.claude-instant-v1',
    'anthropic.claude-v2:1',
    'anthropic.claude-v2',
    'anthropic.claude-3-sonnet-20240229-v1:0',
    'us.anthropic.claude-3-sonnet-20240229-v1:0',
    'anthropic.claude-3-haiku-20240307-v1:0',
    'us.anthropic.claude-3-haiku-20240307-v1:0',
    'anthropic.claude-3-opus-20240229-v1:0',
    'us.anthropic.claude-3-opus-20240229-v1:0',
    'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
    'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'cohere.command-text-v14',
    'cohere.command-r-v1:0',
    'cohere.command-r-plus-v1:0',
    'cohere.command-light-text-v14',
    'meta.llama3-8b-instruct-v1:0',
    'meta.llama3-70b-instruct-v1:0',
    'meta.llama3-1-8b-instruct-v1:0',
    'us.meta.llama3-1-8b-instruct-v1:0',
    'meta.llama3-1-70b-instruct-v1:0',
    'us.meta.llama3-1-70b-instruct-v1:0',
    'meta.llama3-1-405b-instruct-v1:0',
    'us.meta.llama3-2-11b-instruct-v1:0',
    'us.meta.llama3-2-90b-instruct-v1:0',
    'us.meta.llama3-2-1b-instruct-v1:0',
    'us.meta.llama3-2-3b-instruct-v1:0',
    'us.meta.llama3-3-70b-instruct-v1:0',
    'mistral.mistral-7b-instruct-v0:2',
    'mistral.mixtral-8x7b-instruct-v0:1',
    'mistral.mistral-large-2402-v1:0',
    'mistral.mistral-large-2407-v1:0',
]
"""Latest Bedrock models."""

BedrockModelName = Union[str, LatestBedrockModelNames]
"""Possible Bedrock model names.

Since Bedrock supports a variety of date-stamped models, we explicitly list the latest models but allow any name in the type hints.
See [the Bedrock docs](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for a full list.
"""


P = ParamSpec('P')
T = typing.TypeVar('T')


class BedrockModelSettings(ModelSettings, total=False):
    """Settings for Bedrock models.

    ALL FIELDS MUST BE `bedrock_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    See [the Bedrock Converse API docs](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax) for a full list.
    See [the boto3 implementation](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-runtime/client/converse.html) of the Bedrock Converse API.
    """

    bedrock_guardrail_config: GuardrailConfigurationTypeDef
    """Content moderation and safety settings for Bedrock API requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConfiguration.html>.
    """

    bedrock_performance_configuration: PerformanceConfigurationTypeDef
    """Performance optimization settings for model inference.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PerformanceConfiguration.html>.
    """

    bedrock_request_metadata: dict[str, str]
    """Additional metadata to attach to Bedrock API requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_RequestSyntax>.
    """

    bedrock_additional_model_response_fields_paths: list[str]
    """JSON paths to extract additional fields from model responses.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.
    """

    bedrock_prompt_variables: Mapping[str, PromptVariableValuesTypeDef]
    """Variables for substitution into prompt templates.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_PromptVariableValues.html>.
    """

    bedrock_additional_model_requests_fields: Mapping[str, Any]
    """Additional model-specific parameters to include in requests.

    See more about it on <https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html>.
    """


@dataclass(init=False)
class BedrockConverseModel(Model):
    """A model that uses the Bedrock Converse API."""

    client: BedrockRuntimeClient

    _model_name: BedrockModelName = field(repr=False)
    _system: str = field(default='bedrock', repr=False)

    @property
    def model_name(self) -> str:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider, ex: openai."""
        return self._system

    def __init__(
        self,
        model_name: BedrockModelName,
        *,
        provider: Literal['bedrock'] | Provider[BaseClient] = 'bedrock',
    ):
        """Initialize a Bedrock model.

        Args:
            model_name: The name of the model to use.
            model_name: The name of the Bedrock model to use. List of model names available
                [here](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html).
            provider: The provider to use for authentication and API access. Can be either the string
                'bedrock' or an instance of `Provider[BaseClient]`. If not provided, a new provider will be
                created using the other parameters.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self.client = cast('BedrockRuntimeClient', provider.client)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolTypeDef]:
        tools = [self._map_tool_definition(r) for r in model_request_parameters.function_tools]
        if model_request_parameters.output_tools:
            tools += [self._map_tool_definition(r) for r in model_request_parameters.output_tools]
        return tools

    @staticmethod
    def _map_tool_definition(f: ToolDefinition) -> ToolTypeDef:
        return {
            'toolSpec': {
                'name': f.name,
                'description': f.description,
                'inputSchema': {'json': f.parameters_json_schema},
            }
        }

    @property
    def base_url(self) -> str:
        return str(self.client.meta.endpoint_url)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        settings = cast(BedrockModelSettings, model_settings or {})
        response = await self._messages_create(messages, False, settings, model_request_parameters)
        model_response = await self._process_response(response)
        model_response.usage.requests = 1
        return model_response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        settings = cast(BedrockModelSettings, model_settings or {})
        response = await self._messages_create(messages, True, settings, model_request_parameters)
        yield BedrockStreamedResponse(_model_name=self.model_name, _event_stream=response)

    async def _process_response(self, response: ConverseResponseTypeDef) -> ModelResponse:
        items: list[ModelResponsePart] = []
        if message := response['output'].get('message'):
            for item in message['content']:
                if text := item.get('text'):
                    items.append(TextPart(content=text))
                else:
                    tool_use = item.get('toolUse')
                    assert tool_use is not None, f'Found a content that is not a text or tool use: {item}'
                    items.append(
                        ToolCallPart(
                            tool_name=tool_use['name'],
                            args=tool_use['input'],
                            tool_call_id=tool_use['toolUseId'],
                        ),
                    )
        u = usage.Usage(
            request_tokens=response['usage']['inputTokens'],
            response_tokens=response['usage']['outputTokens'],
            total_tokens=response['usage']['totalTokens'],
        )
        return ModelResponse(items, usage=u, model_name=self.model_name)

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> EventStream[ConverseStreamOutputTypeDef]:
        pass

    @overload
    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef:
        pass

    async def _messages_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: BedrockModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ConverseResponseTypeDef | EventStream[ConverseStreamOutputTypeDef]:
        tools = self._get_tools(model_request_parameters)
        support_tools_choice = self.model_name.startswith(('anthropic', 'us.anthropic'))
        if not tools or not support_tools_choice:
            tool_choice: ToolChoiceTypeDef = {}
        elif not model_request_parameters.allow_text_output:
            tool_choice = {'any': {}}
        else:
            tool_choice = {'auto': {}}

        system_prompt, bedrock_messages = await self._map_messages(messages)
        inference_config = self._map_inference_config(model_settings)

        params: ConverseRequestTypeDef = {
            'modelId': self.model_name,
            'messages': bedrock_messages,
            'system': system_prompt,
            'inferenceConfig': inference_config,
        }

        # Bedrock supports a set of specific extra parameters
        if model_settings:
            if guardrail_config := model_settings.get('bedrock_guardrail_config', None):
                params['guardrailConfig'] = guardrail_config
            if performance_configuration := model_settings.get('bedrock_performance_configuration', None):
                params['performanceConfig'] = performance_configuration
            if request_metadata := model_settings.get('bedrock_request_metadata', None):
                params['requestMetadata'] = request_metadata
            if additional_model_response_fields_paths := model_settings.get(
                'bedrock_additional_model_response_fields_paths', None
            ):
                params['additionalModelResponseFieldPaths'] = additional_model_response_fields_paths
            if additional_model_requests_fields := model_settings.get('bedrock_additional_model_requests_fields', None):
                params['additionalModelRequestFields'] = additional_model_requests_fields
            if prompt_variables := model_settings.get('bedrock_prompt_variables', None):
                params['promptVariables'] = prompt_variables

        if tools:
            params['toolConfig'] = {'tools': tools}
            if tool_choice:
                params['toolConfig']['toolChoice'] = tool_choice

        if stream:
            model_response = await anyio.to_thread.run_sync(functools.partial(self.client.converse_stream, **params))
            model_response = model_response['stream']
        else:
            model_response = await anyio.to_thread.run_sync(functools.partial(self.client.converse, **params))
        return model_response

    @staticmethod
    def _map_inference_config(
        model_settings: ModelSettings | None,
    ) -> InferenceConfigurationTypeDef:
        model_settings = model_settings or {}
        inference_config: InferenceConfigurationTypeDef = {}

        if max_tokens := model_settings.get('max_tokens'):
            inference_config['maxTokens'] = max_tokens
        if (temperature := model_settings.get('temperature')) is not None:
            inference_config['temperature'] = temperature
        if top_p := model_settings.get('top_p'):
            inference_config['topP'] = top_p
        if stop_sequences := model_settings.get('stop_sequences'):
            inference_config['stopSequences'] = stop_sequences

        return inference_config

    async def _map_messages(
        self, messages: list[ModelMessage]
    ) -> tuple[list[SystemContentBlockTypeDef], list[MessageUnionTypeDef]]:
        """Maps a `pydantic_ai.Message` to the Bedrock `MessageUnionTypeDef`.

        Groups consecutive ToolReturnPart objects into a single user message as required by Bedrock Claude/Nova models.
        """
        system_prompt: list[SystemContentBlockTypeDef] = []
        bedrock_messages: list[MessageUnionTypeDef] = []
        document_count: Iterator[int] = count(1)
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, SystemPromptPart):
                        system_prompt.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        bedrock_messages.extend(await self._map_user_prompt(part, document_count))
                    elif isinstance(part, ToolReturnPart):
                        assert part.tool_call_id is not None
                        bedrock_messages.append(
                            {
                                'role': 'user',
                                'content': [
                                    {
                                        'toolResult': {
                                            'toolUseId': part.tool_call_id,
                                            'content': [{'text': part.model_response_str()}],
                                            'status': 'success',
                                        }
                                    }
                                ],
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        # TODO(Marcelo): We need to add a test here.
                        if part.tool_name is None:  # pragma: no cover
                            bedrock_messages.append({'role': 'user', 'content': [{'text': part.model_response()}]})
                        else:
                            assert part.tool_call_id is not None
                            bedrock_messages.append(
                                {
                                    'role': 'user',
                                    'content': [
                                        {
                                            'toolResult': {
                                                'toolUseId': part.tool_call_id,
                                                'content': [{'text': part.model_response()}],
                                                'status': 'error',
                                            }
                                        }
                                    ],
                                }
                            )
            elif isinstance(message, ModelResponse):
                content: list[ContentBlockOutputTypeDef] = []
                for item in message.parts:
                    if isinstance(item, TextPart):
                        content.append({'text': item.content})
                    else:
                        assert isinstance(item, ToolCallPart)
                        content.append(self._map_tool_call(item))
                bedrock_messages.append({'role': 'assistant', 'content': content})
            else:
                assert_never(message)

        # Merge together sequential user messages.
        processed_messages: list[MessageUnionTypeDef] = []
        last_message: dict[str, Any] | None = None
        for current_message in bedrock_messages:
            if (
                last_message is not None
                and current_message['role'] == last_message['role']
                and current_message['role'] == 'user'
            ):
                # Add the new user content onto the existing user message.
                last_content = list(last_message['content'])
                last_content.extend(current_message['content'])
                last_message['content'] = last_content
                continue

            # Add the entire message to the list of messages.
            processed_messages.append(current_message)
            last_message = cast(dict[str, Any], current_message)

        if instructions := self._get_instructions(messages):
            system_prompt.insert(0, {'text': instructions})

        return system_prompt, processed_messages

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart, document_count: Iterator[int]) -> list[MessageUnionTypeDef]:
        content: list[ContentBlockUnionTypeDef] = []
        if isinstance(part.content, str):
            content.append({'text': part.content})
        else:
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    format = item.format
                    if item.is_document:
                        name = f'Document {next(document_count)}'
                        assert format in ('pdf', 'txt', 'csv', 'doc', 'docx', 'xls', 'xlsx', 'html', 'md')
                        content.append({'document': {'name': name, 'format': format, 'source': {'bytes': item.data}}})
                    elif item.is_image:
                        assert format in ('jpeg', 'png', 'gif', 'webp')
                        content.append({'image': {'format': format, 'source': {'bytes': item.data}}})
                    elif item.is_video:
                        assert format in ('mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp')
                        content.append({'video': {'format': format, 'source': {'bytes': item.data}}})
                    else:
                        raise NotImplementedError('Binary content is not supported yet.')
                elif isinstance(item, (ImageUrl, DocumentUrl, VideoUrl)):
                    response = await cached_async_http_client().get(item.url)
                    response.raise_for_status()
                    if item.kind == 'image-url':
                        format = item.media_type.split('/')[1]
                        assert format in ('jpeg', 'png', 'gif', 'webp'), f'Unsupported image format: {format}'
                        image: ImageBlockTypeDef = {'format': format, 'source': {'bytes': response.content}}
                        content.append({'image': image})

                    elif item.kind == 'document-url':
                        name = f'Document {next(document_count)}'
                        data = response.content
                        content.append({'document': {'name': name, 'format': item.format, 'source': {'bytes': data}}})

                    elif item.kind == 'video-url':
                        format = item.media_type.split('/')[1]
                        assert format in ('mkv', 'mov', 'mp4', 'webm', 'flv', 'mpeg', 'mpg', 'wmv', 'three_gp'), (
                            f'Unsupported video format: {format}'
                        )
                        video: VideoBlockTypeDef = {'format': format, 'source': {'bytes': response.content}}
                        content.append({'video': video})
                elif isinstance(item, AudioUrl):  # pragma: no cover
                    raise NotImplementedError('Audio is not supported yet.')
                else:
                    assert_never(item)
        return [{'role': 'user', 'content': content}]

    @staticmethod
    def _map_tool_call(t: ToolCallPart) -> ContentBlockOutputTypeDef:
        return {
            'toolUse': {'toolUseId': _utils.guard_tool_call_id(t=t), 'name': t.tool_name, 'input': t.args_as_dict()}
        }


@dataclass
class BedrockStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for Bedrock models."""

    _model_name: BedrockModelName
    _event_stream: EventStream[ConverseStreamOutputTypeDef]
    _timestamp: datetime = field(default_factory=_utils.now_utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.
        """
        chunk: ConverseStreamOutputTypeDef
        tool_id: str | None = None
        async for chunk in _AsyncIteratorWrapper(self._event_stream):
            # TODO(Marcelo): Switch this to `match` when we drop Python 3.9 support.
            if 'messageStart' in chunk:
                continue
            if 'messageStop' in chunk:
                continue
            if 'metadata' in chunk:
                if 'usage' in chunk['metadata']:
                    self._usage += self._map_usage(chunk['metadata'])
                continue
            if 'contentBlockStart' in chunk:
                index = chunk['contentBlockStart']['contentBlockIndex']
                start = chunk['contentBlockStart']['start']
                if 'toolUse' in start:
                    tool_use_start = start['toolUse']
                    tool_id = tool_use_start['toolUseId']
                    tool_name = tool_use_start['name']
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=index,
                        tool_name=tool_name,
                        args=None,
                        tool_call_id=tool_id,
                    )
                    if maybe_event:
                        yield maybe_event
            if 'contentBlockDelta' in chunk:
                index = chunk['contentBlockDelta']['contentBlockIndex']
                delta = chunk['contentBlockDelta']['delta']
                if 'text' in delta:
                    yield self._parts_manager.handle_text_delta(vendor_part_id=index, content=delta['text'])
                if 'toolUse' in delta:
                    tool_use = delta['toolUse']
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=index,
                        tool_name=tool_use.get('name'),
                        args=tool_use.get('input'),
                        tool_call_id=tool_id,
                    )
                    if maybe_event:
                        yield maybe_event

    @property
    def timestamp(self) -> datetime:
        return self._timestamp

    @property
    def model_name(self) -> str:
        """Get the model name of the response."""
        return self._model_name

    def _map_usage(self, metadata: ConverseStreamMetadataEventTypeDef) -> usage.Usage:
        return usage.Usage(
            request_tokens=metadata['usage']['inputTokens'],
            response_tokens=metadata['usage']['outputTokens'],
            total_tokens=metadata['usage']['totalTokens'],
        )


class _AsyncIteratorWrapper(Generic[T]):
    """Wrap a synchronous iterator in an async iterator."""

    def __init__(self, sync_iterator: Iterable[T]):
        self.sync_iterator = iter(sync_iterator)

    def __aiter__(self):
        return self

    async def __anext__(self) -> T:
        try:
            return await anyio.to_thread.run_sync(next, self.sync_iterator)
        except RuntimeError as e:
            if type(e.__cause__) is StopIteration:
                raise StopAsyncIteration
            else:
                raise e
