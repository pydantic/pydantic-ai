"""Google's stateless Interactions API."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, assert_never

import anyio
import httpx
from pydantic import ConfigDict, JsonValue, TypeAdapter
from typing_extensions import TypedDict

from ... import _model_errors
from ..._thinking_part import render_replayed_thinking
from ...exceptions import UnexpectedModelBehavior, UserError
from ...messages import (
    ERROR_OUTCOMES,
    BinaryContent,
    CachePoint,
    FilePart,
    FileUrl,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    NativeToolCallPart,
    NativeToolReturnPart,
    SpeechPart,
    StructuredContent,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UploadedFile,
    UserPromptPart,
)
from ...native_tools import (
    AbstractNativeTool,
    CodeExecutionTool,
    FileSearchTool,
    ImageGenerationTool,
    MCPServerTool,
    WebFetchTool,
    WebSearchTool,
)
from ...profiles import ModelProfile, ModelProfileSpec, merge_profile
from ...providers import Provider, infer_provider
from ...settings import ModelSettings, ThinkingEffort
from .. import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    _sendable_request_parts,  # pyright: ignore[reportPrivateUsage]
    _unconverted_speech_part_error,  # pyright: ignore[reportPrivateUsage]
    _unprepared_part_error,  # pyright: ignore[reportPrivateUsage]
    _UnpreparedPart,  # pyright: ignore[reportPrivateUsage]
    _unprojected_instruction_delta_error,  # pyright: ignore[reportPrivateUsage]
    check_allow_model_requests,
    download_item,
    get_user_agent,
)
from .._tool_choice import resolve_tool_choice

try:
    from google.genai import Client, interactions as gi
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError, APIStatusError, APITimeoutError
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google Interactions API: `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

from ._replay import replay_step, step_index
from ._response import JSON_OBJECT, media_kind, process_response, replay_steps
from ._streaming import GoogleInteractionsStreamedResponse

__all__ = (
    'GoogleInteractionsModel',
    'GoogleInteractionsModelName',
    'GoogleInteractionsModelSettings',
    'GoogleInteractionsStreamedResponse',
)

GoogleInteractionsModelName = str | gi.ModelParam
"""Public Interactions model names, or another model ID supported by the API."""

_MODEL_NAME = TypeAdapter(gi.ModelParam)
_CONTENT_PARAM = TypeAdapter(gi.ContentParam, config=ConfigDict(arbitrary_types_allowed=True))
_RESPONSE_FORMAT = TypeAdapter(
    gi.CreateModelInteractionResponseFormatParam, config=ConfigDict(arbitrary_types_allowed=True)
)


class GoogleInteractionsModelSettings(ModelSettings, total=False):
    """Settings for Google's Interactions API.

    Sampling settings unsupported by this API (`top_k`, penalties, `logit_bias` and
    `parallel_tool_calls`) are ignored. Explicit caching and token counting are unavailable.
    """

    google_thinking_level: gi.ThinkingLevel
    """Native thinking level; overrides the shared `thinking` setting."""
    google_thinking_summaries: gi.ThinkingSummaries
    """Whether the API includes thought summaries."""
    google_safety_settings: list[gi.SafetySettingParam]
    """Safety settings in the Interactions API format. Support depends on the model and API."""
    google_response_format: gi.CreateModelInteractionResponseFormatParam
    """Native response format for text, image, audio or video generation.

    Structured output and `ImageGenerationTool` configure this automatically.
    """
    google_generation_config: gi.GenerationConfigParam
    """Additional typed generation configuration, including speech, transcription and video.

    Values override the corresponding shared model settings.
    """
    google_service_tier: gi.ServiceTier
    """Native service tier; overrides the shared `service_tier` setting."""
    google_native_tools: list[gi.ToolParam]
    """Additional typed native tools, such as Google Maps and Vertex retrieval.

    Use `NativeTool` with the shared native-tool types when one represents the tool.
    """
    google_environment: gi.CreateModelInteractionEnvironmentParam
    """A provider environment for a model that supports remote execution."""
    google_labels: dict[str, str]
    """Labels attached to the request."""


@dataclass(init=False)
class GoogleInteractionsModel(Model[Client]):
    """A Google model accessed through the Interactions API, with `store=False`.

    Select explicitly with `google-interactions:<model>`, or pass an existing Google provider.
    Every request sends the complete conversation, including signed thought and native-tool steps.
    This adapter does not store interactions or continue them by ID.
    """

    _model_name: GoogleInteractionsModelName = field(repr=False)
    unsupported_settings = frozenset(
        {'top_k', 'presence_penalty', 'frequency_penalty', 'logit_bias', 'parallel_tool_calls', 'max_retries'}
    )

    def __init__(
        self,
        model_name: GoogleInteractionsModelName,
        *,
        provider: Literal['google', 'google-cloud', 'gateway'] | Provider[Client] = 'google',
        profile: ModelProfileSpec | None = None,
        settings: GoogleInteractionsModelSettings | None = None,
    ):
        """Initialize an Interactions model using the existing Google provider and its client.

        Args:
            model_name: The public Google model ID.
            provider: Gemini API, Google Cloud, gateway, or a caller-configured Google provider.
            profile: Overrides for the model profile.
            settings: Default request settings.
        """
        self._model_name = model_name
        self._provider = (
            infer_provider('gateway/google-cloud' if provider == 'gateway' else provider)
            if isinstance(provider, str)
            else provider
        )
        super().__init__(settings=settings, profile=profile)

    @property
    def client(self) -> Client:
        return self._provider.client

    @property
    def model_name(self) -> GoogleInteractionsModelName:
        return self._model_name

    @property
    def system(self) -> str:
        return self._provider.name

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    def _instance_profile_defaults(self, profile: ModelProfile) -> ModelProfile:
        return merge_profile(profile, ModelProfile(supports_tool_return_schema=False))

    @cached_property
    def profile(self) -> ModelProfile:
        profile = super().profile
        if not profile.get('supports_image_output', False):
            return merge_profile(
                profile,
                ModelProfile(
                    supported_native_tools=profile.get('supported_native_tools', self.supported_native_tools())
                    - {ImageGenerationTool}
                ),
            )
        return profile

    def _refresh_http_client(self) -> None:
        resource = self.client.aio._interactions  # pyright: ignore[reportPrivateUsage]
        previous = self.client._api_client._async_httpx_client  # pyright: ignore[reportPrivateUsage]
        super()._refresh_http_client()
        current = self.client._api_client._async_httpx_client  # pyright: ignore[reportPrivateUsage]
        if resource is not None and current is not previous and resource.sdk_configuration.async_client is previous:
            self.client.aio._interactions = None  # pyright: ignore[reportPrivateUsage]

    async def _prepare_client(self) -> None:
        api_client = self.client._api_client  # pyright: ignore[reportPrivateUsage]
        if self.client.vertexai and not api_client.api_key:
            # Interactions calls the synchronous token provider even from its async request path.
            # Resolve and refresh ADC outside the event loop before the SDK reads the cached token.
            await anyio.to_thread.run_sync(api_client._access_token)  # pyright: ignore[reportPrivateUsage]

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        return frozenset(
            {WebSearchTool, WebFetchTool, CodeExecutionTool, FileSearchTool, ImageGenerationTool, MCPServerTool}
        )

    def unsupported_native_tool_settings(self, native_tool: AbstractNativeTool) -> frozenset[str]:
        return native_tool.restricting_settings

    def _settings_overriding_thinking(self, model_settings: Mapping[str, object]) -> tuple[str, ...]:
        overrides: list[str] = []
        if 'google_thinking_level' in model_settings:
            overrides.append('google_thinking_level')
        if 'google_generation_config' in model_settings:
            config = TypeAdapter(gi.GenerationConfigParam).validate_python(model_settings['google_generation_config'])
            if 'thinking_level' in config:
                overrides.append('google_generation_config')
        return tuple(overrides)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, params = self.prepare_request(model_settings, model_request_parameters)
        request, steps = await self._build_request(messages, model_settings or {}, params)
        with self._map_api_errors():
            await self._prepare_client()
            response = await self.client.aio.interactions.create(
                request={'body': request}, **self._http_options(model_settings or {}, steps)
            )
        if not isinstance(response, gi.Interaction):
            raise UnexpectedModelBehavior('Expected a non-streaming Google interaction.')
        return await process_response(
            response, self.model_name, self.system, self.base_url, self._canonical_system_stamp
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: object = None,
    ) -> AsyncGenerator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, params = self.prepare_request(model_settings, model_request_parameters)
        request, steps = await self._build_request(messages, model_settings or {}, params)
        request['stream'] = True
        with self._map_api_errors():
            await self._prepare_client()
            response = await self.client.aio.interactions.create(
                request={'body': request}, **self._http_options(model_settings or {}, steps)
            )
            if isinstance(response, gi.Interaction):
                raise UnexpectedModelBehavior('Expected a streaming Google interaction.')
            stream = GoogleInteractionsStreamedResponse(params, response, self.model_name, self.system, self.base_url)
            stream.canonical_system = self._canonical_system_stamp
            try:
                yield stream
            finally:
                await response.close()

    @contextmanager
    def _map_api_errors(self) -> Generator[None]:
        try:
            yield
        except APITimeoutError as exc:
            raise _model_errors.connection_error(self.model_name, str(exc), exc, timeout=True) from exc
        except APIConnectionError as exc:
            raise _model_errors.connection_error(self.model_name, str(exc), exc, timeout=False) from exc
        except APIStatusError as exc:
            raise _model_errors.transport_status_error(self.model_name, exc.response) from exc
        except APIError as exc:
            raise _model_errors.stream_error(self.model_name, str(exc), exc.body) from exc

    def _http_options(self, settings: ModelSettings, steps: list[dict[str, JsonValue]]) -> _HttpOptions:
        options: _HttpOptions = {'extra_headers': {'User-Agent': get_user_agent(), **settings.get('extra_headers', {})}}
        if (timeout := settings.get('timeout')) is not None:
            options['timeout'] = (
                timeout
                if isinstance(timeout, (int, float))
                else httpx.Timeout(connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool)
            )
        if (extra_body := settings.get('extra_body')) is not None:
            options['extra_body'] = TypeAdapter(dict[str, object]).validate_python(extra_body)
        # The SDK's StepParam drops replay fields it has not typed yet, including MCP signatures.
        # Merge the validated steps after SDK serialization so every opaque field survives.
        options['extra_body'] = {**options.get('extra_body', {}), 'input': steps, 'store': False}
        return options

    async def _build_request(
        self, messages: list[ModelMessage], settings: ModelSettings, params: ModelRequestParameters
    ) -> tuple[gi.CreateModelInteractionParam, list[dict[str, JsonValue]]]:
        instructions, steps = await self._map_messages(messages, params)
        request: gi.CreateModelInteractionParam = {
            'model': _MODEL_NAME.validate_python(self.model_name),
            'store': False,
        }
        if instructions:
            request['system_instruction'] = '\n\n'.join(instructions)
        response_format: gi.CreateModelInteractionResponseFormatParam | None
        tools, response_format = self._map_tools(params)
        if 'google_native_tools' in settings:
            tools.extend(
                TypeAdapter(list[gi.ToolParam], config=ConfigDict(arbitrary_types_allowed=True)).validate_python(
                    settings['google_native_tools']
                )
            )
        if 'google_environment' in settings:
            request['environment'] = TypeAdapter(gi.CreateModelInteractionEnvironmentParam).validate_python(
                settings['google_environment']
            )
        if tools:
            request['tools'] = tools
        config = self._generation_config(settings, params)
        if config:
            request['generation_config'] = config
        response_format = self._response_format(settings, params, response_format)
        if response_format is not None:
            request['response_format'] = response_format
        if 'google_safety_settings' in settings:
            request['safety_settings'] = TypeAdapter(list[gi.SafetySettingParam]).validate_python(
                settings['google_safety_settings']
            )
        if 'google_labels' in settings:
            request['labels'] = TypeAdapter(dict[str, str]).validate_python(settings['google_labels'])
        tier = settings.get('google_service_tier')
        if tier is None:
            tier = {'default': 'standard', 'flex': 'flex', 'priority': 'priority'}.get(
                settings.get('service_tier', 'auto')
            )
        if tier is not None:
            request['service_tier'] = TypeAdapter(gi.ServiceTier).validate_python(tier)
        return request, TypeAdapter(list[dict[str, JsonValue]]).validate_python(steps)

    def _response_format(
        self,
        settings: ModelSettings,
        params: ModelRequestParameters,
        response_format: gi.CreateModelInteractionResponseFormatParam | None,
    ) -> gi.CreateModelInteractionResponseFormatParam | None:
        if params.allow_image_output and response_format is None:
            response_format = [{'type': 'image'}, {'type': 'text'}] if params.allow_text_output else {'type': 'image'}
        if params.output_mode == 'native':
            assert params.output_object is not None
            response_format = {
                'type': 'text',
                'mime_type': 'application/json',
                'schema_': params.output_object.json_schema,
            }
        elif params.output_mode == 'prompted':
            response_format = {'type': 'text', 'mime_type': 'application/json'}
        if 'google_response_format' in settings:
            response_format = _RESPONSE_FORMAT.validate_python(settings['google_response_format'])
        return response_format

    def _generation_config(self, settings: ModelSettings, params: ModelRequestParameters) -> gi.GenerationConfigParam:
        config: gi.GenerationConfigParam = {}
        for key in ('temperature', 'top_p', 'max_tokens', 'stop_sequences', 'seed'):
            if key in settings:
                native_key = 'max_output_tokens' if key == 'max_tokens' else key
                config.update(TypeAdapter(gi.GenerationConfigParam).validate_python({native_key: settings.get(key)}))
        config.update(self._thinking_config(params.thinking))
        if 'google_thinking_level' in settings:
            config['thinking_level'] = TypeAdapter(gi.ThinkingLevel).validate_python(settings['google_thinking_level'])
        if 'google_thinking_summaries' in settings:
            config['thinking_summaries'] = TypeAdapter(gi.ThinkingSummaries).validate_python(
                settings['google_thinking_summaries']
            )
        if params.declared_tool_defs:
            config['tool_choice'] = self._tool_choice(settings, params)
        if 'google_generation_config' in settings:
            config.update(TypeAdapter(gi.GenerationConfigParam).validate_python(settings['google_generation_config']))
        return config

    def _tool_choice(self, settings: ModelSettings, params: ModelRequestParameters) -> gi.ToolChoiceParam:
        choice = resolve_tool_choice(settings, params)
        modes: dict[Literal['auto', 'none', 'required'], gi.ToolChoiceType] = {
            'auto': 'auto',
            'none': 'none',
            'required': 'any',
        }
        if self.profile.get('google_supports_strict_tool_definition') and not any(
            tool.strict is False for tool in params.declared_tool_defs.values()
        ):
            modes['auto'] = 'validated'
        if isinstance(choice, tuple):
            mode, names = choice
            return {'allowed_tools': {'mode': modes[mode], 'tools': sorted(names)}}
        else:
            return modes[choice]

    def _thinking_config(self, thinking: ThinkingEffort | bool | None) -> gi.GenerationConfigParam:
        config: gi.GenerationConfigParam = {}
        if isinstance(thinking, str):
            levels: dict[ThinkingEffort, gi.ThinkingLevel] = {
                'minimal': 'minimal',
                'low': 'low',
                'medium': 'medium',
                'high': 'high',
                'xhigh': 'high',
                'max': 'high',
            }
            config['thinking_level'] = levels[thinking]
            config['thinking_summaries'] = 'auto'
        elif thinking is True:
            config['thinking_summaries'] = 'auto'
        elif thinking is False:
            config['thinking_level'] = 'minimal'
            config['thinking_summaries'] = 'none'
        return config

    def _map_tools(
        self, params: ModelRequestParameters
    ) -> tuple[list[gi.ToolParam], gi.CreateModelInteractionResponseFormatParam | None]:
        tools: list[gi.ToolParam] = [
            {
                'type': 'function',
                'name': tool.name,
                'description': tool.description or '',
                'parameters': tool.parameters_json_schema,
            }
            for tool in params.declared_tool_defs.values()
        ]
        response_format: gi.CreateModelInteractionResponseFormatParam | None = None
        for tool in params.native_tools:
            match tool:
                case WebSearchTool():
                    tools.append({'type': 'google_search'})
                case WebFetchTool():
                    tools.append({'type': 'url_context'})
                case CodeExecutionTool():
                    tools.append({'type': 'code_execution'})
                case FileSearchTool():
                    tools.append(self._map_file_search_tool(tool))
                case MCPServerTool():
                    tools.append(self._map_mcp_tool(tool))
                case ImageGenerationTool():
                    image_format = self._map_image_generation_tool(tool)
                    response_format = [image_format, {'type': 'text'}] if params.allow_text_output else image_format
                case _:
                    raise UserError(f'`{type(tool).__name__}` is not supported by `GoogleInteractionsModel`.')
        return tools, response_format

    def _map_file_search_tool(self, tool: FileSearchTool) -> gi.FileSearchParam:
        file_search: gi.FileSearchParam = {
            'type': 'file_search',
            'file_search_store_names': list(tool.file_store_ids),
        }
        if tool.max_num_results is not None:
            file_search['top_k'] = tool.max_num_results
        return file_search

    def _map_mcp_tool(self, tool: MCPServerTool) -> gi.MCPServerParam:
        mcp: gi.MCPServerParam = {'type': 'mcp_server', 'url': tool.url, 'name': tool.id}
        if tool.headers:
            mcp['headers'] = tool.headers
        if tool.authorization_token:
            mcp['headers'] = {**mcp.get('headers', {}), 'Authorization': tool.authorization_token}
        if tool.allowed_tools is not None:
            mcp['allowed_tools'] = [{'tools': tool.allowed_tools}]
        return mcp

    def _map_image_generation_tool(self, tool: ImageGenerationTool) -> gi.ImageResponseFormatParam:
        image_format: gi.ImageResponseFormatParam = {'type': 'image'}
        if tool.aspect_ratio:
            image_format['aspect_ratio'] = tool.aspect_ratio
        if tool.size not in (None, 'auto'):
            image_format['image_size'] = TypeAdapter(gi.ImageResponseFormatImageSize).validate_python(tool.size)
        if tool.output_format not in (None, 'png'):
            image_format['mime_type'] = TypeAdapter(gi.ImageResponseFormatMimeType).validate_python(
                f'image/{tool.output_format}'
            )
        return image_format

    async def _map_messages(
        self, messages: list[ModelMessage], params: ModelRequestParameters
    ) -> tuple[list[str], list[gi.StepParam | dict[str, JsonValue]]]:
        context = _MapMessagesContext()
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in _sendable_request_parts(message.parts):
                    await self._map_request_part(part, context)
            elif isinstance(message, ModelResponse):
                context.steps.extend(await self._map_model_response(message))
            else:
                assert_never(message)
        if parts := self._get_instruction_parts(messages, params):
            context.instructions.extend(part.content for part in parts)
        return context.instructions, context.steps

    async def _map_request_part(self, part: ModelRequestPart, context: _MapMessagesContext) -> None:
        if isinstance(part, _UnpreparedPart):
            raise _unprepared_part_error(part)
        match part:
            case SystemPromptPart():
                self._map_system_prompt(part, context)
            case UserPromptPart():
                await self._map_user_prompt_part(part, context)
            case ToolReturnPart():
                context.steps.extend(await self._map_tool_return(part))
            case InstructionBaselinePart() | InstructionDeltaPart():
                raise _unprojected_instruction_delta_error()
            case SpeechPart():
                raise _unconverted_speech_part_error()
            case _:
                assert_never(part)

    def _map_system_prompt(self, part: SystemPromptPart, context: _MapMessagesContext) -> None:
        context.instructions.append(part.content)

    async def _map_user_prompt_part(self, part: UserPromptPart, context: _MapMessagesContext) -> None:
        if content := await self._map_user_prompt(part):
            context.steps.append({'type': 'user_input', 'content': content})

    async def _map_user_prompt(self, part: UserPromptPart) -> list[gi.ContentParam]:
        if isinstance(part.content, str):
            return [{'type': 'text', 'text': part.content}] if part.content else []
        content: list[gi.ContentParam] = []
        for item in part.content:
            if isinstance(item, str):
                if item:
                    content.append({'type': 'text', 'text': item})
            elif isinstance(item, (TextContent, StructuredContent)):
                content.append({'type': 'text', 'text': item.content})
            elif isinstance(item, CachePoint):
                continue
            elif isinstance(item, (FileUrl, BinaryContent, UploadedFile)):
                content.append(await self._map_file(item))
            else:
                assert_never(item)
        return content

    async def _map_file(self, item: FileUrl | BinaryContent | UploadedFile) -> gi.ContentParam:
        if isinstance(item, UploadedFile):
            self._validate_uploaded_file_provider(item)
            if self.client.vertexai:
                if not item.file_id.startswith('gs://'):
                    raise UserError('UploadedFile for Google Interactions on Google Cloud must use a GCS URI (gs://).')
            elif not item.file_id.startswith('https://'):
                raise UserError(
                    'UploadedFile for Google Interactions on the Gemini API must use a Files API URI (https://).'
                )
            value: dict[str, object] = {'uri': item.file_id, 'mime_type': item.media_type}
        elif isinstance(item, BinaryContent):
            value = {'data': base64.b64encode(item.data).decode(), 'mime_type': item.media_type}
        else:
            if item.force_download:
                downloaded = await download_item(item, data_format='bytes')
                value = {'data': base64.b64encode(downloaded['data']).decode(), 'mime_type': downloaded['data_type']}
            else:
                value = {'uri': item.url, 'mime_type': item.media_type}
        value['type'] = media_kind(str(value['mime_type']))
        return _CONTENT_PARAM.validate_python(value)

    async def _map_tool_return(self, part: ToolReturnPart) -> list[dict[str, JsonValue]]:
        content: list[gi.ContentParam] = []
        for item in part.content_items(mode='str', wrap_if_error=False):
            if isinstance(item, str):
                content.append({'type': 'text', 'text': item})
            else:
                content.append(await self._map_file(item))
        result: _FunctionResultStep = {
            'type': 'function_result',
            'name': part.tool_name,
            'call_id': part.tool_call_id,
            'is_error': part.outcome in ERROR_OUTCOMES,
            'result': content,
        }
        return [TypeAdapter(dict[str, JsonValue]).validate_python(result)]

    async def _map_model_response(self, message: ModelResponse) -> list[gi.StepParam | dict[str, JsonValue]]:
        original = replay_steps(message) if self.accepts_stateless_artifacts(message) else None
        groups: dict[int, list[ModelResponsePart]] = {}
        if original is not None:
            for part in message.parts:
                index = step_index(part)
                if index is not None and 0 <= index < len(original) and self.accepts_stateless_artifacts(message, part):
                    groups.setdefault(index, []).append(part)
        steps: list[gi.StepParam | dict[str, JsonValue]] = []
        sent: set[int] = set()
        for part in message.parts:
            index = step_index(part)
            if original is not None and index in groups:
                assert index is not None
                if index not in sent:
                    steps.append(replay_step(original[index], groups[index]))
                    sent.add(index)
                continue
            steps.extend(await self._map_response_part(message, part))
        return steps

    async def _map_response_part(
        self, message: ModelResponse, part: ModelResponsePart
    ) -> list[gi.StepParam | dict[str, JsonValue]]:
        steps: list[gi.StepParam | dict[str, JsonValue]] = []
        if part.provider_details and self.accepts_stateless_artifacts(message, part):
            signature = part.provider_details.get('thought_signature')
            if isinstance(signature, str):
                steps.append({'type': 'thought', 'signature': signature})
        if isinstance(part, _UnpreparedPart):
            raise _unprepared_part_error(part)
        match part:
            case TextPart():
                steps.extend(self._map_text_part(part))
            case ThinkingPart():
                steps.extend(self._map_thinking_part(message, part))
            case ToolCallPart():
                steps.extend(self._map_tool_call_part(part))
            case FilePart():
                steps.extend(await self._map_file_part(part))
            case NativeToolCallPart() | NativeToolReturnPart():
                steps.extend(self._map_native_part(message, part))
            case SpeechPart():
                raise _unconverted_speech_part_error()
            case _:
                assert_never(part)
        return steps

    def _map_text_part(self, part: TextPart) -> list[gi.StepParam]:
        return [{'type': 'model_output', 'content': [{'type': 'text', 'text': part.content}]}] if part.content else []

    def _map_tool_call_part(self, part: ToolCallPart) -> list[gi.StepParam]:
        return [
            {'type': 'function_call', 'id': part.tool_call_id, 'name': part.tool_name, 'arguments': part.args_as_dict()}
        ]

    async def _map_file_part(self, part: FilePart) -> list[gi.StepParam]:
        return [{'type': 'model_output', 'content': [await self._map_file(part.content)]}]

    def _map_native_part(
        self, message: ModelResponse, part: NativeToolCallPart | NativeToolReturnPart
    ) -> list[dict[str, JsonValue]]:
        if not self.accepts_stateless_artifacts(message, part):
            return []
        raw = JSON_OBJECT.validate_python(part.provider_details or {})
        return [
            replay_step(
                {key: value for key, value in raw.items() if not key.startswith('google_interactions_')}, [part]
            )
        ]

    def _map_thinking_part(self, message: ModelResponse, part: ThinkingPart) -> list[gi.StepParam]:
        if part.signature and self.accepts_stateless_artifacts(message, part):
            thought: gi.ThoughtStepParam = {'type': 'thought', 'signature': part.signature}
            if part.content:
                thought['summary'] = [{'type': 'text', 'text': part.content}]
            return [thought]
        elif part.content:
            text, user_turn = render_replayed_thinking(part.content, self.profile)
            if user_turn:
                return [{'type': 'user_input', 'content': [{'type': 'text', 'text': text}]}]
            return [{'type': 'model_output', 'content': [{'type': 'text', 'text': text}]}]
        return []


@dataclass
class _MapMessagesContext:
    instructions: list[str] = field(default_factory=list[str])
    steps: list[gi.StepParam | dict[str, JsonValue]] = field(default_factory=list[gi.StepParam | dict[str, JsonValue]])


class _FunctionResultStep(TypedDict):
    # The public API accepts all five media kinds; the SDK's result union still only types text and images.
    type: Literal['function_result']
    name: str
    call_id: str
    is_error: bool
    result: list[gi.ContentParam]


class _HttpOptions(TypedDict, total=False):
    extra_headers: dict[str, str]
    extra_body: Mapping[str, object]
    timeout: float | httpx.Timeout
