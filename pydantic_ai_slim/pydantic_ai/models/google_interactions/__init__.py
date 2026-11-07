"""Google's stateless Interactions API."""

from __future__ import annotations

import base64
from collections.abc import AsyncGenerator, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal, assert_never

import anyio
import httpx
import httpx2
from pydantic import ConfigDict, JsonValue, TypeAdapter
from typing_extensions import TypedDict

from ... import _model_errors, _utils
from ..._run_context import RunContext
from ..._thinking_part import render_replayed_thinking
from ...exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError
from ...messages import (
    ERROR_OUTCOMES,
    BinaryContent,
    CachePoint,
    CompactionPart,
    FilePart,
    FileUrl,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    MultiModalContent,
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
from ...profiles.google import GoogleModelProfile, google_interactions_model_profile
from ...providers import Provider, infer_provider
from ...settings import ModelSettings, ThinkingEffort
from .. import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    _google_common,
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

    # Interactions uses separate compatibility errors that the public google.genai.errors does not export.
    from google.genai._gaos.lib.compat_errors import APIConnectionError, APIError, APIStatusError, APITimeoutError
    from google.genai.types import MediaResolution
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

_GENERATION_CONFIG = TypeAdapter(gi.GenerationConfigParam)
_EXTRA_BODY = TypeAdapter(dict[str, object])
_SAFETY_SETTINGS = TypeAdapter(list[gi.SafetySettingParam])
_LABELS = TypeAdapter(dict[str, str])
_SERVICE_TIER = TypeAdapter(gi.ServiceTier)
_STEPS = TypeAdapter(list[dict[str, JsonValue]])
_THINKING_LEVEL = TypeAdapter(gi.ThinkingLevel)
_THINKING_SUMMARIES = TypeAdapter(gi.ThinkingSummaries)
_IMAGE_SIZE = TypeAdapter(gi.ImageResponseFormatImageSize)
_IMAGE_MIME_TYPE = TypeAdapter(gi.ImageResponseFormatMimeType)
_NATIVE_TOOLS = TypeAdapter(list[gi.ToolParam], config=ConfigDict(arbitrary_types_allowed=True))
_VIDEO_RESOLUTION = TypeAdapter(MediaResolution)
_CONTENT_RESOLUTION = TypeAdapter(gi.MediaResolution)


class GoogleInteractionsModelSettings(ModelSettings, total=False):
    """Settings for Google's Interactions API.

    Sampling settings unsupported by this API (`top_k`, penalties, `logit_bias` and
    `parallel_tool_calls`) are ignored. Explicit caching and token counting are unavailable.
    Models that use thinking budgets retain their default effort: Interactions exposes only thinking levels.

    These native Interactions settings are ignored by `GoogleModel`, except for the shared
    `google_labels` and `google_video_resolution`.
    Use the shared `thinking` and `service_tier` settings when switching between the two APIs.
    """

    google_interactions_thinking_level: gi.ThinkingLevel
    """Native thinking level; overrides the shared `thinking` setting."""
    google_interactions_thinking_summaries: gi.ThinkingSummaries
    """Whether the API includes thought summaries."""
    google_interactions_safety_settings: list[gi.SafetySettingParam]
    """Safety settings in the Interactions API format. Support depends on the model and API."""
    google_interactions_response_format: gi.CreateModelInteractionResponseFormatParam
    """Native response format for text, image, audio or video generation.

    Structured output and `ImageGenerationTool` configure this automatically.
    Use inline delivery: video URI delivery requires server-side storage, which this model disables.
    """
    google_interactions_generation_config: gi.GenerationConfigParam
    """Additional typed generation configuration, including speech, transcription and video.

    Values override the corresponding shared model settings.
    """
    google_interactions_service_tier: gi.ServiceTier
    """Native service tier; overrides the shared `service_tier` setting."""
    google_interactions_native_tools: list[gi.ToolParam]
    """Additional typed native tools, such as Google Maps and Vertex retrieval.

    Use `NativeTool` with the shared native-tool types when one represents the tool.
    """
    google_labels: dict[str, str]
    """Labels attached to the request."""
    google_video_resolution: MediaResolution
    """Default resolution for video inputs. Per-file `vendor_metadata` takes precedence."""


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
        provider: Literal['google', 'google-cloud'] | Provider[Client] = 'google',
        profile: ModelProfileSpec | None = None,
        settings: GoogleInteractionsModelSettings | None = None,
    ):
        """Initialize an Interactions model using the existing Google provider and its client.

        Args:
            model_name: The public Google model ID.
            provider: Gemini API, Google Cloud, or a caller-configured Google provider.
            profile: Overrides for the model profile.
            settings: Default request settings.
        """
        self._model_name = model_name
        self._provider = infer_provider(provider) if isinstance(provider, str) else provider
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

    def _provider_model_profile(self) -> ModelProfile:
        model_name = self.model_name.rsplit('/', 1)[-1]
        return self._provider._configured_model_profile(model_name) or {}  # pyright: ignore[reportPrivateUsage]

    def _instance_profile_defaults(self, profile: ModelProfile) -> ModelProfile:
        return merge_profile(profile, google_interactions_model_profile(self.model_name))

    @cached_property
    def profile(self) -> GoogleModelProfile:
        profile = GoogleModelProfile(**super().profile)
        if not profile.get('supports_image_output', False):
            return GoogleModelProfile(
                **merge_profile(
                    profile,
                    ModelProfile(
                        supported_native_tools=profile.get('supported_native_tools', self.supported_native_tools())
                        - {ImageGenerationTool}
                    ),
                )
            )
        return profile

    def _refresh_http_client(self) -> None:
        # The SDK caches its transport on the lazy Interactions resource, so owned-client refresh must invalidate it.
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
        if 'google_interactions_thinking_level' in model_settings:
            overrides.append('google_interactions_thinking_level')
        if 'google_interactions_generation_config' in model_settings:
            config = _GENERATION_CONFIG.validate_python(model_settings['google_interactions_generation_config'])
            if 'thinking_level' in config:
                overrides.append('google_interactions_generation_config')
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
        return process_response(
            response,
            self.model_name,
            self.system,
            self.base_url,
            self._canonical_system_stamp,
            web_search_billed_per_prompt=self.profile.get('google_web_search_billed_per_prompt', False),
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
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
            stream = GoogleInteractionsStreamedResponse(
                params,
                response,
                self.model_name,
                self.system,
                self.base_url,
                _close_response=response.close,
                _web_search_billed_per_prompt=self.profile.get('google_web_search_billed_per_prompt', False),
            )
            stream.canonical_system = self._canonical_system_stamp
            try:
                yield stream
            finally:
                await stream.close_stream()

    @contextmanager
    def _map_api_errors(self) -> Generator[None]:
        try:
            yield
        except APITimeoutError as exc:
            raise _model_errors.connection_error(self.model_name, str(exc), exc, timeout=True) from exc
        except APIConnectionError as exc:
            if isinstance(exc.__cause__, httpx.HTTPStatusError | httpx2.HTTPStatusError):
                raise _model_errors.transport_status_error(self.model_name, exc.__cause__.response) from exc
            raise _model_errors.connection_error(self.model_name, str(exc), exc, timeout=False) from exc
        except APIStatusError as exc:
            raise self._map_status_error(exc) from exc
        except APIError as exc:
            raise _model_errors.stream_error(self.model_name, str(exc), exc.body) from exc

    def _map_status_error(self, exc: APIStatusError) -> ModelHTTPError:
        body = exc.body
        error = body.get('error', body) if _utils.is_str_dict(body) else body
        status = error.get('status') if _utils.is_str_dict(error) else None
        code = error.get('code') if _utils.is_str_dict(error) else None
        status = status if isinstance(status, str) else None
        code = str(code) if isinstance(code, str | int) else None
        message = error.get('message') if _utils.is_str_dict(error) else str(exc)
        category = _model_errors.google_status_category(status, message, body)
        return ModelHTTPError.for_category(
            category or _model_errors.http_status_category(exc.response.status_code, message),
            status_code=exc.response.status_code,
            model_name=self.model_name,
            body=body,
            headers=dict(exc.response.headers),
            provider_error_code=status or code,
        )

    def _http_options(self, settings: ModelSettings, steps: list[dict[str, JsonValue]]) -> _HttpOptions:
        options: _HttpOptions = {'extra_headers': {'User-Agent': get_user_agent(), **settings.get('extra_headers', {})}}
        if (timeout := settings.get('timeout')) is not None:
            options['timeout'] = (
                timeout
                if isinstance(timeout, (int, float))
                else httpx.Timeout(connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool)
            )
        if (extra_body := settings.get('extra_body')) is not None:
            options['extra_body'] = _EXTRA_BODY.validate_python(extra_body)
        # The SDK's StepParam drops replay fields it has not typed yet, including MCP signatures.
        # Merge the validated steps after SDK serialization so every opaque field survives.
        options['extra_body'] = {**options.get('extra_body', {}), 'input': steps, 'store': False}
        return options

    async def _build_request(
        self, messages: list[ModelMessage], settings: ModelSettings, params: ModelRequestParameters
    ) -> tuple[gi.CreateModelInteractionParam, list[dict[str, JsonValue]]]:
        video_resolution = (
            _VIDEO_RESOLUTION.validate_python(settings['google_video_resolution'])
            if 'google_video_resolution' in settings
            else None
        )
        instructions, steps = await self._map_messages(messages, params, video_resolution=video_resolution)
        request: gi.CreateModelInteractionParam = {
            'model': _MODEL_NAME.validate_python(self.model_name),
            'store': False,
        }
        if instructions:
            request['system_instruction'] = '\n\n'.join(instructions)
        response_format: gi.CreateModelInteractionResponseFormatParam | None
        tools, response_format = self._map_tools(params)
        if 'google_interactions_native_tools' in settings:
            tools.extend(_NATIVE_TOOLS.validate_python(settings['google_interactions_native_tools']))
        if tools:
            request['tools'] = tools
        config = self._generation_config(settings, params)
        if config:
            request['generation_config'] = config
        response_format = self._response_format(settings, params, response_format)
        if response_format is not None:
            request['response_format'] = response_format
        if 'google_interactions_safety_settings' in settings:
            request['safety_settings'] = _SAFETY_SETTINGS.validate_python(
                settings['google_interactions_safety_settings']
            )
        if 'google_labels' in settings:
            request['labels'] = _LABELS.validate_python(settings['google_labels'])
        tier = settings.get('google_interactions_service_tier')
        if tier is None:
            tier = {'default': 'standard', 'flex': 'flex', 'priority': 'priority'}.get(
                settings.get('service_tier', 'auto')
            )
        if tier is not None:
            request['service_tier'] = _SERVICE_TIER.validate_python(tier)
        return request, _STEPS.validate_python(steps)

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
        if 'google_interactions_response_format' in settings:
            response_format = _RESPONSE_FORMAT.validate_python(settings['google_interactions_response_format'])
        return response_format

    def _generation_config(self, settings: ModelSettings, params: ModelRequestParameters) -> gi.GenerationConfigParam:
        config: gi.GenerationConfigParam = {}
        for key in ('temperature', 'top_p', 'max_tokens', 'stop_sequences', 'seed'):
            if key in settings:
                native_key = 'max_output_tokens' if key == 'max_tokens' else key
                config.update(_GENERATION_CONFIG.validate_python({native_key: settings.get(key)}))
        config.update(self._thinking_config(params.thinking))
        if 'google_interactions_thinking_level' in settings:
            config['thinking_level'] = _THINKING_LEVEL.validate_python(settings['google_interactions_thinking_level'])
        if 'google_interactions_thinking_summaries' in settings:
            config['thinking_summaries'] = _THINKING_SUMMARIES.validate_python(
                settings['google_interactions_thinking_summaries']
            )
        if params.declared_tool_defs:
            config['tool_choice'] = self._tool_choice(settings, params)
        if 'google_interactions_generation_config' in settings:
            config.update(_GENERATION_CONFIG.validate_python(settings['google_interactions_generation_config']))
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
        if thinking is None:
            return config
        config['thinking_summaries'] = 'none' if thinking is False else 'auto'
        if thinking is not True and self.profile.get('google_supports_thinking_level', True):
            effort = 'minimal' if thinking is False else thinking
            level = _google_common.resolve_google_thinking_level(effort, self.profile)
            config['thinking_level'] = _THINKING_LEVEL.validate_python(level.lower())
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
            image_format['image_size'] = _IMAGE_SIZE.validate_python(tool.size)
        if tool.output_format not in (None, 'png'):
            image_format['mime_type'] = _IMAGE_MIME_TYPE.validate_python(f'image/{tool.output_format}')
        return image_format

    async def _map_messages(
        self,
        messages: list[ModelMessage],
        params: ModelRequestParameters,
        *,
        video_resolution: MediaResolution | None = None,
    ) -> tuple[list[str], list[gi.StepParam | dict[str, JsonValue]]]:
        context = _MapMessagesContext(video_resolution=video_resolution)
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in _sendable_request_parts(message.parts):
                    await self._map_request_part(part, context)
                if context.tool_files:
                    context.steps.append({'type': 'user_input', 'content': context.tool_files})
                    context.tool_files = []
            elif isinstance(message, ModelResponse):
                context.steps.extend(await self._map_model_response(message, context))
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
                context.instructions.append(part.content)
            case UserPromptPart():
                if content := await self._map_user_prompt(part, context):
                    context.steps.append({'type': 'user_input', 'content': content})
            case ToolReturnPart():
                result, files = await self._map_tool_return(part, context)
                context.steps.append(result)
                context.tool_files.extend(files)
            case InstructionBaselinePart() | InstructionDeltaPart():
                raise _unprojected_instruction_delta_error()
            case SpeechPart():
                raise _unconverted_speech_part_error()
            case _:
                assert_never(part)

    async def _map_user_prompt(self, part: UserPromptPart, context: _MapMessagesContext) -> list[gi.ContentParam]:
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
                content.append(await self._map_file(item, context))
            else:
                assert_never(item)
        return content

    async def _map_file(
        self, item: FileUrl | BinaryContent | UploadedFile, context: _MapMessagesContext
    ) -> gi.ContentParam:
        if isinstance(item, UploadedFile):
            _utils.validate_uploaded_file_provider(
                item, system=self.canonical_system, model_type_name=type(self).__name__
            )
            if self.client.vertexai:
                if not item.file_id.startswith('gs://'):
                    raise UserError('UploadedFile for Google Interactions on Google Cloud must use a GCS URI (gs://).')
            elif not item.file_id.startswith('https://'):
                raise UserError(
                    'UploadedFile for Google Interactions on the Gemini API must use an HTTPS URI (https://).'
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
        self._apply_file_metadata(value, item.vendor_metadata or {}, context.video_resolution)
        return _CONTENT_PARAM.validate_python(value)

    def _apply_file_metadata(
        self, value: dict[str, object], metadata: Mapping[str, object], video_resolution: MediaResolution | None
    ) -> None:
        if value['type'] == 'video':
            self._apply_video_metadata(value, metadata, video_resolution)
        if 'media_resolution' in metadata:
            resolution = metadata['media_resolution']
            level = resolution.get('level') if _utils.is_str_dict(resolution) else resolution
            if isinstance(level, str):
                value['resolution'] = _CONTENT_RESOLUTION.validate_python(
                    level.removeprefix('MEDIA_RESOLUTION_').lower()
                )
        if 'media_processing' in metadata:
            value['processing'] = metadata['media_processing']

    def _apply_video_metadata(
        self, value: dict[str, object], metadata: Mapping[str, object], video_resolution: MediaResolution | None
    ) -> None:
        processing: dict[str, object] = {
            key: metadata[key] for key in ('start_offset', 'end_offset', 'fps') if key in metadata
        }
        if processing:
            value['processing'] = {'type': 'static', **processing}
        if video_resolution is not None and video_resolution != MediaResolution.MEDIA_RESOLUTION_UNSPECIFIED:
            value['resolution'] = _CONTENT_RESOLUTION.validate_python(
                video_resolution.value.removeprefix('MEDIA_RESOLUTION_').lower()
            )

    async def _map_tool_return(
        self, part: ToolReturnPart, context: _MapMessagesContext
    ) -> tuple[dict[str, JsonValue], list[gi.ContentParam]]:
        content: list[gi.ContentParam] = []
        fallback_files: list[MultiModalContent] = []
        supported_mime_types = self.profile.get('google_supported_mime_types_in_tool_returns', ())
        for item in part.content_items(mode='str', wrap_if_error=False):
            if isinstance(item, str):
                content.append({'type': 'text', 'text': item})
            elif item.media_type in supported_mime_types:
                content.append(await self._map_file(item, context))
            else:
                fallback_files.append(item)
        files: list[gi.ContentParam] = []
        if fallback_files:
            text, user_content = replace(part, content=fallback_files).model_response_str_and_user_content(
                wrap_if_error=False
            )
            content.append({'type': 'text', 'text': text})
            files = await self._map_user_prompt(UserPromptPart(user_content), context)
        result: _FunctionResultStep = {
            'type': 'function_result',
            'name': part.tool_name,
            'call_id': part.tool_call_id,
            'is_error': part.outcome in ERROR_OUTCOMES,
            'result': content,
        }
        return JSON_OBJECT.validate_python(result), files

    async def _map_model_response(
        self, message: ModelResponse, context: _MapMessagesContext
    ) -> list[gi.StepParam | dict[str, JsonValue]]:
        original = replay_steps(message) if self.accepts_stateless_artifacts(message) else None
        groups = self._replay_groups(message, original) if original is not None else {}
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
            signed = any(step.get('signature') or step.get('type') == 'function_call' for step in steps)
            steps.extend(await self._map_response_part(message, part, context, signed=signed))
        return steps

    def _replay_groups(
        self, message: ModelResponse, original: list[dict[str, JsonValue]]
    ) -> dict[int, list[ModelResponsePart]]:
        groups: dict[int, list[ModelResponsePart]] = {}
        for part in message.parts:
            index = step_index(part)
            if index is not None and 0 <= index < len(original) and self.accepts_stateless_artifacts(message, part):
                groups.setdefault(index, []).append(part)
        return groups

    async def _map_response_part(
        self, message: ModelResponse, part: ModelResponsePart, context: _MapMessagesContext, *, signed: bool
    ) -> list[gi.StepParam | dict[str, JsonValue]]:
        if isinstance(part, CompactionPart):
            return []
        steps: list[gi.StepParam | dict[str, JsonValue]] = []
        if part.provider_details and self.accepts_stateless_artifacts(message, part):
            signature = part.provider_details.get('thought_signature')
            if isinstance(signature, str):
                steps.append({'type': 'thought', 'signature': signature})
        match part:
            case TextPart():
                if part.content:
                    steps.append({'type': 'model_output', 'content': [{'type': 'text', 'text': part.content}]})
            case ThinkingPart():
                steps.extend(self._map_thinking_part(message, part))
            case ToolCallPart():
                steps.append(self._map_tool_call_part(part, signed=signed or bool(steps)))
            case FilePart():
                steps.append({'type': 'model_output', 'content': [await self._map_file(part.content, context)]})
            case NativeToolCallPart() | NativeToolReturnPart():
                steps.extend(self._map_native_part(message, part))
            case SpeechPart():
                raise _unconverted_speech_part_error()
            case _:
                assert_never(part)
        return steps

    def _map_tool_call_part(self, part: ToolCallPart, *, signed: bool) -> dict[str, JsonValue]:
        step: dict[str, JsonValue] = {
            'type': 'function_call',
            'id': part.tool_call_id,
            'name': part.tool_name,
            'arguments': JSON_OBJECT.validate_python(part.args_as_dict()),
        }
        if not signed:
            # Gemini rejects a turn whose first function call has no signature, which is the case for history from
            # another provider or transport. The SDK's step type has no `signature`, but the API accepts the documented
            # placeholder that `GoogleModel` also sends: https://ai.google.dev/gemini-api/docs/thought-signatures#faqs
            step['signature'] = _SKIP_THOUGHT_SIGNATURE
        return step

    def _map_native_part(
        self, message: ModelResponse, part: NativeToolCallPart | NativeToolReturnPart
    ) -> list[dict[str, JsonValue]]:
        raw = JSON_OBJECT.validate_python(part.provider_details or {})
        # GenerateContent history shares the canonical system but carries no Interactions step to replay.
        if 'type' not in raw or not self.accepts_stateless_artifacts(message, part):
            return []
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


_SKIP_THOUGHT_SIGNATURE = base64.b64encode(b'skip_thought_signature_validator').decode()


@dataclass
class _MapMessagesContext:
    video_resolution: MediaResolution | None = None
    instructions: list[str] = field(default_factory=list[str])
    steps: list[gi.StepParam | dict[str, JsonValue]] = field(default_factory=list[gi.StepParam | dict[str, JsonValue]])
    tool_files: list[gi.ContentParam] = field(default_factory=list[gi.ContentParam])


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
