from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal, cast, overload
from uuid import uuid4

from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, usage
from .._output import OutputObjectDefinition
from .._run_context import RunContext
from ..builtin_tools import CodeExecutionTool, ImageGenerationTool, UrlContextTool, WebSearchTool
from ..exceptions import ModelHTTPError, UserError
from ..messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    FilePart,
    FileUrl,
    FinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseStreamEvent,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    VideoUrl,
)
from ..profiles import ModelProfileSpec
from ..profiles.google import GoogleModelProfile
from ..providers import Provider, infer_provider
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    check_allow_model_requests,
    download_item,
    get_user_agent,
)

try:
    from google.genai import Client, errors
    from google.genai.types import (
        BlobDict,
        CodeExecutionResult,
        CodeExecutionResultDict,
        ContentDict,
        ContentUnionDict,
        CountTokensConfigDict,
        ExecutableCode,
        ExecutableCodeDict,
        FileDataDict,
        FinishReason as GoogleFinishReason,
        FunctionCallDict,
        FunctionCallingConfigDict,
        FunctionCallingConfigMode,
        FunctionDeclarationDict,
        GenerateContentConfigDict,
        GenerateContentResponse,
        GenerationConfigDict,
        GoogleSearchDict,
        GroundingMetadata,
        HttpOptionsDict,
        MediaResolution,
        Modality,
        Part,
        PartDict,
        SafetySettingDict,
        ThinkingConfigDict,
        ToolCodeExecutionDict,
        ToolConfigDict,
        ToolDict,
        ToolListUnionDict,
        UrlContextDict,
        VideoMetadataDict,
    )
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

LatestGoogleModelNames = Literal[
    'gemini-flash-latest',
    'gemini-flash-lite-latest',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-flash-preview-09-2025',
    'gemini-2.5-flash-image',
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash-lite-preview-09-2025',
    'gemini-2.5-pro',
    'gemini-3-pro-preview',
]
"""Latest Gemini models."""

GoogleModelName = str | LatestGoogleModelNames
"""Possible Gemini model names.

Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.
"""

_FINISH_REASON_MAP: dict[GoogleFinishReason, FinishReason | None] = {
    GoogleFinishReason.FINISH_REASON_UNSPECIFIED: None,
    GoogleFinishReason.STOP: 'stop',
    GoogleFinishReason.MAX_TOKENS: 'length',
    GoogleFinishReason.SAFETY: 'content_filter',
    GoogleFinishReason.RECITATION: 'content_filter',
    GoogleFinishReason.LANGUAGE: 'error',
    GoogleFinishReason.OTHER: None,
    GoogleFinishReason.BLOCKLIST: 'content_filter',
    GoogleFinishReason.PROHIBITED_CONTENT: 'content_filter',
    GoogleFinishReason.SPII: 'content_filter',
    GoogleFinishReason.MALFORMED_FUNCTION_CALL: 'error',
    GoogleFinishReason.IMAGE_SAFETY: 'content_filter',
    GoogleFinishReason.UNEXPECTED_TOOL_CALL: 'error',
    GoogleFinishReason.IMAGE_PROHIBITED_CONTENT: 'content_filter',
    GoogleFinishReason.NO_IMAGE: 'error',
}


class GoogleModelSettings(ModelSettings, total=False):
    """Settings used for a Gemini model request."""

    # ALL FIELDS MUST BE `gemini_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    google_safety_settings: list[SafetySettingDict]
    """The safety settings to use for the model.

    See <https://ai.google.dev/gemini-api/docs/safety-settings> for more information.
    """

    google_thinking_config: ThinkingConfigDict
    """The thinking configuration to use for the model.

    See <https://ai.google.dev/gemini-api/docs/thinking> for more information.
    """

    google_labels: dict[str, str]
    """User-defined metadata to break down billed charges. Only supported by the Vertex AI API.

    See the [Gemini API docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls) for use cases and limitations.
    """

    google_video_resolution: MediaResolution
    """The video resolution to use for the model.

    See <https://ai.google.dev/api/generate-content#MediaResolution> for more information.
    """

    google_cached_content: str
    """The name of the cached content to use for the model.

    See <https://ai.google.dev/gemini-api/docs/caching> for more information.
    """


@dataclass(init=False)
class GoogleModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: Client = field(repr=False)

    _model_name: GoogleModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)
    _url: str | None = field(repr=False)

    def __init__(
        self,
        model_name: GoogleModelName,
        *,
        provider: Literal['google-gla', 'google-vertex', 'gateway'] | Provider[Client] = 'google-gla',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Gemini model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'google-gla' or 'google-vertex' or an instance of `Provider[google.genai.AsyncClient]`.
                Defaults to 'google-gla'.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: The model settings to use. Defaults to None.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider('gateway/google-vertex' if provider == 'gateway' else provider)
        self._provider = provider
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> GoogleModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

    def prepare_request(
        self, model_settings: ModelSettings | None, model_request_parameters: ModelRequestParameters
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        supports_native_output_with_builtin_tools = GoogleModelProfile.from_profile(
            self.profile
        ).google_supports_native_output_with_builtin_tools
        if model_request_parameters.builtin_tools and model_request_parameters.output_tools:
            if model_request_parameters.output_mode == 'auto':
                output_mode = 'native' if supports_native_output_with_builtin_tools else 'prompted'
                model_request_parameters = replace(model_request_parameters, output_mode=output_mode)
            else:
                output_mode = 'NativeOutput' if supports_native_output_with_builtin_tools else 'PromptedOutput'
                raise UserError(
                    f'Google does not support output tools and built-in tools at the same time. Use `output_type={output_mode}(...)` instead.'
                )
        return super().prepare_request(model_settings, model_request_parameters)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, False, model_settings, model_request_parameters)
        return self._process_response(response)

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> usage.RequestUsage:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        contents, generation_config = await self._build_content_and_config(
            messages, model_settings, model_request_parameters
        )

        # Annoyingly, the type of `GenerateContentConfigDict.get` is "partially `Unknown`" because `response_schema` includes `typing._UnionGenericAlias`,
        # so without this we'd need `pyright: ignore[reportUnknownMemberType]` on every line and wouldn't get type checking anyway.
        generation_config = cast(dict[str, Any], generation_config)

        config = CountTokensConfigDict(
            http_options=generation_config.get('http_options'),
        )
        if self._provider.name != 'google-gla':
            # The fields are not supported by the Gemini API per https://github.com/googleapis/python-genai/blob/7e4ec284dc6e521949626f3ed54028163ef9121d/google/genai/models.py#L1195-L1214
            config.update(  # pragma: lax no cover
                system_instruction=generation_config.get('system_instruction'),
                tools=cast(list[ToolDict], generation_config.get('tools')),
                # Annoyingly, GenerationConfigDict has fewer fields than GenerateContentConfigDict, and no extra fields are allowed.
                generation_config=GenerationConfigDict(
                    temperature=generation_config.get('temperature'),
                    top_p=generation_config.get('top_p'),
                    max_output_tokens=generation_config.get('max_output_tokens'),
                    stop_sequences=generation_config.get('stop_sequences'),
                    presence_penalty=generation_config.get('presence_penalty'),
                    frequency_penalty=generation_config.get('frequency_penalty'),
                    seed=generation_config.get('seed'),
                    thinking_config=generation_config.get('thinking_config'),
                    media_resolution=generation_config.get('media_resolution'),
                    response_mime_type=generation_config.get('response_mime_type'),
                    response_json_schema=generation_config.get('response_json_schema'),
                ),
            )

        response = await self.client.aio.models.count_tokens(
            model=self._model_name,
            contents=contents,
            config=config,
        )
        if response.total_tokens is None:
            raise UnexpectedModelBehavior(  # pragma: no cover
                'Total tokens missing from Gemini response', str(response)
            )
        return usage.RequestUsage(
            input_tokens=response.total_tokens,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings, model_request_parameters = self.prepare_request(
            model_settings,
            model_request_parameters,
        )
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, True, model_settings, model_request_parameters)
        yield await self._process_streamed_response(response, model_request_parameters)  # type: ignore

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolDict] | None:
        tools: list[ToolDict] = [
            ToolDict(function_declarations=[_function_declaration_from_tool(t)])
            for t in model_request_parameters.tool_defs.values()
        ]

        if model_request_parameters.builtin_tools:
            if model_request_parameters.function_tools:
                raise UserError('Google does not support function tools and built-in tools at the same time.')

            for tool in model_request_parameters.builtin_tools:
                if isinstance(tool, WebSearchTool):
                    tools.append(ToolDict(google_search=GoogleSearchDict()))
                elif isinstance(tool, UrlContextTool):
                    tools.append(ToolDict(url_context=UrlContextDict()))
                elif isinstance(tool, CodeExecutionTool):
                    tools.append(ToolDict(code_execution=ToolCodeExecutionDict()))
                elif isinstance(tool, ImageGenerationTool):  # pragma: no branch
                    if not self.profile.supports_image_output:
                        raise UserError(
                            "`ImageGenerationTool` is not supported by this model. Use a model with 'image' in the name instead."
                        )
                else:  # pragma: no cover
                    raise UserError(
                        f'`{tool.__class__.__name__}` is not supported by `GoogleModel`. If it should be, please file an issue.'
                    )
        return tools or None

    def _get_tool_config(
        self, model_request_parameters: ModelRequestParameters, tools: list[ToolDict] | None
    ) -> ToolConfigDict | None:
        if not model_request_parameters.allow_text_output and tools:
            names: list[str] = []
            for tool in tools:
                for function_declaration in tool.get('function_declarations') or []:
                    if name := function_declaration.get('name'):  # pragma: no branch
                        names.append(name)
            return _tool_config(names)
        else:
            return None

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse: ...

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]: ...

    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse | Awaitable[AsyncIterator[GenerateContentResponse]]:
        contents, config = await self._build_content_and_config(messages, model_settings, model_request_parameters)
        func = self.client.aio.models.generate_content_stream if stream else self.client.aio.models.generate_content
        try:
            return await func(model=self._model_name, contents=contents, config=config)  # type: ignore
        except errors.APIError as e:
            if (status_code := e.code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code,
                    model_name=self._model_name,
                    body=cast(Any, e.details),  # pyright: ignore[reportUnknownMemberType]
                ) from e
            raise  # pragma: lax no cover

    async def _build_content_and_config(
        self,
        messages: list[ModelMessage],
        model_settings: GoogleModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[list[ContentUnionDict], GenerateContentConfigDict]:
        tools = self._get_tools(model_request_parameters)
        if tools and not self.profile.supports_tools:
            raise UserError('Tools are not supported by this model.')

        response_mime_type = None
        response_schema = None
        if model_request_parameters.output_mode == 'native':
            if model_request_parameters.function_tools:
                raise UserError(
                    'Google does not support `NativeOutput` and function tools at the same time. Use `output_type=ToolOutput(...)` instead.'
                )
            response_mime_type = 'application/json'
            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_schema = self._map_response_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and not tools:
            if not self.profile.supports_json_object_output:
                raise UserError('JSON output is not supported by this model.')
            response_mime_type = 'application/json'

        tool_config = self._get_tool_config(model_request_parameters, tools)
        system_instruction, contents = await self._map_messages(messages, model_request_parameters)

        modalities = [Modality.TEXT.value]
        if self.profile.supports_image_output:
            modalities.append(Modality.IMAGE.value)

        http_options: HttpOptionsDict = {
            'headers': {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}
        }
        if timeout := model_settings.get('timeout'):
            if isinstance(timeout, int | float):
                http_options['timeout'] = int(1000 * timeout)
            else:
                raise UserError('Google does not support setting ModelSettings.timeout to a httpx.Timeout')

        config = GenerateContentConfigDict(
            http_options=http_options,
            system_instruction=system_instruction,
            temperature=model_settings.get('temperature'),
            top_p=model_settings.get('top_p'),
            max_output_tokens=model_settings.get('max_tokens'),
            stop_sequences=model_settings.get('stop_sequences'),
            presence_penalty=model_settings.get('presence_penalty'),
            frequency_penalty=model_settings.get('frequency_penalty'),
            seed=model_settings.get('seed'),
            safety_settings=model_settings.get('google_safety_settings'),
            thinking_config=model_settings.get('google_thinking_config'),
            labels=model_settings.get('google_labels'),
            media_resolution=model_settings.get('google_video_resolution'),
            cached_content=model_settings.get('google_cached_content'),
            tools=cast(ToolListUnionDict, tools),
            tool_config=tool_config,
            response_mime_type=response_mime_type,
            response_json_schema=response_schema,
            response_modalities=modalities,
        )
        return contents, config

    def _process_response(self, response: GenerateContentResponse) -> ModelResponse:
        if not response.candidates:
            raise UnexpectedModelBehavior('Expected at least one candidate in Gemini response')  # pragma: no cover

        candidate = response.candidates[0]

        vendor_id = response.response_id
        vendor_details: dict[str, Any] | None = None
        finish_reason: FinishReason | None = None
        raw_finish_reason = candidate.finish_reason
        if raw_finish_reason:  # pragma: no branch
            vendor_details = {'finish_reason': raw_finish_reason.value}
            finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

        if candidate.content is None or candidate.content.parts is None:
            if finish_reason == 'content_filter' and raw_finish_reason:
                raise UnexpectedModelBehavior(
                    f'Content filter {raw_finish_reason.value!r} triggered', response.model_dump_json()
                )
            parts = []  # pragma: no cover
        else:
            parts = candidate.content.parts or []

        usage = _metadata_as_usage(response, provider=self._provider.name, provider_url=self._provider.base_url)
        return _process_response_from_parts(
            parts,
            candidate.grounding_metadata,
            response.model_version or self._model_name,
            self._provider.name,
            usage,
            vendor_id=vendor_id,
            vendor_details=vendor_details,
            finish_reason=finish_reason,
        )

    async def _process_streamed_response(
        self, response: AsyncIterator[GenerateContentResponse], model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        return GeminiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=first_chunk.model_version or self._model_name,
            _response=peekable_response,
            _timestamp=first_chunk.create_time or _utils.now_utc(),
            _provider_name=self._provider.name,
            _provider_url=self._provider.base_url,
        )

    async def _map_messages(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> tuple[ContentDict | None, list[ContentUnionDict]]:
        contents: list[ContentUnionDict] = []
        system_parts: list[PartDict] = []

        for m in messages:
            if isinstance(m, ModelRequest):
                message_parts: list[PartDict] = []

                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        message_parts.extend(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        message_parts.append(
                            {
                                'function_response': {
                                    'name': part.tool_name,
                                    'response': part.model_response_object(),
                                    'id': part.tool_call_id,
                                }
                            }
                        )
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            message_parts.append({'text': part.model_response()})  # pragma: no cover
                        else:
                            message_parts.append(
                                {
                                    'function_response': {
                                        'name': part.tool_name,
                                        'response': {'call_error': part.model_response()},
                                        'id': part.tool_call_id,
                                    }
                                }
                            )
                    else:
                        assert_never(part)

                # Google GenAI requires at least one part in the message.
                if not message_parts:
                    message_parts = [{'text': ''}]
                contents.append({'role': 'user', 'parts': message_parts})
            elif isinstance(m, ModelResponse):
                contents.append(_content_model_response(m, self.system))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages, model_request_parameters):
            system_parts.insert(0, {'text': instructions})
        system_instruction = ContentDict(role='user', parts=system_parts) if system_parts else None
        return system_instruction, contents

    async def _map_user_prompt(self, part: UserPromptPart) -> list[PartDict]:
        if isinstance(part.content, str):
            return [{'text': part.content}]
        else:
            content: list[PartDict] = []
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    inline_data_dict: BlobDict = {'data': item.data, 'mime_type': item.media_type}
                    part_dict: PartDict = {'inline_data': inline_data_dict}
                    if item.vendor_metadata:
                        part_dict['video_metadata'] = cast(VideoMetadataDict, item.vendor_metadata)
                    content.append(part_dict)
                elif isinstance(item, VideoUrl) and item.is_youtube:
                    file_data_dict: FileDataDict = {'file_uri': item.url, 'mime_type': item.media_type}
                    part_dict: PartDict = {'file_data': file_data_dict}
                    if item.vendor_metadata:  # pragma: no branch
                        part_dict['video_metadata'] = cast(VideoMetadataDict, item.vendor_metadata)
                    content.append(part_dict)
                elif isinstance(item, FileUrl):
                    if item.force_download or (
                        # google-gla does not support passing file urls directly, except for youtube videos
                        # (see above) and files uploaded to the file API (which cannot be downloaded anyway)
                        self.system == 'google-gla'
                        and not item.url.startswith(r'https://generativelanguage.googleapis.com/v1beta/files')
                    ):
                        downloaded_item = await download_item(item, data_format='bytes')
                        inline_data: BlobDict = {
                            'data': downloaded_item['data'],
                            'mime_type': downloaded_item['data_type'],
                        }
                        content.append({'inline_data': inline_data})
                    else:
                        file_data_dict: FileDataDict = {'file_uri': item.url, 'mime_type': item.media_type}
                        content.append({'file_data': file_data_dict})  # pragma: lax no cover
                elif isinstance(item, CachePoint):
                    # Google Gemini doesn't support prompt caching via CachePoint
                    pass
                else:
                    assert_never(item)
        return content

    def _map_response_schema(self, o: OutputObjectDefinition) -> dict[str, Any]:
        response_schema = o.json_schema.copy()
        if o.name:
            response_schema['title'] = o.name
        if o.description:
            response_schema['description'] = o.description

        return response_schema


@dataclass
class GeminiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the Gemini model."""

    _model_name: GoogleModelName
    _response: AsyncIterator[GenerateContentResponse]
    _timestamp: datetime
    _provider_name: str
    _provider_url: str

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        code_execution_tool_call_id: str | None = None
        async for chunk in self._response:
            self._usage = _metadata_as_usage(chunk, self._provider_name, self._provider_url)

            if not chunk.candidates:
                continue  # pragma: no cover

            candidate = chunk.candidates[0]

            if chunk.response_id:  # pragma: no branch
                self.provider_response_id = chunk.response_id

            raw_finish_reason = candidate.finish_reason
            if raw_finish_reason:
                self.provider_details = {'finish_reason': raw_finish_reason.value}
                self.finish_reason = _FINISH_REASON_MAP.get(raw_finish_reason)

            # Google streams the grounding metadata (including the web search queries and results)
            # _after_ the text that was generated using it, so it would show up out of order in the stream,
            # and cause issues with the logic that doesn't consider text ahead of built-in tool calls as output.
            # If that gets fixed (or we have a workaround), we can uncomment this:
            # web_search_call, web_search_return = _map_grounding_metadata(
            #     candidate.grounding_metadata, self.provider_name
            # )
            # if web_search_call and web_search_return:
            #     yield self._parts_manager.handle_part(vendor_part_id=uuid4(), part=web_search_call)
            #     yield self._parts_manager.handle_part(
            #         vendor_part_id=uuid4(), part=web_search_return
            #     )

            if candidate.content is None or candidate.content.parts is None:
                if self.finish_reason == 'content_filter' and raw_finish_reason:  # pragma: no cover
                    raise UnexpectedModelBehavior(
                        f'Content filter {raw_finish_reason.value!r} triggered', chunk.model_dump_json()
                    )
                else:  # pragma: no cover
                    continue

            parts = candidate.content.parts
            if not parts:
                continue  # pragma: no cover

            for part in parts:
                if part.thought_signature:
                    # Per https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thought-signatures:
                    # - Always send the thought_signature back to the model inside its original Part.
                    # - Don't merge a Part containing a signature with one that does not. This breaks the positional context of the thought.
                    # - Don't combine two Parts that both contain signatures, as the signature strings cannot be merged.

                    signature = base64.b64encode(part.thought_signature).decode('utf-8')
                    # Attach signature to most recent thinking part, if there was one
                    yield self._parts_manager.handle_thinking_delta(
                        vendor_part_id=None,
                        signature=signature,
                        provider_name=self.provider_name,
                    )

                if part.text is not None:
                    if len(part.text) > 0:
                        if part.thought:
                            yield self._parts_manager.handle_thinking_delta(vendor_part_id=None, content=part.text)
                        else:
                            maybe_event = self._parts_manager.handle_text_delta(vendor_part_id=None, content=part.text)
                            if maybe_event is not None:  # pragma: no branch
                                yield maybe_event
                elif part.function_call:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=uuid4(),
                        tool_name=part.function_call.name,
                        args=part.function_call.args,
                        tool_call_id=part.function_call.id,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                elif part.inline_data is not None:
                    data = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    assert data and mime_type, 'Inline data must have data and mime type'
                    content = BinaryContent(data=data, media_type=mime_type)
                    yield self._parts_manager.handle_part(
                        vendor_part_id=uuid4(),
                        part=FilePart(content=BinaryContent.narrow_type(content)),
                    )
                elif part.executable_code is not None:
                    code_execution_tool_call_id = _utils.generate_tool_call_id()
                    yield self._parts_manager.handle_part(
                        vendor_part_id=uuid4(),
                        part=_map_executable_code(
                            part.executable_code, self.provider_name, code_execution_tool_call_id
                        ),
                    )
                elif part.code_execution_result is not None:
                    assert code_execution_tool_call_id is not None
                    yield self._parts_manager.handle_part(
                        vendor_part_id=uuid4(),
                        part=_map_code_execution_result(
                            part.code_execution_result, self.provider_name, code_execution_tool_call_id
                        ),
                    )
                else:
                    assert part.function_response is not None, f'Unexpected part: {part}'  # pragma: no cover

    @property
    def model_name(self) -> GoogleModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return self._provider_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _content_model_response(m: ModelResponse, provider_name: str) -> ContentDict:  # noqa: C901
    parts: list[PartDict] = []
    thought_signature: bytes | None = None
    function_call_requires_signature: bool = True
    for item in m.parts:
        part: PartDict = {}
        if thought_signature:
            part['thought_signature'] = thought_signature
            thought_signature = None

        if isinstance(item, ToolCallPart):
            function_call = FunctionCallDict(name=item.tool_name, args=item.args_as_dict(), id=item.tool_call_id)
            part['function_call'] = function_call
            if function_call_requires_signature and not part.get('thought_signature'):
                # Per https://ai.google.dev/gemini-api/docs/gemini-3?thinking=high#migrating_from_other_models:
                # > If you are transferring a conversation trace from another model (e.g., Gemini 2.5) or injecting
                # > a custom function call that was not generated by Gemini 3, you will not have a valid signature.
                # > To bypass strict validation in these specific scenarios, populate the field with this specific
                # > dummy string: "thoughtSignature": "context_engineering_is_the_way_to_go"
                part['thought_signature'] = b'context_engineering_is_the_way_to_go'
            # Only the first function call requires a signature
            function_call_requires_signature = False
        elif isinstance(item, TextPart):
            part['text'] = item.content
        elif isinstance(item, ThinkingPart):
            if item.provider_name == provider_name and item.signature:
                # The thought signature is to be included on the _next_ part, not the thought part itself
                thought_signature = base64.b64decode(item.signature)

            if item.content:
                part['text'] = item.content
                part['thought'] = True
        elif isinstance(item, BuiltinToolCallPart):
            if item.provider_name == provider_name:
                if item.tool_name == CodeExecutionTool.kind:
                    part['executable_code'] = cast(ExecutableCodeDict, item.args_as_dict())
                elif item.tool_name == WebSearchTool.kind:
                    # Web search calls are not sent back
                    pass
        elif isinstance(item, BuiltinToolReturnPart):
            if item.provider_name == provider_name:
                if item.tool_name == CodeExecutionTool.kind and isinstance(item.content, dict):
                    part['code_execution_result'] = cast(CodeExecutionResultDict, item.content)  # pyright: ignore[reportUnknownMemberType]
                elif item.tool_name == WebSearchTool.kind:
                    # Web search results are not sent back
                    pass
        elif isinstance(item, FilePart):
            content = item.content
            inline_data_dict: BlobDict = {'data': content.data, 'mime_type': content.media_type}
            part['inline_data'] = inline_data_dict
        else:
            assert_never(item)

        if part:
            parts.append(part)
    return ContentDict(role='model', parts=parts)


def _process_response_from_parts(
    parts: list[Part],
    grounding_metadata: GroundingMetadata | None,
    model_name: GoogleModelName,
    provider_name: str,
    usage: usage.RequestUsage,
    vendor_id: str | None,
    vendor_details: dict[str, Any] | None = None,
    finish_reason: FinishReason | None = None,
) -> ModelResponse:
    items: list[ModelResponsePart] = []

    # We don't currently turn `candidate.url_context_metadata` into BuiltinToolCallPart and BuiltinToolReturnPart for UrlContextTool.
    # Please file an issue if you need this.

    web_search_call, web_search_return = _map_grounding_metadata(grounding_metadata, provider_name)
    if web_search_call and web_search_return:
        items.append(web_search_call)
        items.append(web_search_return)

    item: ModelResponsePart | None = None
    code_execution_tool_call_id: str | None = None
    for part in parts:
        if part.thought_signature:
            signature = base64.b64encode(part.thought_signature).decode('utf-8')
            if not isinstance(item, ThinkingPart):
                item = ThinkingPart(content='')
                items.append(item)
            item.signature = signature
            item.provider_name = provider_name

        if part.executable_code is not None:
            code_execution_tool_call_id = _utils.generate_tool_call_id()
            item = _map_executable_code(part.executable_code, provider_name, code_execution_tool_call_id)
        elif part.code_execution_result is not None:
            assert code_execution_tool_call_id is not None
            item = _map_code_execution_result(part.code_execution_result, provider_name, code_execution_tool_call_id)
        elif part.text is not None:
            # Google sometimes sends empty text parts, we don't want to add them to the response
            if len(part.text) == 0:
                continue
            if part.thought:
                item = ThinkingPart(content=part.text)
            else:
                item = TextPart(content=part.text)
        elif part.function_call:
            assert part.function_call.name is not None
            item = ToolCallPart(tool_name=part.function_call.name, args=part.function_call.args)
            if part.function_call.id is not None:
                item.tool_call_id = part.function_call.id  # pragma: no cover
        elif inline_data := part.inline_data:
            data = inline_data.data
            mime_type = inline_data.mime_type
            assert data and mime_type, 'Inline data must have data and mime type'
            content = BinaryContent(data=data, media_type=mime_type)
            item = FilePart(content=BinaryContent.narrow_type(content))
        else:  # pragma: no cover
            raise UnexpectedModelBehavior(f'Unsupported response from Gemini: {part!r}')

        items.append(item)
    return ModelResponse(
        parts=items,
        model_name=model_name,
        usage=usage,
        provider_response_id=vendor_id,
        provider_details=vendor_details,
        provider_name=provider_name,
        finish_reason=finish_reason,
    )


def _function_declaration_from_tool(tool: ToolDefinition) -> FunctionDeclarationDict:
    json_schema = tool.parameters_json_schema
    f = FunctionDeclarationDict(
        name=tool.name,
        description=tool.description or '',
        parameters_json_schema=json_schema,
    )
    return f


def _tool_config(function_names: list[str]) -> ToolConfigDict:
    mode = FunctionCallingConfigMode.ANY
    function_calling_config = FunctionCallingConfigDict(mode=mode, allowed_function_names=function_names)
    return ToolConfigDict(function_calling_config=function_calling_config)


def _metadata_as_usage(response: GenerateContentResponse, provider: str, provider_url: str) -> usage.RequestUsage:
    metadata = response.usage_metadata
    if metadata is None:
        return usage.RequestUsage()
    details: dict[str, int] = {}
    if cached_content_token_count := metadata.cached_content_token_count:
        details['cached_content_tokens'] = cached_content_token_count

    if thoughts_token_count := (metadata.thoughts_token_count or 0):
        details['thoughts_tokens'] = thoughts_token_count

    if tool_use_prompt_token_count := metadata.tool_use_prompt_token_count:
        details['tool_use_prompt_tokens'] = tool_use_prompt_token_count

    for prefix, metadata_details in [
        ('prompt', metadata.prompt_tokens_details),
        ('cache', metadata.cache_tokens_details),
        ('candidates', metadata.candidates_tokens_details),
        ('tool_use_prompt', metadata.tool_use_prompt_tokens_details),
    ]:
        assert getattr(metadata, f'{prefix}_tokens_details') is metadata_details
        if not metadata_details:
            continue
        for detail in metadata_details:
            if not detail.modality or not detail.token_count:
                continue
            details[f'{detail.modality.lower()}_{prefix}_tokens'] = detail.token_count

    return usage.RequestUsage.extract(
        response.model_dump(include={'model_version', 'usage_metadata'}, by_alias=True),
        provider=provider,
        provider_url=provider_url,
        provider_fallback='google',
        details=details,
    )


def _map_executable_code(executable_code: ExecutableCode, provider_name: str, tool_call_id: str) -> BuiltinToolCallPart:
    return BuiltinToolCallPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        args=executable_code.model_dump(mode='json'),
        tool_call_id=tool_call_id,
    )


def _map_code_execution_result(
    code_execution_result: CodeExecutionResult, provider_name: str, tool_call_id: str
) -> BuiltinToolReturnPart:
    return BuiltinToolReturnPart(
        provider_name=provider_name,
        tool_name=CodeExecutionTool.kind,
        content=code_execution_result.model_dump(mode='json'),
        tool_call_id=tool_call_id,
    )


def _map_grounding_metadata(
    grounding_metadata: GroundingMetadata | None, provider_name: str
) -> tuple[BuiltinToolCallPart, BuiltinToolReturnPart] | tuple[None, None]:
    if grounding_metadata and (web_search_queries := grounding_metadata.web_search_queries):
        tool_call_id = _utils.generate_tool_call_id()
        return (
            BuiltinToolCallPart(
                provider_name=provider_name,
                tool_name=WebSearchTool.kind,
                tool_call_id=tool_call_id,
                args={'queries': web_search_queries},
            ),
            BuiltinToolReturnPart(
                provider_name=provider_name,
                tool_name=WebSearchTool.kind,
                tool_call_id=tool_call_id,
                content=[chunk.web.model_dump(mode='json') for chunk in grounding_chunks if chunk.web]
                if (grounding_chunks := grounding_metadata.grounding_chunks)
                else None,
            ),
        )
    else:
        return None, None
