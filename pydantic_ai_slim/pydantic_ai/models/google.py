from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Union, cast, overload
from uuid import uuid4

from typing_extensions import assert_never

from .. import UnexpectedModelBehavior, _utils, usage
from .._output import OutputObjectDefinition
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    FileUrl,
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
from ..providers import Provider
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
    from google import genai
    from google.genai.types import (
        ContentDict,
        ContentUnionDict,
        FunctionCallDict,
        FunctionCallingConfigDict,
        FunctionCallingConfigMode,
        FunctionDeclarationDict,
        GenerateContentConfigDict,
        GenerateContentResponse,
        HttpOptionsDict,
        MediaResolution,
        Part,
        PartDict,
        SafetySettingDict,
        ThinkingConfigDict,
        ToolConfigDict,
        ToolDict,
        ToolListUnionDict,
    )

    from ..providers.google import GoogleProvider
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error

LatestGoogleModelNames = Literal[
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite-preview-06-17',
    'gemini-2.5-pro',
]
"""Latest Gemini models."""

GoogleModelName = Union[str, LatestGoogleModelNames]
"""Possible Gemini model names.

Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.
"""


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


@dataclass(init=False)
class GoogleModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: genai.Client = field(repr=False)

    _model_name: GoogleModelName = field(repr=False)
    _provider: Provider[genai.Client] = field(repr=False)
    _url: str | None = field(repr=False)
    _system: str = field(default='google', repr=False)

    def __init__(
        self,
        model_name: GoogleModelName,
        *,
        provider: Literal['google-gla', 'google-vertex'] | Provider[genai.Client] = 'google-gla',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Gemini model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`.
                If not provided, a new provider will be created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: The model settings to use. Defaults to None.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = GoogleProvider(vertexai=provider == 'google-vertex')

        self._provider = provider
        self._system = provider.name
        self.client = provider.client

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        check_allow_model_requests()
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, False, model_settings, model_request_parameters)
        return self._process_response(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings = cast(GoogleModelSettings, model_settings or {})
        response = await self._generate_content(messages, True, model_settings, model_request_parameters)
        yield await self._process_streamed_response(response)  # type: ignore

    @property
    def model_name(self) -> GoogleModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolDict] | None:
        tools: list[ToolDict] = [
            ToolDict(function_declarations=[_function_declaration_from_tool(t)])
            for t in model_request_parameters.function_tools
        ]
        if model_request_parameters.output_tools:
            tools += [
                ToolDict(function_declarations=[_function_declaration_from_tool(t)])
                for t in model_request_parameters.output_tools
            ]
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
        tools = self._get_tools(model_request_parameters)

        response_mime_type = None
        response_schema = None
        if model_request_parameters.output_mode == 'native':
            if tools:
                raise UserError('Gemini does not support structured output and tools at the same time.')

            response_mime_type = 'application/json'

            output_object = model_request_parameters.output_object
            assert output_object is not None
            response_schema = self._map_response_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and not tools:
            response_mime_type = 'application/json'

        tool_config = self._get_tool_config(model_request_parameters, tools)
        system_instruction, contents = await self._map_messages(messages)

        http_options: HttpOptionsDict = {
            'headers': {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}
        }
        if timeout := model_settings.get('timeout'):
            if isinstance(timeout, (int, float)):
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
            safety_settings=model_settings.get('google_safety_settings'),
            thinking_config=model_settings.get('google_thinking_config'),
            labels=model_settings.get('google_labels'),
            media_resolution=model_settings.get('google_video_resolution'),
            tools=cast(ToolListUnionDict, tools),
            tool_config=tool_config,
            response_mime_type=response_mime_type,
            response_schema=response_schema,
        )

        func = self.client.aio.models.generate_content_stream if stream else self.client.aio.models.generate_content
        return await func(model=self._model_name, contents=contents, config=config)  # type: ignore

    def _process_response(self, response: GenerateContentResponse) -> ModelResponse:
        if not response.candidates or len(response.candidates) != 1:
            raise UnexpectedModelBehavior('Expected exactly one candidate in Gemini response')  # pragma: no cover
        if response.candidates[0].content is None or response.candidates[0].content.parts is None:
            if response.candidates[0].finish_reason == 'SAFETY':
                raise UnexpectedModelBehavior('Safety settings triggered', str(response))
            else:
                raise UnexpectedModelBehavior(
                    'Content field missing from Gemini response', str(response)
                )  # pragma: no cover
        parts = response.candidates[0].content.parts or []
        vendor_id = response.response_id or None
        vendor_details: dict[str, Any] | None = None
        finish_reason = response.candidates[0].finish_reason
        if finish_reason:  # pragma: no branch
            vendor_details = {'finish_reason': finish_reason.value}
        usage = _metadata_as_usage(response)
        usage.requests = 1
        return _process_response_from_parts(
            parts, response.model_version or self._model_name, usage, vendor_id=vendor_id, vendor_details=vendor_details
        )

    async def _process_streamed_response(self, response: AsyncIterator[GenerateContentResponse]) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')  # pragma: no cover

        return GeminiStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=first_chunk.create_time or _utils.now_utc(),
        )

    async def _map_messages(self, messages: list[ModelMessage]) -> tuple[ContentDict | None, list[ContentUnionDict]]:
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
                contents.append(_content_model_response(m))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages):
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
                    # NOTE: The type from Google GenAI is incorrect, it should be `str`, not `bytes`.
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    inline_data_dict = {'inline_data': {'data': base64_encoded, 'mime_type': item.media_type}}
                    if item.vendor_metadata:
                        inline_data_dict['video_metadata'] = item.vendor_metadata
                    content.append(inline_data_dict)  # type: ignore
                elif isinstance(item, VideoUrl) and item.is_youtube:
                    file_data_dict = {'file_data': {'file_uri': item.url, 'mime_type': item.media_type}}
                    if item.vendor_metadata:
                        file_data_dict['video_metadata'] = item.vendor_metadata
                    content.append(file_data_dict)  # type: ignore
                elif isinstance(item, FileUrl):
                    if item.force_download or (
                        # google-gla does not support passing file urls directly, except for youtube videos
                        # (see above) and files uploaded to the file API (which cannot be downloaded anyway)
                        self.system == 'google-gla'
                        and not item.url.startswith(r'https://generativelanguage.googleapis.com/v1beta/files')
                    ):
                        downloaded_item = await download_item(item, data_format='base64')
                        inline_data = {'data': downloaded_item['data'], 'mime_type': downloaded_item['data_type']}
                        content.append({'inline_data': inline_data})  # type: ignore
                    else:
                        content.append({'file_data': {'file_uri': item.url, 'mime_type': item.media_type}})
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

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage = _metadata_as_usage(chunk)

            assert chunk.candidates is not None
            candidate = chunk.candidates[0]
            if candidate.content is None:
                raise UnexpectedModelBehavior('Streamed response has no content field')  # pragma: no cover
            assert candidate.content.parts is not None
            for part in candidate.content.parts:
                if part.text is not None:
                    if part.thought:
                        yield self._parts_manager.handle_thinking_delta(vendor_part_id='thinking', content=part.text)
                    else:
                        maybe_event = self._parts_manager.handle_text_delta(vendor_part_id='content', content=part.text)
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
                else:
                    assert part.function_response is not None, f'Unexpected part: {part}'  # pragma: no cover

    @property
    def model_name(self) -> GoogleModelName:
        """Get the model name of the response."""
        return self._model_name

    @property
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        return self._timestamp


def _content_model_response(m: ModelResponse) -> ContentDict:
    parts: list[PartDict] = []
    for item in m.parts:
        if isinstance(item, ToolCallPart):
            function_call = FunctionCallDict(name=item.tool_name, args=item.args_as_dict(), id=item.tool_call_id)
            parts.append({'function_call': function_call})
        elif isinstance(item, TextPart):
            parts.append({'text': item.content})
        elif isinstance(item, ThinkingPart):  # pragma: no cover
            # NOTE: We don't send ThinkingPart to the providers yet. If you are unsatisfied with this,
            # please open an issue. The below code is the code to send thinking to the provider.
            # parts.append({'text': item.content, 'thought': True})
            pass
        else:
            assert_never(item)
    return ContentDict(role='model', parts=parts)


def _process_response_from_parts(
    parts: list[Part],
    model_name: GoogleModelName,
    usage: usage.Usage,
    vendor_id: str | None,
    vendor_details: dict[str, Any] | None = None,
) -> ModelResponse:
    items: list[ModelResponsePart] = []
    for part in parts:
        if part.text is not None:
            if part.thought:
                items.append(ThinkingPart(content=part.text))
            else:
                items.append(TextPart(content=part.text))
        elif part.function_call:
            assert part.function_call.name is not None
            tool_call_part = ToolCallPart(tool_name=part.function_call.name, args=part.function_call.args)
            if part.function_call.id is not None:
                tool_call_part.tool_call_id = part.function_call.id  # pragma: no cover
            items.append(tool_call_part)
        elif part.function_response:  # pragma: no cover
            raise UnexpectedModelBehavior(
                f'Unsupported response from Gemini, expected all parts to be function calls or text, got: {part!r}'
            )
    return ModelResponse(
        parts=items, model_name=model_name, usage=usage, vendor_id=vendor_id, vendor_details=vendor_details
    )


def _function_declaration_from_tool(tool: ToolDefinition) -> FunctionDeclarationDict:
    json_schema = tool.parameters_json_schema
    f = FunctionDeclarationDict(
        name=tool.name,
        description=tool.description or '',
        parameters=json_schema,  # type: ignore
    )
    return f


def _tool_config(function_names: list[str]) -> ToolConfigDict:
    mode = FunctionCallingConfigMode.ANY
    function_calling_config = FunctionCallingConfigDict(mode=mode, allowed_function_names=function_names)
    return ToolConfigDict(function_calling_config=function_calling_config)


def _metadata_as_usage(response: GenerateContentResponse) -> usage.Usage:
    metadata = response.usage_metadata
    if metadata is None:
        return usage.Usage()  # pragma: no cover
    metadata = metadata.model_dump(exclude_defaults=True)

    details: dict[str, int] = {}
    if cached_content_token_count := metadata.get('cached_content_token_count'):
        details['cached_content_tokens'] = cached_content_token_count  # pragma: no cover

    if thoughts_token_count := metadata.get('thoughts_token_count'):
        details['thoughts_tokens'] = thoughts_token_count

    if tool_use_prompt_token_count := metadata.get('tool_use_prompt_token_count'):
        details['tool_use_prompt_tokens'] = tool_use_prompt_token_count  # pragma: no cover

    for key, metadata_details in metadata.items():
        if key.endswith('_details') and metadata_details:
            suffix = key.removesuffix('_details')
            for detail in metadata_details:
                details[f'{detail["modality"].lower()}_{suffix}'] = detail['token_count']

    return usage.Usage(
        request_tokens=metadata.get('prompt_token_count', 0),
        response_tokens=metadata.get('candidates_token_count', 0),
        total_tokens=metadata.get('total_token_count', 0),
        details=details,
    )
