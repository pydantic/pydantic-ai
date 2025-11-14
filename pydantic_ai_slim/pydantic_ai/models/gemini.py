from __future__ import annotations as _annotations

import base64
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol, cast
from uuid import uuid4

import httpx
import pydantic
from httpx import USE_CLIENT_DEFAULT, Response as HTTPResponse
from typing_extensions import NotRequired, TypedDict, assert_never, deprecated

from .. import ModelHTTPError, UnexpectedModelBehavior, _utils, usage
from .._output import OutputObjectDefinition
from .._run_context import RunContext
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    CachePoint,
    FilePart,
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
from . import Model, ModelRequestParameters, StreamedResponse, check_allow_model_requests, download_item, get_user_agent

LatestGeminiModelNames = Literal[
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite',
    'gemini-2.5-flash',
    'gemini-2.5-flash-preview-09-2025',
    'gemini-2.5-flash-lite',
    'gemini-2.5-flash-lite-preview-09-2025',
    'gemini-flash-latest',
    'gemini-flash-lite-latest',
    'gemini-2.5-pro',
]
"""Latest Gemini models."""

GeminiModelName = str | LatestGeminiModelNames
"""Possible Gemini model names.

Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.
"""


class GeminiModelSettings(ModelSettings, total=False):
    """Settings used for a Gemini model request."""

    # ALL FIELDS MUST BE `gemini_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    gemini_safety_settings: list[GeminiSafetySettings]
    """Safety settings options for Gemini model request."""

    gemini_thinking_config: ThinkingConfig
    """Thinking is "on" by default in both the API and AI Studio.

    Being on by default doesn't mean the model will send back thoughts. For that, you would need to set `include_thoughts`
    to `True`, but since end of January 2025, `thoughts` are not returned anymore, and are only displayed in the Google
    AI Studio. See https://discuss.ai.google.dev/t/thoughts-are-missing-cot-not-included-anymore/63653 for more details.

    If you want to avoid the model spending any tokens on thinking, you can set `thinking_budget` to `0`.

    See more about it on <https://ai.google.dev/gemini-api/docs/thinking>.
    """

    gemini_labels: dict[str, str]
    """User-defined metadata to break down billed charges. Only supported by the Vertex AI provider.

    See the [Gemini API docs](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/add-labels-to-api-calls) for use cases and limitations.
    """


@deprecated('Use `GoogleModel` instead. See <https://ai.pydantic.dev/models/google/> for more details.')
@dataclass(init=False)
class GeminiModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: httpx.AsyncClient = field(repr=False)

    _model_name: GeminiModelName = field(repr=False)
    _provider: Provider[httpx.AsyncClient] = field(repr=False)
    _auth: AuthProtocol | None = field(repr=False)
    _url: str | None = field(repr=False)

    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        provider: Literal['google-gla', 'google-vertex'] | Provider[httpx.AsyncClient] = 'google-gla',
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
            settings: Default model settings for this model instance.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            if provider == 'google-gla':
                from pydantic_ai.providers.google_gla import GoogleGLAProvider  # type: ignore[reportDeprecated]

                provider = GoogleGLAProvider()  # type: ignore[reportDeprecated]
            else:
                from pydantic_ai.providers.google_vertex import GoogleVertexProvider  # type: ignore[reportDeprecated]

                provider = GoogleVertexProvider()  # type: ignore[reportDeprecated]
        self._provider = provider
        self.client = provider.client
        self._url = str(self.client.base_url)

        super().__init__(settings=settings, profile=profile or provider.model_profile)

    @property
    def base_url(self) -> str:
        assert self._url is not None, 'URL not initialized'  # pragma: no cover
        return self._url  # pragma: no cover

    @property
    def model_name(self) -> GeminiModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The model provider."""
        return self._provider.name

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
        async with self._make_request(
            messages, False, cast(GeminiModelSettings, model_settings or {}), model_request_parameters
        ) as http_response:
            data = await http_response.aread()
            response = _gemini_response_ta.validate_json(data)
        return self._process_response(response)

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
        async with self._make_request(
            messages, True, cast(GeminiModelSettings, model_settings or {}), model_request_parameters
        ) as http_response:
            yield await self._process_streamed_response(http_response, model_request_parameters)

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> _GeminiTools | None:
        tools = [_function_from_abstract_tool(t) for t in model_request_parameters.tool_defs.values()]
        return _GeminiTools(function_declarations=tools) if tools else None

    def _get_tool_config(
        self, model_request_parameters: ModelRequestParameters, tools: _GeminiTools | None
    ) -> _GeminiToolConfig | None:
        if not model_request_parameters.allow_text_output and tools:
            return _tool_config([t['name'] for t in tools['function_declarations']])
        else:
            return None

    @asynccontextmanager
    async def _make_request(
        self,
        messages: list[ModelMessage],
        streamed: bool,
        model_settings: GeminiModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[HTTPResponse]:
        tools = self._get_tools(model_request_parameters)
        tool_config = self._get_tool_config(model_request_parameters, tools)
        sys_prompt_parts, contents = await self._message_to_gemini_content(messages, model_request_parameters)

        request_data = _GeminiRequest(contents=contents)
        if sys_prompt_parts:
            request_data['systemInstruction'] = _GeminiTextContent(role='user', parts=sys_prompt_parts)
        if tools is not None:
            request_data['tools'] = tools
        if tool_config is not None:
            request_data['toolConfig'] = tool_config

        generation_config = _settings_to_generation_config(model_settings)
        if model_request_parameters.output_mode == 'native':
            if tools:
                raise UserError(
                    'Gemini does not support `NativeOutput` and tools at the same time. Use `output_type=ToolOutput(...)` instead.'
                )

            generation_config['response_mime_type'] = 'application/json'

            output_object = model_request_parameters.output_object
            assert output_object is not None
            generation_config['response_schema'] = self._map_response_schema(output_object)
        elif model_request_parameters.output_mode == 'prompted' and not tools:
            generation_config['response_mime_type'] = 'application/json'

        if generation_config:
            request_data['generationConfig'] = generation_config

        if gemini_safety_settings := model_settings.get('gemini_safety_settings'):
            request_data['safetySettings'] = gemini_safety_settings

        if gemini_labels := model_settings.get('gemini_labels'):
            if self._provider.name == 'google-vertex':
                request_data['labels'] = gemini_labels  # pragma: lax no cover

        headers = {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}
        url = f'/{self._model_name}:{"streamGenerateContent" if streamed else "generateContent"}'

        request_json = _gemini_request_ta.dump_json(request_data, by_alias=True)
        async with self.client.stream(
            'POST',
            url,
            content=request_json,
            headers=headers,
            timeout=model_settings.get('timeout', USE_CLIENT_DEFAULT),
        ) as r:
            if (status_code := r.status_code) != 200:
                await r.aread()
                if status_code >= 400:
                    raise ModelHTTPError(status_code=status_code, model_name=self.model_name, body=r.text)
                raise UnexpectedModelBehavior(  # pragma: no cover
                    f'Unexpected response from gemini {status_code}', r.text
                )
            yield r

    def _process_response(self, response: _GeminiResponse) -> ModelResponse:
        vendor_details: dict[str, Any] | None = None

        if len(response['candidates']) != 1:
            raise UnexpectedModelBehavior('Expected exactly one candidate in Gemini response')  # pragma: no cover
        if 'content' not in response['candidates'][0]:
            if response['candidates'][0].get('finish_reason') == 'SAFETY':
                raise UnexpectedModelBehavior('Safety settings triggered', str(response))
            else:
                raise UnexpectedModelBehavior(  # pragma: no cover
                    'Content field missing from Gemini response', str(response)
                )
        parts = response['candidates'][0]['content']['parts']
        vendor_id = response.get('vendor_id', None)
        finish_reason = response['candidates'][0].get('finish_reason')
        if finish_reason:
            vendor_details = {'finish_reason': finish_reason}
        usage = _metadata_as_usage(response)
        return _process_response_from_parts(
            parts,
            response.get('model_version', self._model_name),
            usage,
            vendor_id=vendor_id,
            vendor_details=vendor_details,
        )

    async def _process_streamed_response(
        self, http_response: HTTPResponse, model_request_parameters: ModelRequestParameters
    ) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        aiter_bytes = http_response.aiter_bytes()
        start_response: _GeminiResponse | None = None
        content = bytearray()

        async for chunk in aiter_bytes:
            content.extend(chunk)
            responses = _gemini_streamed_response_ta.validate_json(
                _ensure_decodeable(content),
                experimental_allow_partial='trailing-strings',
            )
            if responses:  # pragma: no branch
                last = responses[-1]
                if last['candidates'] and last['candidates'][0].get('content', {}).get('parts'):
                    start_response = last
                    break

        if start_response is None:
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return GeminiStreamedResponse(
            model_request_parameters=model_request_parameters,
            _model_name=self._model_name,
            _content=content,
            _stream=aiter_bytes,
            _provider_name=self._provider.name,
        )

    async def _message_to_gemini_content(
        self, messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> tuple[list[_GeminiTextPart], list[_GeminiContent]]:
        sys_prompt_parts: list[_GeminiTextPart] = []
        contents: list[_GeminiContent] = []
        for m in messages:
            if isinstance(m, ModelRequest):
                message_parts: list[_GeminiPartUnion] = []

                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        sys_prompt_parts.append(_GeminiTextPart(text=part.content))
                    elif isinstance(part, UserPromptPart):
                        message_parts.extend(await self._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        message_parts.append(_response_part_from_response(part.tool_name, part.model_response_object()))
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            message_parts.append(_GeminiTextPart(text=part.model_response()))  # pragma: no cover
                        else:
                            response = {'call_error': part.model_response()}
                            message_parts.append(_response_part_from_response(part.tool_name, response))
                    else:
                        assert_never(part)

                if message_parts:  # pragma: no branch
                    contents.append(_GeminiContent(role='user', parts=message_parts))
            elif isinstance(m, ModelResponse):
                contents.append(_content_model_response(m))
            else:
                assert_never(m)
        if instructions := self._get_instructions(messages, model_request_parameters):
            sys_prompt_parts.insert(0, _GeminiTextPart(text=instructions))
        return sys_prompt_parts, contents

    async def _map_user_prompt(self, part: UserPromptPart) -> list[_GeminiPartUnion]:
        if isinstance(part.content, str):
            return [{'text': part.content}]
        else:
            content: list[_GeminiPartUnion] = []
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data).decode('utf-8')
                    content.append(
                        _GeminiInlineDataPart(inline_data={'data': base64_encoded, 'mime_type': item.media_type})
                    )
                elif isinstance(item, VideoUrl) and item.is_youtube:
                    file_data = _GeminiFileDataPart(file_data={'file_uri': item.url, 'mime_type': item.media_type})
                    content.append(file_data)
                elif isinstance(item, FileUrl):
                    if self.system == 'google-gla' or item.force_download:
                        downloaded_item = await download_item(item, data_format='base64')
                        inline_data = _GeminiInlineDataPart(
                            inline_data={'data': downloaded_item['data'], 'mime_type': downloaded_item['data_type']}
                        )
                        content.append(inline_data)
                    else:  # pragma: lax no cover
                        file_data = _GeminiFileDataPart(file_data={'file_uri': item.url, 'mime_type': item.media_type})
                        content.append(file_data)
                elif isinstance(item, CachePoint):
                    # Gemini doesn't support prompt caching via CachePoint
                    pass
                else:
                    assert_never(item)  # pragma: lax no cover
        return content

    def _map_response_schema(self, o: OutputObjectDefinition) -> dict[str, Any]:
        response_schema = o.json_schema.copy()
        if o.name:
            response_schema['title'] = o.name
        if o.description:
            response_schema['description'] = o.description

        return response_schema


def _settings_to_generation_config(model_settings: GeminiModelSettings) -> _GeminiGenerationConfig:
    config: _GeminiGenerationConfig = {}
    if (max_tokens := model_settings.get('max_tokens')) is not None:
        config['max_output_tokens'] = max_tokens
    if (stop_sequences := model_settings.get('stop_sequences')) is not None:
        config['stop_sequences'] = stop_sequences  # pragma: no cover
    if (temperature := model_settings.get('temperature')) is not None:
        config['temperature'] = temperature
    if (top_p := model_settings.get('top_p')) is not None:
        config['top_p'] = top_p
    if (presence_penalty := model_settings.get('presence_penalty')) is not None:
        config['presence_penalty'] = presence_penalty
    if (frequency_penalty := model_settings.get('frequency_penalty')) is not None:
        config['frequency_penalty'] = frequency_penalty
    if (thinkingConfig := model_settings.get('gemini_thinking_config')) is not None:
        config['thinking_config'] = thinkingConfig
    return config


class AuthProtocol(Protocol):
    """Abstract definition for Gemini authentication."""

    async def headers(self) -> dict[str, str]: ...


@dataclass
class ApiKeyAuth:
    """Authentication using an API key for the `X-Goog-Api-Key` header."""

    api_key: str

    async def headers(self) -> dict[str, str]:
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        return {'X-Goog-Api-Key': self.api_key}  # pragma: no cover


@dataclass
class GeminiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the Gemini model."""

    _model_name: GeminiModelName
    _content: bytearray
    _stream: AsyncIterator[bytes]
    _provider_name: str
    _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for gemini_response in self._get_gemini_responses():
            candidate = gemini_response['candidates'][0]
            if 'content' not in candidate:
                raise UnexpectedModelBehavior('Streamed response has no content field')  # pragma: no cover
            gemini_part: _GeminiPartUnion
            for gemini_part in candidate['content']['parts']:
                if 'text' in gemini_part:
                    # Using vendor_part_id=None means we can produce multiple text parts if their deltas are sprinkled
                    # amongst the tool call deltas
                    maybe_event = self._parts_manager.handle_text_delta(
                        vendor_part_id=None, content=gemini_part['text']
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event

                elif 'function_call' in gemini_part:
                    # Here, we assume all function_call parts are complete and don't have deltas.
                    # We do this by assigning a unique randomly generated "vendor_part_id".
                    # We need to confirm whether this is actually true, but if it isn't, we can still handle it properly
                    # it would just be a bit more complicated. And we'd need to confirm the intended semantics.
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=uuid4(),
                        tool_name=gemini_part['function_call']['name'],
                        args=gemini_part['function_call']['args'],
                        tool_call_id=None,
                    )
                    if maybe_event is not None:  # pragma: no branch
                        yield maybe_event
                else:
                    if not any([key in gemini_part for key in ['function_response', 'thought']]):
                        raise AssertionError(f'Unexpected part: {gemini_part}')  # pragma: no cover

    async def _get_gemini_responses(self) -> AsyncIterator[_GeminiResponse]:
        # This method exists to ensure we only yield completed items, so we don't need to worry about
        # partial gemini responses, which would make everything more complicated

        gemini_responses: list[_GeminiResponse] = []
        current_gemini_response_index = 0
        # Right now, there are some circumstances where we will have information that could be yielded sooner than it is
        # But changing that would make things a lot more complicated.
        async for chunk in self._stream:
            self._content.extend(chunk)

            gemini_responses = _gemini_streamed_response_ta.validate_json(
                _ensure_decodeable(self._content),
                experimental_allow_partial='trailing-strings',
            )

            # The idea: yield only up to the latest response, which might still be partial.
            # Note that if the latest response is complete, we could yield it immediately, but there's not a good
            # allow_partial API to determine if the last item in the list is complete.
            responses_to_yield = gemini_responses[:-1]
            for r in responses_to_yield[current_gemini_response_index:]:
                current_gemini_response_index += 1
                yield r

        # Now yield the final response, which should be complete
        if gemini_responses:  # pragma: no branch
            r = gemini_responses[-1]
            self._usage = _metadata_as_usage(r)
            yield r

    @property
    def model_name(self) -> GeminiModelName:
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


# We use typed dicts to define the Gemini API response schema
# once Pydantic partial validation supports, dataclasses, we could revert to using them
# TypeAdapters take care of validation and serialization


@pydantic.with_config(pydantic.ConfigDict(defer_build=True))
class _GeminiRequest(TypedDict):
    """Schema for an API request to the Gemini API.

    See <https://ai.google.dev/api/generate-content#request-body> for API docs.
    """

    # Note: Even though Google supposedly supports camelCase and snake_case, we've had user report misbehavior
    # when using snake_case, which is why this typeddict now uses camelCase. And anyway, the plan is to replace this
    # with an official google SDK in the near future anyway.
    contents: list[_GeminiContent]
    tools: NotRequired[_GeminiTools]
    toolConfig: NotRequired[_GeminiToolConfig]
    safetySettings: NotRequired[list[GeminiSafetySettings]]
    systemInstruction: NotRequired[_GeminiTextContent]
    """
    Developer generated system instructions, see
    <https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest>
    """
    generationConfig: NotRequired[_GeminiGenerationConfig]
    labels: NotRequired[dict[str, str]]


class GeminiSafetySettings(TypedDict):
    """Safety settings options for Gemini model request.

    See [Gemini API docs](https://ai.google.dev/gemini-api/docs/safety-settings) for safety category and threshold descriptions.
    For an example on how to use `GeminiSafetySettings`, see [here](../../agents.md#model-specific-settings).
    """

    category: Literal[
        'HARM_CATEGORY_UNSPECIFIED',
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
        'HARM_CATEGORY_CIVIC_INTEGRITY',
    ]
    """
    Safety settings category.
    """

    threshold: Literal[
        'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
        'BLOCK_LOW_AND_ABOVE',
        'BLOCK_MEDIUM_AND_ABOVE',
        'BLOCK_ONLY_HIGH',
        'BLOCK_NONE',
        'OFF',
    ]
    """
    Safety settings threshold.
    """


class ThinkingConfig(TypedDict, total=False):
    """The thinking features configuration."""

    include_thoughts: Annotated[bool, pydantic.Field(alias='includeThoughts')]
    """Indicates whether to include thoughts in the response. If true, thoughts are returned only if the model supports thought and thoughts are available."""

    thinking_budget: Annotated[int, pydantic.Field(alias='thinkingBudget')]
    """Indicates the thinking budget in tokens."""


class _GeminiGenerationConfig(TypedDict, total=False):
    """Schema for an API request to the Gemini API.

    Note there are many additional fields available that have not been added yet.

    See <https://ai.google.dev/api/generate-content#generationconfig> for API docs.
    """

    max_output_tokens: int
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    stop_sequences: list[str]
    thinking_config: ThinkingConfig
    response_mime_type: str
    response_schema: dict[str, Any]


class _GeminiContent(TypedDict):
    role: Literal['user', 'model']
    parts: list[_GeminiPartUnion]


def _content_model_response(m: ModelResponse) -> _GeminiContent:
    parts: list[_GeminiPartUnion] = []
    for item in m.parts:
        if isinstance(item, ToolCallPart):
            parts.append(_function_call_part_from_call(item))
        elif isinstance(item, ThinkingPart):
            # NOTE: We don't send ThinkingPart to the providers yet. If you are unsatisfied with this,
            # please open an issue. The below code is the code to send thinking to the provider.
            # parts.append(_GeminiTextPart(text=item.content, thought=True))
            pass
        elif isinstance(item, TextPart):
            if item.content:
                parts.append(_GeminiTextPart(text=item.content))
        elif isinstance(item, BuiltinToolCallPart | BuiltinToolReturnPart):  # pragma: no cover
            # This is currently never returned from gemini
            pass
        elif isinstance(item, FilePart):  # pragma: no cover
            # Files generated by models are not sent back to models that don't themselves generate files.
            pass
        else:
            assert_never(item)
    return _GeminiContent(role='model', parts=parts)


class _BasePart(TypedDict):
    thought: NotRequired[bool]
    """Indicates if the part is thought from the model."""


class _GeminiTextPart(_BasePart):
    text: str


class _GeminiInlineData(_BasePart):
    data: str
    mime_type: Annotated[str, pydantic.Field(alias='mimeType')]


class _GeminiInlineDataPart(_BasePart):
    """See <https://ai.google.dev/api/caching#Blob>."""

    inline_data: Annotated[_GeminiInlineData, pydantic.Field(alias='inlineData')]


class _GeminiFileData(_BasePart):
    """See <https://ai.google.dev/api/caching#FileData>."""

    file_uri: Annotated[str, pydantic.Field(alias='fileUri')]
    mime_type: Annotated[str, pydantic.Field(alias='mimeType')]


class _GeminiFileDataPart(_BasePart):
    file_data: Annotated[_GeminiFileData, pydantic.Field(alias='fileData')]


class _GeminiThoughtPart(TypedDict):
    thought: bool
    thought_signature: Annotated[str, pydantic.Field(alias='thoughtSignature')]


class _GeminiFunctionCallPart(_BasePart):
    function_call: Annotated[_GeminiFunctionCall, pydantic.Field(alias='functionCall')]


def _function_call_part_from_call(tool: ToolCallPart) -> _GeminiFunctionCallPart:
    return _GeminiFunctionCallPart(function_call=_GeminiFunctionCall(name=tool.tool_name, args=tool.args_as_dict()))


def _process_response_from_parts(
    parts: Sequence[_GeminiPartUnion],
    model_name: GeminiModelName,
    usage: usage.RequestUsage,
    vendor_id: str | None,
    vendor_details: dict[str, Any] | None = None,
) -> ModelResponse:
    items: list[ModelResponsePart] = []
    for part in parts:
        if 'text' in part:
            # NOTE: Google doesn't include the `thought` field anymore. We handle this here in case they decide to
            # change their mind and start including it again.
            if part.get('thought'):  # pragma: no cover
                items.append(ThinkingPart(content=part['text']))
            else:
                items.append(TextPart(content=part['text']))
        elif 'function_call' in part:
            items.append(ToolCallPart(tool_name=part['function_call']['name'], args=part['function_call']['args']))
        elif 'function_response' in part:  # pragma: no cover
            raise UnexpectedModelBehavior(
                f'Unsupported response from Gemini, expected all parts to be function calls or text, got: {part!r}'
            )
    return ModelResponse(
        parts=items, usage=usage, model_name=model_name, provider_response_id=vendor_id, provider_details=vendor_details
    )


class _GeminiFunctionCall(TypedDict):
    """See <https://ai.google.dev/api/caching#FunctionCall>."""

    name: str
    args: dict[str, Any]


class _GeminiFunctionResponsePart(TypedDict):
    function_response: Annotated[_GeminiFunctionResponse, pydantic.Field(alias='functionResponse')]


def _response_part_from_response(name: str, response: dict[str, Any]) -> _GeminiFunctionResponsePart:
    return _GeminiFunctionResponsePart(function_response=_GeminiFunctionResponse(name=name, response=response))


class _GeminiFunctionResponse(TypedDict):
    """See <https://ai.google.dev/api/caching#FunctionResponse>."""

    name: str
    response: dict[str, Any]


def _part_discriminator(v: Any) -> str:
    if isinstance(v, dict):  # pragma: no branch
        if 'text' in v:
            return 'text'
        elif 'inlineData' in v:
            return 'inline_data'  # pragma: no cover
        elif 'fileData' in v:
            return 'file_data'  # pragma: no cover
        elif 'thought' in v:
            return 'thought'
        elif 'functionCall' in v or 'function_call' in v:
            return 'function_call'
        elif 'functionResponse' in v or 'function_response' in v:
            return 'function_response'
    return 'text'


# See <https://ai.google.dev/api/caching#Part>
# we don't currently support other part types
_GeminiPartUnion = Annotated[
    Annotated[_GeminiTextPart, pydantic.Tag('text')]
    | Annotated[_GeminiFunctionCallPart, pydantic.Tag('function_call')]
    | Annotated[_GeminiFunctionResponsePart, pydantic.Tag('function_response')]
    | Annotated[_GeminiInlineDataPart, pydantic.Tag('inline_data')]
    | Annotated[_GeminiFileDataPart, pydantic.Tag('file_data')]
    | Annotated[_GeminiThoughtPart, pydantic.Tag('thought')],
    pydantic.Discriminator(_part_discriminator),
]


class _GeminiTextContent(TypedDict):
    role: Literal['user', 'model']
    parts: list[_GeminiTextPart]


class _GeminiTools(TypedDict):
    function_declarations: Annotated[list[_GeminiFunction], pydantic.Field(alias='functionDeclarations')]


class _GeminiFunction(TypedDict):
    name: str
    description: str
    parameters: NotRequired[dict[str, Any]]
    """
    ObjectJsonSchema isn't really true since Gemini only accepts a subset of JSON Schema
    <https://ai.google.dev/gemini-api/docs/function-calling#function_declarations>
    and
    <https://ai.google.dev/api/caching#FunctionDeclaration>
    """


def _function_from_abstract_tool(tool: ToolDefinition) -> _GeminiFunction:
    json_schema = tool.parameters_json_schema
    f = _GeminiFunction(name=tool.name, description=tool.description or '', parameters=json_schema)
    return f


class _GeminiToolConfig(TypedDict):
    function_calling_config: _GeminiFunctionCallingConfig


def _tool_config(function_names: list[str]) -> _GeminiToolConfig:
    return _GeminiToolConfig(
        function_calling_config=_GeminiFunctionCallingConfig(mode='ANY', allowed_function_names=function_names)
    )


class _GeminiFunctionCallingConfig(TypedDict):
    mode: Literal['ANY', 'AUTO']
    allowed_function_names: list[str]


@pydantic.with_config(pydantic.ConfigDict(defer_build=True))
class _GeminiResponse(TypedDict):
    """Schema for the response from the Gemini API.

    See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>
    and <https://cloud.google.com/vertex-ai/docs/reference/rest/v1/GenerateContentResponse>
    """

    candidates: list[_GeminiCandidates]
    # usageMetadata appears to be required by both APIs but is omitted when streaming responses until the last response
    usage_metadata: NotRequired[Annotated[_GeminiUsageMetaData, pydantic.Field(alias='usageMetadata')]]
    prompt_feedback: NotRequired[Annotated[_GeminiPromptFeedback, pydantic.Field(alias='promptFeedback')]]
    model_version: NotRequired[Annotated[str, pydantic.Field(alias='modelVersion')]]
    vendor_id: NotRequired[Annotated[str, pydantic.Field(alias='responseId')]]


class _GeminiCandidates(TypedDict):
    """See <https://ai.google.dev/api/generate-content#v1beta.Candidate>."""

    content: NotRequired[_GeminiContent]
    finish_reason: NotRequired[Annotated[Literal['STOP', 'MAX_TOKENS', 'SAFETY'], pydantic.Field(alias='finishReason')]]
    """
    See <https://ai.google.dev/api/generate-content#FinishReason>, lots of other values are possible,
    but let's wait until we see them and know what they mean to add them here.
    """
    avg_log_probs: NotRequired[Annotated[float, pydantic.Field(alias='avgLogProbs')]]
    index: NotRequired[int]
    safety_ratings: NotRequired[Annotated[list[_GeminiSafetyRating], pydantic.Field(alias='safetyRatings')]]


class _GeminiModalityTokenCount(TypedDict):
    """See <https://ai.google.dev/api/generate-content#modalitytokencount>."""

    modality: Annotated[
        Literal['MODALITY_UNSPECIFIED', 'TEXT', 'IMAGE', 'VIDEO', 'AUDIO', 'DOCUMENT'], pydantic.Field(alias='modality')
    ]
    token_count: Annotated[int, pydantic.Field(alias='tokenCount', default=0)]


class _GeminiUsageMetaData(TypedDict, total=False):
    """See <https://ai.google.dev/api/generate-content#UsageMetadata>.

    The docs suggest all fields are required, but some are actually not required, so we assume they are all optional.
    """

    prompt_token_count: Annotated[int, pydantic.Field(alias='promptTokenCount')]
    candidates_token_count: NotRequired[Annotated[int, pydantic.Field(alias='candidatesTokenCount')]]
    total_token_count: Annotated[int, pydantic.Field(alias='totalTokenCount')]
    cached_content_token_count: NotRequired[Annotated[int, pydantic.Field(alias='cachedContentTokenCount')]]
    thoughts_token_count: NotRequired[Annotated[int, pydantic.Field(alias='thoughtsTokenCount')]]
    tool_use_prompt_token_count: NotRequired[Annotated[int, pydantic.Field(alias='toolUsePromptTokenCount')]]
    prompt_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='promptTokensDetails')]
    ]
    cache_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='cacheTokensDetails')]
    ]
    candidates_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='candidatesTokensDetails')]
    ]
    tool_use_prompt_tokens_details: NotRequired[
        Annotated[list[_GeminiModalityTokenCount], pydantic.Field(alias='toolUsePromptTokensDetails')]
    ]


def _metadata_as_usage(response: _GeminiResponse) -> usage.RequestUsage:
    metadata = response.get('usage_metadata')
    if metadata is None:
        return usage.RequestUsage()
    details: dict[str, int] = {}
    if cached_content_token_count := metadata.get('cached_content_token_count', 0):
        details['cached_content_tokens'] = cached_content_token_count

    if thoughts_token_count := metadata.get('thoughts_token_count', 0):
        details['thoughts_tokens'] = thoughts_token_count

    if tool_use_prompt_token_count := metadata.get('tool_use_prompt_token_count', 0):
        details['tool_use_prompt_tokens'] = tool_use_prompt_token_count

    input_audio_tokens = 0
    output_audio_tokens = 0
    cache_audio_read_tokens = 0
    for key, metadata_details in metadata.items():
        if key.endswith('_details') and metadata_details:
            metadata_details = cast(list[_GeminiModalityTokenCount], metadata_details)
            suffix = key.removesuffix('_details')
            for detail in metadata_details:
                modality = detail['modality']
                details[f'{modality.lower()}_{suffix}'] = value = detail.get('token_count', 0)
                if value and modality == 'AUDIO':
                    if key == 'prompt_tokens_details':
                        input_audio_tokens = value
                    elif key == 'candidates_tokens_details':
                        output_audio_tokens = value
                    elif key == 'cache_tokens_details':  # pragma: no branch
                        cache_audio_read_tokens = value

    return usage.RequestUsage(
        input_tokens=metadata.get('prompt_token_count', 0),
        output_tokens=metadata.get('candidates_token_count', 0) + thoughts_token_count,
        cache_read_tokens=cached_content_token_count,
        input_audio_tokens=input_audio_tokens,
        output_audio_tokens=output_audio_tokens,
        cache_audio_read_tokens=cache_audio_read_tokens,
        details=details,
    )


class _GeminiSafetyRating(TypedDict):
    """See <https://ai.google.dev/gemini-api/docs/safety-settings#safety-filters>."""

    category: Literal[
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
        'HARM_CATEGORY_CIVIC_INTEGRITY',
    ]
    probability: Literal['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH']
    blocked: NotRequired[bool]


class _GeminiPromptFeedback(TypedDict):
    """See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>."""

    block_reason: Annotated[str, pydantic.Field(alias='blockReason')]
    safety_ratings: Annotated[list[_GeminiSafetyRating], pydantic.Field(alias='safetyRatings')]


_gemini_request_ta = pydantic.TypeAdapter(_GeminiRequest)
_gemini_response_ta = pydantic.TypeAdapter(_GeminiResponse)

# steam requests return a list of https://ai.google.dev/api/generate-content#method:-models.streamgeneratecontent
_gemini_streamed_response_ta = pydantic.TypeAdapter(list[_GeminiResponse], config=pydantic.ConfigDict(defer_build=True))


def _ensure_decodeable(content: bytearray) -> bytearray:
    """Trim any invalid unicode point bytes off the end of a bytearray.

    This is necessary before attempting to parse streaming JSON bytes.

    This is a temporary workaround until https://github.com/pydantic/pydantic-core/issues/1633 is resolved
    """
    try:
        content.decode()
    except UnicodeDecodeError as e:
        # e.start marks the start of the invalid decoded bytes, so cut up to before the first invalid byte
        return content[: e.start]
    else:
        return content
