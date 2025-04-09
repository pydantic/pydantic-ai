from __future__ import annotations as _annotations

import base64
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol, Union, cast
from uuid import uuid4

import httpx
import pydantic
from httpx import USE_CLIENT_DEFAULT, Response as HTTPResponse
from typing_extensions import NotRequired, TypedDict, assert_never

from pydantic_ai.providers import Provider, infer_provider

from .. import ModelHTTPError, UnexpectedModelBehavior, UserError, _utils, usage
from ..messages import (
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
)
from ..settings import ModelSettings
from ..tools import ToolDefinition
from . import (
    Model,
    ModelRequestParameters,
    StreamedResponse,
    cached_async_http_client,
    check_allow_model_requests,
    get_user_agent,
)

LatestGeminiModelNames = Literal[
    'gemini-1.5-flash',
    'gemini-1.5-flash-8b',
    'gemini-1.5-pro',
    'gemini-1.0-pro',
    'gemini-2.0-flash-exp',
    'gemini-2.0-flash-thinking-exp-01-21',
    'gemini-exp-1206',
    'gemini-2.0-flash',
    'gemini-2.0-flash-lite-preview-02-05',
    'gemini-2.0-pro-exp-02-05',
    'gemini-2.5-pro-exp-03-25',
]
"""Latest Gemini models."""

GeminiModelName = Union[str, LatestGeminiModelNames]
"""Possible Gemini model names.

Since Gemini supports a variety of date-stamped models, we explicitly list the latest models but
allow any name in the type hints.
See [the Gemini API docs](https://ai.google.dev/gemini-api/docs/models/gemini#model-variations) for a full list.
"""


class GeminiModelSettings(ModelSettings):
    """Settings used for a Gemini model request.

    ALL FIELDS MUST BE `gemini_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    gemini_safety_settings: list[GeminiSafetySettings]


@dataclass(init=False)
class GeminiModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: httpx.AsyncClient = field(repr=False)

    _model_name: GeminiModelName = field(repr=False)
    _provider: Literal['google-gla', 'google-vertex'] | Provider[httpx.AsyncClient] | None = field(repr=False)
    _auth: AuthProtocol | None = field(repr=False)
    _url: str | None = field(repr=False)
    _system: str = field(default='gemini', repr=False)

    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        provider: Literal['google-gla', 'google-vertex'] | Provider[httpx.AsyncClient] = 'google-gla',
    ):
        """Initialize a Gemini model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`.
                If not provided, a new provider will be created using the other parameters.
        """
        self._model_name = model_name
        self._provider = provider

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._system = provider.name
        self.client = provider.client
        self._url = str(self.client.base_url)

    @property
    def base_url(self) -> str:
        assert self._url is not None, 'URL not initialized'
        return self._url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        check_allow_model_requests()
        async with self._make_request(
            messages, False, cast(GeminiModelSettings, model_settings or {}), model_request_parameters
        ) as http_response:
            data = await http_response.aread()
            response = _gemini_response_ta.validate_json(data)
        return self._process_response(response), _metadata_as_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        async with self._make_request(
            messages, True, cast(GeminiModelSettings, model_settings or {}), model_request_parameters
        ) as http_response:
            yield await self._process_streamed_response(http_response)

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        def _customize_tool_def(t: ToolDefinition):
            return replace(t, parameters_json_schema=_GeminiJsonSchema(t.parameters_json_schema).simplify())

        return ModelRequestParameters(
            function_tools=[_customize_tool_def(tool) for tool in model_request_parameters.function_tools],
            allow_text_result=model_request_parameters.allow_text_result,
            result_tools=[_customize_tool_def(tool) for tool in model_request_parameters.result_tools],
        )

    @property
    def model_name(self) -> GeminiModelName:
        """The model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The system / model provider."""
        return self._system

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> _GeminiTools | None:
        tools = [_function_from_abstract_tool(t) for t in model_request_parameters.function_tools]
        if model_request_parameters.result_tools:
            tools += [_function_from_abstract_tool(t) for t in model_request_parameters.result_tools]
        return _GeminiTools(function_declarations=tools) if tools else None

    def _get_tool_config(
        self, model_request_parameters: ModelRequestParameters, tools: _GeminiTools | None
    ) -> _GeminiToolConfig | None:
        if model_request_parameters.allow_text_result:
            return None
        elif tools:
            return _tool_config([t['name'] for t in tools['function_declarations']])
        else:
            return _tool_config([])

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
        sys_prompt_parts, contents = await self._message_to_gemini_content(messages)

        request_data = _GeminiRequest(contents=contents)
        if sys_prompt_parts:
            request_data['system_instruction'] = _GeminiTextContent(role='user', parts=sys_prompt_parts)
        if tools is not None:
            request_data['tools'] = tools
        if tool_config is not None:
            request_data['tool_config'] = tool_config

        generation_config: _GeminiGenerationConfig = {}
        if model_settings:
            if (max_tokens := model_settings.get('max_tokens')) is not None:
                generation_config['max_output_tokens'] = max_tokens
            if (temperature := model_settings.get('temperature')) is not None:
                generation_config['temperature'] = temperature
            if (top_p := model_settings.get('top_p')) is not None:
                generation_config['top_p'] = top_p
            if (presence_penalty := model_settings.get('presence_penalty')) is not None:
                generation_config['presence_penalty'] = presence_penalty
            if (frequency_penalty := model_settings.get('frequency_penalty')) is not None:
                generation_config['frequency_penalty'] = frequency_penalty
            if (gemini_safety_settings := model_settings.get('gemini_safety_settings')) != []:
                request_data['safety_settings'] = gemini_safety_settings
        if generation_config:
            request_data['generation_config'] = generation_config

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
                raise UnexpectedModelBehavior(f'Unexpected response from gemini {status_code}', r.text)
            yield r

    def _process_response(self, response: _GeminiResponse) -> ModelResponse:
        if len(response['candidates']) != 1:
            raise UnexpectedModelBehavior('Expected exactly one candidate in Gemini response')
        if 'content' not in response['candidates'][0]:
            if response['candidates'][0].get('finish_reason') == 'SAFETY':
                raise UnexpectedModelBehavior('Safety settings triggered', str(response))
            else:
                raise UnexpectedModelBehavior('Content field missing from Gemini response', str(response))
        parts = response['candidates'][0]['content']['parts']
        return _process_response_from_parts(parts, model_name=response.get('model_version', self._model_name))

    async def _process_streamed_response(self, http_response: HTTPResponse) -> StreamedResponse:
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
            if responses:
                last = responses[-1]
                if last['candidates'] and last['candidates'][0].get('content', {}).get('parts'):
                    start_response = last
                    break

        if start_response is None:
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        return GeminiStreamedResponse(_model_name=self._model_name, _content=content, _stream=aiter_bytes)

    @classmethod
    async def _message_to_gemini_content(
        cls, messages: list[ModelMessage]
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
                        message_parts.extend(await cls._map_user_prompt(part))
                    elif isinstance(part, ToolReturnPart):
                        message_parts.append(_response_part_from_response(part.tool_name, part.model_response_object()))
                    elif isinstance(part, RetryPromptPart):
                        if part.tool_name is None:
                            message_parts.append(_GeminiTextPart(text=part.model_response()))
                        else:
                            response = {'call_error': part.model_response()}
                            message_parts.append(_response_part_from_response(part.tool_name, response))
                    else:
                        assert_never(part)

                if message_parts:
                    contents.append(_GeminiContent(role='user', parts=message_parts))
            elif isinstance(m, ModelResponse):
                contents.append(_content_model_response(m))
            else:
                assert_never(m)

        return sys_prompt_parts, contents

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> list[_GeminiPartUnion]:
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
                elif isinstance(item, (AudioUrl, ImageUrl, DocumentUrl)):
                    client = cached_async_http_client()
                    response = await client.get(item.url, follow_redirects=True)
                    response.raise_for_status()
                    mime_type = response.headers['Content-Type'].split(';')[0]
                    inline_data = _GeminiInlineDataPart(
                        inline_data={'data': base64.b64encode(response.content).decode('utf-8'), 'mime_type': mime_type}
                    )
                    content.append(inline_data)
                else:
                    assert_never(item)
        return content


class AuthProtocol(Protocol):
    """Abstract definition for Gemini authentication."""

    async def headers(self) -> dict[str, str]: ...


@dataclass
class ApiKeyAuth:
    """Authentication using an API key for the `X-Goog-Api-Key` header."""

    api_key: str

    async def headers(self) -> dict[str, str]:
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        return {'X-Goog-Api-Key': self.api_key}


@dataclass
class GeminiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the Gemini model."""

    _model_name: GeminiModelName
    _content: bytearray
    _stream: AsyncIterator[bytes]
    _timestamp: datetime = field(default_factory=_utils.now_utc, init=False)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for gemini_response in self._get_gemini_responses():
            candidate = gemini_response['candidates'][0]
            if 'content' not in candidate:
                raise UnexpectedModelBehavior('Streamed response has no content field')
            gemini_part: _GeminiPartUnion
            for gemini_part in candidate['content']['parts']:
                if 'text' in gemini_part:
                    # Using vendor_part_id=None means we can produce multiple text parts if their deltas are sprinkled
                    # amongst the tool call deltas
                    yield self._parts_manager.handle_text_delta(vendor_part_id=None, content=gemini_part['text'])

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
                    if maybe_event is not None:
                        yield maybe_event
                else:
                    assert 'function_response' in gemini_part, f'Unexpected part: {gemini_part}'

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
                self._usage += _metadata_as_usage(r)
                yield r

        # Now yield the final response, which should be complete
        if gemini_responses:
            r = gemini_responses[-1]
            self._usage += _metadata_as_usage(r)
            yield r

    @property
    def model_name(self) -> GeminiModelName:
        """Get the model name of the response."""
        return self._model_name

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

    contents: list[_GeminiContent]
    tools: NotRequired[_GeminiTools]
    tool_config: NotRequired[_GeminiToolConfig]
    safety_settings: NotRequired[list[GeminiSafetySettings]]
    # we don't implement `generationConfig`, instead we use a named tool for the response
    system_instruction: NotRequired[_GeminiTextContent]
    """
    Developer generated system instructions, see
    <https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest>
    """
    generation_config: NotRequired[_GeminiGenerationConfig]


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


class _GeminiContent(TypedDict):
    role: Literal['user', 'model']
    parts: list[_GeminiPartUnion]


def _content_model_response(m: ModelResponse) -> _GeminiContent:
    parts: list[_GeminiPartUnion] = []
    for item in m.parts:
        if isinstance(item, ToolCallPart):
            parts.append(_function_call_part_from_call(item))
        elif isinstance(item, TextPart):
            if item.content:
                parts.append(_GeminiTextPart(text=item.content))
        else:
            assert_never(item)
    return _GeminiContent(role='model', parts=parts)


class _GeminiTextPart(TypedDict):
    text: str


class _GeminiInlineData(TypedDict):
    data: str
    mime_type: Annotated[str, pydantic.Field(alias='mimeType')]


class _GeminiInlineDataPart(TypedDict):
    """See <https://ai.google.dev/api/caching#Blob>."""

    inline_data: Annotated[_GeminiInlineData, pydantic.Field(alias='inlineData')]


class _GeminiFileData(TypedDict):
    """See <https://ai.google.dev/api/caching#FileData>."""

    file_uri: Annotated[str, pydantic.Field(alias='fileUri')]
    mime_type: Annotated[str, pydantic.Field(alias='mimeType')]


class _GeminiFileDataPart(TypedDict):
    file_data: Annotated[_GeminiFileData, pydantic.Field(alias='fileData')]


class _GeminiFunctionCallPart(TypedDict):
    function_call: Annotated[_GeminiFunctionCall, pydantic.Field(alias='functionCall')]


def _function_call_part_from_call(tool: ToolCallPart) -> _GeminiFunctionCallPart:
    return _GeminiFunctionCallPart(function_call=_GeminiFunctionCall(name=tool.tool_name, args=tool.args_as_dict()))


def _process_response_from_parts(
    parts: Sequence[_GeminiPartUnion], model_name: GeminiModelName, timestamp: datetime | None = None
) -> ModelResponse:
    items: list[ModelResponsePart] = []
    for part in parts:
        if 'text' in part:
            items.append(TextPart(content=part['text']))
        elif 'function_call' in part:
            items.append(ToolCallPart(tool_name=part['function_call']['name'], args=part['function_call']['args']))
        elif 'function_response' in part:
            raise UnexpectedModelBehavior(
                f'Unsupported response from Gemini, expected all parts to be function calls or text, got: {part!r}'
            )
    return ModelResponse(parts=items, model_name=model_name, timestamp=timestamp or _utils.now_utc())


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
    if isinstance(v, dict):
        if 'text' in v:
            return 'text'
        elif 'inlineData' in v:
            return 'inline_data'
        elif 'fileData' in v:
            return 'file_data'
        elif 'functionCall' in v or 'function_call' in v:
            return 'function_call'
        elif 'functionResponse' in v or 'function_response' in v:
            return 'function_response'
    return 'text'


# See <https://ai.google.dev/api/caching#Part>
# we don't currently support other part types
# TODO discriminator
_GeminiPartUnion = Annotated[
    Union[
        Annotated[_GeminiTextPart, pydantic.Tag('text')],
        Annotated[_GeminiFunctionCallPart, pydantic.Tag('function_call')],
        Annotated[_GeminiFunctionResponsePart, pydantic.Tag('function_response')],
        Annotated[_GeminiInlineDataPart, pydantic.Tag('inline_data')],
        Annotated[_GeminiFileDataPart, pydantic.Tag('file_data')],
    ],
    pydantic.Discriminator(_part_discriminator),
]


class _GeminiTextContent(TypedDict):
    role: Literal['user', 'model']
    parts: list[_GeminiTextPart]


class _GeminiTools(TypedDict):
    function_declarations: list[Annotated[_GeminiFunction, pydantic.Field(alias='functionDeclarations')]]


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
    f = _GeminiFunction(name=tool.name, description=tool.description)
    if json_schema.get('properties'):
        f['parameters'] = json_schema
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


class _GeminiUsageMetaData(TypedDict, total=False):
    """See <https://ai.google.dev/api/generate-content#FinishReason>.

    The docs suggest all fields are required, but some are actually not required, so we assume they are all optional.
    """

    prompt_token_count: Annotated[int, pydantic.Field(alias='promptTokenCount')]
    candidates_token_count: NotRequired[Annotated[int, pydantic.Field(alias='candidatesTokenCount')]]
    total_token_count: Annotated[int, pydantic.Field(alias='totalTokenCount')]
    cached_content_token_count: NotRequired[Annotated[int, pydantic.Field(alias='cachedContentTokenCount')]]


def _metadata_as_usage(response: _GeminiResponse) -> usage.Usage:
    metadata = response.get('usage_metadata')
    if metadata is None:
        return usage.Usage()
    details: dict[str, int] = {}
    if cached_content_token_count := metadata.get('cached_content_token_count'):
        details['cached_content_token_count'] = cached_content_token_count
    return usage.Usage(
        request_tokens=metadata.get('prompt_token_count', 0),
        response_tokens=metadata.get('candidates_token_count', 0),
        total_tokens=metadata.get('total_token_count', 0),
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


class _GeminiJsonSchema:
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """

    def __init__(self, schema: _utils.ObjectJsonSchema):
        self.schema = deepcopy(schema)
        self.defs = self.schema.pop('$defs', {})

    def simplify(self) -> dict[str, Any]:
        self._simplify(self.schema, refs_stack=())
        return self.schema

    def _simplify(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        schema.pop('title', None)
        schema.pop('default', None)
        schema.pop('$schema', None)
        schema.pop('exclusiveMaximum', None)
        schema.pop('exclusiveMinimum', None)
        if ref := schema.pop('$ref', None):
            # noinspection PyTypeChecker
            key = re.sub(r'^#/\$defs/', '', ref)
            if key in refs_stack:
                raise UserError('Recursive `$ref`s in JSON Schema are not supported by Gemini')
            refs_stack += (key,)
            schema_def = self.defs[key]
            self._simplify(schema_def, refs_stack)
            schema.update(schema_def)
            return

        if any_of := schema.get('anyOf'):
            for item_schema in any_of:
                self._simplify(item_schema, refs_stack)
            if len(any_of) == 2 and {'type': 'null'} in any_of:
                for item_schema in any_of:
                    if item_schema != {'type': 'null'}:
                        schema.clear()
                        schema.update(item_schema)
                        schema['nullable'] = True
                        return

        type_ = schema.get('type')

        if type_ == 'object':
            self._object(schema, refs_stack)
        elif type_ == 'array':
            return self._array(schema, refs_stack)
        elif type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

    def _object(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        ad_props = schema.pop('additionalProperties', None)
        if ad_props:
            raise UserError('Additional properties in JSON Schema are not supported by Gemini')

        if properties := schema.get('properties'):  # pragma: no branch
            for value in properties.values():
                self._simplify(value, refs_stack)

    def _array(self, schema: dict[str, Any], refs_stack: tuple[str, ...]) -> None:
        if prefix_items := schema.get('prefixItems'):
            # TODO I think this not is supported by Gemini, maybe we should raise an error?
            for prefix_item in prefix_items:
                self._simplify(prefix_item, refs_stack)

        if items_schema := schema.get('items'):  # pragma: no branch
            self._simplify(items_schema, refs_stack)


def _ensure_decodeable(content: bytearray) -> bytearray:
    """Trim any invalid unicode point bytes off the end of a bytearray.

    This is necessary before attempting to parse streaming JSON bytes.

    This is a temporary workaround until https://github.com/pydantic/pydantic-core/issues/1633 is resolved
    """
    while True:
        try:
            content.decode()
        except UnicodeDecodeError:
            content = content[:-1]  # this will definitely succeed before we run out of bytes
        else:
            return content
