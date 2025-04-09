from __future__ import annotations as _annotations

import base64
import re
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Annotated, Any, Literal, Protocol, Union, cast, overload
from uuid import uuid4

import httpx
import pydantic
from google.genai.types import (
    ContentDict,
    ContentUnionDict,
    FunctionCallDict,
    FunctionCallingConfigDict,
    FunctionCallingConfigMode,
    GenerateContentConfigDict,
    Part,
    PartDict,
    SafetySettingDict,
    ToolConfigDict,
    ToolDict,
    ToolListUnionDict,
)
from typing_extensions import NotRequired, TypedDict, assert_never, deprecated

from pydantic_ai.providers import Provider, infer_provider

from .. import UnexpectedModelBehavior, UserError, _utils, usage
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

try:
    from google import genai
    from google.genai.types import FunctionDeclarationDict, GenerateContentResponse
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Gemini model, '
        'you can use the `gemini` optional group — `pip install "pydantic-ai-slim[gemini]"`'
    ) from _import_error

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

    gemini_safety_settings: list[SafetySettingDict]


@dataclass(init=False)
class GeminiModel(Model):
    """A model that uses Gemini via `generativelanguage.googleapis.com` API.

    This is implemented from scratch rather than using a dedicated SDK, good API documentation is
    available [here](https://ai.google.dev/api).

    Apart from `__init__`, all methods are private or match those of the base class.
    """

    client: genai.Client = field(repr=False)

    _model_name: GeminiModelName = field(repr=False)
    _provider: Provider[genai.Client] = field(repr=False)
    _url: str | None = field(repr=False)
    _system: str = field(default='gemini', repr=False)

    @overload
    @deprecated(
        'Use `pydantic_ai.providers.google.GoogleProvider` instead of `GoogleGLAProvider` or `GoogleVertexProvider`.'
    )
    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        provider: Provider[httpx.AsyncClient],
    ) -> None: ...

    @overload
    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        provider: Literal['google-gla', 'google-vertex'] | Provider[genai.Client] = 'google-gla',
    ) -> None: ...

    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        provider: Literal['google-gla', 'google-vertex']
        | Provider[httpx.AsyncClient]
        | Provider[genai.Client] = 'google-gla',
    ):
        """Initialize a Gemini model.

        Args:
            model_name: The name of the model to use.
            provider: The provider to use for authentication and API access. Can be either the string
                'google-gla' or 'google-vertex' or an instance of `Provider[httpx.AsyncClient]`.
                If not provided, a new provider will be created using the other parameters.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        if isinstance(provider.client, httpx.AsyncClient):
            raise UserError('Use `GoogleProvider` instead of `GoogleGLAProvider` or `GoogleVertexProvider`.')
        provider = cast(Provider[genai.Client], provider)

        self._provider = provider
        self._system = provider.name
        self.client = provider.client

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelResponse, usage.Usage]:
        check_allow_model_requests()
        model_settings = cast(GeminiModelSettings, model_settings or {})
        response = await self._generate_content(messages, False, model_settings, model_request_parameters)
        return self._process_response(response), _metadata_as_usage(response)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        check_allow_model_requests()
        model_settings = cast(GeminiModelSettings, model_settings or {})
        response = await self._generate_content(messages, True, model_settings, model_request_parameters)
        yield await self._process_streamed_response(response)

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

    def _get_tools(self, model_request_parameters: ModelRequestParameters) -> list[ToolDict] | None:
        tools: list[ToolDict] = [
            ToolDict(function_declarations=[_function_declaration_from_tool(t)])
            for t in model_request_parameters.function_tools
        ]
        if model_request_parameters.result_tools:
            tools += [
                ToolDict(function_declarations=[_function_declaration_from_tool(t)])
                for t in model_request_parameters.result_tools
            ]
        return tools

    def _get_tool_config(
        self, model_request_parameters: ModelRequestParameters, tools: list[ToolDict] | None
    ) -> ToolConfigDict | None:
        if model_request_parameters.allow_text_result:
            return None
        elif tools:
            return _tool_config([t['name'] for t in tools['function_declarations']])  # type: ignore
        else:
            return _tool_config([])

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: GeminiModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse: ...

    @overload
    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: GeminiModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> Awaitable[AsyncIterator[GenerateContentResponse]]: ...

    async def _generate_content(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: GeminiModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> GenerateContentResponse | Awaitable[AsyncIterator[GenerateContentResponse]]:
        tools = self._get_tools(model_request_parameters)
        tool_config = self._get_tool_config(model_request_parameters, tools)
        system_instruction, contents = await self._map_messages(messages)

        config = GenerateContentConfigDict(
            http_options={'headers': {'Content-Type': 'application/json', 'User-Agent': get_user_agent()}},
            system_instruction=system_instruction,
            temperature=model_settings.get('temperature'),
            top_p=model_settings.get('top_p'),
            max_output_tokens=model_settings.get('max_tokens'),
            presence_penalty=model_settings.get('presence_penalty'),
            frequency_penalty=model_settings.get('frequency_penalty'),
            safety_settings=model_settings.get('gemini_safety_settings'),
            tools=cast(ToolListUnionDict, tools),
            tool_config=tool_config,
        )

        func = self.client.aio.models.generate_content_stream if stream else self.client.aio.models.generate_content
        return await func(model=self._model_name, contents=contents, config=config)

    def _process_response(self, response: GenerateContentResponse) -> ModelResponse:
        if not response.candidates or len(response.candidates) != 1:
            raise UnexpectedModelBehavior('Expected exactly one candidate in Gemini response')
        if response.candidates[0].content is None:
            if response.candidates[0].finish_reason == 'SAFETY':
                raise UnexpectedModelBehavior('Safety settings triggered', str(response))
            else:
                raise UnexpectedModelBehavior('Content field missing from Gemini response', str(response))
        parts = response.candidates[0].content.parts or []
        return _process_response_from_parts(parts, response.model_version or self._model_name)

    async def _process_streamed_response(self, response: AsyncIterator[GenerateContentResponse]) -> StreamedResponse:
        """Process a streamed response, and prepare a streaming response to return."""
        peekable_response = _utils.PeekableAsyncStream(response)
        first_chunk = await peekable_response.peek()
        if isinstance(first_chunk, _utils.Unset):
            raise UnexpectedModelBehavior('Streamed response ended without content or tool calls')

        assert first_chunk.create_time is not None

        return GeminiStreamedResponse(
            _model_name=self._model_name,
            _response=peekable_response,
            _timestamp=first_chunk.create_time,
        )

    @classmethod
    async def _map_messages(cls, messages: list[ModelMessage]) -> tuple[ContentDict | None, list[ContentUnionDict]]:
        contents: list[ContentUnionDict] = []
        system_parts: list[PartDict] = []

        for m in messages:
            if isinstance(m, ModelRequest):
                message_parts: list[PartDict] = []

                for part in m.parts:
                    if isinstance(part, SystemPromptPart):
                        system_parts.append({'text': part.content})
                    elif isinstance(part, UserPromptPart):
                        message_parts.extend(await cls._map_user_prompt(part))
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
                            message_parts.append({'text': part.model_response()})
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

                if message_parts:
                    contents.append({'role': 'user', 'parts': message_parts})
            elif isinstance(m, ModelResponse):
                contents.append(_content_model_response(m))
            else:
                assert_never(m)

        system_instruction = ContentDict(role='user', parts=system_parts) if system_parts else None
        return system_instruction, contents

    @staticmethod
    async def _map_user_prompt(part: UserPromptPart) -> list[PartDict]:
        if isinstance(part.content, str):
            return [{'text': part.content}]
        else:
            content: list[PartDict] = []
            for item in part.content:
                if isinstance(item, str):
                    content.append({'text': item})
                elif isinstance(item, BinaryContent):
                    base64_encoded = base64.b64encode(item.data)
                    content.append({'inline_data': {'data': base64_encoded, 'mime_type': item.media_type}})
                elif isinstance(item, (AudioUrl, ImageUrl, DocumentUrl)):
                    client = cached_async_http_client()
                    response = await client.get(item.url, follow_redirects=True)
                    response.raise_for_status()
                    mime_type = response.headers['Content-Type'].split(';')[0]
                    base64_encoded = base64.b64encode(response.content)
                    content.append({'inline_data': {'data': base64_encoded, 'mime_type': mime_type}})
                else:
                    assert_never(item)
        return content


class AuthProtocol(Protocol):
    """Abstract definition for Gemini authentication."""

    async def headers(self) -> dict[str, str]: ...


@dataclass
class GeminiStreamedResponse(StreamedResponse):
    """Implementation of `StreamedResponse` for the Gemini model."""

    _model_name: GeminiModelName
    _response: AsyncIterator[GenerateContentResponse]
    _timestamp: datetime

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        async for chunk in self._response:
            self._usage += _metadata_as_usage(chunk)

            assert chunk.candidates is not None
            candidate = chunk.candidates[0]
            if candidate.content is None:
                raise UnexpectedModelBehavior('Streamed response has no content field')
            assert candidate.content.parts is not None
            for part in candidate.content.parts:
                if part.text:
                    yield self._parts_manager.handle_text_delta(vendor_part_id='content', content=part.text)
                elif part.function_call:
                    maybe_event = self._parts_manager.handle_tool_call_delta(
                        vendor_part_id=uuid4(),
                        tool_name=part.function_call.name,
                        args=part.function_call.args,
                        tool_call_id=part.function_call.id,
                    )
                    if maybe_event is not None:
                        yield maybe_event
                else:
                    assert part.function_response is not None, f'Unexpected part: {part}'

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


def _content_model_response(m: ModelResponse) -> ContentDict:
    parts: list[PartDict] = []
    for item in m.parts:
        if isinstance(item, ToolCallPart):
            function_call = FunctionCallDict(name=item.tool_name, args=item.args_as_dict(), id=item.tool_call_id)
            parts.append({'function_call': function_call})
        elif isinstance(item, TextPart):
            if item.content:
                parts.append({'text': item.content})
        else:
            assert_never(item)
    return ContentDict(role='model', parts=parts)


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
    parts: list[Part], model_name: GeminiModelName, timestamp: datetime | None = None
) -> ModelResponse:
    items: list[ModelResponsePart] = []
    for part in parts:
        if part.text:
            items.append(TextPart(content=part.text))
        elif part.function_call:
            assert part.function_call.name is not None
            tool_call_part = ToolCallPart(tool_name=part.function_call.name, args=part.function_call.args or {})
            if part.function_call.id is not None:
                tool_call_part.tool_call_id = part.function_call.id
            items.append(tool_call_part)
        elif part.function_response:
            raise UnexpectedModelBehavior(
                f'Unsupported response from Gemini, expected all parts to be function calls or text, got: {part!r}'
            )
    return ModelResponse(parts=items, model_name=model_name, timestamp=timestamp or _utils.now_utc())


def _function_declaration_from_tool(tool: ToolDefinition) -> FunctionDeclarationDict:
    json_schema = tool.parameters_json_schema
    f = FunctionDeclarationDict(name=tool.name, description=tool.description)
    if json_schema.get('properties'):
        f['parameters'] = json_schema
    return f


def _tool_config(function_names: list[str]) -> ToolConfigDict:
    mode = FunctionCallingConfigMode.ANY
    function_calling_config = FunctionCallingConfigDict(mode=mode, allowed_function_names=function_names)
    return ToolConfigDict(function_calling_config=function_calling_config)


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


def _metadata_as_usage(response: GenerateContentResponse) -> usage.Usage:
    metadata = response.usage_metadata
    if metadata is None:
        return usage.Usage()
    # TODO(Marcelo): We exclude the `prompt_tokens_details` and `candidate_token_details` fields because on
    # `usage.Usage.incr``, it will try to sum non-integer values with integers, which will fail. We should probably
    # handle this in the `Usage` class.
    details = metadata.model_dump(exclude={'prompt_tokens_details', 'candidates_tokens_details'}, exclude_defaults=True)
    return usage.Usage(
        request_tokens=details.pop('prompt_token_count', 0),
        response_tokens=details.pop('candidates_token_count', 0),
        total_tokens=details.pop('total_token_count', 0),
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
