"""Custom interface to the `generativelanguage.googleapis.com` API using HTTPX and Pydantic.

The Google SDK for interacting with the `generativelanguage.googleapis.com` API
[`google-generativeai`](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) reads like it was written by a
Java developer who thought they knew everything about OOP, spent 30 minutes trying to learn Python,
gave up and decided to build the library to prove how horrible Python is. It also doesn't use httpx for HTTP requests,
and tries to implement tool calling itself, but doesn't use Pydantic or equivalent for validation.

We could also use the Google Vertex SDK,
[`google-cloud-aiplatform`](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk)
which uses the `*-aiplatform.googleapis.com` API, but that requires a service account for authentication
which is a faff to set up and manage. The big advantages of `*-aiplatform.googleapis.com` is that it claims API
compatibility with OpenAI's API, but I suspect Gemini's limited support for JSON Schema means you'd need to
hack around its limitations anyway for tool calls.
"""

from __future__ import annotations as _annotations

import os
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, cast

from httpx import AsyncClient as AsyncHTTPClient
from pydantic import Field
from typing_extensions import assert_never

from .. import _pydantic, _utils, shared
from ..messages import (
    ArgsObject,
    LLMMessage,
    LLMResponse,
    LLMToolCalls,
    Message,
    RetryPrompt,
    ToolCall,
    ToolReturn,
)
from . import AbstractToolDefinition, AgentModel, Model, cached_async_http_client

__all__ = 'GeminiModel', 'GeminiModelName'

# https://ai.google.dev/gemini-api/docs/models/gemini#model-variations
GeminiModelName = Literal['gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-1.0-pro']


@dataclass(init=False)
class GeminiModel(Model):
    model_name: GeminiModelName
    api_key: str
    http_client: AsyncHTTPClient
    url_template: str

    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        api_key: str | None = None,
        http_client: AsyncHTTPClient | None = None,
        # https://ai.google.dev/gemini-api/docs/quickstart?lang=rest#make-first-request
        url_template: str = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',
    ):
        self.model_name = model_name
        if api_key is None:
            if env_api_key := os.getenv('GEMINI_API_KEY'):
                api_key = env_api_key
            else:
                raise shared.UserError('API key must be provided or set in the GEMINI_API_KEY environment variable')
        self.api_key = api_key
        self.http_client = http_client or cached_async_http_client()
        self.url_template = url_template

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tools: Sequence[AbstractToolDefinition] | None,
    ) -> GeminiAgentModel:
        tools = [_GeminiFunction.from_abstract_tool(t) for t in retrievers.values()]
        if result_tools is not None:
            tools += [_GeminiFunction.from_abstract_tool(t) for t in result_tools]

        if allow_text_result:
            tool_config = None
        else:
            tool_config = _GeminiToolConfig.call_required([t.name for t in tools])

        return GeminiAgentModel(
            http_client=self.http_client,
            model_name=self.model_name,
            api_key=self.api_key,
            tools=_GeminiTools(function_declarations=tools) if tools else None,
            tool_config=tool_config,
            url_template=self.url_template,
        )

    def name(self) -> str:
        return self.model_name


@dataclass
class GeminiAgentModel(AgentModel):
    http_client: AsyncHTTPClient
    model_name: GeminiModelName
    api_key: str
    tools: _GeminiTools | None
    tool_config: _GeminiToolConfig | None
    url_template: str

    async def request(self, messages: list[Message]) -> tuple[LLMMessage, shared.Cost]:
        response = await self.make_request(messages)
        return self.process_response(response), response.usage_metadata.as_cost()

    async def make_request(self, messages: list[Message]) -> _GeminiResponse:
        contents: list[_GeminiContent] = []
        sys_prompt_parts: list[_GeminiTextPart] = []
        for m in messages:
            either_content = self.message_to_gemini(m)
            if left := either_content.left:
                sys_prompt_parts.append(left.value)
            else:
                contents.append(either_content.right)

        request_data = _GeminiRequest(
            contents=contents,
            system_instruction=_GeminiTextContent(role='user', parts=sys_prompt_parts) if sys_prompt_parts else None,
            tools=self.tools if self.tools is not None else None,
            tool_config=self.tool_config if self.tool_config is not None else None,
        )
        request_json = _gemini_request_ta.dump_json(request_data, exclude_none=True, by_alias=True)
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        headers = {
            'X-Goog-Api-Key': self.api_key,
            'Content-Type': 'application/json',
        }
        url = self.url_template.format(model=self.model_name)
        r = await self.http_client.post(url, content=request_json, headers=headers)
        if r.status_code != 200:
            raise shared.UnexpectedModelBehaviour(f'Unexpected response from gemini {r.status_code}', r.text)
        return _gemini_response_ta.validate_json(r.content)

    @staticmethod
    def process_response(response: _GeminiResponse) -> LLMMessage:
        assert len(response.candidates) == 1, 'Expected exactly one candidate'
        parts = response.candidates[0].content.parts
        if all(isinstance(part, _GeminiFunctionCallPart) for part in parts):
            parts = cast(list[_GeminiFunctionCallPart], parts)
            calls = [ToolCall.from_object(part.function_call.name, part.function_call.args) for part in parts]
            return LLMToolCalls(calls)
        elif all(isinstance(part, _GeminiTextPart) for part in parts):
            parts = cast(list[_GeminiTextPart], parts)
            return LLMResponse(content=''.join(part.text for part in parts))
        else:
            raise shared.UnexpectedModelBehaviour(
                f'Unexpected response from Gemini, expected all parts to be function calls or text, got: {parts!r}'
            )

    @staticmethod
    def message_to_gemini(m: Message) -> _utils.Either[_GeminiTextPart, _GeminiContent]:
        """Convert a message to a _GeminiTextPart for "system_instructions" or _GeminiContent for "contents"."""
        if m.role == 'system':
            # SystemPrompt ->
            return _utils.Either(left=_GeminiTextPart(text=m.content))
        elif m.role == 'user':
            # UserPrompt ->
            return _utils.Either(right=_GeminiContent.user_text(m.content))
        elif m.role == 'tool-return':
            # ToolReturn ->
            return _utils.Either(right=_GeminiContent.function_return(m))
        elif m.role == 'retry-prompt':
            # RetryPrompt ->
            return _utils.Either(right=_GeminiContent.function_retry(m))
        elif m.role == 'llm-response':
            # LLMResponse ->
            return _utils.Either(right=_GeminiContent.model_text(m.content))
        elif m.role == 'llm-tool-calls':
            # LLMToolCalls ->
            return _utils.Either(right=_GeminiContent.function_call(m))
        else:
            assert_never(m)


@dataclass
class _GeminiRequest:
    """Schema for an API request to the Gemini API.

    See <https://ai.google.dev/api/generate-content#request-body> for API docs.
    """

    contents: list[_GeminiContent]
    tools: _GeminiTools | None = None
    tool_config: _GeminiToolConfig | None = None
    # we don't implement `generationConfig`, instead we use a named tool for the response
    system_instruction: _GeminiTextContent | None = None
    """
    Developer generated system instructions, see
    <https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest>
    """


# We use dataclasses, not typed dicts to define the Gemini API schema
# so we can include custom constructors etc.
# TypeAdapters take care of validation and serialization


@dataclass
class _GeminiContent:
    role: Literal['user', 'model']
    parts: list[_GeminiPartUnion]

    @classmethod
    def user_text(cls, text: str) -> _GeminiContent:
        return cls(role='user', parts=[_GeminiTextPart(text=text)])

    @classmethod
    def model_text(cls, text: str) -> _GeminiContent:
        return cls(role='model', parts=[_GeminiTextPart(text=text)])

    @classmethod
    def function_call(cls, m: LLMToolCalls) -> _GeminiContent:
        parts: list[_GeminiPartUnion] = [_GeminiFunctionCallPart.from_call(t) for t in m.calls]
        return cls(role='model', parts=parts)

    @classmethod
    def function_return(cls, m: ToolReturn) -> _GeminiContent:
        f_response = _GeminiFunctionResponsePart.from_response(m.tool_name, m.model_response_object())
        return cls(role='user', parts=[f_response])

    @classmethod
    def function_retry(cls, m: RetryPrompt) -> _GeminiContent:
        if m.tool_name is None:
            part = _GeminiTextPart(text=m.model_response())
        else:
            response = {'call_error': m.model_response()}
            part = _GeminiFunctionResponsePart.from_response(m.tool_name, response)
        return cls(role='user', parts=[part])


@dataclass
class _GeminiTextPart:
    text: str


@dataclass
class _GeminiFunctionCallPart:
    function_call: Annotated[_GeminiFunctionCall, Field(alias='functionCall')]

    @classmethod
    def from_call(cls, tool: ToolCall) -> _GeminiFunctionCallPart:
        assert isinstance(tool.args, ArgsObject), f'Expected ArgsObject, got {tool.args}'
        return cls(function_call=_GeminiFunctionCall(name=tool.tool_name, args=tool.args.args_object))


@dataclass
class _GeminiFunctionCall:
    """See <https://ai.google.dev/api/caching#FunctionCall>."""

    name: str
    args: dict[str, Any]


@dataclass
class _GeminiFunctionResponsePart:
    function_response: Annotated[_GeminiFunctionResponse, Field(alias='functionResponse')]

    @classmethod
    def from_response(cls, name: str, response: dict[str, Any]) -> _GeminiFunctionResponsePart:
        return cls(function_response=_GeminiFunctionResponse(name=name, response=response))


@dataclass
class _GeminiFunctionResponse:
    """See <https://ai.google.dev/api/caching#FunctionResponse>."""

    name: str
    response: dict[str, Any]


# See <https://ai.google.dev/api/caching#Part>
# we don't currently support other part types
# TODO discriminator
_GeminiPartUnion = Union[_GeminiTextPart, _GeminiFunctionCallPart, _GeminiFunctionResponsePart]


@dataclass
class _GeminiTextContent:
    role: Literal['user', 'model']
    parts: list[_GeminiTextPart]


@dataclass
class _GeminiTools:
    function_declarations: list[_GeminiFunction]


@dataclass
class _GeminiFunction:
    name: str
    description: str
    parameters: dict[str, Any]
    """
    ObjectJsonSchema isn't really true since Gemini only accepts a subset of JSON Schema
    <https://ai.google.dev/gemini-api/docs/function-calling#function_declarations>
    """

    @classmethod
    def from_abstract_tool(cls, tool: AbstractToolDefinition) -> _GeminiFunction:
        json_schema = _GeminiJsonSchema(tool.json_schema).simplify()
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=json_schema,
        )


@dataclass
class _GeminiToolConfig:
    function_calling_config: _GeminiFunctionCallingConfig

    @classmethod
    def call_required(cls, function_names: list[str]) -> _GeminiToolConfig:
        return cls(
            function_calling_config=_GeminiFunctionCallingConfig(mode='ANY', allowed_function_names=function_names)
        )


@dataclass
class _GeminiFunctionCallingConfig:
    mode: Literal['ANY', 'AUTO']
    allowed_function_names: list[str]


@dataclass
class _GeminiResponse:
    """Schema for the response from the Gemini API.

    See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>
    """

    candidates: list[_GeminiCandidates]
    usage_metadata: Annotated[_GeminiUsageMetaData, Field(alias='usageMetadata')]
    prompt_feedback: Annotated[_GeminiPromptFeedback | None, Field(alias='promptFeedback')] = None


@dataclass
class _GeminiCandidates:
    content: _GeminiContent
    finish_reason: Annotated[Literal['STOP'], Field(alias='finishReason')]
    """
    See https://ai.google.dev/api/generate-content#FinishReason, lots of other values are possible,
    but let's wait until we see them and know what they mean to add them here.
    """
    index: int
    safety_ratings: Annotated[list[_GeminiSafetyRating], Field(alias='safetyRatings')]


@dataclass
class _GeminiUsageMetaData:
    prompt_token_count: Annotated[int, Field(alias='promptTokenCount')]
    candidates_token_count: Annotated[int, Field(alias='candidatesTokenCount')]
    total_token_count: Annotated[int, Field(alias='totalTokenCount')]
    cached_content_token_count: Annotated[int | None, Field(alias='cachedContentTokenCount')] = None

    def as_cost(self) -> shared.Cost:
        details: dict[str, int] = {}
        if self.cached_content_token_count is not None:
            details['cached_content_token_count'] = self.cached_content_token_count
        return shared.Cost(
            request_tokens=self.prompt_token_count,
            response_tokens=self.candidates_token_count,
            total_tokens=self.total_token_count,
            details=details,
        )


@dataclass
class _GeminiSafetyRating:
    """See <https://ai.google.dev/gemini-api/docs/safety-settings#safety-filters>."""

    category: Literal[
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
        'HARM_CATEGORY_CIVIC_INTEGRITY',
    ]
    probability: Literal['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH']


@dataclass
class _GeminiPromptFeedback:
    """See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>."""

    block_reason: Annotated[str, Field(alias='blockReason')]
    safety_ratings: Annotated[list[_GeminiSafetyRating], Field(alias='safetyRatings')]


_gemini_request_ta = _pydantic.LazyTypeAdapter(_GeminiRequest)
_gemini_response_ta = _pydantic.LazyTypeAdapter(_GeminiResponse)


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
        self._simplify(self.schema, allow_ref=True)
        return self.schema

    def _simplify(self, schema: dict[str, Any], allow_ref: bool) -> None:
        schema.pop('title', None)
        schema.pop('default', None)
        if ref := schema.pop('$ref', None):
            if not allow_ref:
                raise shared.UserError('Recursive `$ref`s in JSON Schema are not supported by Gemini')
            # noinspection PyTypeChecker
            key = re.sub(r'^#/\$defs/', '', ref)
            schema_def = self.defs[key]
            self._simplify(schema_def, allow_ref=False)
            schema.update(schema_def)
            return

        if any_of := schema.get('anyOf'):
            for schema in any_of:
                self._simplify(schema, allow_ref)

        type_ = schema.get('type')

        if type_ == 'object':
            self._object(schema, allow_ref)
        elif type_ == 'array':
            return self._array(schema, allow_ref)

    def _object(self, schema: dict[str, Any], allow_ref: bool) -> None:
        ad_props = schema.pop('additionalProperties', None)
        if ad_props:
            raise shared.UserError('Additional properties in JSON Schema are not supported by Gemini')

        if properties := schema.get('properties'):  # pragma: no branch
            for value in properties.values():
                self._simplify(value, allow_ref)

    def _array(self, schema: dict[str, Any], allow_ref: bool) -> None:
        if prefix_items := schema.get('prefixItems'):
            # TODO I think this not is supported by Gemini, maybe we should raise an error?
            for prefix_item in prefix_items:
                self._simplify(prefix_item, allow_ref)

        if items_schema := schema.get('items'):  # pragma: no branch
            self._simplify(items_schema, allow_ref)
