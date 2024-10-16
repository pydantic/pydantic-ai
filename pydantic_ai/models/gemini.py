"""
The Google SDK for interacting with the `generativelanguage.googleapis.com` API
[`google-generativeai`](https://ai.google.dev/gemini-api/docs/quickstart?lang=python) reads like it was written by a
Java developer who thought they knew everything about OOP, spent 30 minutes trying to learn Python,
gave up and decided to build the library to prove how horrible Python is. It also doesn't use httpx for HTTP requests,
and it tries to implement tool calling itself, but doesn't use Pydantic or equivalent for validation.

We could also use the Google Vertex SDK,
[`google-cloud-aiplatform`](https://cloud.google.com/vertex-ai/docs/python-sdk/use-vertex-ai-python-sdk)
which uses the `*-aiplatform.googleapis.com` API, but that requires a service account for authentication
which is a faff to set up and manage. The big advantages of `*-aiplatform.googleapis.com` is that it claims API
compatibility with OpenAI's API, but I suspect Gemini's limited support for JSON Schema means you'd need to
hack around its limitations anyway for tool calls.

This code is a custom interface to the `generativelanguage.googleapis.com` API using httpx, Pydantic
and just a little bit of Python knowledge.
"""

from __future__ import annotations as _annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, assert_never, cast

from httpx import AsyncClient as AsyncHTTPClient
from pydantic import Field, TypeAdapter

from .. import _utils
from ..messages import (
    ArgsObject,
    LLMMessage,
    LLMResponse,
    LLMToolCalls,
    Message,
    ToolCall,
    ToolRetry,
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

    def __init__(
        self,
        model_name: GeminiModelName,
        *,
        api_key: str | None = None,
        http_client: AsyncHTTPClient | None = None,
    ):
        self.model_name = model_name
        if api_key is None:
            if env_api_key := os.getenv('GEMINI_API_KEY'):
                api_key = env_api_key
            else:
                raise ValueError('API key must be provided or set in the GEMINI_API_KEY environment variable')
        self.api_key = api_key
        self.http_client = http_client or cached_async_http_client()

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tool: AbstractToolDefinition | None,
    ) -> AgentModel:
        tools = [GeminiFunction.from_abstract_tool(t) for t in retrievers.values()]
        if result_tool:
            tools.append(GeminiFunction.from_abstract_tool(result_tool))
            tool_config = GeminiToolConfig.call_required([t.name for t in tools])
        else:
            tool_config = None

        return GeminiAgentModel(
            http_client=self.http_client,
            model_name=self.model_name,
            api_key=self.api_key,
            tools=GeminiTools(function_declarations=tools) if tools else None,
            tool_config=tool_config,
        )


@dataclass
class GeminiAgentModel(AgentModel):
    http_client: AsyncHTTPClient
    model_name: GeminiModelName
    api_key: str
    tools: GeminiTools | None
    tool_config: GeminiToolConfig | None
    # https://ai.google.dev/gemini-api/docs/quickstart?lang=rest#make-first-request
    url_template: str = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

    async def request(self, messages: list[Message]) -> LLMMessage:
        response = await self.make_request(messages)
        return self.process_response(response)

    async def make_request(self, messages: list[Message]) -> GeminiResponse:
        contents: list[GeminiContent] = []
        sys_prompt_parts: list[GeminiTextPart] = []
        for m in messages:
            either_content = self.message_to_gemini(m)
            if left := either_content.left:
                sys_prompt_parts.append(left.value)
            else:
                contents.append(either_content.right)

        request_data = GeminiRequest(
            contents=contents,
            system_instruction=GeminiTextContent(role='user', parts=sys_prompt_parts) if sys_prompt_parts else None,
            tools=self.tools if self.tools is not None else None,
            tool_config=self.tool_config if self.tool_config is not None else None,
        )
        request_json = gemini_request_ta.dump_json(request_data, exclude_none=True, by_alias=True)
        # https://cloud.google.com/docs/authentication/api-keys-use#using-with-rest
        headers = {
            'X-Goog-Api-Key': self.api_key,
            'Content-Type': 'application/json',
        }
        url = self.url_template.format(model=self.model_name)
        response = await self.http_client.post(url, content=request_json, headers=headers)
        if response.status_code != 200:
            # TODO better custom error
            raise RuntimeError(f'Error {response.status_code}: {response.text}')
        return gemini_response_ta.validate_json(response.content)

    @staticmethod
    def process_response(response: GeminiResponse) -> LLMMessage:
        assert len(response.candidates) == 1, 'Expected exactly one candidate'
        parts = response.candidates[0].content.parts
        if all(isinstance(part, GeminiFunctionCallPart) for part in parts):
            parts = cast(list[GeminiFunctionCallPart], parts)
            calls = [ToolCall.from_object(part.function_call.name, part.function_call.args) for part in parts]
            return LLMToolCalls(calls)
        elif all(isinstance(part, GeminiTextPart) for part in parts):
            parts = cast(list[GeminiTextPart], parts)
            return LLMResponse(content=''.join(part.text for part in parts))
        else:
            raise RuntimeError(
                f'Unexpected response from Gemini, expected all parts to be function calls or text, ' f'got: {parts}'
            )

    @staticmethod
    def message_to_gemini(m: Message) -> _utils.Either[GeminiTextPart, GeminiContent]:
        """
        Convert a message to a GeminiTextPart for "system_instructions" or GeminiContent for "contents".
        """
        if m.role == 'system':
            # SystemPrompt ->
            return _utils.Either(left=GeminiTextPart(text=m.content))
        elif m.role == 'user':
            # UserPrompt ->
            return _utils.Either(right=GeminiContent.user_text(m.content))
        elif m.role == 'tool-return':
            # ToolReturn ->
            return _utils.Either(right=GeminiContent.function_return(m))
        elif m.role == 'tool-retry':
            # ToolRetry ->
            return _utils.Either(right=GeminiContent.function_retry(m))
        elif m.role == 'plain-response-forbidden':
            return _utils.Either(right=GeminiContent.user_text(m.llm_response()))
        elif m.role == 'llm-response':
            # LLMResponse ->
            return _utils.Either(right=GeminiContent.model_text(m.content))
        elif m.role == 'llm-tool-calls':
            # LLMToolCalls ->
            return _utils.Either(right=GeminiContent.function_call(m))
        else:
            assert_never(m)


@dataclass
class GeminiRequest:
    """Schema for an API request to the Gemini API.

    See <https://ai.google.dev/api/generate-content#request-body> for API docs.
    """

    contents: list[GeminiContent]
    tools: GeminiTools | None = None
    tool_config: GeminiToolConfig | None = None
    # we don't implement `generationConfig`, instead we use a named tool for the response
    system_instruction: GeminiTextContent | None = None
    """
    Developer generated system instructions, see
    <https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest>
    """


# We use dataclasses, not typed dicts to define the Gemini API schema
# so we can include custom constructors etc.
# TypeAdapters take care of validation and serialization


@dataclass
class GeminiContent:
    role: Literal['user', 'model']
    parts: list[GeminiPartUnion]

    @classmethod
    def user_text(cls, text: str) -> GeminiContent:
        return cls(role='user', parts=[GeminiTextPart(text=text)])

    @classmethod
    def model_text(cls, text: str) -> GeminiContent:
        return cls(role='model', parts=[GeminiTextPart(text=text)])

    @classmethod
    def function_call(cls, m: LLMToolCalls) -> GeminiContent:
        parts: list[GeminiPartUnion] = [GeminiFunctionCallPart.from_call(t) for t in m.calls]
        return cls(role='model', parts=parts)

    @classmethod
    def function_return(cls, m: ToolReturn) -> GeminiContent:
        # TODO non string responses
        response = {'return_value': m.llm_response()}
        f_response = GeminiFunctionResponsePart.from_response(m.tool_name, response)
        return GeminiContent(role='user', parts=[f_response])

    @classmethod
    def function_retry(cls, m: ToolRetry) -> GeminiContent:
        response = {'call_error': m.llm_response()}
        f_response = GeminiFunctionResponsePart.from_response(m.tool_name, response)
        return GeminiContent(role='user', parts=[f_response])


@dataclass
class GeminiTextPart:
    text: str


@dataclass
class GeminiFunctionCallPart:
    function_call: Annotated[GeminiFunctionCall, Field(alias='functionCall')]

    @classmethod
    def from_call(cls, tool: ToolCall) -> GeminiFunctionCallPart:
        assert isinstance(tool.args, ArgsObject), f'Expected ArgsObject, got {tool.args}'
        return cls(function_call=GeminiFunctionCall(name=tool.tool_name, args=tool.args.args_object))


@dataclass
class GeminiFunctionCall:
    """See <https://ai.google.dev/api/caching#FunctionCall>"""

    name: str
    args: dict[str, Any]


@dataclass
class GeminiFunctionResponsePart:
    function_response: Annotated[GeminiFunctionResponse, Field(alias='functionResponse')]

    @classmethod
    def from_response(cls, name: str, response: dict[str, Any]) -> GeminiFunctionResponsePart:
        return cls(function_response=GeminiFunctionResponse(name=name, response=response))


@dataclass
class GeminiFunctionResponse:
    """See <https://ai.google.dev/api/caching#FunctionResponse>"""

    name: str
    response: dict[str, Any]


# See <https://ai.google.dev/api/caching#Part>
# we don't currently support other part types
# TODO discriminator
GeminiPartUnion = Union[GeminiTextPart, GeminiFunctionCallPart, GeminiFunctionResponsePart]


@dataclass
class GeminiTextContent:
    role: Literal['user', 'model']
    parts: list[GeminiTextPart]


@dataclass
class GeminiTools:
    function_declarations: list[GeminiFunction]


@dataclass
class GeminiFunction:
    name: str
    description: str
    parameters: _utils.ObjectJsonSchema
    """
    ObjectJsonSchema isn't really true since Gemini only accepts a subset of JSON Schema
    <https://ai.google.dev/gemini-api/docs/function-calling#function_declarations>
    """

    @classmethod
    def from_abstract_tool(cls, tool: AbstractToolDefinition) -> GeminiFunction:
        json_schema = deepcopy(tool.json_schema)
        json_schema.pop('title', None)  # pyright: ignore[reportArgumentType]
        for value in json_schema.get('properties', {}).values():
            value.pop('title', None)
        return cls(
            name=tool.name,
            description=tool.description,
            parameters=json_schema,
        )


@dataclass
class GeminiToolConfig:
    function_calling_config: GeminiFunctionCallingConfig

    @classmethod
    def call_required(cls, function_names: list[str]) -> GeminiToolConfig:
        return cls(
            function_calling_config=GeminiFunctionCallingConfig(mode='ANY', allowed_function_names=function_names)
        )


@dataclass
class GeminiFunctionCallingConfig:
    mode: Literal['ANY', 'AUTO']
    allowed_function_names: list[str]


@dataclass
class GeminiResponse:
    """
    Schema for the response from the Gemini API.

    See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>
    """

    candidates: list[GeminiCandidates]
    usage_metadata: Annotated[GeminiUsageMetaData, Field(alias='usageMetadata')]
    prompt_feedback: Annotated[GeminiPromptFeedback | None, Field(alias='promptFeedback')] = None


@dataclass
class GeminiCandidates:
    content: GeminiContent
    finish_reason: Annotated[Literal['STOP'], Field(alias='finishReason')]
    """
    See https://ai.google.dev/api/generate-content#FinishReason, lots of other values are possible,
    but let's wait until we see them and know what they mean to add them here.
    """
    index: int
    safety_ratings: Annotated[list[GeminiSafetyRating], Field(alias='safetyRatings')]


@dataclass
class GeminiUsageMetaData:
    prompt_token_count: Annotated[int, Field(alias='promptTokenCount')]
    candidates_token_count: Annotated[int, Field(alias='candidatesTokenCount')]
    total_token_count: Annotated[int, Field(alias='totalTokenCount')]
    cached_content_token_count: Annotated[int | None, Field(alias='cachedContentTokenCount')] = None


@dataclass
class GeminiSafetyRating:
    """See https://ai.google.dev/gemini-api/docs/safety-settings#safety-filters"""

    category: Literal[
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
        'HARM_CATEGORY_CIVIC_INTEGRITY',
    ]
    probability: Literal['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH']


@dataclass
class GeminiPromptFeedback:
    """See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>"""

    block_reason: Annotated[str, Field(alias='blockReason')]
    safety_ratings: Annotated[list[GeminiSafetyRating], Field(alias='safetyRatings')]


gemini_request_ta = TypeAdapter(GeminiRequest)
gemini_response_ta = TypeAdapter(GeminiResponse)
