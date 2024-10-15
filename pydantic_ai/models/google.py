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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict, Union, assert_never

from httpx import AsyncClient as AsyncHTTPClient
from pydantic import Field, TypeAdapter
from typing_extensions import NotRequired

from .. import _utils
from ..messages import (
    LLMMessage,
    Message,
)
from . import AbstractToolDefinition, AgentModel, Model, cached_async_http_client

# https://ai.google.dev/gemini-api/docs/models/gemini#model-variations
GeminiModelName = Literal['gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-1.5-pro', 'gemini-1.0-pro']


@dataclass(init=False)
class GoogleModel(Model):
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
        self.api_key = api_key or 'TODO'
        self.http_client = http_client or cached_async_http_client()

    def agent_model(
        self,
        retrievers: Mapping[str, AbstractToolDefinition],
        allow_text_result: bool,
        result_tool: AbstractToolDefinition | None,
    ) -> AgentModel:
        # TODO
        raise NotImplementedError()


@dataclass
class OpenAIAgentModel(AgentModel):
    http_client: AsyncHTTPClient
    model_name: GeminiModelName
    api_key: str
    allow_text_result: bool
    tools: GeminiTools | None
    tool_config: GeminiToolConfig | None
    # https://ai.google.dev/gemini-api/docs/quickstart?lang=rest#make-first-request
    url_template: str = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

    async def request(self, messages: list[Message]) -> LLMMessage:
        response = await self.make_request(messages)
        return self.process_response(response)

    async def make_request(self, messages: list[Message]) -> GeminiResponse:
        contents: list[GeminiContent] = []
        system_prompt_parts: list[GeminiTextPart] = []
        for m in messages:
            either_content = self.message_to_gemini(m)
            if left := either_content.left:
                system_prompt_parts.append(left.value)
            else:
                contents.append(either_content.right)

        request_data = GeminiRequest(contents=contents)
        if system_prompt_parts:
            request_data['system_instructions'] = GeminiTextContent(role='user', parts=system_prompt_parts)
        if self.tools is not None:
            request_data['tools'] = self.tools
        if self.tool_config is not None:
            request_data['tool_config'] = self.tool_config
        request_json = gemini_request_ta.dump_json(request_data)
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

    def process_response(self, response: GeminiResponse) -> LLMMessage:
        # TODO
        raise NotImplementedError()

    @staticmethod
    def message_to_gemini(m: Message) -> _utils.Either[GeminiTextPart, GeminiContent]:
        """
        Convert a message to a GeminiTextPart for "system_instructions" or GeminiContent for "contents".
        """
        if m.role == 'system':
            # SystemPrompt ->
            return _utils.Either(left=GeminiTextPart(text=m.content))

        if m.role == 'user':
            # UserPrompt ->
            return _utils.Either(right=GeminiContent(role='user', parts=[GeminiTextPart(text=m.content)]))
        elif m.role == 'tool-return':
            # ToolReturn ->
            # TODO non string responses
            function_response: tuple[str, dict[str, Any]] = (m.tool_name, {'return_value': m.llm_response()})
        elif m.role == 'tool-retry':
            # ToolRetry ->
            function_response = (m.tool_name, {'call_error': m.llm_response()})
        elif m.role == 'plain-response-forbidden':
            return _utils.Either(right=GeminiContent(role='user', parts=[GeminiTextPart(text=m.llm_response())]))
        elif m.role == 'llm-response':
            # LLMResponse ->
            return _utils.Either(right=GeminiContent(role='model', parts=[GeminiTextPart(text=m.content)]))
        elif m.role == 'llm-tool-calls':
            # LLMToolCalls ->
            parts: list[GeminiPartUnion] = [
                GeminiFunctionCallPart(name=t.tool_name, args=json.loads(t.arguments)) for t in m.calls
            ]
            return _utils.Either(right=GeminiContent(role='model', parts=parts))
        else:
            assert_never(m)

        name, response = function_response
        f_response = GeminiFunctionResponsePart(name=name, response=response)
        return _utils.Either(right=GeminiContent(role='model', parts=[f_response]))

    @staticmethod
    def gemini_to_message(gemini: GeminiContent) -> Message:
        raise NotImplementedError()


class GeminiRequest(TypedDict):
    """Schema for an API request to the Gemini API.

    See <https://ai.google.dev/api/generate-content#request-body> for API docs.
    """

    contents: list[GeminiContent]
    tools: NotRequired[GeminiTools]
    tool_config: NotRequired[GeminiToolConfig]
    # we don't implement `generationConfig`, instead we use a named tool for the response
    system_instructions: NotRequired[Annotated[GeminiTextContent, Field(alias='systemInstructions')]]
    """
    Developer generated system instructions, see
    <https://ai.google.dev/gemini-api/docs/system-instructions?lang=rest>
    """


class GeminiContent(TypedDict):
    role: Literal['user', 'model']
    parts: list[GeminiPartUnion]


class GeminiTextPart(TypedDict):
    text: str


class GeminiFunctionCallPart(TypedDict):
    """See <https://ai.google.dev/api/caching#FunctionCall>"""

    name: str
    args: dict[str, Any]


class GeminiFunctionResponsePart(TypedDict):
    """See <https://ai.google.dev/api/caching#FunctionResponse>"""

    name: str
    response: dict[str, Any]


# we don't currently support other part types
GeminiPartUnion = Union[GeminiTextPart, GeminiFunctionCallPart, GeminiFunctionResponsePart]


class GeminiTextContent(TypedDict):
    role: Literal['user', 'model']
    parts: list[GeminiTextPart]


class GeminiTools(TypedDict):
    function_declarations: list[GeminiFunction]


class GeminiFunction(TypedDict):
    name: str
    description: str
    parameters: _utils.ObjectJsonSchema
    """
    ObjectJsonSchema isn't really true since Gemini only accepts a subset of JSON Schema
    <https://ai.google.dev/gemini-api/docs/function-calling#function_declarations>
    """


class GeminiToolConfig(TypedDict):
    function_calling_config: GeminiFunctionCallingConfig


class GeminiFunctionCallingConfig(TypedDict):
    mode: Literal['ANY', 'AUTO']
    allowed_function_names: list[str]


class GeminiResponse(TypedDict):
    """
    Schema for the response from the Gemini API.

    See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>
    """

    candidates: list[GeminiCandidates]
    usage_metadata: Annotated[GeminiUsageMetaData, Field(alias='usageMetadata')]
    prompt_feedback: Annotated[GeminiPromptFeedback, Field(alias='promptFeedback')]


class GeminiCandidates(TypedDict):
    content: GeminiContent
    finish_reason: Annotated[Literal['STOP'], Field(alias='finishReason')]
    """
    See https://ai.google.dev/api/generate-content#FinishReason, lots of other values are possible,
    but let's wait until we see them and know what they mean to add them here.
    """
    index: int
    safety_ratings: Annotated[list[GeminiSafetyRating], Field(alias='safetyRatings')]


class GeminiUsageMetaData(TypedDict):
    prompt_token_count: Annotated[int, Field(alias='promptTokenCount')]
    candidate_token_count: Annotated[int, Field(alias='candidateTokenCount')]
    cached_content_token_count: Annotated[int, Field(alias='cachedContentTokenCount')]
    total_token_count: Annotated[int, Field(alias='totalTokenCount')]


class GeminiSafetyRating(TypedDict):
    """See https://ai.google.dev/gemini-api/docs/safety-settings#safety-filters"""

    category: Literal[
        'HARM_CATEGORY_HARASSMENT',
        'HARM_CATEGORY_HATE_SPEECH',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'HARM_CATEGORY_DANGEROUS_CONTENT',
        'HARM_CATEGORY_CIVIC_INTEGRITY',
    ]
    probability: Literal['NEGLIGIBLE', 'LOW', 'MEDIUM', 'HIGH']


class GeminiPromptFeedback(TypedDict):
    """See <https://ai.google.dev/api/generate-content#v1beta.GenerateContentResponse>"""

    block_reason: Annotated[str, Field(alias='blockReason')]
    safety_ratings: Annotated[list[GeminiSafetyRating], Field(alias='safetyRatings')]


gemini_request_ta = TypeAdapter(GeminiRequest)
gemini_response_ta = TypeAdapter(GeminiResponse)
