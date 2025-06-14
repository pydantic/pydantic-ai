from typing import Any, Literal, overload

from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from .. import ModelHTTPError
from ..messages import ModelMessage, ModelResponse
from ..profiles import ModelProfileSpec
from ..providers.openrouter import OpenRouterProvider
from . import ModelRequestParameters
from .openai import OpenAIModel, OpenAIModelName, OpenAIModelSettings, OpenAISystemPromptRole


class OpenRouterErrorResponse(BaseModel):
    """Represents error responses from upstream LLM provider relayed by OpenRouter.

    Attributes:
        code: The error code returned by LLM provider.
        message: The error message returned by OpenRouter
        metadata: Additional error context provided by OpenRouter.

    See: https://openrouter.ai/docs/api-reference/errors
    """

    code: int
    message: str
    metadata: dict[str, Any] | None


class OpenRouterChatCompletion(ChatCompletion):
    """Extends ChatCompletion with OpenRouter-specific attributes.

    This class extends the base ChatCompletion model to include additional
    fields returned specifically by the OpenRouter API.

    Attributes:
        provider: The name of the upstream LLM provider (e.g., "Anthropic",
            "OpenAI", etc.) that processed the request through OpenRouter.
    """

    provider: str


class OpenRouterModel(OpenAIModel):
    """Extends OpenAIModel to capture extra metadata for Openrouter."""

    def __init__(
        self,
        model_name: OpenAIModelName,
        *,
        provider: Literal['openrouter'] | OpenRouterProvider = 'openrouter',
        profile: ModelProfileSpec | None = None,
        system_prompt_role: OpenAISystemPromptRole | None = None,
    ):
        super().__init__(model_name, provider=provider, profile=profile, system_prompt_role=system_prompt_role)

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[True],
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncStream[ChatCompletionChunk]: ...

    @overload
    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: Literal[False],
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletion: ...

    async def _completions_create(
        self,
        messages: list[ModelMessage],
        stream: bool,
        model_settings: OpenAIModelSettings,
        model_request_parameters: ModelRequestParameters,
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        response = await super()._completions_create(
            messages=messages,
            stream=stream,
            model_settings=model_settings,
            model_request_parameters=model_request_parameters,
        )
        if error := getattr(response, 'error', None):
            parsed_error = OpenRouterErrorResponse.model_validate(error)
            raise ModelHTTPError(
                status_code=parsed_error.code, model_name=self.model_name, body=parsed_error.model_dump()
            )
        else:
            return response

    def _process_response(self, response: ChatCompletion) -> ModelResponse:
        response = OpenRouterChatCompletion.construct(**response.model_dump())
        model_response = super()._process_response(response=response)
        openrouter_provider: str | None = getattr(response, 'provider', None)
        if openrouter_provider:
            vendor_details: dict[str, Any] = model_response.vendor_details or {}
            vendor_details['provider'] = openrouter_provider
            model_response.vendor_details = vendor_details
        return model_response
