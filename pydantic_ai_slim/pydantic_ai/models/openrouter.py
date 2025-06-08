from typing import Any, cast

from openai.types import chat
from typing_extensions import TypedDict

from .. import ModelHTTPError
from ..messages import ModelResponse
from .openai import OpenAIModel


class OpenRouterErrorResponse(TypedDict):
    """Represents error responses from upstream LLM providers returned by OpenRouter.

    Attributes:
        code: The error code returned by OpenRouter.
        message: The error message returned by OpenRouter
        metadata: Additional error context provided by OpenRouter.

    See: https://openrouter.ai/docs/api-reference/errors
    """

    code: int
    message: str
    metadata: dict[str, Any] | None


class OpenRouterChatCompletion(chat.ChatCompletion):
    """Extends ChatCompletion with OpenRouter-specific attributes.

    This class extends the base ChatCompletion model to include additional
    fields returned specifically by the OpenRouter API.

    Attributes:
        provider: The name of the upstream LLM provider (e.g., "Anthropic",
            "OpenAI", etc.) that processed the request through OpenRouter.
        error: Optional error information returned by the upstream LLM provider.
    """

    provider: str
    error: OpenRouterErrorResponse | None


class OpenRouterModel(OpenAIModel):
    """Extends OpenAIModel to capture extra metadata for Openrouter."""

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        response = cast(OpenRouterChatCompletion, response)
        if error := getattr(response, 'error', None):
            raise ModelHTTPError(status_code=error['code'], model_name=self.model_name, body=error)
        else:
            model_response = super()._process_response(response=response)
        openrouter_provider: str | None = response.provider if hasattr(response, 'provider') else None
        if openrouter_provider:
            vendor_details: dict[str, Any] = model_response.vendor_details or {}
            vendor_details['provider'] = openrouter_provider
            model_response.vendor_details = vendor_details
        return model_response
