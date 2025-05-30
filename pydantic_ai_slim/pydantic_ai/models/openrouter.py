from typing import Any, cast

from openai.types import chat

from ..messages import ModelResponse
from .openai import OpenAIModel


class OpenRouterChatCompletion(chat.ChatCompletion):
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

    def _process_response(self, response: chat.ChatCompletion) -> ModelResponse:
        response = cast(OpenRouterChatCompletion, response)
        model_response = super()._process_response(response=response)
        openrouter_provider: str | None = response.provider if hasattr(response, 'provider') else None
        if openrouter_provider:
            vendor_details: dict[str, Any] = getattr(model_response, 'vendor_details') or {}
            vendor_details['provider'] = openrouter_provider
            model_response.vendor_details = vendor_details
        return model_response
