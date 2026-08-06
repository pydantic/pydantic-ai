"""Vercel AI Gateway model implementation using the OpenAI-compatible API."""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, Literal

from typing_extensions import override

from ..profiles import ModelProfileSpec
from ..settings import ModelSettings
from ..usage import RequestUsage

try:
    from openai import AsyncOpenAI
    from openai.types import chat, completion_usage

    from ..providers import Provider
    from .openai import OpenAIChatModel, OpenAIStreamedResponse
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Vercel AI Gateway model, '
        'you can use the `vercel` optional group — `pip install "pydantic-ai-slim[vercel]"`'
    ) from _import_error

__all__ = ('VercelModel',)


class _VercelUsage(completion_usage.CompletionUsage):
    """Usage payload extended with the gateway's non-standard total-cost field."""

    cost: float | None = None
    """Total cost of the request in USD, as billed by the gateway."""


def _cost_details(usage_data: completion_usage.CompletionUsage | None) -> dict[str, Any]:
    if usage_data is None:
        return {}
    validated = _VercelUsage.model_validate(usage_data.model_dump())
    return {'cost': validated.cost} if validated.cost is not None else {}


@dataclass
class VercelStreamedResponse(OpenAIStreamedResponse):
    """Implementation of `StreamedResponse` for the Vercel AI Gateway."""

    @override
    def _map_usage(self, response: chat.ChatCompletionChunk) -> RequestUsage:
        # Hooked here rather than in `_map_provider_details`: the gateway can report usage on a
        # spec-shaped final chunk with empty `choices`, which the base event loop skips before
        # `_map_provider_details` is reached, while `_map_usage` sees every chunk.
        if details := _cost_details(response.usage):
            self.provider_details = {**(self.provider_details or {}), **details}
        return super()._map_usage(response)


class VercelModel(OpenAIChatModel):
    """Extends `OpenAIChatModel` to surface Vercel AI Gateway metadata.

    The gateway reports the billed cost of every request in its non-standard `usage.cost` field
    (on streamed responses too, riding the final usage chunk). The generic OpenAI mapping drops
    unknown usage fields, so this model lifts it into `ModelResponse.provider_details['cost']`,
    matching `OpenRouterModel`'s behavior for the equivalent OpenRouter field.
    """

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['vercel'] | Provider[AsyncOpenAI] = 'vercel',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Vercel AI Gateway model.

        Args:
            model_name: The name of the model to use, in `creator/model` format (e.g. `anthropic/claude-sonnet-4-5`).
            provider: The provider to use for authentication and API access. If not provided, a new provider will be created with the default settings.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        super().__init__(model_name, provider=provider, profile=profile, settings=settings)

    @override
    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        provider_details = super()._process_provider_details(response) or {}
        provider_details.update(_cost_details(response.usage))
        return provider_details or None

    @property
    @override
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return VercelStreamedResponse
