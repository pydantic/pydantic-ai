from __future__ import annotations as _annotations

from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import override

from ..native_tools import AbstractNativeTool, WebSearchTool
from ..profiles import ModelProfileSpec
from ..providers import Provider
from ..providers.perplexity import PerplexityProvider
from ..settings import ModelSettings
from . import ModelRequestParameters

try:
    from openai import AsyncOpenAI
    from openai.types import chat
    from openai.types.chat.completion_create_params import WebSearchOptions

    from .openai import OpenAIChatModel, _ChatCompletion  # pyright: ignore[reportPrivateUsage]
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the Perplexity model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class _PerplexitySearchResult(BaseModel):
    title: str | None = None
    url: str | None = None
    date: str | None = None
    last_updated: str | None = None
    snippet: str | None = None
    source: str | None = None


class _PerplexityChatCompletion(_ChatCompletion):
    citations: list[str] | None = None
    search_results: list[_PerplexitySearchResult] | None = None


class PerplexityModel(OpenAIChatModel):
    """A Perplexity Sonar chat model using the OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        model_name: str,
        *,
        provider: Literal['perplexity'] | Provider[AsyncOpenAI] = 'perplexity',
        profile: ModelProfileSpec | None = None,
        settings: ModelSettings | None = None,
    ):
        """Initialize a Perplexity model."""
        super().__init__(
            model_name,
            provider=provider or PerplexityProvider(),
            profile=profile,
            settings=settings,
        )

    @classmethod
    @override
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the native tool types this model can handle.

        Perplexity performs web search natively for Sonar chat requests.
        """
        return frozenset({WebSearchTool})

    @override
    def _get_web_search_options(self, model_request_parameters: ModelRequestParameters) -> WebSearchOptions | None:
        """Perplexity handles search natively, not via the OpenAI `web_search_options` parameter."""
        return None

    @override
    def _validate_completion(self, response: chat.ChatCompletion) -> _PerplexityChatCompletion:
        return _PerplexityChatCompletion.model_validate(response.model_dump())

    @override
    def _process_provider_details(self, response: chat.ChatCompletion) -> dict[str, Any] | None:
        # `_validate_completion` has already parsed the response into `_PerplexityChatCompletion`.
        assert isinstance(response, _PerplexityChatCompletion)

        provider_details = super()._process_provider_details(response) or {}
        if response.citations:
            provider_details['citations'] = response.citations
        if response.search_results:
            provider_details['search_results'] = [
                search_result.model_dump(exclude_none=True) for search_result in response.search_results
            ]

        return provider_details or None
