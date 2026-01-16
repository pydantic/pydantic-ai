from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

try:
    from voyageai.client_async import AsyncClient
    from voyageai.error import VoyageError
except ImportError as _import_error:
    raise ImportError(
        'Please install `voyageai` to use the VoyageAI embeddings model, '
        'you can use the `voyageai` optional group — `pip install "pydantic-ai-slim[voyageai]"`'
    ) from _import_error

LatestVoyageAIEmbeddingModelNames = Literal[
    'voyage-4-large',
    'voyage-4',
    'voyage-4-lite',
    'voyage-3-large',
    'voyage-3.5',
    'voyage-3.5-lite',
    'voyage-code-3',
    'voyage-finance-2',
    'voyage-law-2',
    'voyage-code-2',
]
"""Latest VoyageAI embedding models.

See [VoyageAI Embeddings](https://docs.voyageai.com/docs/embeddings)
for available models and their capabilities.
"""

VoyageAIEmbeddingModelName = str | LatestVoyageAIEmbeddingModelNames
"""Possible VoyageAI embedding model names."""

VoyageAIEmbedInputType = Literal['query', 'document', 'none']
"""VoyageAI embedding input types.

- `'query'`: For search queries; prepends retrieval-optimized prefix.
- `'document'`: For documents; prepends document retrieval prefix.
- `'none'`: Direct embedding without any prefix.
"""


class VoyageAIEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a VoyageAI embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus VoyageAI-specific settings prefixed with `voyageai_`.
    """

    # ALL FIELDS MUST BE `voyageai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    voyageai_input_type: VoyageAIEmbedInputType
    """The VoyageAI-specific input type for the embedding.

    Overrides the standard `input_type` argument. Options include:
    `'query'`, `'document'`, or `'none'` for direct embedding without prefix.
    """


_MAX_INPUT_TOKENS: dict[VoyageAIEmbeddingModelName, int] = {
    'voyage-4-large': 32000,
    'voyage-4': 32000,
    'voyage-4-lite': 32000,
    'voyage-3-large': 32000,
    'voyage-3.5': 32000,
    'voyage-3.5-lite': 32000,
    'voyage-code-3': 32000,
    'voyage-finance-2': 32000,
    'voyage-law-2': 16000,
    'voyage-code-2': 16000,
}


@dataclass(init=False)
class VoyageAIEmbeddingModel(EmbeddingModel):
    """VoyageAI embedding model implementation.

    VoyageAI provides state-of-the-art embedding models optimized for
    retrieval, with specialized models for code, finance, and legal domains.

    Example:
    ```python
    from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingModel

    model = VoyageAIEmbeddingModel('voyage-3.5')
    ```
    """

    _model_name: VoyageAIEmbeddingModelName = field(repr=False)
    _provider: Provider[AsyncClient] = field(repr=False)

    def __init__(
        self,
        model_name: VoyageAIEmbeddingModelName,
        *,
        provider: Literal['voyageai'] | Provider[AsyncClient] = 'voyageai',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize a VoyageAI embedding model.

        Args:
            model_name: The name of the VoyageAI model to use.
                See [VoyageAI models](https://docs.voyageai.com/docs/embeddings)
                for available options.
            provider: The provider to use for authentication and API access. Can be:

                - `'voyageai'` (default): Uses the standard VoyageAI API
                - A [`VoyageAIProvider`][pydantic_ai.providers.voyageai.VoyageAIProvider] instance
                  for custom configuration
            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str:
        """The base URL for the provider API."""
        return self._provider.base_url

    @property
    def model_name(self) -> VoyageAIEmbeddingModelName:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return self._provider.name

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        settings = cast(VoyageAIEmbeddingSettings, settings)

        voyageai_input_type: VoyageAIEmbedInputType = settings.get(
            'voyageai_input_type', 'document' if input_type == 'document' else 'query'
        )
        # Convert 'none' string to None for the API
        api_input_type = None if voyageai_input_type == 'none' else voyageai_input_type

        try:
            response = await self._provider.client.embed(
                texts=list(inputs),
                model=self.model_name,
                input_type=api_input_type,
                truncation=settings.get('truncate', False),
                output_dimension=settings.get('dimensions'),
            )
        except VoyageError as e:
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

        return EmbeddingResult(
            embeddings=response.embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=_map_usage(response.total_tokens),
            model_name=self.model_name,
            provider_name=self.system,
        )

    async def max_input_tokens(self) -> int | None:
        return _MAX_INPUT_TOKENS.get(self.model_name)


def _map_usage(total_tokens: int) -> RequestUsage:
    return RequestUsage(input_tokens=total_tokens)
