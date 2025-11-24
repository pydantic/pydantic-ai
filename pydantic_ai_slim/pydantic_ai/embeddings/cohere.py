from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast, overload

from pydantic_ai.embeddings.base import EmbeddingModel, EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.providers import Provider, infer_provider

try:
    from cohere import AsyncClientV2
    from cohere.core.request_options import RequestOptions
except ImportError as _import_error:
    raise ImportError(
        'Please install `cohere` to use the Cohere embeddings model, '
        'you can use the `cohere` optional group — `pip install "pydantic-ai-slim[cohere]"`'
    ) from _import_error

LatestCohereEmbeddingModelNames = Literal[
    'cohere:embed-v4.0',
    'embed-english-v3.0embed-english-light-v3.0',
    'embed-multilingual-v3.0',
    'embed-multilingual-light-v3.0',
]
"""Latest Cohere embeddings models."""

CohereEmbeddingModelName = str | LatestCohereEmbeddingModelNames
"""Possible Cohere embeddings model names."""


class CohereEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Cohere embedding model request."""

    # ALL FIELDS MUST BE `cohere_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    # TODO: Possibly move to base EmbeddingSettings if supported by more providers
    cohere_max_tokens: int
    """The maximum number of tokens to generate before stopping."""

    # We don't support `embedding_types for now because it doesn't affect the user-facing API today..
    # cohere_embedding_types: Literal["float", "int8", "uint8", "binary", "ubinary", "base64"]


@dataclass(init=False)
class CohereEmbeddingModel(EmbeddingModel):
    """Cohere embedding model."""

    _model_name: CohereEmbeddingModelName = field(repr=False)
    _provider: Provider[AsyncClientV2] = field(repr=False)

    def __init__(
        self,
        model_name: CohereEmbeddingModelName,
        *,
        provider: Literal['cohere'] | Provider[AsyncClientV2] = 'cohere',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize an Cohere model.

        Args:
            model_name: The name of the Cohere model to use. List of model names
                available [here](https://docs.cohere.com/docs/cohere-embed).
            provider: The provider to use for authentication and API access. Can be either the string
                'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
                created using the other parameters.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self._client = provider.client

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str:
        """The base URL for the provider API, if available."""
        return self._provider.base_url

    @property
    def model_name(self) -> CohereEmbeddingModelName:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return self._provider.name

    @overload
    async def embed(
        self, documents: str, *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float]:
        pass

    @overload
    async def embed(
        self, documents: Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]:
        pass

    async def embed(
        self, documents: Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        documents, is_single_document, settings = self.prepare_embed(documents, settings)
        embeddings = await self._embed(documents, input_type, cast(CohereEmbeddingSettings, settings))
        return embeddings[0] if is_single_document else embeddings

    async def _embed(
        self, documents: Sequence[str], input_type: EmbedInputType, settings: CohereEmbeddingSettings
    ) -> list[list[float]]:
        request_options = RequestOptions()
        if extra_headers := settings.get('extra_headers'):
            request_options['additional_headers'] = extra_headers
        if extra_body := settings.get('extra_body'):
            request_options['additional_body_parameters'] = cast(dict[str, Any], extra_body)

        response = await self._client.embed(
            model=self.model_name,
            texts=documents,
            output_dimension=settings.get('dimensions'),
            input_type='search_document' if input_type == 'document' else 'search_query',
            max_tokens=settings.get('cohere_max_tokens'),
            request_options=request_options,
        )
        embeddings = response.embeddings.float_
        assert embeddings is not None, 'This is a bug in cohere?'

        return embeddings
