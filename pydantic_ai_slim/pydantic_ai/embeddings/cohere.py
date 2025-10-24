from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast, overload

from pydantic_ai.embeddings.embedding_model import EmbeddingModel
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.providers import Provider, infer_provider

from .settings import merge_embedding_settings

try:
    from cohere import AsyncClientV2
except ImportError as _import_error:
    raise ImportError(
        'Please install `cohere` to use the Cohere embeddings model, '
        'you can use the `cohere` optional group — `pip install "pydantic-ai-slim[cohere]"`'
    ) from _import_error

LatestCohereEmbeddingModelNames = Literal[
    'cohere:embed-v4.0',
    # TODO: Add the others
]
"""Latest Cohere embeddings models."""

CohereEmbeddingModelName = str | LatestCohereEmbeddingModelNames
"""Possible Cohere embeddings model names."""


@dataclass(init=False)
class CohereEmbeddingModel(EmbeddingModel):
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
                available [here](https://docs.cohere.com/docs/models#command).
            provider: The provider to use for authentication and API access. Can be either the string
                'cohere' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
                created using the other parameters.
            profile: The model profile to use. Defaults to a profile picked by the provider based on the model name.
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
    async def embed(self, documents: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    async def embed(self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None) -> list[list[float]]:
        pass

    async def embed(
        self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        input_is_string = isinstance(documents, str)
        if input_is_string:
            documents = [documents]

        settings = merge_embedding_settings(self._settings, settings) or {}
        response = await self._client.embed(
            model=self.model_name,
            input_type=settings.get('input_type', 'search_document'),
            texts=cast(Sequence[str], documents),
            output_dimension=settings.get('output_dimension'),
        )
        embeddings = response.embeddings.float_
        assert embeddings is not None, 'This is a bug in cohere?'

        if input_is_string:
            return embeddings[0]

        return embeddings
