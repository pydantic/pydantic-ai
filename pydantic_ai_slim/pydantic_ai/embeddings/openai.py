from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, overload

from pydantic_ai.embeddings.embedding_model import EmbeddingModel
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.providers import Provider, infer_provider

from .settings import merge_embedding_settings

try:
    from openai import NOT_GIVEN, AsyncOpenAI
    from openai.types import EmbeddingModel as LatestOpenAIEmbeddingModelNames
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI embeddings model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenAIEmbeddingModelName = str | LatestOpenAIEmbeddingModelNames
"""Possible OpenAI embeddings model names."""


@dataclass(init=False)
class OpenAIEmbeddingModel(EmbeddingModel):
    _model_name: OpenAIEmbeddingModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIEmbeddingModelName,
        *,
        provider: Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize an OpenAI model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names
                available [here](https://docs.OpenAI.com/docs/models#command).
            provider: The provider to use for authentication and API access. Can be either the string
                'OpenAI' or an instance of `Provider[AsyncClientV2]`. If not provided, a new provider will be
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
        return str(self._client.base_url)

    @property
    def model_name(self) -> OpenAIEmbeddingModelName:
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
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        input_is_string = isinstance(documents, str)
        if input_is_string:
            documents = [documents]

        settings = merge_embedding_settings(self._settings, settings) or {}
        response = await self._client.embeddings.create(
            input=documents,  # pyright: ignore[reportArgumentType]  # Sequence[str] not compatible with SequenceNotStr[str] :/
            model=self.model_name,
            dimensions=settings.get('output_dimension') or NOT_GIVEN,
        )
        result = [item.embedding for item in response.data]

        if input_is_string:
            return result[0]
        return result
