from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, overload

from pydantic_ai.embeddings.base import EmbeddingModel, EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.providers import Provider, infer_provider

from . import OpenAIEmbeddingsCompatibleProvider

try:
    from openai import AsyncOpenAI
    from openai.types import EmbeddingModel as LatestOpenAIEmbeddingModelNames

    from pydantic_ai.models.openai import OMIT
except ImportError as _import_error:
    raise ImportError(
        'Please install `openai` to use the OpenAI embeddings model, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

OpenAIEmbeddingModelName = str | LatestOpenAIEmbeddingModelNames
"""Possible OpenAI embeddings model names."""


class OpenAIEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for an OpenAI embedding model request."""

    # ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.


@dataclass(init=False)
class OpenAIEmbeddingModel(EmbeddingModel):
    """OpenAI embedding model."""

    _model_name: OpenAIEmbeddingModelName = field(repr=False)
    _provider: Provider[AsyncOpenAI] = field(repr=False)

    def __init__(
        self,
        model_name: OpenAIEmbeddingModelName,
        *,
        provider: OpenAIEmbeddingsCompatibleProvider | Literal['openai'] | Provider[AsyncOpenAI] = 'openai',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize an OpenAI embedding model.

        Args:
            model_name: The name of the OpenAI model to use. List of model names
                available [here](https://platform.openai.com/docs/guides/embeddings#embedding-models).
            provider: The provider to use for authentication and API access. Can be either the string
                'openai' or an instance of `Provider[AsyncOpenAI]`. If not provided, a new provider will be
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
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        documents, is_single_document, settings = self.prepare_embed(documents, settings)
        # API doesn't currently distinguish between query and document input types
        embeddings = await self._embed(documents, settings)
        return embeddings[0] if is_single_document else embeddings

    async def _embed(self, documents: Sequence[str], settings: OpenAIEmbeddingSettings) -> list[list[float]]:
        response = await self._client.embeddings.create(
            input=documents,  # pyright: ignore[reportArgumentType]  # Sequence[str] not compatible with SequenceNotStr[str] :/
            model=self.model_name,
            dimensions=settings.get('dimensions') or OMIT,
            extra_headers=settings.get('extra_headers'),
            extra_body=settings.get('extra_body'),
        )
        return [item.embedding for item in response.data]
