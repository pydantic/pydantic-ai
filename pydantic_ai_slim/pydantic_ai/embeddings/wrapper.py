from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

if TYPE_CHECKING:
    pass


@dataclass(init=False)
class WrapperEmbeddingModel(EmbeddingModel):
    """Base class for embedding models that wrap another model.

    Use this as a base class to create custom embedding model wrappers
    that modify behavior (e.g., caching, logging, rate limiting) while
    delegating to an underlying model.

    By default, all methods are passed through to the wrapped model.
    Override specific methods to customize behavior.
    """

    wrapped: EmbeddingModel
    """The underlying embedding model being wrapped."""

    def __init__(self, wrapped: EmbeddingModel | str):
        """Initialize the wrapper with an embedding model.

        Args:
            wrapped: The model to wrap. Can be an
                [`EmbeddingModel`][pydantic_ai.embeddings.EmbeddingModel] instance
                or a model name string (e.g., `'openai:text-embedding-3-small'`).
        """
        from . import infer_embedding_model

        super().__init__()
        self.wrapped = infer_embedding_model(wrapped) if isinstance(wrapped, str) else wrapped

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        return await self.wrapped.embed(inputs, input_type=input_type, settings=settings)

    async def max_input_tokens(self) -> int | None:
        return await self.wrapped.max_input_tokens()

    async def count_tokens(self, text: str) -> int:
        return await self.wrapped.count_tokens(text)

    @property
    def model_name(self) -> str:
        return self.wrapped.model_name

    @property
    def system(self) -> str:
        return self.wrapped.system

    @property
    def settings(self) -> EmbeddingSettings | None:
        """Get the settings from the wrapped embedding model."""
        return self.wrapped.settings

    @property
    def base_url(self) -> str | None:
        return self.wrapped.base_url

    def __getattr__(self, item: str):
        return getattr(self.wrapped, item)  # pragma: no cover
