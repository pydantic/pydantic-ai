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
    """Embedding model which wraps another embedding model.

    Does nothing on its own, used as a base class.
    """

    wrapped: EmbeddingModel
    """The underlying embedding model being wrapped."""

    def __init__(self, wrapped: EmbeddingModel | str):
        from . import infer_model

        super().__init__()
        self.wrapped = infer_model(wrapped) if isinstance(wrapped, str) else wrapped

    async def embed(
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        return await self.wrapped.embed(documents, input_type=input_type, settings=settings)

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
        return getattr(self.wrapped, item)
