from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast, overload

import pydantic_ai._utils as _utils
from pydantic_ai.embeddings.base import EmbeddingModel, EmbedInputType
from pydantic_ai.embeddings.settings import EmbeddingSettings
from pydantic_ai.exceptions import UnexpectedModelBehavior

try:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as _import_error:
    raise ImportError(
        'Please install `sentence-transformers` to use the Sentence-Transformers embeddings model, '
        'you can use the `sentence-transformers` optional group — '
        'pip install "pydantic-ai-slim[sentence-transformers]"'
    ) from _import_error


class SentenceTransformersEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Sentence-Transformers embedding model request.

    All fields are `sentence_transformers_`-prefixed so settings can be merged across providers safely.
    """

    sentence_transformers_device: str
    """Device to run inference on, e.g. "cpu", "cuda", "cuda:0", "mps"."""

    sentence_transformers_normalize_embeddings: bool
    """Whether to L2-normalize embeddings. Mirrors `normalize_embeddings` in SentenceTransformer.encode."""

    sentence_transformers_batch_size: int
    """Batch size to use during encoding."""


@dataclass(init=False)
class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Local embeddings using `sentence-transformers` models.

    Example models include "all-MiniLM-L6-v2" and many others hosted on Hugging Face.
    """

    _model_name: str = field(repr=False)
    _model: SentenceTransformer = field(repr=False)

    def __init__(self, model_name: str, *, settings: EmbeddingSettings | None = None) -> None:
        """Initialize a Sentence-Transformers embedding model.

        Args:
            model_name: The model name or local path to load with `SentenceTransformer`.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name
        # Defer device selection to encode() where we can override via settings
        self._model = SentenceTransformer(model_name)

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str | None:
        """No base URL — runs locally."""
        return None

    @property
    def model_name(self) -> str:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider/system identifier."""
        return 'sentence-transformers'

    @overload
    async def embed(
        self, documents: str, *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float]: ...

    @overload
    async def embed(
        self, documents: Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]: ...

    async def embed(
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        docs, is_single_document, settings = self.prepare_embed(documents, settings)
        embeddings = await self._embed(docs, input_type, cast(SentenceTransformersEmbeddingSettings, settings))
        return embeddings[0] if is_single_document else embeddings

    async def _embed(
        self, documents: Sequence[str], input_type: EmbedInputType, settings: SentenceTransformersEmbeddingSettings
    ) -> list[list[float]]:
        device = settings.get('sentence_transformers_device', None)
        normalize = settings.get('sentence_transformers_normalize_embeddings', False)
        batch_size = settings.get('sentence_transformers_batch_size', None)

        # TODO: Update /typings so we can remove the type ignores
        encode_func = self._model.encode_query if input_type == 'query' else self._model.encode_document  # type: ignore[reportUnknownReturnType]

        np_embeddings: np.ndarray[Any, float] = await _utils.run_in_executor(  # type: ignore[reportAssignmentType]
            encode_func,  # type: ignore[reportArgumentType]
            documents if isinstance(documents, str) else list(documents),
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=device,
            normalize_embeddings=normalize,
            **{'batch_size': batch_size} if batch_size is not None else {},  # type: ignore[reportArgumentType]
        )
        return np_embeddings.tolist()

    async def max_input_tokens(self) -> int | None:
        return self._model.get_max_seq_length()

    async def count_tokens(self, text: str) -> int:
        result: dict[str, torch.Tensor] = await _utils.run_in_executor(
            self._model.tokenize,  # type: ignore[reportArgumentType]
            [text],
        )
        if 'input_ids' not in result or not isinstance(result['input_ids'], torch.Tensor):
            raise UnexpectedModelBehavior(
                'The SentenceTransformers tokenizer output did not have an `input_ids` field holding a tensor',
                str(result),
            )
        return len(result['input_ids'][0])
