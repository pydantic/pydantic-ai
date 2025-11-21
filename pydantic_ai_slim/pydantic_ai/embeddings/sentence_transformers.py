from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, overload

from pydantic_ai.embeddings.base import EmbeddingModel
from pydantic_ai.embeddings.settings import EmbeddingSettings

# Optional dependency: sentence-transformers
try:  # pragma: no cover - depends on optional install
    from sentence_transformers import SentenceTransformer
except ImportError as _import_error:  # pragma: no cover - depends on optional install
    SentenceTransformer = None
    raise ImportError(
        'Please install `sentence-transformers` to use the Sentence-Transformers embeddings model, '
        'you can use the `sentence-transformers` optional group — '
        'pip install "pydantic-ai-slim[sentence-transformers]"'
    ) from _import_error


class SentenceTransformersEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Sentence-Transformers embedding model request.

    All fields are `sentence_transformers_`-prefixed so settings can be merged across providers safely.
    """

    # Device to run inference on, e.g. "cpu", "cuda", "cuda:0", "mps".
    sentence_transformers_device: str

    # Whether to L2-normalize embeddings. Mirrors `normalize_embeddings` in SentenceTransformer.encode.
    sentence_transformers_normalize_embeddings: bool

    # Batch size to use during encoding.
    sentence_transformers_batch_size: int


@dataclass(init=False)
class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Local embeddings using `sentence-transformers` models.

    Example models include "all-MiniLM-L6-v2" and many others hosted on Hugging Face.
    """

    _model_name: str = field(repr=False)
    _model: Any = field(repr=False)

    def __init__(self, model_name: str, *, settings: EmbeddingSettings | None = None) -> None:
        """Initialize a Sentence-Transformers embedding model.

        Args:
            model_name: The model name or local path to load with `SentenceTransformer`.
            settings: Model-specific settings that will be used as defaults for this model.
        """
        self._model_name = model_name
        if SentenceTransformer is None:  # pragma: no cover - depends on optional install
            raise ImportError(
                'Please install `sentence-transformers` to use this embeddings model, '
                'you can use the `sentence-transformers` optional group — '
                'pip install "pydantic-ai-slim[sentence-transformers]"'
            )
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
    async def embed(self, documents: str, *, settings: EmbeddingSettings | None = None) -> list[float]: ...

    @overload
    async def embed(
        self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]: ...

    async def embed(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        docs, is_single_document, settings = self.prepare_embed(documents, settings)
        embeddings = await self._embed(docs, settings)
        return embeddings[0] if is_single_document else embeddings

    async def _embed(
        self, documents: Sequence[str], settings: SentenceTransformersEmbeddingSettings
    ) -> list[list[float]]:
        device = settings.get('sentence_transformers_device', None)
        normalize = settings.get('sentence_transformers_normalize_embeddings', False)
        batch_size = settings.get('sentence_transformers_batch_size')

        encode_kwargs: dict[str, Any] = {
            'show_progress_bar': False,
            'convert_to_numpy': True,
            'convert_to_tensor': False,
            'device': device,
            'normalize_embeddings': normalize,
        }
        if batch_size is not None:
            encode_kwargs['batch_size'] = batch_size

        np_embeddings = self._model.encode(list(documents), **encode_kwargs)
        return np_embeddings.tolist()
