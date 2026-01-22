from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, cast

import pydantic_ai._utils as _utils
from pydantic_ai.exceptions import UnexpectedModelBehavior

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

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

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus Sentence-Transformers-specific settings prefixed with `sentence_transformers_`.
    """

    sentence_transformers_device: str
    """Device to run inference on.

    Examples: `'cpu'`, `'cuda'`, `'cuda:0'`, `'mps'` (Apple Silicon).
    """

    sentence_transformers_normalize_embeddings: bool
    """Whether to L2-normalize embeddings.

    When `True`, all embeddings will have unit length, which is useful for
    cosine similarity calculations.
    """

    sentence_transformers_batch_size: int
    """Batch size to use during encoding.

    Larger batches may be faster but require more memory.
    """


@dataclass(init=False)
class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Local embedding model using the `sentence-transformers` library.

    This model runs embeddings locally on your machine, which is useful for:

    - Privacy-sensitive applications where data shouldn't leave your infrastructure
    - Reducing API costs for high-volume embedding workloads
    - Offline or air-gapped environments

    Models are downloaded from Hugging Face on first use.
    See the [Sentence-Transformers documentation](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
    for available models.

    Example:
    ```python
    from sentence_transformers import SentenceTransformer

    from pydantic_ai.embeddings.sentence_transformers import (
        SentenceTransformerEmbeddingModel,
    )

    # Using a model name (downloads from Hugging Face)
    model = SentenceTransformerEmbeddingModel('all-MiniLM-L6-v2')

    # Using an existing SentenceTransformer instance
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    model = SentenceTransformerEmbeddingModel(st_model)
    ```
    """

    _model_name: str = field(repr=False)
    _model: SentenceTransformer | None = field(repr=False, default=None)

    def __init__(self, model: SentenceTransformer | str, *, settings: EmbeddingSettings | None = None) -> None:
        """Initialize a Sentence-Transformers embedding model.

        Args:
            model: The model to use. Can be:

                - A model name from Hugging Face (e.g., `'all-MiniLM-L6-v2'`)
                - A local path to a saved model
                - An existing `SentenceTransformer` instance
            settings: Model-specific
                [`SentenceTransformersEmbeddingSettings`][pydantic_ai.embeddings.sentence_transformers.SentenceTransformersEmbeddingSettings]
                to use as defaults for this model.
        """
        if isinstance(model, str):
            self._model_name = model
        else:
            self._model = deepcopy(model)
            self._model_name = model.model_card_data.model_id or model.model_card_data.base_model or 'unknown'

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

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        settings = cast(SentenceTransformersEmbeddingSettings, settings)

        device = settings.get('sentence_transformers_device', None)
        normalize = settings.get('sentence_transformers_normalize_embeddings', False)
        batch_size = settings.get('sentence_transformers_batch_size', None)
        dimensions = settings.get('dimensions', None)

        model = await self._get_model()
        encode_func = model.encode_query if input_type == 'query' else model.encode_document  # type: ignore[reportUnknownReturnType]

        np_embeddings: np.ndarray[Any, float] = await _utils.run_in_executor(  # type: ignore[reportAssignmentType]
            encode_func,  # type: ignore[reportArgumentType]
            inputs,
            show_progress_bar=False,
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=device,
            normalize_embeddings=normalize,
            truncate_dim=dimensions,
            **{'batch_size': batch_size} if batch_size is not None else {},  # type: ignore[reportArgumentType]
        )
        embeddings = np_embeddings.tolist()

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
        )

    async def max_input_tokens(self) -> int | None:
        model = await self._get_model()
        return model.get_max_seq_length()

    async def count_tokens(self, text: str) -> int:
        model = await self._get_model()
        result: dict[str, torch.Tensor] = await _utils.run_in_executor(
            model.tokenize,  # type: ignore[reportArgumentType]
            [text],
        )
        if 'input_ids' not in result or not isinstance(result['input_ids'], torch.Tensor):  # pragma: no cover
            raise UnexpectedModelBehavior(
                'The SentenceTransformers tokenizer output did not have an `input_ids` field holding a tensor',
                str(result),
            )
        return len(result['input_ids'][0])

    async def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            # This may download the model from Hugging Face, so we do it in a thread
            self._model = await _utils.run_in_executor(SentenceTransformer, self.model_name)  # pragma: no cover
        return self._model
