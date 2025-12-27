from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel
from .result import EmbeddingResult, EmbedInputType
from .settings import EmbeddingSettings

try:
    from voyageai.client_async import AsyncClient
    from voyageai.error import VoyageError
except ImportError as _import_error:
    raise ImportError(
        'Please install `voyageai` to use the VoyageAI embeddings model, you can use — `pip install voyageai`'
    ) from _import_error

LatestVoyageAIEmbeddingModelNames = Literal[
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


class VoyageAIEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a VoyageAI embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus VoyageAI-specific settings prefixed with `voyageai_`.
    """

    # ALL FIELDS MUST BE `voyageai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    voyageai_truncation: bool
    """Whether to truncate inputs that exceed the model's context length.

    Defaults to True. If False, an error is raised for inputs that are too long.
    """

    voyageai_output_dtype: Literal['float', 'int8', 'uint8', 'binary', 'ubinary']
    """The output data type for embeddings.

    - `'float'` (default): 32-bit floats
    - `'int8'`: Signed 8-bit integers (quantized)
    - `'uint8'`: Unsigned 8-bit integers (quantized)
    - `'binary'`: Binary embeddings
    - `'ubinary'`: Unsigned binary embeddings
    """


_MAX_INPUT_TOKENS: dict[VoyageAIEmbeddingModelName, int] = {
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
    _client: AsyncClient = field(repr=False)

    def __init__(
        self,
        model_name: VoyageAIEmbeddingModelName,
        *,
        api_key: str | None = None,
        max_retries: int = 0,
        timeout: int | None = None,
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize a VoyageAI embedding model.

        Args:
            model_name: The name of the VoyageAI model to use.
                See [VoyageAI models](https://docs.voyageai.com/docs/embeddings)
                for available options.
            api_key: The VoyageAI API key. If not provided, uses the
                `VOYAGE_API_KEY` environment variable.
            max_retries: Maximum number of retries for failed requests.
            timeout: Request timeout in seconds.
            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name
        self._client = AsyncClient(
            api_key=api_key,
            max_retries=max_retries,
            timeout=timeout,
        )

        super().__init__(settings=settings)

    @property
    def model_name(self) -> VoyageAIEmbeddingModelName:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return 'voyageai'

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        settings = cast(VoyageAIEmbeddingSettings, settings)

        voyageai_input_type = 'document' if input_type == 'document' else 'query'

        try:
            response = await self._client.embed(
                texts=list(inputs),
                model=self.model_name,
                input_type=voyageai_input_type,
                truncation=settings.get('voyageai_truncation', True),
                output_dtype=settings.get('voyageai_output_dtype', 'float'),
                output_dimension=settings.get('dimensions'),
            )
        except VoyageError as e:
            raise ModelAPIError(model_name=self.model_name, message=str(e)) from e

        return EmbeddingResult(
            embeddings=response.embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=_map_usage(response.total_tokens, self.model_name),
            model_name=self.model_name,
            provider_name=self.system,
        )

    async def max_input_tokens(self) -> int | None:
        return _MAX_INPUT_TOKENS.get(self.model_name)


def _map_usage(total_tokens: int, model: str) -> RequestUsage:
    usage_data = {'total_tokens': total_tokens}
    response_data = {'model': model, 'usage': usage_data}

    return RequestUsage.extract(
        response_data,
        provider='voyageai',
        provider_url='https://api.voyageai.com',
        provider_fallback='voyageai',
        api_flavor='embeddings',
    )
