from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel, EmbedInputType
from .result import EmbeddingResult
from .settings import EmbeddingSettings

try:
    from google.genai import Client, errors
    from google.genai.types import ContentListUnion, EmbedContentConfig, EmbedContentResponse
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google embeddings model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


LatestGoogleGLAEmbeddingModelNames = Literal['gemini-embedding-001']
"""Latest Google Gemini API (GLA) embedding models.

See the [Google Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
for available models and their capabilities.
"""

LatestGoogleVertexEmbeddingModelNames = Literal[
    'gemini-embedding-001',
    'text-embedding-005',
    'text-multilingual-embedding-002',
]
"""Latest Google Vertex AI embedding models.

See the [Vertex AI Embeddings documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
for available models and their capabilities.
"""

LatestGoogleEmbeddingModelNames = LatestGoogleGLAEmbeddingModelNames | LatestGoogleVertexEmbeddingModelNames
"""All latest Google embedding models (union of GLA and Vertex AI models)."""

GoogleEmbeddingModelName = str | LatestGoogleEmbeddingModelNames
"""Possible Google embeddings model names."""


@dataclass
class _GoogleEmbeddingModelConfig:
    max_input_tokens: int
    """Maximum number of input tokens supported by the model."""


_GOOGLE_EMBEDDING_MODELS: dict[GoogleEmbeddingModelName, _GoogleEmbeddingModelConfig] = {
    'gemini-embedding-001': _GoogleEmbeddingModelConfig(max_input_tokens=2048),
    'text-embedding-005': _GoogleEmbeddingModelConfig(max_input_tokens=2048),
    'text-multilingual-embedding-002': _GoogleEmbeddingModelConfig(max_input_tokens=2048),
}


GoogleTaskType = Literal[
    'SEMANTIC_SIMILARITY',
    'CLASSIFICATION',
    'CLUSTERING',
    'RETRIEVAL_DOCUMENT',
    'RETRIEVAL_QUERY',
    'CODE_RETRIEVAL_QUERY',
    'QUESTION_ANSWERING',
    'FACT_VERIFICATION',
]
"""Task types for Google embeddings.

Different task types optimize embeddings for specific use cases:

- `SEMANTIC_SIMILARITY`: Optimized for measuring text similarity
- `CLASSIFICATION`: Optimized for text categorization
- `CLUSTERING`: Optimized for grouping similar texts
- `RETRIEVAL_DOCUMENT`: Optimized for document indexing in search
- `RETRIEVAL_QUERY`: Optimized for search queries
- `CODE_RETRIEVAL_QUERY`: Optimized for code search queries
- `QUESTION_ANSWERING`: Optimized for QA systems
- `FACT_VERIFICATION`: Optimized for fact-checking tasks
"""


class GoogleEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Google embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus Google-specific settings prefixed with `google_`.
    """

    # ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    google_task_type: GoogleTaskType
    """The task type for the embedding.

    Overrides the automatic task type selection based on `input_type`.
    See [`GoogleTaskType`][pydantic_ai.embeddings.google.GoogleTaskType] for available options.
    """

    google_title: str
    """Optional title for the content being embedded.

    Only applicable when task_type is `RETRIEVAL_DOCUMENT`.
    """


@dataclass(init=False)
class GoogleEmbeddingModel(EmbeddingModel):
    """Google embedding model implementation.

    This model works with Google's embeddings API via the `google-genai` SDK,
    supporting both the Gemini API (Google AI Studio) and Vertex AI.

    Example:
    ```python
    from pydantic_ai.embeddings.google import GoogleEmbeddingModel
    from pydantic_ai.providers.google import GoogleProvider

    # Using Gemini API (requires GOOGLE_API_KEY env var)
    model = GoogleEmbeddingModel('gemini-embedding-001')

    # Using Vertex AI
    model = GoogleEmbeddingModel(
        'gemini-embedding-001',
        provider=GoogleProvider(vertexai=True, project='my-project', location='us-central1'),
    )
    ```
    """

    _model_name: GoogleEmbeddingModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)

    def __init__(
        self,
        model_name: GoogleEmbeddingModelName = 'gemini-embedding-001',
        *,
        provider: Literal['google-gla', 'google-vertex'] | Provider[Client] = 'google-gla',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize a Google embedding model.

        Args:
            model_name: The name of the Google model to use.
                See [Google Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
                for available models.
            provider: The provider to use for authentication and API access. Can be:

                - `'google-gla'` (default): Uses the Gemini API (Google AI Studio)
                - `'google-vertex'`: Uses Vertex AI
                - A [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] instance
                  for custom configuration
            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider
        self._client = provider.client

        super().__init__(settings=settings)

    @property
    def base_url(self) -> str:
        return self._provider.base_url

    @property
    def model_name(self) -> GoogleEmbeddingModelName:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return self._provider.name

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)
        settings = cast(GoogleEmbeddingSettings, settings)

        google_task_type = settings.get('google_task_type')
        if google_task_type is None:
            google_task_type = 'RETRIEVAL_DOCUMENT' if input_type == 'document' else 'RETRIEVAL_QUERY'

        config = EmbedContentConfig(
            task_type=google_task_type,
            output_dimensionality=settings.get('dimensions'),
            title=settings.get('google_title'),
        )

        try:
            response = await self._client.aio.models.embed_content(
                model=self._model_name,
                contents=cast(ContentListUnion, inputs),
                config=config,
            )
        except errors.APIError as e:
            if (status_code := e.code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code,
                    model_name=self._model_name,
                    body=cast(object, e.details),  # pyright: ignore[reportUnknownMemberType]
                ) from e
            raise  # pragma: no cover

        embeddings: list[list[float]] = []
        if response.embeddings:
            for emb in response.embeddings:
                if emb.values is not None:
                    embeddings.append(emb.values)

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            usage=_map_usage(response, self.system, self.base_url, self._model_name),
            model_name=self._model_name,
            provider_name=self.system,
        )

    async def max_input_tokens(self) -> int | None:
        config = _GOOGLE_EMBEDDING_MODELS.get(self._model_name)
        return config.max_input_tokens if config else None

    async def count_tokens(self, text: str) -> int:
        response = await self._client.aio.models.count_tokens(
            model=self._model_name,
            contents=text,
        )
        if response.total_tokens is None:
            raise NotImplementedError('Token counting returned no result')  # pragma: no cover
        return response.total_tokens


def _map_usage(
    response: EmbedContentResponse,
    provider: str,
    provider_url: str,
    model: str,
) -> RequestUsage:
    """Map Google embedding response to RequestUsage.

    Note: The Gemini API (google-gla) doesn't return token usage information.
    Vertex AI (google-vertex) returns token_count in embedding statistics.
    """
    total_tokens = 0
    if response.embeddings:
        for emb in response.embeddings:
            if emb.statistics and emb.statistics.token_count:
                total_tokens += int(emb.statistics.token_count)

    if total_tokens > 0:
        return RequestUsage(input_tokens=total_tokens)
    return RequestUsage()
