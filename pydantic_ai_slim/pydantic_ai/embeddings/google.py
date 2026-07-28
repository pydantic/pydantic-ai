import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import BinaryContent, TextContent
from pydantic_ai.models import download_item
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.usage import RequestUsage

from .base import EmbeddingModel
from .input import EmbeddingContentPart, EmbeddingInput, EmbeddingModality, embedding_parts
from .result import EmbeddingResult, EmbedInputType
from .settings import EmbeddingSettings

try:
    from google.genai import Client, errors
    from google.genai.types import Blob, Content, ContentUnion, EmbedContentConfig, EmbedContentResponse, Part
except ImportError as _import_error:
    raise ImportError(
        'Please install `google-genai` to use the Google embeddings model, '
        'you can use the `google` optional group — `pip install "pydantic-ai-slim[google]"`'
    ) from _import_error


LatestGoogleGLAEmbeddingModelNames = Literal['gemini-embedding-001', 'gemini-embedding-2-preview', 'gemini-embedding-2']
"""Latest Gemini API embedding models.

See the [Google Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
for available models and their capabilities.
"""

LatestGoogleVertexEmbeddingModelNames = Literal[
    'gemini-embedding-001',
    'gemini-embedding-2-preview',
    'gemini-embedding-2',
    'text-embedding-005',
    'text-multilingual-embedding-002',
]
"""Latest Google Cloud (formerly known as Vertex AI) embedding models.

See the [Google Cloud Embeddings documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings)
for available models and their capabilities.
"""

LatestGoogleEmbeddingModelNames = LatestGoogleGLAEmbeddingModelNames | LatestGoogleVertexEmbeddingModelNames
"""All latest Google embedding models (union of Gemini API and Google Cloud models)."""

GoogleEmbeddingModelName = str | LatestGoogleEmbeddingModelNames
"""Possible Google embeddings model names."""


GoogleEmbeddingTask = Literal[
    'search result',
    'question answering',
    'fact checking',
    'code retrieval',
    'classification',
    'clustering',
    'sentence similarity',
    'raw',
]
"""Task the embedding is optimized for, applied as a text prefix by `gemini-embedding-2`.

Unlike other Google embedding models (which condition on the [`google_task_type`][pydantic_ai.embeddings.google.GoogleEmbeddingSettings.google_task_type]
field), `gemini-embedding-2` is conditioned by prepending a task instruction to the input text.

Asymmetric tasks prefix queries and documents differently, so the same task can be used for both
sides of a retrieval pair:

- `'search result'`: retrieval; find documents relevant to a search query (the default).
- `'question answering'`: retrieval; find passages that answer a question.
- `'fact checking'`: retrieval; find evidence that supports or refutes a claim.
- `'code retrieval'`: retrieval; find code relevant to a natural-language query.

Symmetric tasks prefix both inputs the same way, since both sides play the same role:

- `'classification'`: assign inputs to predefined categories.
- `'clustering'`: group inputs by similarity.
- `'sentence similarity'`: measure semantic similarity between inputs.

- `'raw'`: embed the text verbatim, without any prefix.
"""

_SYMMETRIC_TASKS: frozenset[GoogleEmbeddingTask] = frozenset({'classification', 'clustering', 'sentence similarity'})

# The only model that conditions on a task via a text prefix rather than the `task_type` field.
_TASK_PREFIX_MODEL = 'gemini-embedding-2'

# Models that accept more than text. See https://ai.google.dev/gemini-api/docs/embeddings#multimodal
# Derived from `_TASK_PREFIX_MODEL` because the branch of `embed()` that maps non-text parts is the
# task-prefix branch: a model added here without a multimodal request mapping would advertise a
# capability and then have `prepare_text_embed()` refuse it.
_MULTIMODAL_MODELS: frozenset[str] = frozenset({_TASK_PREFIX_MODEL})

_MULTIMODAL_MODALITIES: frozenset[EmbeddingModality] = frozenset({'text', 'image', 'audio', 'video', 'document'})


_MAX_INPUT_TOKENS: dict[GoogleEmbeddingModelName, int] = {
    'gemini-embedding-001': 2048,
    'gemini-embedding-2-preview': 8192,
    'gemini-embedding-2': 8192,
    'text-embedding-005': 2048,
    'text-multilingual-embedding-002': 2048,
}


class GoogleEmbeddingSettings(EmbeddingSettings, total=False):
    """Settings used for a Google embedding model request.

    All fields from [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] are supported,
    plus Google-specific settings prefixed with `google_`.
    """

    # ALL FIELDS MUST BE `google_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.

    google_task: GoogleEmbeddingTask
    """Task to condition `gemini-embedding-2` on, applied as a text prefix.

    Only supported by `gemini-embedding-2`; on other models it is ignored with a warning (they use
    [`google_task_type`][pydantic_ai.embeddings.google.GoogleEmbeddingSettings.google_task_type] instead).
    When unset on `gemini-embedding-2`, defaults to `'search result'`.

    For asymmetric tasks the prefix depends on `input_type`: a `'query'` becomes `task: {task} | query: {text}`,
    while a `'document'` becomes `title: {title} | text: {text}`, using
    [`google_title`][pydantic_ai.embeddings.google.GoogleEmbeddingSettings.google_title] (or `none` when no
    title is set). Symmetric tasks use the `task: {task} | query: {text}` form for both. `'raw'` embeds the
    text verbatim. See [`GoogleEmbeddingTask`][pydantic_ai.embeddings.google.GoogleEmbeddingTask] for the per-task semantics.
    """

    google_task_type: str
    """The task type for the embedding.

    Overrides the automatic task type selection based on `input_type`.
    See [Google's task type documentation](https://ai.google.dev/gemini-api/docs/embeddings#task-types)
    for available options.
    """

    google_title: str
    """Optional title for the content being embedded.

    Only applicable when task_type is `RETRIEVAL_DOCUMENT`.
    """


@dataclass(init=False)
class GoogleEmbeddingModel(EmbeddingModel):
    """Google embedding model implementation.

    This model works with Google's embeddings API via the `google-genai` SDK,
    supporting both the Gemini API (Google AI Studio) and Google Cloud (formerly known as Vertex AI).

    Example:
    ```python
    from pydantic_ai.embeddings.google import GoogleEmbeddingModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    # Using the Gemini API (requires GOOGLE_API_KEY env var)
    model = GoogleEmbeddingModel('gemini-embedding-001', provider=GoogleProvider())

    # Using Google Cloud
    model = GoogleEmbeddingModel(
        'gemini-embedding-001',
        provider=GoogleCloudProvider(project='my-project', location='us-central1'),
    )
    ```
    """

    _model_name: GoogleEmbeddingModelName = field(repr=False)
    _provider: Provider[Client] = field(repr=False)

    def __init__(
        self,
        model_name: GoogleEmbeddingModelName,
        *,
        provider: Literal['google', 'google-cloud'] | Provider[Client] = 'google',
        settings: EmbeddingSettings | None = None,
    ):
        """Initialize a Google embedding model.

        Args:
            model_name: The name of the Google model to use.
                See [Google Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings)
                for available models.
            provider: The provider to use for authentication and API access. Can be:

                - `'google'` (default): Uses the Gemini API (Google AI Studio)
                - `'google-cloud'`: Uses Google Cloud (formerly known as Vertex AI)
                - A [`GoogleProvider`][pydantic_ai.providers.google.GoogleProvider] or
                  [`GoogleCloudProvider`][pydantic_ai.providers.google_cloud.GoogleCloudProvider] instance
                  for custom configuration
            settings: Model-specific [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for this model.
        """
        self._model_name = model_name

        if isinstance(provider, str):
            provider = infer_provider(provider)
        self._provider = provider

        super().__init__(settings=settings)

    @property
    def _client(self) -> Client:
        return self._provider.client

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

    @property
    def supported_modalities(self) -> frozenset[EmbeddingModality]:
        """The modalities this model can embed; `gemini-embedding-2` also accepts images, audio, video and PDFs."""
        if self._model_name in _MULTIMODAL_MODELS:
            return _MULTIMODAL_MODALITIES
        return super().supported_modalities

    async def embed(
        self,
        inputs: EmbeddingInput | Sequence[EmbeddingInput],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        items: Sequence[EmbeddingInput]
        contents: list[ContentUnion]

        if self._model_name == _TASK_PREFIX_MODEL:
            items, merged_settings = self.prepare_embed(inputs, settings)
            settings = cast(GoogleEmbeddingSettings, merged_settings)

            google_task = settings.get('google_task')
            if (google_task_type := settings.get('google_task_type')) is not None:
                warnings.warn(
                    f'`google_task_type` is not supported by `{_TASK_PREFIX_MODEL}` and is ignored; '
                    'this model conditions on a task via the `google_task` text prefix instead.',
                    UserWarning,
                    stacklevel=2,
                )
            task = google_task if google_task is not None else 'search result'

            contents = []
            skipped_file_conditioning = False
            skipped_text_conditioning = False
            for item in items:
                parts = embedding_parts(item)
                # The prefix conditions text, so it only applies to an input that is a single text part;
                # Google doesn't define task instructions for other modalities, nor for a multi-part input.
                # See https://ai.google.dev/gemini-api/docs/embeddings#multimodal
                if len(parts) == 1 and isinstance(text_part := parts[0], str | TextContent):
                    text = text_part if isinstance(text_part, str) else text_part.content
                    # `'raw'` opts out of conditioning (verbatim passthrough). Named `'raw'`, not `'none'`:
                    # the prefix is applied client-side (no provider API value to mirror, unlike VoyageAI's
                    # `'none'` which maps to a null `input_type`), and `'raw'` avoids the `google_task=None`
                    # footgun where `None` would silently fall back to the `'search result'` default.
                    if task == 'raw':
                        pass
                    elif input_type == 'document' and task not in _SYMMETRIC_TASKS:
                        title = settings.get('google_title') or 'none'
                        text = f'title: {title} | text: {text}'
                    else:
                        text = f'task: {task} | query: {text}'
                    contents.append(Content(parts=[Part(text=text)]))
                else:
                    if all(isinstance(part, str | TextContent) for part in parts):
                        skipped_text_conditioning = True
                    else:
                        skipped_file_conditioning = True
                    contents.append(Content(parts=[await _map_content_part(part) for part in parts]))

            if task != 'raw':
                # Text that goes unconditioned lands in a different region of the space than text that
                # doesn't, so warn even on the default task — for an all-text input, conditioning was
                # meaningful and got skipped. An input that also holds a file is deliberately silent
                # here: its vector is a multimodal one, which Google defines no task instruction for,
                # and warning would fire on every caption-plus-image embed, the common case.
                if skipped_text_conditioning:
                    warnings.warn(
                        f'`{self._model_name}` conditions on a task by prefixing the text, so an `EmbeddingContent` '
                        'of several text parts is embedded unconditioned; pass it as a single text part to condition '
                        'it, or set `google_task` to `raw` to opt out of conditioning entirely.',
                        UserWarning,
                        stacklevel=2,
                    )
                if google_task is not None and skipped_file_conditioning:
                    warnings.warn(
                        f'`google_task` only conditions inputs that are a single text part and is ignored for the '
                        f'others; `{self._model_name}` has no task conditioning for other modalities.',
                        UserWarning,
                        stacklevel=2,
                    )

            config = EmbedContentConfig(
                task_type=None,
                output_dimensionality=settings.get('dimensions'),
                title=None,
            )
        else:
            items, texts, merged_settings = self.prepare_text_embed(inputs, settings)
            settings = cast(GoogleEmbeddingSettings, merged_settings)

            if settings.get('google_task') is not None:
                warnings.warn(
                    f'`google_task` is only supported by `{_TASK_PREFIX_MODEL}` and is ignored; '
                    f'`{self._model_name}` conditions on a task via the `google_task_type` setting instead.',
                    UserWarning,
                    stacklevel=2,
                )
            google_task_type = settings.get('google_task_type')
            if google_task_type is None:
                google_task_type = 'RETRIEVAL_DOCUMENT' if input_type == 'document' else 'RETRIEVAL_QUERY'

            contents = [Content(parts=[Part(text=text)]) for text in texts]
            config = EmbedContentConfig(
                task_type=google_task_type,
                output_dimensionality=settings.get('dimensions'),
                title=settings.get('google_title'),
            )

        try:
            response = await self._client.aio.models.embed_content(
                model=self._model_name,
                contents=contents,
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

        embeddings: list[list[float]] = [emb.values for emb in (response.embeddings or []) if emb.values is not None]

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=items,
            input_type=input_type,
            usage=_map_usage(response, self.system, self.base_url, self._model_name),
            model_name=self._model_name,
            provider_name=self.system,
        )

    async def max_input_tokens(self) -> int | None:
        return _MAX_INPUT_TOKENS.get(self._model_name)

    async def count_tokens(self, text: str) -> int:
        try:
            response = await self._client.aio.models.count_tokens(
                model=self._model_name,
                contents=text,
            )
        except errors.APIError as e:
            if (status_code := e.code) >= 400:
                raise ModelHTTPError(
                    status_code=status_code,
                    model_name=self._model_name,
                    body=cast(object, e.details),  # pyright: ignore[reportUnknownMemberType]
                ) from e
            raise  # pragma: no cover

        if response.total_tokens is None:
            raise UnexpectedModelBehavior('Token counting returned no result')  # pragma: no cover
        return response.total_tokens


async def _map_content_part(part: EmbeddingContentPart) -> Part:
    """Map a content part to a Google `Part`, downloading URLs as the embeddings API only takes inline data."""
    if isinstance(part, str):
        return Part(text=part)
    elif isinstance(part, TextContent):
        return Part(text=part.content)
    elif isinstance(part, BinaryContent):
        return Part(inline_data=Blob(data=part.data, mime_type=part.media_type))
    else:
        downloaded = await download_item(part, data_format='bytes')
        return Part(inline_data=Blob(data=downloaded['data'], mime_type=downloaded['data_type']))


def _map_usage(
    response: EmbedContentResponse,
    provider: str,
    provider_url: str,
    model: str,
) -> RequestUsage:
    """Map Google embedding response to RequestUsage.

    Note: the Gemini API does return usage — the recorded responses carry `usageMetadata.promptTokenCount`,
    broken down per modality — but `google-genai` drops it: `EmbedContentResponse` exposes only
    `embeddings` and a `metadata` holding `billable_character_count`. So usage is reported as 0 tokens
    on the Gemini API, which matters most for images and documents, where tokens are not billed per character.
    Google Cloud (formerly known as Vertex AI) returns token_count in embedding statistics.
    """
    total_tokens = 0
    if response.embeddings:  # pragma: no branch
        for emb in response.embeddings:
            if emb.statistics and emb.statistics.token_count:
                total_tokens += int(emb.statistics.token_count)  # pragma: lax no cover -- requires vertexai

    return RequestUsage(input_tokens=total_tokens)
