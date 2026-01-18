from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args

from typing_extensions import TypeAliasType

from pydantic_ai import _utils
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import OpenAIChatCompatibleProvider, OpenAIResponsesCompatibleProvider
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.providers import Provider, infer_provider

from .base import EmbeddingModel
from .instrumented import InstrumentedEmbeddingModel, instrument_embedding_model
from .result import EmbeddingResult, EmbedInputType
from .settings import EmbeddingSettings, merge_embedding_settings
from .test import TestEmbeddingModel
from .wrapper import WrapperEmbeddingModel

__all__ = [
    'Embedder',
    'EmbeddingModel',
    'EmbeddingSettings',
    'EmbeddingResult',
    'merge_embedding_settings',
    'KnownEmbeddingModelName',
    'infer_embedding_model',
    'WrapperEmbeddingModel',
    'InstrumentedEmbeddingModel',
    'instrument_embedding_model',
    'TestEmbeddingModel',
]

KnownEmbeddingModelName = TypeAliasType(
    'KnownEmbeddingModelName',
    Literal[
        'google-gla:gemini-embedding-001',
        'google-vertex:gemini-embedding-001',
        'google-vertex:text-embedding-005',
        'google-vertex:text-multilingual-embedding-002',
        'openai:text-embedding-ada-002',
        'openai:text-embedding-3-small',
        'openai:text-embedding-3-large',
        'cohere:embed-v4.0',
        'cohere:embed-english-v3.0',
        'cohere:embed-english-light-v3.0',
        'cohere:embed-multilingual-v3.0',
        'cohere:embed-multilingual-light-v3.0',
    ],
)
"""Known model names that can be used with the `model` parameter of [`Embedder`][pydantic_ai.embeddings.Embedder].

`KnownEmbeddingModelName` is provided as a concise way to specify an embedding model.
"""

# For now, we assume that every chat and completions-compatible provider also
# supports the embeddings endpoint, as at worst the user would get an `ModelHTTPError`.
OpenAIEmbeddingsCompatibleProvider = OpenAIChatCompatibleProvider | OpenAIResponsesCompatibleProvider


def infer_embedding_model(
    model: EmbeddingModel | KnownEmbeddingModelName | str,
    *,
    provider_factory: Callable[[str], Provider[Any]] = infer_provider,
) -> EmbeddingModel:
    """Infer the model from the name."""
    if isinstance(model, EmbeddingModel):
        return model

    try:
        provider_name, model_name = model.split(':', maxsplit=1)
    except ValueError as e:
        raise ValueError('You must provide a provider prefix when specifying an embedding model name') from e

    provider = provider_factory(provider_name)

    model_kind = provider_name
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        model_kind = normalize_gateway_provider(model_kind)

    if model_kind in (
        'openai',
        # For now, we assume that every chat and completions-compatible provider also
        # supports the embeddings endpoint, as at worst the user would get an `ModelHTTPError`.
        *get_args(OpenAIChatCompatibleProvider.__value__),
        *get_args(OpenAIResponsesCompatibleProvider.__value__),
    ):
        from .openai import OpenAIEmbeddingModel

        return OpenAIEmbeddingModel(model_name, provider=provider)
    elif model_kind == 'cohere':
        from .cohere import CohereEmbeddingModel

        return CohereEmbeddingModel(model_name, provider=provider)
    elif model_kind in ('google-gla', 'google-vertex'):
        from .google import GoogleEmbeddingModel

        return GoogleEmbeddingModel(model_name, provider=provider)
    elif model_kind == 'sentence-transformers':
        from .sentence_transformers import SentenceTransformerEmbeddingModel

        return SentenceTransformerEmbeddingModel(model_name)
    else:
        raise UserError(f'Unknown embeddings model: {model}')  # pragma: no cover


@dataclass(init=False)
class Embedder:
    """High-level interface for generating text embeddings.

    The `Embedder` class provides a convenient way to generate vector embeddings from text
    using various embedding model providers. It handles model inference, settings management,
    and optional OpenTelemetry instrumentation.

    Example:
    ```python
    from pydantic_ai import Embedder

    embedder = Embedder('openai:text-embedding-3-small')


    async def main():
        result = await embedder.embed_query('What is machine learning?')
        print(result.embeddings[0][:5])  # First 5 dimensions
        #> [1.0, 1.0, 1.0, 1.0, 1.0]
    ```
    """

    instrument: InstrumentationSettings | bool | None
    """Options to automatically instrument with OpenTelemetry.

    Set to `True` to use default instrumentation settings, which will use Logfire if it's configured.
    Set to an instance of [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings] to customize.
    If this isn't set, then the last value set by
    [`Embedder.instrument_all()`][pydantic_ai.embeddings.Embedder.instrument_all]
    will be used, which defaults to False.
    See the [Debugging and Monitoring guide](https://ai.pydantic.dev/logfire/) for more info.
    """

    _instrument_default: ClassVar[InstrumentationSettings | bool] = False

    def __init__(
        self,
        model: EmbeddingModel | KnownEmbeddingModelName | str,
        *,
        settings: EmbeddingSettings | None = None,
        defer_model_check: bool = True,
        instrument: InstrumentationSettings | bool | None = None,
    ) -> None:
        """Initialize an Embedder.

        Args:
            model: The embedding model to use. Can be specified as:

                - A model name string in the format `'provider:model-name'`
                  (e.g., `'openai:text-embedding-3-small'`)
                - An [`EmbeddingModel`][pydantic_ai.embeddings.EmbeddingModel] instance
            settings: Optional [`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings]
                to use as defaults for all embed calls.
            defer_model_check: Whether to defer model validation until first use.
                Set to `False` to validate the model immediately on construction.
            instrument: OpenTelemetry instrumentation settings. Set to `True` to enable with defaults,
                or pass an [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
                instance to customize. If `None`, uses the value from
                [`Embedder.instrument_all()`][pydantic_ai.embeddings.Embedder.instrument_all].
        """
        self._model = model if defer_model_check else infer_embedding_model(model)
        self._settings = settings
        self.instrument = instrument

        self._override_model: ContextVar[EmbeddingModel | None] = ContextVar('_override_model', default=None)

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the default instrumentation options for all embedders where `instrument` is not explicitly set.

        This is useful for enabling instrumentation globally without modifying each embedder individually.

        Args:
            instrument: Instrumentation settings to use as the default. Set to `True` for default settings,
                `False` to disable, or pass an
                [`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]
                instance to customize.
        """
        Embedder._instrument_default = instrument

    @property
    def model(self) -> EmbeddingModel | KnownEmbeddingModelName | str:
        """The embedding model used by this embedder."""
        return self._model

    @contextmanager
    def override(
        self,
        *,
        model: EmbeddingModel | KnownEmbeddingModelName | str | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        """Context manager to temporarily override the embedding model.

        Useful for testing or dynamically switching models.

        Args:
            model: The embedding model to use within this context.

        Example:
        ```python
        from pydantic_ai import Embedder

        embedder = Embedder('openai:text-embedding-3-small')


        async def main():
            # Temporarily use a different model
            with embedder.override(model='openai:text-embedding-3-large'):
                result = await embedder.embed_query('test')
                print(len(result.embeddings[0]))  # 3072 dimensions for large model
                #> 3072
        ```
        """
        if _utils.is_set(model):
            model_token = self._override_model.set(infer_embedding_model(model))
        else:
            model_token = None

        try:
            yield
        finally:
            if model_token is not None:
                self._override_model.reset(model_token)

    async def embed_query(
        self, query: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Embed one or more query texts.

        Use this method when embedding search queries that will be compared against document embeddings.
        Some models optimize embeddings differently based on whether the input is a query or document.

        Args:
            query: A single query string or sequence of query strings to embed.
            settings: Optional settings to override the embedder's default settings for this call.

        Returns:
            An [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing the embeddings
            and metadata about the operation.
        """
        return await self.embed(query, input_type='query', settings=settings)

    async def embed_documents(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Embed one or more document texts.

        Use this method when embedding documents that will be stored and later searched against.
        Some models optimize embeddings differently based on whether the input is a query or document.

        Args:
            documents: A single document string or sequence of document strings to embed.
            settings: Optional settings to override the embedder's default settings for this call.

        Returns:
            An [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing the embeddings
            and metadata about the operation.
        """
        return await self.embed(documents, input_type='document', settings=settings)

    async def embed(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Embed text inputs with explicit input type specification.

        This is the low-level embedding method. For most use cases, prefer
        [`embed_query()`][pydantic_ai.embeddings.Embedder.embed_query] or
        [`embed_documents()`][pydantic_ai.embeddings.Embedder.embed_documents].

        Args:
            inputs: A single string or sequence of strings to embed.
            input_type: The type of input, either `'query'` or `'document'`.
            settings: Optional settings to override the embedder's default settings for this call.

        Returns:
            An [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing the embeddings
            and metadata about the operation.
        """
        model = self._get_model()
        settings = merge_embedding_settings(self._settings, settings)
        return await model.embed(inputs, input_type=input_type, settings=settings)

    async def max_input_tokens(self) -> int | None:
        """Get the maximum number of tokens the model can accept as input.

        Returns:
            The maximum token count, or `None` if the limit is unknown for this model.
        """
        model = self._get_model()
        return await model.max_input_tokens()

    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text: The text to tokenize and count.

        Returns:
            The number of tokens in the text.

        Raises:
            NotImplementedError: If the model doesn't support token counting.
            UserError: If the model or tokenizer is not supported.
        """
        model = self._get_model()
        return await model.count_tokens(text)

    def embed_query_sync(
        self, query: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Synchronous version of [`embed_query()`][pydantic_ai.embeddings.Embedder.embed_query]."""
        return _utils.get_event_loop().run_until_complete(self.embed_query(query, settings=settings))

    def embed_documents_sync(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Synchronous version of [`embed_documents()`][pydantic_ai.embeddings.Embedder.embed_documents]."""
        return _utils.get_event_loop().run_until_complete(self.embed_documents(documents, settings=settings))

    def embed_sync(
        self, inputs: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> EmbeddingResult:
        """Synchronous version of [`embed()`][pydantic_ai.embeddings.Embedder.embed]."""
        return _utils.get_event_loop().run_until_complete(self.embed(inputs, input_type=input_type, settings=settings))

    def max_input_tokens_sync(self) -> int | None:
        """Synchronous version of [`max_input_tokens()`][pydantic_ai.embeddings.Embedder.max_input_tokens]."""
        return _utils.get_event_loop().run_until_complete(self.max_input_tokens())

    def count_tokens_sync(self, text: str) -> int:
        """Synchronous version of [`count_tokens()`][pydantic_ai.embeddings.Embedder.count_tokens]."""
        return _utils.get_event_loop().run_until_complete(self.count_tokens(text))

    def _get_model(self) -> EmbeddingModel:
        """Create a model configured for this embedder.

        Returns:
            The embedding model to use, with instrumentation applied if configured.
        """
        model_: EmbeddingModel
        if some_model := self._override_model.get():
            model_ = some_model
        else:
            model_ = self._model = infer_embedding_model(self.model)

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        return instrument_embedding_model(model_, instrument)
