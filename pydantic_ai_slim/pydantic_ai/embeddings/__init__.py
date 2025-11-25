from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args, overload

from typing_extensions import TypeAliasType

from pydantic_ai import _utils
from pydantic_ai.embeddings.base import EmbeddingModel
from pydantic_ai.embeddings.settings import EmbeddingSettings, merge_embedding_settings
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import OpenAIChatCompatibleProvider, OpenAIResponsesCompatibleProvider
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.providers import Provider, infer_provider

from .base import EmbedInputType
from .instrumented import InstrumentedEmbeddingModel, instrument_embedding_model
from .wrapper import WrapperEmbeddingModel

__all__ = [
    'Embedder',
    'EmbeddingModel',
    'EmbeddingSettings',
    'merge_embedding_settings',
    'KnownEmbeddingModelName',
    'infer_model',
    'WrapperEmbeddingModel',
    'InstrumentedEmbeddingModel',
    'instrument_embedding_model',
]

KnownEmbeddingModelName = TypeAliasType(
    'KnownEmbeddingModelName',
    Literal[
        'openai:text-embedding-ada-002',
        'openai:text-embedding-3-small',
        'openai:text-embedding-3-largecohere:embed-v4.0',
    ],
)
"""Known model names that can be used with the `model` parameter of [`Embedder`][pydantic_ai.embeddings.Embedder].

`KnownEmbeddingModelName` is provided as a concise way to specify an embedding model.
"""

# For now, we assume that every chat and completions-compatible provider also
# supports the embeddings endpoint, as at worst the user would get an `ModelHTTPError`.
OpenAIEmbeddingsCompatibleProvider = OpenAIChatCompatibleProvider | OpenAIResponsesCompatibleProvider


def infer_model(
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
    elif model_kind == 'sentence-transformers':
        from .sentence_transformers import SentenceTransformerEmbeddingModel

        return SentenceTransformerEmbeddingModel(model_name)
    else:
        raise UserError(f'Unknown embeddings model: {model}')  # pragma: no cover


@dataclass(init=False)
class Embedder:
    """TODO: Docstring."""

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
            model: The embedding model to use - can be a model instance, model name, or string.
            settings: Optional embedding settings to use as defaults.
            defer_model_check: Whether to defer model validation until first use.
            instrument: OpenTelemetry instrumentation settings. Set to `True` to enable with defaults,
                or pass an `InstrumentationSettings` instance to customize. If `None`, uses the value
                from `Embedder.instrument_all()`.
        """
        self._model = model if defer_model_check else infer_model(model)
        self._settings = settings
        self.instrument = instrument

        self._override_model: ContextVar[EmbeddingModel | None] = ContextVar('_override_model', default=None)

    @staticmethod
    def instrument_all(instrument: InstrumentationSettings | bool = True) -> None:
        """Set the instrumentation options for all embedders where `instrument` is not set.

        Args:
            instrument: Instrumentation settings to use as the default. Set to `True` for default settings,
                `False` to disable, or pass an `InstrumentationSettings` instance to customize.
        """
        Embedder._instrument_default = instrument

    @property
    def model(self) -> EmbeddingModel | KnownEmbeddingModelName | str:
        return self._model

    @contextmanager
    def override(
        self,
        *,
        model: EmbeddingModel | KnownEmbeddingModelName | str | _utils.Unset = _utils.UNSET,
    ) -> Iterator[None]:
        if _utils.is_set(model):
            model_token = self._override_model.set(infer_model(model))
        else:
            model_token = None

        try:
            yield
        finally:
            if model_token is not None:
                self._override_model.reset(model_token)

    @overload
    async def embed_query(self, query: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    async def embed_query(
        self, query: Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]:
        pass

    async def embed_query(
        self, query: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        return await self.embed(query, input_type='query', settings=settings)

    @overload
    async def embed_documents(self, documents: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    async def embed_documents(
        self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]:
        pass

    async def embed_documents(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        return await self.embed(documents, input_type='document', settings=settings)

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
        model = self._get_model()
        settings = merge_embedding_settings(self._settings, settings)
        return await model.embed(documents, input_type=input_type, settings=settings)

    @overload
    def embed_query_sync(self, query: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    def embed_query_sync(self, query: Sequence[str], *, settings: EmbeddingSettings | None = None) -> list[list[float]]:
        pass

    def embed_query_sync(
        self, query: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        return _utils.get_event_loop().run_until_complete(self.embed_query(query, settings=settings))

    @overload
    def embed_documents_sync(self, documents: str, *, settings: EmbeddingSettings | None = None) -> list[float]:
        pass

    @overload
    def embed_documents_sync(
        self, documents: Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]:
        pass

    def embed_documents_sync(
        self, documents: str | Sequence[str], *, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        return _utils.get_event_loop().run_until_complete(self.embed_documents(documents, settings=settings))

    @overload
    def embed_sync(
        self, documents: str, *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float]:
        pass

    @overload
    def embed_sync(
        self, documents: Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[list[float]]:
        pass

    def embed_sync(
        self, documents: str | Sequence[str], *, input_type: EmbedInputType, settings: EmbeddingSettings | None = None
    ) -> list[float] | list[list[float]]:
        return _utils.get_event_loop().run_until_complete(
            self.embed(documents, input_type=input_type, settings=settings)
        )

    async def count_tokens(self, text: str) -> int:
        model = self._get_model()
        return await model.count_tokens(text)

    def count_tokens_sync(self, text: str) -> int:
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
            model_ = self._model = infer_model(self.model)

        instrument = self.instrument
        if instrument is None:
            instrument = self._instrument_default

        return instrument_embedding_model(model_, instrument)
