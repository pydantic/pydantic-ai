from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal, get_args, overload

from typing_extensions import TypeAliasType

from pydantic_ai import _utils
from pydantic_ai.embeddings.base import EmbeddingModel
from pydantic_ai.embeddings.settings import EmbeddingSettings, merge_embedding_settings
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import OpenAIChatCompatibleProvider, OpenAIResponsesCompatibleProvider
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.providers import Provider, infer_provider

from .base import EmbedInputType

__all__ = [
    'Embedder',
    'EmbeddingModel',
    'EmbeddingSettings',
    'merge_embedding_settings',
    'KnownEmbeddingModelName',
    'infer_model',
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
    """Options to automatically instrument with OpenTelemetry."""

    def __init__(
        self,
        model: EmbeddingModel | KnownEmbeddingModelName | str,
        *,
        settings: EmbeddingSettings | None = None,
        defer_model_check: bool = True,
        # TODO: Figure out instrumentation later..
        instrument: InstrumentationSettings | bool | None = None,
    ) -> None:
        self._model = model if defer_model_check else infer_model(model)
        self._settings = settings
        self._instrument = instrument

        self._override_model: ContextVar[EmbeddingModel | None] = ContextVar('_override_model', default=None)

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

    def _get_model(self) -> EmbeddingModel:
        """Create a model configured for this agent.

        Returns:
            The embedding model to use
        """
        model_: EmbeddingModel
        if some_model := self._override_model.get():
            model_ = some_model
        else:
            model_ = self._model = infer_model(self.model)

        # TODO: Port the instrumentation logic from Model once we settle on an embeddings API
        # instrument = self.instrument
        # if instrument is None:
        #     instrument = Agent._instrument_default
        #
        # return instrument_model(model_, instrument)

        return model_
