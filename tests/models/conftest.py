from __future__ import annotations

from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

    from tests.cassette_utils import CassetteContext

with try_import() as google_imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

    GoogleModelFactory = Callable[..., GoogleModel]

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    AnthropicModelFactory = Callable[..., AnthropicModel]

with try_import() as openrouter_imports_successful:
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    OpenRouterModelFactory = Callable[..., OpenRouterModel]

with try_import() as openai_imports_successful:
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.cerebras import CerebrasProvider
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    OpenAIChatModelFactory = Callable[..., OpenAIChatModel]
    OpenAIResponsesModelFactory = Callable[..., OpenAIResponsesModel]
    DeepSeekModelFactory = Callable[..., OpenAIChatModel]
    CerebrasModelFactory = Callable[..., CerebrasModel]

with try_import() as groq_imports_successful:
    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider

    GroqModelFactory = Callable[..., GroqModel]

with try_import() as mistral_imports_successful:
    from pydantic_ai.models.mistral import MistralModel
    from pydantic_ai.providers.mistral import MistralProvider

    MistralModelFactory = Callable[..., MistralModel]

with try_import() as cohere_imports_successful:
    from pydantic_ai.models.cohere import CohereModel
    from pydantic_ai.providers.cohere import CohereProvider

    CohereModelFactory = Callable[..., CohereModel]


@pytest.fixture
def google_model(gemini_api_key: str) -> GoogleModelFactory:
    """Factory to create Google models. Used by VCR-recorded integration tests."""

    @cache
    def _cached(model_name: str, api_key: str | None = None) -> GoogleModel:
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key or gemini_api_key))

    def _create_model(
        model_name: str, *, settings: ModelSettings | None = None, api_key: str | None = None
    ) -> GoogleModel:
        if settings is None:
            return _cached(model_name, api_key)
        return GoogleModel(model_name, provider=GoogleProvider(api_key=api_key or gemini_api_key), settings=settings)

    return _create_model


@pytest.fixture
def anthropic_model(anthropic_api_key: str) -> AnthropicModelFactory:
    """Factory to create Anthropic models. Used by VCR-recorded integration tests."""

    @cache
    def _cached(model_name: str, api_key: str | None = None) -> AnthropicModel:
        return AnthropicModel(model_name, provider=AnthropicProvider(api_key=api_key or anthropic_api_key))

    def _create_model(
        model_name: str, *, settings: ModelSettings | None = None, api_key: str | None = None
    ) -> AnthropicModel:
        if settings is None:
            return _cached(model_name, api_key)
        return AnthropicModel(
            model_name, provider=AnthropicProvider(api_key=api_key or anthropic_api_key), settings=settings
        )

    return _create_model


@pytest.fixture
def openrouter_model(openrouter_api_key: str) -> OpenRouterModelFactory:
    """Factory to create OpenRouter models. Used by VCR-recorded integration tests."""

    @cache
    def _cached(model_name: str, api_key: str | None = None) -> OpenRouterModel:
        return OpenRouterModel(model_name, provider=OpenRouterProvider(api_key=api_key or openrouter_api_key))

    def _create_model(
        model_name: str, *, settings: ModelSettings | None = None, api_key: str | None = None
    ) -> OpenRouterModel:
        if settings is None:
            return _cached(model_name, api_key)
        return OpenRouterModel(
            model_name, provider=OpenRouterProvider(api_key=api_key or openrouter_api_key), settings=settings
        )

    return _create_model


@pytest.fixture
def openai_chat_model(openai_api_key: str) -> OpenAIChatModelFactory:
    """Factory to create OpenAI Chat Completions models. Used by VCR-recorded integration tests."""

    @cache
    def _create_model(model_name: str, api_key: str | None = None) -> OpenAIChatModel:
        return OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key or openai_api_key))

    return _create_model


@pytest.fixture
def openai_responses_model(openai_api_key: str) -> OpenAIResponsesModelFactory:
    """Factory to create OpenAI Responses API models. Used by VCR-recorded integration tests."""

    @cache
    def _cached(model_name: str, api_key: str | None = None) -> OpenAIResponsesModel:
        return OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key=api_key or openai_api_key))

    def _create_model(
        model_name: str, *, settings: ModelSettings | None = None, api_key: str | None = None
    ) -> OpenAIResponsesModel:
        if settings is None:
            return _cached(model_name, api_key)
        return OpenAIResponsesModel(
            model_name, provider=OpenAIProvider(api_key=api_key or openai_api_key), settings=settings
        )

    return _create_model


@pytest.fixture
def deepseek_model(deepseek_api_key: str) -> DeepSeekModelFactory:
    """Factory to create DeepSeek models (an `OpenAIChatModel` on the DeepSeek provider).

    Used by VCR-recorded integration tests.
    """

    @cache
    def _create_model(model_name: str, api_key: str | None = None) -> OpenAIChatModel:
        return OpenAIChatModel(model_name, provider=DeepSeekProvider(api_key=api_key or deepseek_api_key))

    return _create_model


@pytest.fixture
def cerebras_model(cerebras_api_key: str) -> CerebrasModelFactory:
    """Factory to create Cerebras models. Used by VCR-recorded integration tests."""

    @cache
    def _create_model(model_name: str, api_key: str | None = None) -> CerebrasModel:
        return CerebrasModel(model_name, provider=CerebrasProvider(api_key=api_key or cerebras_api_key))

    return _create_model


@pytest.fixture
def groq_model(groq_api_key: str) -> GroqModelFactory:
    """Factory to create Groq models. Used by VCR-recorded integration tests."""

    @cache
    def _create_model(model_name: str, api_key: str | None = None) -> GroqModel:
        return GroqModel(model_name, provider=GroqProvider(api_key=api_key or groq_api_key))

    return _create_model


@pytest.fixture
def mistral_model(mistral_api_key: str) -> MistralModelFactory:
    """Factory to create Mistral models. Used by VCR-recorded integration tests."""

    @cache
    def _create_model(model_name: str, api_key: str | None = None) -> MistralModel:
        return MistralModel(model_name, provider=MistralProvider(api_key=api_key or mistral_api_key))

    return _create_model


@pytest.fixture
def cohere_model(co_api_key: str) -> CohereModelFactory:
    """Factory to create Cohere models. Used by VCR-recorded integration tests."""

    @cache
    def _create_model(model_name: str, api_key: str | None = None) -> CohereModel:
        return CohereModel(model_name, provider=CohereProvider(api_key=api_key or co_api_key))

    return _create_model


@pytest.fixture(scope='function')
def cassette_ctx(request: pytest.FixtureRequest, vcr: Cassette) -> CassetteContext:
    """Unified cassette verification context for model tests.

    Returns a CassetteContext for tests with a 'provider' parameter, or for
    non-parametrized tests (defaulting to 'vcr' provider).
    """
    from tests.cassette_utils import CassetteContext

    provider = 'vcr'
    if callspec := getattr(request.node, 'callspec', None):  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        params = cast(dict[str, object], callspec.params)
        p = params.get('provider')
        if isinstance(p, str):  # pragma: no branch
            provider = p

    test_module: str = request.node.fspath.basename.replace('.py', '')  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    test_dir = Path(request.node.fspath).parent  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
    return CassetteContext(
        provider=provider,
        vcr=vcr,
        test_name=request.node.name,  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        test_module=test_module,  # pyright: ignore[reportUnknownArgumentType]
        test_dir=test_dir,
    )
