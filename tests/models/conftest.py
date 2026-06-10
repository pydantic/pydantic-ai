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
