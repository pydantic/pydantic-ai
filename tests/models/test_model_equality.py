"""Model equality includes default settings.

Regression tests for https://github.com/pydantic/pydantic-ai/issues/8191: `_settings` is
declared on the non-dataclass `Model`/`EmbeddingModel` bases, so the generated dataclass
`__eq__` ignored it and two models differing only in settings compared equal — which made
capability merging silently drop the later model's settings.

These are unit tests rather than VCR tests: equality is internal, definitory behavior with
no network involvement, and worth pinning against drift directly.
"""

from __future__ import annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.capabilities import XSearch
    from pydantic_ai.embeddings.test import TestEmbeddingModel
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.settings import ModelSettings

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _openai_model(provider: OpenAIProvider, settings: ModelSettings | None) -> OpenAIChatModel:
    return OpenAIChatModel('gpt-5', provider=provider, settings=settings)


@pytest.fixture
def openai_provider() -> OpenAIProvider:
    return OpenAIProvider(api_key='test')


def test_model_equality_includes_settings(openai_provider: OpenAIProvider) -> None:
    first = _openai_model(openai_provider, {'temperature': 0.0})
    later = _openai_model(openai_provider, {'temperature': 1.0})

    assert first != later
    assert not first == later

    assert first == _openai_model(openai_provider, {'temperature': 0.0})
    assert first != _openai_model(openai_provider, None)
    assert first != 'gpt-5'
    assert first != TestModel()


def test_model_equality_still_compares_dataclass_fields(openai_provider: OpenAIProvider) -> None:
    assert _openai_model(openai_provider, None) != OpenAIChatModel('gpt-4', provider=openai_provider, settings=None)
    # Providers are plain classes, so distinct instances never compare equal — as before.
    assert _openai_model(openai_provider, None) != _openai_model(OpenAIProvider(api_key='test'), None)


def test_embedding_model_equality_includes_settings() -> None:
    assert TestEmbeddingModel(settings={'dimensions': 8}) != TestEmbeddingModel(settings={'dimensions': 16})
    assert TestEmbeddingModel(settings={'dimensions': 8}) == TestEmbeddingModel(settings={'dimensions': 8})


def test_capability_merge_keeps_later_fallback_subagent_model_settings(openai_provider: OpenAIProvider) -> None:
    """The merge symptom from the issue: the later capability's model settings must survive."""
    first = _openai_model(openai_provider, {'temperature': 0.0})
    later = _openai_model(openai_provider, {'temperature': 1.0})

    merged = XSearch.combine(
        [
            XSearch(native=False, fallback_subagent_model=first),
            XSearch(native=False, fallback_subagent_model=later),
        ]
    )

    merged_model = merged.fallback_subagent_model
    assert merged_model is later
    assert isinstance(merged_model, OpenAIChatModel)
    assert merged_model.settings == {'temperature': 1.0}
