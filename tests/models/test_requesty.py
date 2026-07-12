from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.requesty import RequestyModel
    from pydantic_ai.providers.requesty import RequestyProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_requesty_model_init_with_provider_name(requesty_api_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('REQUESTY_API_KEY', requesty_api_key)
    model = RequestyModel('openai/gpt-4o-mini')
    assert model.model_name == 'openai/gpt-4o-mini'
    assert model.system == 'requesty'
    assert isinstance(model.client, openai.AsyncOpenAI)
    assert str(model.client.base_url) == 'https://router.requesty.ai/v1/'


def test_requesty_model_init_with_provider_instance() -> None:
    provider = RequestyProvider(api_key='api-key')
    model = RequestyModel('anthropic/claude-sonnet-4-5', provider=provider)
    assert model.model_name == 'anthropic/claude-sonnet-4-5'
    assert model.system == 'requesty'
    assert model.client == provider.client
