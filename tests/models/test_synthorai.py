"""Tests for `SynthoraiModel`.

Synthorai routes to models from several upstreams behind one OpenAI-compatible endpoint, so
what is worth testing is which profile `SynthoraiProvider.model_profile()` resolves for a
given model id and that a standard Chat Completions request round-trips. Model ids here carry
no vendor prefix - they are flat names such as `claude-opus-5` - so the family is matched on a
leading substring rather than split on '/'.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.synthorai import SynthoraiModel
    from pydantic_ai.providers.synthorai import SynthoraiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def test_synthorai_provider_base_url(synthorai_api_key: str):
    provider = SynthoraiProvider(api_key=synthorai_api_key)
    assert provider.name == 'synthorai'
    assert provider.base_url == 'https://synthorai.io/v1'


def test_synthorai_provider_needs_a_key(monkeypatch: pytest.MonkeyPatch):
    from pydantic_ai.exceptions import UserError

    monkeypatch.delenv('SYNTHORAI_API_KEY', raising=False)
    with pytest.raises(UserError, match='SYNTHORAI_API_KEY'):
        SynthoraiProvider()


async def test_synthorai_model_simple(allow_model_requests: None, synthorai_api_key: str):
    """A plain Chat Completions round-trip through the gateway."""
    model = SynthoraiModel('claude-opus-5', provider=SynthoraiProvider(api_key=synthorai_api_key))
    agent = Agent(model)
    result = await agent.run('What is 2 + 2?')
    assert result.output
