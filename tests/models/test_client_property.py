"""Test that model classes use a property for `client` to avoid stale references.

When a provider's client is replaced (e.g., in durable execution environments
like Temporal, or in testing with dependency injection), the model should reflect
the updated client, not hold a stale reference from construction time.

Covers: https://github.com/pydantic/pydantic-ai/issues/4336
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import importlib
from typing import Any, Protocol, cast

import pytest

from pydantic_ai.providers import Provider


class FakeClient:
    """A simple fake client for testing."""

    def __init__(self, label: str = 'default'):
        self.label = label


class FakeProvider(Provider[FakeClient]):
    """A provider whose client can be swapped after construction."""

    def __init__(self, client: FakeClient):
        self._client = client

    @property
    def name(self) -> str:
        return 'fake'

    @property
    def base_url(self) -> str:
        return 'http://fake'

    @property
    def client(self) -> FakeClient:
        return self._client


class _ClientPropertyModel(Protocol):
    @property
    def client(self) -> FakeClient: ...


class FakeModel:
    def __init__(self, provider: FakeProvider) -> None:
        self._provider = provider

    @property
    def client(self) -> FakeClient:
        return self._provider.client


def _assert_property_reflects_provider(model: _ClientPropertyModel, provider: FakeProvider) -> None:
    first = provider.client
    assert model.client is first
    cast(Any, provider)._client = FakeClient('updated')
    assert model.client is provider.client


def test_fake_provider_client_property_reflects_provider_changes() -> None:
    provider = FakeProvider(FakeClient('initial'))
    model = FakeModel(provider)

    _assert_property_reflects_provider(model, provider)

    assert provider.name == 'fake'
    assert provider.base_url == 'http://fake'


def test_openai_chat_model_client_property_reflects_provider_changes():
    """OpenAIChatModel.client should always return the provider's current client."""
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client_a = AsyncOpenAI(api_key='test-key-a')
    provider = OpenAIProvider(openai_client=client_a)
    model = OpenAIChatModel('gpt-4o', provider=provider)

    assert model.client is client_a

    # Swap the provider's internal client
    client_b = AsyncOpenAI(api_key='test-key-b')
    cast(Any, provider)._client = client_b

    # Model should now reflect the new client, not the stale one
    assert model.client is client_b
    assert model.client is not client_a


def test_openai_responses_model_client_property_reflects_provider_changes():
    """OpenAIResponsesModel.client should always return the provider's current client."""
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client_a = AsyncOpenAI(api_key='test-key-a')
    provider = OpenAIProvider(openai_client=client_a)
    model = OpenAIResponsesModel('gpt-4o', provider=provider)

    assert model.client is client_a

    client_b = AsyncOpenAI(api_key='test-key-b')
    cast(Any, provider)._client = client_b

    assert model.client is client_b
    assert model.client is not client_a


def test_openai_chat_model_client_not_in_dataclass_fields():
    """client should not be a dataclass field since it is now a property."""
    from dataclasses import fields

    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    client = AsyncOpenAI(api_key='test-key')
    provider = OpenAIProvider(openai_client=client)
    model = OpenAIChatModel('gpt-4o', provider=provider)

    field_names = {f.name for f in fields(model)}
    assert 'client' not in field_names, 'client should be a property, not a dataclass field'


@pytest.mark.parametrize(
    'model_module,model_class,provider_module,provider_class,model_name',
    [
        pytest.param(
            'pydantic_ai.models.openai',
            'OpenAIChatModel',
            'pydantic_ai.providers.openai',
            'OpenAIProvider',
            'gpt-4o',
            id='openai-chat',
        ),
        pytest.param(
            'pydantic_ai.models.openai',
            'OpenAIResponsesModel',
            'pydantic_ai.providers.openai',
            'OpenAIProvider',
            'gpt-4o',
            id='openai-responses',
        ),
    ],
)
def test_client_property_pattern(
    model_module: str,
    model_class: str,
    provider_module: str,
    provider_class: str,
    model_name: str,
) -> None:
    """Verify the client property pattern is implemented correctly across OpenAI models."""
    mod = importlib.import_module(model_module)
    prov_mod = importlib.import_module(provider_module)

    model_cls = getattr(mod, model_class)
    provider_cls = getattr(prov_mod, provider_class)

    from openai import AsyncOpenAI

    client_a = AsyncOpenAI(api_key='key-a')
    provider = provider_cls(openai_client=client_a)
    model = model_cls(model_name, provider=provider)

    assert model.client is client_a

    client_b = AsyncOpenAI(api_key='key-b')
    provider._client = client_b
    assert model.client is client_b
