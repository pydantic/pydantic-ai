"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel


def test_model_is_exported_from_realtime_package() -> None:
    from pydantic_ai.realtime import AzureRealtimeModel as ExportedAzureRealtimeModel

    assert ExportedAzureRealtimeModel is AzureRealtimeModel


def test_non_azure_provider_instance_is_rejected() -> None:
    # A non-Azure `Provider` *instance* (not just the `provider='...'` string) must fail fast with a clear
    # `UserError` at construction, rather than an `AssertionError` deep inside later.
    with pytest.raises(UserError, match='requires an `AzureProvider`'):
        AzureRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='x'))


@pytest.mark.anyio
async def test_url_and_auth_headers() -> None:
    provider = AzureProvider(
        azure_endpoint='https://resource.openai.azure.com/openai/v1/',
        api_key='azure-key',
    )
    model = AzureRealtimeModel('gpt realtime', provider=provider)

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt+realtime'
    )
    assert await model._auth_headers() == {'api-key': 'azure-key'}  # pyright: ignore[reportPrivateUsage]


def test_infer_provider_from_bare_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    # The realtime model speaks only the GA `/openai/v1` protocol and never uses the provider's SDK
    # client, so inferring the provider from a bare resource endpoint must not demand the unrelated
    # `api_version` the SDK client would need.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    model = AzureRealtimeModel('gpt-realtime')

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )


def test_infer_provider_with_api_version_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # With `OPENAI_API_VERSION` set, the standard provider inference works and the realtime URL is
    # still derived from the endpoint's host.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.setenv('OPENAI_API_VERSION', '2024-10-01')

    model = AzureRealtimeModel('gpt-realtime')

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )


def test_infer_provider_with_v1_endpoint_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    model = AzureRealtimeModel('gpt-realtime')

    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )
