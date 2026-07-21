"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.realtime.azure import AzureRealtimeModel


def test_model_is_exported_from_realtime_package() -> None:
    from pydantic_ai.realtime import AzureRealtimeModel as ExportedAzureRealtimeModel

    assert ExportedAzureRealtimeModel is AzureRealtimeModel


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
