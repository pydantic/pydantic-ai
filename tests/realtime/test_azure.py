"""Network-free tests for the Azure OpenAI realtime model."""

from __future__ import annotations as _annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.realtime import RealtimeSessionErrorEvent
from pydantic_ai.realtime.azure import AzureRealtimeConnection, AzureRealtimeModel


class _DroppedWebSocket:
    """A socket that reports an abnormal close as soon as it is iterated."""

    async def __aiter__(self) -> AsyncIterator[str]:
        raise OSError('dropped')
        yield ''  # pragma: no cover


def test_model_is_not_exported_from_the_realtime_package() -> None:
    # Concrete providers live in submodules (see the package docstring), so neither this model nor
    # `OpenAIRealtimeModel` is reachable from `pydantic_ai.realtime` itself.
    import pydantic_ai.realtime

    with pytest.raises(AttributeError):
        getattr(pydantic_ai.realtime, 'AzureRealtimeModel')


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
        'wss://resource.openai.azure.com/openai/v1/realtime?model=gpt%20realtime'
    )
    assert await model._auth_headers() == {'api-key': 'azure-key'}  # pyright: ignore[reportPrivateUsage]


def test_sideband_url_uses_the_ga_realtime_path() -> None:
    """The sideband control URL follows the resource endpoint, like the session URL.

    Regression: deriving it from the provider's `base_url` instead of the shared
    `_realtime_ws_base()` seam dropped Azure's `/openai/v1` whenever the endpoint wasn't already in
    the GA form, and the sideband dialed a path the resource doesn't serve.
    """
    provider = AzureProvider(
        azure_endpoint='https://resource.openai.azure.com', api_version='2024-10-01', api_key='azure-key'
    )
    model = AzureRealtimeModel('gpt-realtime', provider=provider)

    # The provider's own `base_url` is the SDK's `/openai/` form, which is exactly what made deriving
    # the sideband URL from it wrong.
    assert provider.base_url == 'https://resource.openai.azure.com/openai/'

    assert model._sideband_url('rtc_123') == (  # pyright: ignore[reportPrivateUsage]
        'wss://resource.openai.azure.com/openai/v1/realtime?call_id=rtc_123'
    )


@pytest.mark.anyio
async def test_connection_names_azure_not_openai() -> None:
    # The GA protocol is shared with OpenAI, but the vendor in a connection's messages must not be:
    # someone debugging a dropped or rejected Azure call would be sent to the wrong service.
    conn = AzureRealtimeConnection(_DroppedWebSocket())  # type: ignore[arg-type]
    events = [event async for event in conn]
    assert events == [
        RealtimeSessionErrorEvent(message='Azure OpenAI Realtime connection closed: dropped', recoverable=False)
    ]
    with pytest.raises(UserError, match='Azure OpenAI Realtime does not support'):
        await conn.send(cast('Any', object()))
    # The provider stamped onto content the connection can't carry names Azure too.
    assert conn._provider_name == 'azure'  # pyright: ignore[reportPrivateUsage]
    assert AzureRealtimeModel._connection_type is AzureRealtimeConnection  # pyright: ignore[reportPrivateUsage]


def test_profile_override_corrects_a_deployment_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # Azure's `model` is the *deployment* name, a user-chosen string that need not name the model, and
    # the profile is inferred from it (reasoning effort is only accepted by `gpt-realtime-2*`). A
    # deployment named anything else therefore loses `thinking` — `profile=` is the way to correct it,
    # mirroring `profile=` on a standard `Model`.
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://resource.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'azure-key')

    inferred = AzureRealtimeModel('voice-prod', settings={'thinking': 'low'})
    assert inferred.profile.get('supports_thinking') is False
    assert 'reasoning' not in inferred._session_config('', None, None)  # pyright: ignore[reportPrivateUsage]

    corrected = AzureRealtimeModel('voice-prod', settings={'thinking': 'low'}, profile={'supports_thinking': True})
    assert corrected.profile.get('supports_thinking') is True
    assert corrected._session_config('', None, None)['reasoning'] == {'effort': 'low'}  # pyright: ignore[reportPrivateUsage]
    # Everything the provider said is still there — `profile=` is a layer, not a replacement.
    assert corrected.profile.get('supports_image_input') is True


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
