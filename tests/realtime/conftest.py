"""Realtime test configuration."""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from ..conftest import sanitize_filename, try_import
from .ws_cassettes import ProviderName, RealtimeCassette, patched_ws_connect, realtime_cassette_plan

with try_import() as imports_successful:
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider

with try_import() as xai_imports_successful:
    from pydantic_ai.providers.xai import XaiProvider

with try_import() as azure_imports_successful:
    from pydantic_ai.providers.azure import AzureProvider

if TYPE_CHECKING:
    from pydantic_ai.providers import Provider

CASSETTES_DIR = Path(__file__).parent / 'cassettes'

# Our Azure OpenAI dev resource, hardcoded (not a secret — like `test_azure_provider_call`) so a recorded
# HTTP host stays stable between recording (real key) and offline replay (placeholder key).
_AZURE_REALTIME_DEV_ENDPOINT = 'https://pydantic-ai-realtime-dev.openai.azure.com/openai/v1'

# A real WebRTC offer (generated once with `aiortc`, then stripped of host ICE candidates and with all
# addresses zeroed) that the OpenAI/Azure `/realtime/calls` endpoints accept and answer — so the
# signaling round-trip can be recorded against the live APIs instead of mocked. The leftover ICE ufrag /
# password / DTLS fingerprint are random per-session values, meaningless outside a live media session.
REAL_SDP_OFFER = (
    '\r\n'.join(
        """v=0
o=- 3993840254 3993840254 IN IP4 0.0.0.0
s=-
t=0 0
a=group:BUNDLE 0 1
a=msid-semantic:WMS *
m=audio 51603 UDP/TLS/RTP/SAVPF 96 9 0 8
c=IN IP4 0.0.0.0
a=sendrecv
a=extmap:1 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:2 urn:ietf:params:rtp-hdrext:ssrc-audio-level
a=mid:0
a=msid:993a573d-865c-4f89-b4d6-bf0023b36333 28299f45-cf21-4e7d-8945-3fc2846d1979
a=rtcp:9 IN IP4 0.0.0.0
a=rtcp-mux
a=ssrc:596951577 cname:54390b83-64d1-4178-a0ef-a2cafdb3f3a7
a=rtpmap:96 opus/48000/2
a=rtpmap:9 G722/8000
a=rtpmap:0 PCMU/8000
a=rtpmap:8 PCMA/8000
a=ice-ufrag:YnNx
a=ice-pwd:jIsRuXZmV9Yq00qk4a33Xe
a=fingerprint:sha-256 97:A7:E2:EF:70:B3:AD:B9:06:C8:DF:11:61:01:E5:6F:8F:46:EB:15:50:F2:54:D0:72:51:5B:37:0F:00:21:CB
a=setup:actpass
m=application 34376 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 0.0.0.0
a=mid:1
a=sctp-port:5000
a=max-message-size:65536
a=ice-ufrag:YnNx
a=ice-pwd:jIsRuXZmV9Yq00qk4a33Xe
a=fingerprint:sha-256 97:A7:E2:EF:70:B3:AD:B9:06:C8:DF:11:61:01:E5:6F:8F:46:EB:15:50:F2:54:D0:72:51:5B:37:0F:00:21:CB
a=setup:actpass""".strip().splitlines()
    )
    + '\r\n'
)


def _scrub_ephemeral_secret(response: dict[str, Any]) -> dict[str, Any]:
    """Redact the short-lived WebRTC client secret from recorded `/realtime/client_secrets` responses.

    The mint response body carries `{"value": "ek_..."}` — the ephemeral browser token. It expires in
    seconds and is useless offline, but replacing it keeps recorded cassettes free of anything
    secret-shaped. (The api-key / Entra bearer used to mint it are filtered out via `filter_headers`.)
    """
    try:
        raw = response['body']['string']
    except (KeyError, TypeError):  # non-body responses
        return response
    if not raw:  # empty body
        return response
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):  # non-JSON body
        return response
    if not isinstance(data, dict):  # non-object JSON body
        return response
    body_data = cast('dict[str, Any]', data)
    value = body_data.get('value')
    if isinstance(value, str) and value.startswith('ek_'):
        body_data['value'] = 'ek_scrubbed'
        body = json.dumps(body_data)
        response['body']['string'] = body.encode() if isinstance(raw, bytes) else body
    return response


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, Any]:
    """VCR config for realtime HTTP (WebRTC signaling) cassettes.

    Extends the repo default with Azure's `api-key` header (the WebSocket cassettes never record HTTP,
    so the default set omits it) and scrubs the minted ephemeral client secret from response bodies.
    """
    return {
        'ignore_localhost': True,
        'filter_headers': ['authorization', 'x-api-key', 'api-key', 'cookie'],
        'decode_compressed_response': True,
        'before_record_response': _scrub_ephemeral_secret,
    }


@pytest.fixture(autouse=True)
def _realtime_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder API keys so realtime models can resolve their default providers offline.

    The realtime models resolve their provider (and its API client) eagerly at construction, like
    `OpenAIChatModel` / `GoogleModel`. Network-free tests never hit the network, so a placeholder key
    is enough to let `OpenAIRealtimeModel()` / `GoogleRealtimeModel()` build their default providers.

    The cassette fixtures build their provider from the session-scoped `openai_api_key` /
    `gemini_api_key` fixtures, which are resolved before this (function-scoped) override runs and read
    a real key from the environment when recording, so this placeholder doesn't interfere with them.
    """
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')
    monkeypatch.setenv('XAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://mock.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'mock-api-key')


def _record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast('Any', request.config).getoption('record_mode')
    except (ValueError, AttributeError):  # pragma: no cover - depends on pytest-recording being active
        return None


@contextmanager
def _ws_cassette(
    request: pytest.FixtureRequest, provider: ProviderName, *, skip_if_missing: bool = False, subdir: str | None = None
) -> Generator[RealtimeCassette]:
    """Patch the provider's WebSocket transport to replay from / record into this test's cassette.

    `skip_if_missing` skips (rather than errors) when no cassette exists offline, for providers whose
    cassettes may not have been recorded yet (e.g. xAI, gated on realtime API access for our account).
    `subdir` overrides the cassette subdirectory (default: the test module), so a test that also records
    an HTTP VCR cassette (which uses the module-named subdirectory) doesn't collide with the WS cassette.
    """
    module = cast('str', request.node.fspath.basename).replace('.py', '')  # pyright: ignore[reportUnknownMemberType]
    name = sanitize_filename(cast('str', request.node.name), 240)  # pyright: ignore[reportUnknownMemberType]
    path = CASSETTES_DIR / (subdir or module) / f'{name}.yaml'
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=_record_mode(request))
    if plan == 'error_missing':  # pragma: no cover - only when a cassette is missing offline
        if skip_if_missing:
            pytest.skip(f'Missing realtime WebSocket cassette (record with `--record-mode=rewrite`): {path}')
        raise RuntimeError(
            f'Missing realtime WebSocket cassette: {path}\n'
            'Record it with: uv run --env-file .env pytest --record-mode=rewrite <test> -q'
        )
    cassette = RealtimeCassette.load(path) if plan == 'replay' else RealtimeCassette()
    try:
        with patched_ws_connect(provider, cassette, plan):
            yield cassette
    finally:
        # Persist recorded frames even if later assertions fail, so cassettes can be recorded first
        # and snapshots filled from replay afterwards (mirroring the VCR workflow).
        if plan == 'record' and cassette.interactions:  # pragma: no cover - only runs while recording
            cassette.dump(path)


@pytest.fixture
def openai_ws_cassette(
    request: pytest.FixtureRequest, openai_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An `OpenAIProvider` whose realtime WebSocket is backed by a cassette."""
    if not imports_successful():
        pytest.skip('openai / websockets not installed')
    with _ws_cassette(request, 'openai') as cassette:
        yield OpenAIProvider(api_key=openai_api_key), cassette


@pytest.fixture
def openai_ws_sideband_cassette(
    request: pytest.FixtureRequest, openai_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An `OpenAIProvider` whose realtime sideband control WebSocket is cassette-backed.

    Stored under a dedicated subdirectory so the WebSocket cassette doesn't collide with the HTTP VCR
    cassette (SDP offer relay) a WebRTC sideband test records under the module-named subdirectory.
    """
    if not imports_successful():
        pytest.skip('openai / websockets not installed')
    with _ws_cassette(request, 'openai', subdir='test_openai_ws_sideband') as cassette:
        yield OpenAIProvider(api_key=openai_api_key), cassette


@pytest.fixture
def gemini_ws_cassette(
    request: pytest.FixtureRequest, gemini_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """A `GoogleProvider` whose Gemini Live WebSocket is backed by a cassette."""
    if not imports_successful():  # pragma: no cover
        pytest.skip('google-genai not installed')
    with _ws_cassette(request, 'gemini') as cassette:
        yield GoogleProvider(api_key=gemini_api_key), cassette


@pytest.fixture
def xai_ws_cassette(request: pytest.FixtureRequest, xai_api_key: str) -> Iterator[tuple[XaiProvider, RealtimeCassette]]:
    """An `XaiProvider` whose Grok Voice realtime WebSocket is backed by a cassette.

    Skips (rather than errors) when the cassette is missing offline: recording requires xAI realtime
    API access, which our account may not have, so these cassettes may not be present.
    """
    if not xai_imports_successful():  # pragma: no cover
        pytest.skip('xai-sdk / websockets not installed')
    with _ws_cassette(request, 'xai', skip_if_missing=True) as cassette:
        yield XaiProvider(api_key=xai_api_key), cassette


@pytest.fixture(scope='session')
def azure_config() -> tuple[str, str]:
    """Capture real Azure OpenAI configuration before offline placeholders apply."""
    return (
        os.getenv('AZURE_OPENAI_ENDPOINT', 'https://mock.openai.azure.com'),
        os.getenv('AZURE_OPENAI_API_KEY', 'mock-api-key'),
    )


@pytest.fixture
def azure_ws_cassette(
    request: pytest.FixtureRequest, azure_config: tuple[str, str]
) -> Iterator[tuple[AzureProvider, RealtimeCassette]]:
    """An `AzureProvider` whose Azure OpenAI realtime WebSocket is cassette-backed."""
    if not azure_imports_successful():  # pragma: no cover
        pytest.skip('openai / websockets not installed')
    endpoint, api_key = azure_config
    with _ws_cassette(request, 'openai') as cassette:
        yield AzureProvider(azure_endpoint=f'{endpoint.rstrip("/")}/openai/v1', api_key=api_key), cassette


@pytest.fixture
def azure_ws_sideband_cassette(
    request: pytest.FixtureRequest, azure_config: tuple[str, str]
) -> Iterator[tuple[AzureProvider, RealtimeCassette]]:
    """An `AzureProvider` whose realtime sideband control WebSocket is cassette-backed.

    Like `openai_ws_sideband_cassette`, the WebSocket cassette lives under a dedicated subdirectory so it
    doesn't collide with the HTTP VCR cassette (the two-step SDP offer relay) an Azure WebRTC sideband
    test records under the module-named subdirectory.
    """
    if not azure_imports_successful():  # pragma: no cover
        pytest.skip('openai / websockets not installed')
    # A sideband test also records an HTTP VCR cassette (the two-step SDP relay), which matches on host,
    # so pin our dev resource endpoint (not a secret — like `test_azure_provider_call`) rather than the
    # `azure_config` one, which is a placeholder offline. The api-key is filtered out of the cassette.
    _, api_key = azure_config
    with _ws_cassette(request, 'openai', subdir='test_azure_ws_sideband') as cassette:
        yield AzureProvider(azure_endpoint=_AZURE_REALTIME_DEV_ENDPOINT, api_key=api_key), cassette
