"""Realtime test configuration."""

from __future__ import annotations as _annotations

import json
import os
import re
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from ..conftest import sanitize_filename, try_import
from .ws_cassettes import ProviderName, RealtimeCassette, patched_ws_connect, realtime_cassette_plan

# `imports_successful` gates only the `google-genai` import, so a Gemini cassette test is the only one
# that needs it. `gateway_provider` has no optional dependency of its own, so it gets its own flag
# rather than riding on the Google one — otherwise a gateway/OpenAI cassette test (which needs neither
# Google) would skip whenever `google-genai` is absent.
with try_import() as imports_successful:
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as gateway_imports_successful:
    from pydantic_ai.providers.gateway import gateway_provider

# Separate from the flags above so OpenAI cassette tests still run in an environment
# without `google-genai` installed.
with try_import() as openai_imports_successful:
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


# The address fields of an SDP offer: the `c=` connection line and the address in an ICE candidate
# (`a=candidate:<foundation> <component> <transport> <priority> <address> <port> ...`).
_SDP_ADDRESS_RE = re.compile(rb'^(?P<prefix>c=IN IP[46] |a=candidate:\S+ \d+ \S+ \d+ )(?P<address>\S+)', re.MULTILINE)


def _zero_sdp_addresses(request: Any) -> Any:
    """Zero out the network addresses in a recorded SDP offer.

    A cassette recorded against a *live* WebRTC peer (see `_webrtc_media_peer` — the only way to get
    the provider to report playback) otherwise commits the recorder's own machine addresses. Nothing
    replays or matches on a recorded request body, so blanking them costs nothing, and it keeps
    hand-zeroing them (as `REAL_SDP_OFFER` above was) from being a step someone has to remember.
    """
    body = request.body
    if isinstance(body, bytes):
        # Zero every address the regex finds, not just when an ICE candidate is present: an SDP whose
        # only address is the `c=IN IP4/IP6` connection line (no `a=candidate:` lines) would otherwise
        # be recorded with the recorder's real address intact.
        request.body = _SDP_ADDRESS_RE.sub(
            lambda match: match['prefix'] + (b'0.0.0.0' if b'.' in match['address'] else b'::'), body
        )
    return request


@pytest.fixture(scope='module')
def vcr_config() -> dict[str, Any]:
    """VCR config for realtime HTTP (WebRTC signaling) cassettes.

    Extends the repo default with Azure's `api-key` header (the WebSocket cassettes never record HTTP,
    so the default set omits it), scrubs the minted ephemeral client secret from response bodies, and
    zeroes the network addresses in a recorded SDP offer.
    """
    return {
        'ignore_localhost': True,
        'filter_headers': ['authorization', 'x-api-key', 'api-key', 'cookie'],
        'decode_compressed_response': True,
        'before_record_request': _zero_sdp_addresses,
        'before_record_response': _scrub_ephemeral_secret,
    }


@pytest.fixture(autouse=True)
def _realtime_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide placeholder API keys so realtime models can resolve their default providers offline.

    The realtime models resolve their provider (and its API client) eagerly at construction, like
    `OpenAIChatModel` / `GoogleModel`. Network-free tests never hit the network, so a placeholder key
    is enough to let realtime models build their default providers.

    The cassette fixtures build their provider from the session-scoped `openai_api_key` /
    `gemini_api_key` fixtures, which are resolved before this (function-scoped) override runs and read
    a real key from the environment when recording, so this placeholder doesn't interfere with them.
    """
    monkeypatch.setenv('OPENAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('GOOGLE_API_KEY', 'mock-api-key')
    monkeypatch.setenv('XAI_API_KEY', 'mock-api-key')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://mock.openai.azure.com/openai/v1')
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', 'mock-api-key')
    # Voice Live is a distinct resource with its own credential set. Tests build their provider from the
    # Azure OpenAI values above (Voice Live falls back to them), so clear any real `AZURE_VOICELIVE_*` a
    # developer has exported — otherwise it takes precedence and breaks the fallback-URL assertions.
    for _var in ('AZURE_VOICELIVE_ENDPOINT', 'AZURE_VOICELIVE_API_KEY', 'AZURE_VOICELIVE_API_VERSION'):
        monkeypatch.delenv(_var, raising=False)


def _record_mode(request: pytest.FixtureRequest) -> str | None:
    try:
        return cast('Any', request.config).getoption('record_mode')
    # Depends on pytest-recording being active.
    except (ValueError, AttributeError):  # pragma: no cover
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
    if plan == 'error_missing':
        if skip_if_missing:  # pragma: no cover
            # Only reachable in a checkout where the cassette was never recorded, so it can't be
            # covered by a suite that ships the cassettes it replays.
            pytest.skip(f'Missing realtime WebSocket cassette (record with `--record-mode=rewrite`): {path}')
        # A cassette we expect to exist has gone missing.
        raise RuntimeError(  # pragma: no cover
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
        # Only runs while recording.
        if plan == 'record' and cassette.interactions:  # pragma: no cover
            cassette.dump(path)


@pytest.fixture
def openai_ws_cassette(
    request: pytest.FixtureRequest, openai_api_key: str
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An `OpenAIProvider` whose realtime WebSocket is backed by a cassette."""
    if not openai_imports_successful():  # pragma: no cover
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
    if not openai_imports_successful():  # pragma: no cover
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


def _gateway_realtime_provider(kind: str, api_key: str | None) -> Provider[Any]:
    """Build a gateway provider for realtime, mirroring how `gateway/<kind>:...` resolves for a user.

    With a real key, the gateway base URL is inferred from the key's encoded region — the exact path a
    user reaches. Offline (no key), the placeholder encodes no region, so pin an explicit base URL;
    replay never dials, so only its stability matters, not the host.
    """
    # Only while recording.
    if api_key:  # pragma: no cover
        return gateway_provider(kind, api_key=api_key)
    return gateway_provider(kind, api_key='mock-gateway-key', base_url='https://gateway.pydantic.info/proxy')


@pytest.fixture
def gateway_openai_ws_cassette(
    request: pytest.FixtureRequest, gateway_api_key: str | None
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """An OpenAI realtime provider that routes through the Pydantic AI Gateway, cassette-backed.

    The gateway relays OpenAI's realtime WebSocket verbatim, so the same OpenAI transport (and its
    `websockets` reference) is patched; only the provider's base URL and bearer key differ. Recording
    needs a real `PYDANTIC_AI_GATEWAY_API_KEY`; offline replay never dials, so a placeholder is enough.
    """
    if not (gateway_imports_successful() and openai_imports_successful()):  # pragma: no cover
        pytest.skip('gateway / openai / websockets not installed')
    provider = _gateway_realtime_provider('openai', gateway_api_key)
    with _ws_cassette(request, 'openai') as cassette:
        yield provider, cassette


@pytest.fixture
def gateway_gemini_ws_cassette(
    request: pytest.FixtureRequest, gateway_api_key: str | None
) -> Iterator[tuple[Provider[Any], RealtimeCassette]]:
    """A Gemini Live provider that routes through the gateway's Vertex upstream, cassette-backed.

    The `google-genai` SDK dials the native Vertex Bidi path
    (`/proxy/<route>/ws/google.cloud.aiplatform.v1beta1.LlmBidiService/BidiGenerateContent`) rather than
    the OpenAI-shaped `/proxy/<route>/realtime` upgrade the other gateway route uses, so this fixture
    covers the second protocol the gateway relays.
    """
    if not (gateway_imports_successful() and imports_successful()):  # pragma: no cover
        pytest.skip('gateway / google-genai / websockets not installed')
    provider = _gateway_realtime_provider('google', gateway_api_key)
    with _ws_cassette(request, 'gemini') as cassette:
        yield provider, cassette


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
    # Mirror `AzureProvider.for_realtime`'s normalization: only append `/openai/v1` when the
    # configured endpoint doesn't already end with it, so an env already set to the GA form
    # doesn't dial `.../openai/v1/openai/v1`. Replay uses a suffix-less placeholder endpoint, so
    # only recording against a GA-form env ever takes the other branch.
    if not endpoint.rstrip('/').endswith('/openai/v1'):  # pragma: no branch
        endpoint = f'{endpoint.rstrip("/")}/openai/v1'
    with _ws_cassette(request, 'openai') as cassette:
        yield AzureProvider(azure_endpoint=endpoint, api_key=api_key), cassette


@pytest.fixture
def azure_voice_live_ws_cassette(
    request: pytest.FixtureRequest, azure_config: tuple[str, str]
) -> Iterator[tuple[AzureProvider, RealtimeCassette]]:
    """An `AzureProvider` whose Azure AI Voice Live WebSocket is cassette-backed."""
    if not azure_imports_successful():  # pragma: no cover
        pytest.skip('openai / websockets not installed')
    endpoint, api_key = azure_config
    endpoint = endpoint.partition('/openai/')[0].rstrip('/')
    with _ws_cassette(request, 'openai') as cassette:
        yield AzureProvider(azure_endpoint=endpoint, api_version='2026-04-10', api_key=api_key), cassette


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


@pytest.fixture
def parity_ws_cassette(
    request: pytest.FixtureRequest,
    openai_api_key: str,
    gemini_api_key: str,
    xai_api_key: str,
    azure_config: tuple[str, str],
    gateway_api_key: str | None,
) -> Iterator[tuple[Any, Provider[Any], RealtimeCassette]]:
    """Build an indirectly parametrized parity-matrix provider before placeholder keys take effect."""
    case, route = cast('tuple[Any, str]', request.param)
    provider_name: ProviderName
    if route == 'openai':
        provider = OpenAIProvider(api_key=openai_api_key)
        provider_name = 'openai'
    elif route == 'azure':
        endpoint, api_key = azure_config
        # Same GA-form normalization as `azure_ws_cassette` above; replay's placeholder endpoint
        # never carries the suffix, so only live recording takes the other branch.
        if not endpoint.rstrip('/').endswith('/openai/v1'):  # pragma: no branch
            endpoint = f'{endpoint.rstrip("/")}/openai/v1'
        provider = AzureProvider(azure_endpoint=endpoint, api_key=api_key)
        provider_name = 'openai'
    elif route == 'xai':
        provider = XaiProvider(api_key=xai_api_key)
        provider_name = 'xai'
    elif route == 'google':
        provider = GoogleProvider(api_key=gemini_api_key)
        provider_name = 'gemini'
    elif route == 'gateway-openai':
        provider = _gateway_realtime_provider('openai', gateway_api_key)
        provider_name = 'openai'
    else:
        assert route == 'gateway-google'
        provider = _gateway_realtime_provider('google', gateway_api_key)
        provider_name = 'gemini'

    with _ws_cassette(request, provider_name) as cassette:
        yield case, provider, cassette
