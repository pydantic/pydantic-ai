"""Tests for the realtime WebRTC signaling helpers and OpenAI-family model methods.

The browser <-> provider media path is exercised by the runnable example, not here. These tests pin the
server-side signaling that Pydantic AI owns: minting a client secret, relaying an SDP offer (the secure
topology), parsing the `call_id`, Azure Microsoft Entra ID token minting, and the capability gating of a
sideband session. Success paths that depend on provider behavior use recorded HTTP cassettes; focused
unit tests use `httpx.MockTransport` only for our own guards, error formatting, and request shaping.
"""

from __future__ import annotations as _annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import
from .conftest import _scrub_ephemeral_secret  # pyright: ignore[reportPrivateUsage]

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import WebRTCSession
    from pydantic_ai.realtime._openai_webrtc import parse_call_id
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]

SAMPLE_SDP_OFFER = 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
SAMPLE_SDP_ANSWER = 'v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=recvonly\r\n'

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

# Our Azure OpenAI dev resource, hardcoded (not a secret — like `test_azure_provider_call`) so the
# recorded host is stable between recording (real key) and offline replay (placeholder key).
_AZURE_REALTIME_ENDPOINT = 'https://pydantic-ai-realtime-dev.openai.azure.com/openai/v1'


def _mock_provider(handler: Any, *, api_key: str = 'sk-test') -> Any:
    """An `OpenAIProvider` whose HTTP calls are served by `handler` instead of the network."""
    return OpenAIProvider(api_key=api_key, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _unused_handler(request: httpx.Request) -> httpx.Response:
    """A transport handler for tests whose guard raises before any HTTP request is made."""
    raise AssertionError('no HTTP request expected')  # pragma: no cover


# --- call_id parsing --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('location', 'expected'),
    [
        ('/v1/realtime/calls/rtc_abc123', 'rtc_abc123'),
        ('https://api.openai.com/v1/realtime/calls/rtc_XYZ', 'rtc_XYZ'),
        ('https://host/realtime?call_id=rtc_q', 'rtc_q'),
        (None, None),
        ('', None),
        ('/v1/realtime/sessions/sess_1', None),  # not a `.../calls/<id>` path
    ],
)
def test_parse_call_id(location: str | None, expected: str | None) -> None:
    assert parse_call_id(location) == expected


def test_scrub_ephemeral_secret_redacts_client_secret() -> None:
    """The VCR `before_record_response` hook redacts the minted `ek_...` client secret from recorded bodies.

    A unit test because the hook only runs while *recording* a cassette; offline replay never invokes it,
    so a cassette test can't reach it — yet it's the guard that keeps recorded signaling cassettes free of
    anything secret-shaped.
    """
    minted = {'body': {'string': json.dumps({'value': 'ek_live_secret', 'expires_at': 1}).encode()}}
    assert json.loads(_scrub_ephemeral_secret(minted)['body']['string'])['value'] == 'ek_scrubbed'
    # A non-secret JSON body is returned unchanged.
    other = {'body': {'string': b'{"foo": "bar"}'}}
    assert _scrub_ephemeral_secret(other)['body']['string'] == b'{"foo": "bar"}'


# --- client secret minting --------------------------------------------------------------------------


@pytest.mark.vcr
async def test_create_client_secret(openai_api_key: str, request: pytest.FixtureRequest) -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key=openai_api_key))
    secret = await model.create_client_secret(
        instructions='Be brief.',
        model_settings=OpenAIRealtimeModelSettings(voice='marin'),
        expires_after_seconds=60,
    )

    assert secret.value
    assert secret.expires_at.tzinfo is not None
    # The secret expires shortly after recording, so only a live response can remain future-dated.
    if request.config.getoption('record_mode') == 'rewrite':  # pragma: no cover - only while recording
        assert secret.expires_at > datetime.now(timezone.utc)


async def test_create_client_secret_missing_value() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'expires_at': 1_700_000_060})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='did not include a `value`'):
        await model.create_client_secret()


async def test_create_client_secret_non_numeric_expires_at() -> None:
    # A `value` with a non-integer `expires_at` can't be turned into an expiry timestamp, so it's rejected.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'value': 'ek_x', 'expires_at': 'soon'})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='numeric'):
        await model.create_client_secret()


async def test_create_client_secret_through_gateway() -> None:
    # A gateway-routed provider's base URL ends at `.../openai`; the gateway accepts the `/v1`-less
    # signaling path, so the client-secret URL is derived straight from that base without a `/v1` segment.
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['url'] = str(request.url)
        return httpx.Response(200, json={'value': 'ek_gw', 'expires_at': 1_700_000_060})

    provider = gateway_provider(
        'openai',
        api_key='gw-key',
        base_url='https://gateway.pydantic.dev/proxy',
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    secret = await model.create_client_secret()

    assert captured['url'] == 'https://gateway.pydantic.dev/proxy/openai/realtime/client_secrets'
    assert secret.value == 'ek_gw'


async def test_create_client_secret_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='invalid api key')

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='401 error minting realtime client secret: invalid api key'):
        await model.create_client_secret()


# --- WebRTC offer relay -----------------------------------------------------------------------------


@pytest.mark.vcr
async def test_answer_webrtc_offer(openai_api_key: str) -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key=openai_api_key))
    answer = await model.answer_webrtc_offer(
        REAL_SDP_OFFER,
        instructions='Answer in two words.',
        model_settings=OpenAIRealtimeModelSettings(voice='cedar'),
    )

    assert answer.session.provider_name == 'openai'
    assert answer.session.call_id.startswith('rtc_')
    assert answer.sdp.startswith('v=0')


async def test_answer_webrtc_offer_missing_location() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, text=SAMPLE_SDP_ANSWER)  # no Location header

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='did not return a parseable `call_id`'):
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)


async def test_answer_webrtc_offer_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text='bad sdp')

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='400 error negotiating realtime WebRTC call: bad sdp'):
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)


# --- Azure Microsoft Entra ID + endpoints -----------------------------------------------------------


def _azure_mock_provider(handler: Any) -> Any:
    return AzureProvider(
        azure_endpoint='https://resource.openai.azure.com/openai/v1/',
        api_key='azure-key',
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


class _FakeAccessToken:
    def __init__(self, token: str) -> None:
        self.token = token
        self.expires_on = 1_700_000_000


class _FakeCredential:
    """A minimal `TokenCredential` stand-in that records the requested scope."""

    def __init__(self) -> None:
        self.scopes: tuple[str, ...] | None = None

    def get_token(self, *scopes: str, **kwargs: Any) -> _FakeAccessToken:
        self.scopes = scopes
        return _FakeAccessToken('entra-token-xyz')


async def test_azure_entra_credential_mints_client_secret_with_bearer() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['url'] = str(request.url)
        captured['api_key'] = request.headers.get('api-key')
        captured['auth'] = request.headers.get('authorization')
        return httpx.Response(200, json={'value': 'ek_az', 'expires_at': 1_700_000_060})

    credential = _FakeCredential()
    model = AzureRealtimeModel('gpt-realtime', provider=_azure_mock_provider(handler), credential=credential)
    secret = await model.create_client_secret(instructions='Hi.')

    # With an Entra credential, signaling uses a bearer token (data-plane scope) and never the api-key.
    assert credential.scopes == ('https://ai.azure.com/.default',)
    assert captured['url'] == 'https://resource.openai.azure.com/openai/v1/realtime/client_secrets'
    assert captured['auth'] == 'Bearer entra-token-xyz'
    assert captured['api_key'] is None
    assert secret.value == 'ek_az'


# --- recorded signaling round-trips (real APIs) -----------------------------------------------------


@pytest.mark.vcr
async def test_azure_answer_webrtc_offer_records(azure_config: tuple[str, str]) -> None:
    """Azure's two-step WebRTC negotiation, recorded against the real API.

    Azure's `/realtime/calls` rejects the api-key with a 401, so `answer_webrtc_offer` mints an ephemeral
    client secret first, then relays the raw SDP offer with it. This exercises that end to end — the path
    the old `MockTransport` test asserted incorrectly (it mimicked OpenAI's single-step multipart relay,
    which Azure never accepts).
    """
    _, api_key = azure_config
    provider = AzureProvider(azure_endpoint=_AZURE_REALTIME_ENDPOINT, api_key=api_key)
    model = AzureRealtimeModel('gpt-realtime', provider=provider)

    answer = await model.answer_webrtc_offer(REAL_SDP_OFFER, instructions='Answer in two or three words.')

    assert answer.session.provider_name == 'azure'
    assert answer.session.call_id.startswith('rtc_')
    assert answer.sdp.startswith('v=0')


# --- sideband connect guards ------------------------------------------------------------------------


async def test_realtime_session_sideband_rejects_audio_retention() -> None:
    # A sideband session doesn't own the audio transport, so audio retention can never be satisfied.
    from pydantic_ai import Agent

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(_unused_handler))
    agent = Agent()
    call = WebRTCSession(provider_name='openai', session_id='rtc_x')
    with pytest.raises(UserError, match="can't retain audio"):
        async with agent.realtime(model).session(provider_session=call, audio_retention='input_audio'):
            pass  # pragma: no cover - raises before connecting


async def test_connect_webrtc_provider_mismatch() -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(_unused_handler))
    call = WebRTCSession(provider_name='azure', session_id='rtc_x')
    with pytest.raises(UserError, match='was negotiated by provider'):
        async with model.connect_webrtc(
            call, messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
        ):
            pass  # pragma: no cover - the mismatch raises before yielding


async def test_base_model_rejects_webrtc() -> None:
    # WebSocket-only realtime models (Gemini Live, and xAI — which has no `/realtime/calls` sideband)
    # don't override the WebRTC methods, so the base `RealtimeModel` rejects the whole surface.
    from contextlib import AbstractAsyncContextManager

    from pydantic_ai.realtime import RealtimeModel
    from pydantic_ai.realtime.codec import RealtimeConnection

    class _WebSocketOnlyModel(RealtimeModel):
        @property
        def model_name(self) -> str:
            return 'ws-only'

        @property
        def system(self) -> str:
            return 'ws-only'

        def connect(self, **kwargs: Any) -> AbstractAsyncContextManager[RealtimeConnection]:
            raise NotImplementedError  # pragma: no cover - not exercised by these guard tests

    model = _WebSocketOnlyModel()
    # The base `RealtimeModel` reads these to build its "unsupported" errors, so pin the stand-in's identity.
    assert model.system == 'ws-only'
    assert model.model_name == 'ws-only'
    with pytest.raises(NotImplementedError, match='does not support WebRTC'):
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    with pytest.raises(NotImplementedError, match='does not support minting client secrets'):
        await model.create_client_secret()
    with pytest.raises(NotImplementedError, match='does not support WebRTC sideband sessions'):
        async with model.connect_webrtc(
            WebRTCSession(provider_name='ws-only', session_id='x'),
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        ):
            pass  # pragma: no cover - raises before yielding
