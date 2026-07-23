"""Network-free tests for the realtime WebRTC signaling helpers and OpenAI-family model methods.

The browser <-> provider media path is exercised by the runnable example, not here. These tests pin the
server-side signaling that Pydantic AI owns: minting a client secret, relaying an SDP offer (the secure
topology), parsing the `call_id`, Azure Microsoft Entra ID token minting, and the capability gating of a
sideband session. The HTTP is driven through an `httpx.MockTransport` so the exact request shape and
response parsing are asserted deterministically offline.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import httpx
import pytest

from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestParameters

from ..conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime import WebRTCAnswer, WebRTCCall
    from pydantic_ai.realtime._openai_webrtc import parse_call_id
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]

SAMPLE_SDP_OFFER = 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
SAMPLE_SDP_ANSWER = 'v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=recvonly\r\n'


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


# --- client secret minting --------------------------------------------------------------------------


async def test_create_client_secret() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['url'] = str(request.url)
        captured['method'] = request.method
        captured['auth'] = request.headers.get('authorization')
        captured['content_type'] = request.headers.get('content-type')
        captured['body'] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                'value': 'ek_secret_123',
                'expires_at': 1_700_000_060,
                'session': {'type': 'realtime', 'model': 'gpt-realtime'},
            },
        )

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    secret = await model.create_client_secret(
        instructions='Be brief.',
        model_settings=OpenAIRealtimeModelSettings(voice='marin'),
        expires_after_seconds=60,
    )

    assert captured['method'] == 'POST'
    assert captured['url'] == 'https://api.openai.com/v1/realtime/client_secrets'
    assert captured['auth'] == 'Bearer sk-test'
    assert captured['content_type'] == 'application/json'
    # The session config is baked into the minted secret, including the resolved instructions and voice.
    assert captured['body']['expires_after'] == {'anchor': 'created_at', 'seconds': 60}
    session = captured['body']['session']
    assert session['instructions'] == 'Be brief.'
    assert session['audio']['output']['voice'] == 'marin'
    assert session['type'] == 'realtime'
    # The WebRTC signaling endpoints read the model from the session body (not a `?model=` query).
    assert session['model'] == 'gpt-realtime'

    assert secret.value == 'ek_secret_123'
    assert secret.expires_at == IsDatetime()
    assert secret.expires_at.tzinfo is not None
    assert secret.provider_details == {
        'expires_at': 1_700_000_060,
        'session': {'type': 'realtime', 'model': 'gpt-realtime'},
    }


async def test_create_client_secret_missing_value() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'expires_at': 1_700_000_060})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='did not include a `value`'):
        await model.create_client_secret()


async def test_create_client_secret_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='invalid api key')

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='401 error minting realtime client secret: invalid api key'):
        await model.create_client_secret()


# --- WebRTC offer relay -----------------------------------------------------------------------------


async def test_answer_webrtc_offer() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['url'] = str(request.url)
        captured['method'] = request.method
        captured['auth'] = request.headers.get('authorization')
        captured['accept'] = request.headers.get('accept')
        captured['content_type'] = request.headers.get('content-type')
        captured['body'] = request.content.decode()
        return httpx.Response(
            201,
            headers={'Location': '/v1/realtime/calls/rtc_call_9'},
            text=SAMPLE_SDP_ANSWER,
        )

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    answer = await model.answer_webrtc_offer(
        SAMPLE_SDP_OFFER,
        instructions='Answer in two words.',
        model_settings=OpenAIRealtimeModelSettings(voice='cedar'),
    )

    assert captured['method'] == 'POST'
    assert captured['url'] == 'https://api.openai.com/v1/realtime/calls'
    assert captured['auth'] == 'Bearer sk-test'
    assert captured['accept'] == 'application/sdp'
    # The offer and the session config are sent as a multipart body (httpx sets the boundary).
    assert captured['content_type'].startswith('multipart/form-data; boundary=')
    assert SAMPLE_SDP_OFFER in captured['body']
    assert '"instructions": "Answer in two words."' in captured['body']
    assert '"voice": "cedar"' in captured['body']
    assert '"model": "gpt-realtime"' in captured['body']

    assert answer == WebRTCAnswer(
        sdp=SAMPLE_SDP_ANSWER,
        call=WebRTCCall(
            provider_name='openai',
            call_id='rtc_call_9',
            provider_details={'location': '/v1/realtime/calls/rtc_call_9'},
        ),
    )


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


async def test_azure_answer_webrtc_offer_uses_api_key_and_webrtcfilter() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured['url'] = str(request.url)
        captured['api_key'] = request.headers.get('api-key')
        captured['auth'] = request.headers.get('authorization')
        return httpx.Response(201, headers={'Location': '/v1/realtime/calls/rtc_az'}, text=SAMPLE_SDP_ANSWER)

    model = AzureRealtimeModel('gpt-realtime', provider=_azure_mock_provider(handler))
    answer = await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)

    # Azure relays the offer with the `api-key` header (no bearer) and the `webrtcfilter=on` query.
    assert captured['url'] == 'https://resource.openai.azure.com/openai/v1/realtime/calls?webrtcfilter=on'
    assert captured['api_key'] == 'azure-key'
    assert captured['auth'] is None
    assert answer.call == WebRTCCall(
        provider_name='azure', call_id='rtc_az', provider_details={'location': '/v1/realtime/calls/rtc_az'}
    )


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


# --- sideband connect guards ------------------------------------------------------------------------


async def test_realtime_session_sideband_rejects_audio_retention() -> None:
    # A sideband session doesn't own the audio transport, so audio retention can never be satisfied.
    from pydantic_ai import Agent

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(_unused_handler))
    agent = Agent()
    call = WebRTCCall(provider_name='openai', call_id='rtc_x')
    with pytest.raises(UserError, match="can't retain audio"):
        async with agent.realtime_session(model, provider_session=call, audio_retention='input_audio'):
            pass  # pragma: no cover - raises before connecting


async def test_connect_webrtc_provider_mismatch() -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(_unused_handler))
    call = WebRTCCall(provider_name='azure', call_id='rtc_x')
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
    with pytest.raises(UserError, match='does not support WebRTC'):
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    with pytest.raises(UserError, match='does not support minting client secrets'):
        await model.create_client_secret()
    with pytest.raises(UserError, match='does not support WebRTC sideband sessions'):
        async with model.connect_webrtc(
            WebRTCCall(provider_name='ws-only', call_id='x'),
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        ):
            pass  # pragma: no cover - raises before yielding
