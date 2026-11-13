"""What a realtime model raises when the provider rejects the connection with an HTTP status.

A rejected WebSocket upgrade, WebRTC signaling request or REST preflight raises what the same response to a regular
request raises: the error category its status and the provider's error code give it, with the provider's error code
and type, the response headers (so a 429's `Retry-After` reaches `retry_after`), and the body as text.

The upgrades are real: each model dials, through the `websockets` client, a local server that rejects the upgrade
with the case's response. The HTTP requests are served by an `httpx2.MockTransport`.
"""

from __future__ import annotations as _annotations

import platform
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal, assert_never
from unittest import mock

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    import httpx2
    import websockets
    from google.genai import live as genai_live
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.datastructures import Headers
    from websockets.http11 import Request, Response

    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime import RealtimeModel, WebRTCSession
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel
    from pydantic_ai.realtime.elevenlabs_agents import ElevenLabsAgentsRealtimeModel
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel
    from pydantic_ai.realtime.openai_live import OpenAILiveModel
    from pydantic_ai.realtime.xai import XaiRealtimeModel

    from .test_elevenlabs_agents import AGENT_ID, RestRecorder, agent_json

pytestmark = pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed')


@dataclass(frozen=True)
class Rejection:
    status: int
    body: bytes = b''
    headers: tuple[tuple[str, str], ...] = ()


OPENAI_INVALID_KEY = (
    b'{"error": {"message": "Incorrect API key provided: sk-test.", "type": "invalid_request_error", '
    b'"param": null, "code": "invalid_api_key"}}'
)
OPENAI_RATE_LIMIT = (
    b'{"error": {"message": "Rate limit reached.", "type": "requests", "param": null, "code": "rate_limit_exceeded"}}'
)
OPENAI_QUOTA = (
    b'{"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota", "param": null, '
    b'"code": "insufficient_quota"}}'
)
GOOGLE_INVALID_KEY = (
    b'{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", '
    b'"status": "INVALID_ARGUMENT", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", '
    b'"reason": "API_KEY_INVALID", "domain": "googleapis.com"}]}}'
)
GOOGLE_RESOURCE_EXHAUSTED = (
    b'{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", '
    b'"status": "RESOURCE_EXHAUSTED"}}'
)
RETRY_AFTER_5 = (('Retry-After', '5'),)
REPEATED_SET_COOKIE = (('Set-Cookie', 'a=1'), ('Set-Cookie', 'b=2'))
MESSAGES = [ModelRequest(parts=[])]


def describe(error: ModelHTTPError) -> dict[str, object]:
    """The fields of a raised `ModelHTTPError` a caller's handling depends on."""
    return {
        'class': type(error).__name__,
        'status_code': error.status_code,
        'provider_error_code': error.provider_error_code,
        'provider_error_type': error.provider_error_type,
        'retry_after': error.retry_after,
        'body': error.body,
        'headers': error.headers,
    }


class Rejector:
    """A local WebSocket server that rejects every upgrade with the current `rejection`."""

    def __init__(self) -> None:
        self.rejection = Rejection(500)

    def process_request(self, connection: ServerConnection, request: Request) -> Response:
        body = self.rejection.body
        headers = Headers([*self.rejection.headers, ('Content-Length', str(len(body)))])
        return Response(self.rejection.status, 'Rejected', headers, body)


async def _never_accepted(connection: ServerConnection) -> None:
    raise AssertionError('every upgrade is rejected')  # pragma: no cover


@pytest.fixture
async def rejector() -> AsyncIterator[Rejector]:
    """A rejecting server that every realtime model's WebSocket dial reaches, whatever URL it builds."""
    rejector = Rejector()
    async with serve(
        _never_accepted, '127.0.0.1', 0, process_request=rejector.process_request, server_header=None
    ) as server:
        host, port = next(iter(server.sockets)).getsockname()[:2]

        def dial(uri: str, **kwargs: Any) -> websocket_connect:
            # `google-genai` passes a TLS context for its `wss://` URL; the local server is plain text.
            kwargs.pop('ssl', None)
            return websocket_connect(f'ws://{host}:{port}', **kwargs)

        with mock.patch.object(websockets, 'connect', dial), mock.patch.object(genai_live, 'ws_connect', dial):
            yield rejector


def _azure_provider(http_client: httpx2.AsyncClient | None = None) -> AzureProvider:
    return AzureProvider(
        azure_endpoint='https://resource.openai.azure.com/openai/v1/', api_key='k', http_client=http_client
    )


def _elevenlabs_agents_model(http_client: httpx2.AsyncClient) -> ElevenLabsAgentsRealtimeModel:
    return ElevenLabsAgentsRealtimeModel(AGENT_ID, provider=ElevenLabsProvider(api_key='k', http_client=http_client))


MODELS: dict[str, Callable[[], RealtimeModel]] = {
    'openai': lambda: OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='sk-test')),
    'azure': lambda: AzureRealtimeModel('gpt-realtime', provider=_azure_provider()),
    'elevenlabs': lambda: ElevenLabsRealtimeModel('eleven_realtime_v1_mini', provider=ElevenLabsProvider(api_key='k')),
    'xai': lambda: XaiRealtimeModel('grok-voice-latest', provider=XaiProvider(api_key='k')),
    'openai_live': lambda: OpenAILiveModel('gpt-live-1', provider=OpenAIProvider(api_key='sk-test')),
    'google': lambda: GoogleRealtimeModel('gemini-2.5-flash-native-audio-latest', provider=GoogleProvider(api_key='k')),
    # The agent preflight succeeds, so the model goes on to dial the WebSocket.
    'elevenlabs_agents': lambda: _elevenlabs_agents_model(
        httpx2.AsyncClient(transport=httpx2.MockTransport(RestRecorder(agent_json())))
    ),
}


@dataclass(frozen=True)
class UpgradeCase:
    id: str
    model: str
    rejection: Rejection
    expected: dict[str, object]


UPGRADE_CASES = [
    # OpenAI's error code picks the category, as over HTTP: a 429 is a rate limit or an exhausted quota.
    UpgradeCase(
        id='openai-401-invalid-api-key',
        model='openai',
        rejection=Rejection(401, OPENAI_INVALID_KEY),
        expected=snapshot(
            {
                'class': 'ModelAuthenticationError',
                'status_code': 401,
                'provider_error_code': 'invalid_api_key',
                'provider_error_type': 'invalid_request_error',
                'retry_after': None,
                'body': '{"error": {"message": "Incorrect API key provided: sk-test.", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}',
                'headers': {'content-length': '137'},
            }
        ),
    ),
    UpgradeCase(
        id='openai-429-rate-limit',
        model='openai',
        rejection=Rejection(429, OPENAI_RATE_LIMIT, RETRY_AFTER_5),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': 'rate_limit_exceeded',
                'provider_error_type': 'requests',
                'retry_after': 5.0,
                'body': '{"error": {"message": "Rate limit reached.", "type": "requests", "param": null, "code": "rate_limit_exceeded"}}',
                'headers': {'retry-after': '5', 'content-length': '111'},
            }
        ),
    ),
    UpgradeCase(
        id='openai-429-insufficient-quota',
        model='openai',
        rejection=Rejection(429, OPENAI_QUOTA),
        expected=snapshot(
            {
                'class': 'ModelQuotaExceededError',
                'status_code': 429,
                'provider_error_code': 'insufficient_quota',
                'provider_error_type': 'insufficient_quota',
                'retry_after': None,
                'body': '{"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota", "param": null, "code": "insufficient_quota"}}',
                'headers': {'content-length': '133'},
            }
        ),
    ),
    UpgradeCase(
        id='openai-404-plain-body',
        model='openai',
        rejection=Rejection(404, b'Not Found'),
        expected=snapshot(
            {
                'class': 'ModelNotFoundError',
                'status_code': 404,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': 'Not Found',
                'headers': {'content-length': '9'},
            }
        ),
    ),
    UpgradeCase(
        id='openai-503-empty-body',
        model='openai',
        rejection=Rejection(503),
        expected=snapshot(
            {
                'class': 'ModelUnavailableError',
                'status_code': 503,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': 'Rejected',
                'headers': {'content-length': '0'},
            }
        ),
    ),
    # A repeated header can't go into a plain `dict`: its values are joined, as `httpx` joins them.
    UpgradeCase(
        id='openai-429-repeated-header',
        model='openai',
        rejection=Rejection(429, b'Too Many Requests', (*RETRY_AFTER_5, *REPEATED_SET_COOKIE)),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': 5.0,
                'body': 'Too Many Requests',
                'headers': {'retry-after': '5', 'set-cookie': 'a=1, b=2', 'content-length': '17'},
            }
        ),
    ),
    # A status with no error category still raises a plain `ModelHTTPError`, whatever its body's error code.
    UpgradeCase(
        id='openai-200-not-upgraded',
        model='openai',
        rejection=Rejection(200, OPENAI_RATE_LIMIT),
        expected=snapshot(
            {
                'class': 'ModelHTTPError',
                'status_code': 200,
                'provider_error_code': 'rate_limit_exceeded',
                'provider_error_type': 'requests',
                'retry_after': None,
                'body': '{"error": {"message": "Rate limit reached.", "type": "requests", "param": null, "code": "rate_limit_exceeded"}}',
                'headers': {'content-length': '111'},
            }
        ),
    ),
    # Every other model dialing through the OpenAI-protocol connect mapper.
    UpgradeCase(
        id='azure-429-rate-limit',
        model='azure',
        rejection=Rejection(429, OPENAI_RATE_LIMIT, RETRY_AFTER_5),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': 'rate_limit_exceeded',
                'provider_error_type': 'requests',
                'retry_after': 5.0,
                'body': '{"error": {"message": "Rate limit reached.", "type": "requests", "param": null, "code": "rate_limit_exceeded"}}',
                'headers': {'retry-after': '5', 'content-length': '111'},
            }
        ),
    ),
    UpgradeCase(
        id='elevenlabs-401-plain-body',
        model='elevenlabs',
        rejection=Rejection(401, b'Unauthorized'),
        expected=snapshot(
            {
                'class': 'ModelAuthenticationError',
                'status_code': 401,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': 'Unauthorized',
                'headers': {'content-length': '12'},
            }
        ),
    ),
    UpgradeCase(
        id='xai-429-plain-body',
        model='xai',
        rejection=Rejection(429, b'Too Many Requests', RETRY_AFTER_5),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': 5.0,
                'body': 'Too Many Requests',
                'headers': {'retry-after': '5', 'content-length': '17'},
            }
        ),
    ),
    UpgradeCase(
        id='openai-live-429-insufficient-quota',
        model='openai_live',
        rejection=Rejection(429, OPENAI_QUOTA),
        expected=snapshot(
            {
                'class': 'ModelQuotaExceededError',
                'status_code': 429,
                'provider_error_code': 'insufficient_quota',
                'provider_error_type': 'insufficient_quota',
                'retry_after': None,
                'body': '{"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota", "param": null, "code": "insufficient_quota"}}',
                'headers': {'content-length': '133'},
            }
        ),
    ),
    # Gemini's error status picks the category, as for a Gemini API error response; without one, the HTTP status does.
    UpgradeCase(
        id='google-400-api-key-invalid',
        model='google',
        rejection=Rejection(400, GOOGLE_INVALID_KEY),
        expected=snapshot(
            {
                'class': 'ModelAuthenticationError',
                'status_code': 400,
                'provider_error_code': 'INVALID_ARGUMENT',
                'provider_error_type': None,
                'retry_after': None,
                'body': '{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT", "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID", "domain": "googleapis.com"}]}}',
                'headers': {'content-length': '241'},
            }
        ),
    ),
    UpgradeCase(
        id='google-429-resource-exhausted',
        model='google',
        rejection=Rejection(429, GOOGLE_RESOURCE_EXHAUSTED, RETRY_AFTER_5),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': 'RESOURCE_EXHAUSTED',
                'provider_error_type': None,
                'retry_after': 5.0,
                'body': '{"error": {"code": 429, "message": "Resource has been exhausted (e.g. check quota).", "status": "RESOURCE_EXHAUSTED"}}',
                'headers': {'retry-after': '5', 'content-length': '118'},
            }
        ),
    ),
    UpgradeCase(
        id='google-503-plain-body',
        model='google',
        rejection=Rejection(503, b'Service Unavailable'),
        expected=snapshot(
            {
                'class': 'ModelUnavailableError',
                'status_code': 503,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': 'Service Unavailable',
                'headers': {'content-length': '19'},
            }
        ),
    ),
    UpgradeCase(
        id='google-429-repeated-header',
        model='google',
        rejection=Rejection(429, b'Too Many Requests', REPEATED_SET_COOKIE),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': 'Too Many Requests',
                'headers': {'set-cookie': 'a=1, b=2', 'content-length': '17'},
            }
        ),
    ),
    # ElevenLabs Agents speaks its own protocol, so only the status picks the category.
    UpgradeCase(
        id='elevenlabs-agents-403-plain-body',
        model='elevenlabs_agents',
        rejection=Rejection(403, b'denied'),
        expected=snapshot(
            {
                'class': 'ModelAuthenticationError',
                'status_code': 403,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': 'denied',
                'headers': {'content-length': '6'},
            }
        ),
    ),
    UpgradeCase(
        id='elevenlabs-agents-429-repeated-header',
        model='elevenlabs_agents',
        rejection=Rejection(429, b'slow down', (*RETRY_AFTER_5, *REPEATED_SET_COOKIE)),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': 5.0,
                'body': 'slow down',
                'headers': {'retry-after': '5', 'set-cookie': 'a=1, b=2', 'content-length': '9'},
            }
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in UPGRADE_CASES])
async def test_rejected_upgrade(case: UpgradeCase, rejector: Rejector) -> None:
    rejector.rejection = case.rejection
    with pytest.raises(ModelHTTPError) as exc_info:
        async with MODELS[case.model]().connect(
            messages=MESSAGES, model_settings=None, model_request_parameters=ModelRequestParameters()
        ):
            pass  # pragma: no cover
    assert describe(exc_info.value) == case.expected


SDP_OFFER = 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'

SignalingCall = Literal['openai_client_secret', 'openai_webrtc_offer', 'azure_hang_up', 'elevenlabs_agents_preflight']


@dataclass(frozen=True)
class SignalingCase:
    id: str
    call: SignalingCall
    rejection: Rejection
    expected: dict[str, object]


SIGNALING_CASES = [
    SignalingCase(
        id='openai-client-secret-429-insufficient-quota',
        call='openai_client_secret',
        rejection=Rejection(429, OPENAI_QUOTA),
        expected=snapshot(
            {
                'class': 'ModelQuotaExceededError',
                'status_code': 429,
                'provider_error_code': 'insufficient_quota',
                'provider_error_type': 'insufficient_quota',
                'retry_after': None,
                'body': '{"error": {"message": "You exceeded your current quota.", "type": "insufficient_quota", "param": null, "code": "insufficient_quota"}}',
                'headers': {'content-length': '133'},
            }
        ),
    ),
    SignalingCase(
        id='openai-webrtc-offer-429-rate-limit',
        call='openai_webrtc_offer',
        rejection=Rejection(429, OPENAI_RATE_LIMIT, RETRY_AFTER_5),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': 'rate_limit_exceeded',
                'provider_error_type': 'requests',
                'retry_after': 5.0,
                'body': '{"error": {"message": "Rate limit reached.", "type": "requests", "param": null, "code": "rate_limit_exceeded"}}',
                'headers': {'retry-after': '5', 'content-length': '111'},
            }
        ),
    ),
    SignalingCase(
        id='azure-hang-up-401-invalid-api-key',
        call='azure_hang_up',
        rejection=Rejection(401, OPENAI_INVALID_KEY),
        expected=snapshot(
            {
                'class': 'ModelAuthenticationError',
                'status_code': 401,
                'provider_error_code': 'invalid_api_key',
                'provider_error_type': 'invalid_request_error',
                'retry_after': None,
                'body': '{"error": {"message": "Incorrect API key provided: sk-test.", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}}',
                'headers': {'content-length': '137'},
            }
        ),
    ),
    # A redirect isn't a created call, but has no error category.
    SignalingCase(
        id='openai-webrtc-offer-302',
        call='openai_webrtc_offer',
        rejection=Rejection(302, b'', (('Location', '/v1/realtime/calls/rtc_x'),)),
        expected=snapshot(
            {
                'class': 'ModelHTTPError',
                'status_code': 302,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': None,
                'body': None,
                'headers': {'location': '/v1/realtime/calls/rtc_x'},
            }
        ),
    ),
    SignalingCase(
        id='elevenlabs-agents-preflight-429',
        call='elevenlabs_agents_preflight',
        rejection=Rejection(429, b'slow down', RETRY_AFTER_5),
        expected=snapshot(
            {
                'class': 'ModelRateLimitError',
                'status_code': 429,
                'provider_error_code': None,
                'provider_error_type': None,
                'retry_after': 5.0,
                'body': 'slow down',
                'headers': {'retry-after': '5', 'content-length': '9'},
            }
        ),
    ),
]


@pytest.fixture(scope='module')
def platform_cached() -> None:
    """Fill `platform`'s cache before a test reads an OpenAI client's `default_headers` on the event loop.

    The SDK builds its platform headers on that first read, running `uname -p` once per process, and BlockBuster
    stops that blocking call halfway, leaking the child process. The batch path's read of them is
    https://github.com/pydantic/pydantic-ai/issues/12444.
    """
    platform.platform()


async def _signal(call: SignalingCall, http_client: httpx2.AsyncClient) -> None:
    match call:
        case 'openai_client_secret':
            provider = OpenAIProvider(api_key='sk-test', http_client=http_client)
            await OpenAIRealtimeModel('gpt-realtime', provider=provider).create_client_secret()
        case 'openai_webrtc_offer':
            provider = OpenAIProvider(api_key='sk-test', http_client=http_client)
            await OpenAIRealtimeModel('gpt-realtime', provider=provider).answer_webrtc_offer(SDP_OFFER)
        case 'azure_hang_up':
            model = AzureRealtimeModel('gpt-realtime', provider=_azure_provider(http_client))
            await model.hang_up(WebRTCSession(provider_name='azure', session_id='rtc_x'))
        case 'elevenlabs_agents_preflight':
            async with _elevenlabs_agents_model(http_client).connect(
                messages=MESSAGES, model_settings=None, model_request_parameters=ModelRequestParameters()
            ):
                pass  # pragma: no cover
        case _:
            assert_never(call)


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in SIGNALING_CASES])
async def test_rejected_signaling_request(case: SignalingCase, platform_cached: None) -> None:
    rejection = case.rejection
    transport = httpx2.MockTransport(
        lambda request: httpx2.Response(rejection.status, content=rejection.body, headers=list(rejection.headers))
    )
    async with httpx2.AsyncClient(transport=transport) as http_client:
        with pytest.raises(ModelHTTPError) as exc_info:
            await _signal(case.call, http_client)
    assert describe(exc_info.value) == case.expected
