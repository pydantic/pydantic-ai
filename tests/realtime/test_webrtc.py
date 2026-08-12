"""Tests for the realtime WebRTC signaling helpers and OpenAI-family model methods.

The browser <-> provider media path is exercised by the runnable example, not here. These tests pin the
server-side signaling that Pydantic AI owns: minting a client secret, relaying an SDP offer (the secure
topology), parsing the `call_id`, Azure Microsoft Entra ID token minting, and the capability gating of a
sideband session. Success paths that depend on provider behavior use recorded HTTP cassettes; focused
unit tests use `httpx2.MockTransport` only for our own guards, error formatting, and request shaping.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timezone
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime import (
    RealtimeClientSecret,
    RealtimeModel,
    RealtimeModelSettings,
    WebRTCAnswer,
    WebRTCSession,
)
from pydantic_ai.realtime.codec import RealtimeConnection
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import UsageLimits

from ..conftest import try_import
from .conftest import (
    REAL_SDP_OFFER,
    _scrub_ephemeral_secret,  # pyright: ignore[reportPrivateUsage]
    _zero_sdp_addresses,  # pyright: ignore[reportPrivateUsage]
)

with try_import() as imports_successful:
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.realtime._openai_webrtc import parse_call_id
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed'),
]

SAMPLE_SDP_OFFER = 'v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\n'
SAMPLE_SDP_ANSWER = 'v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=-\r\nt=0 0\r\na=recvonly\r\n'

# Our Azure OpenAI dev resource, hardcoded (not a secret — like `test_azure_provider_call`) so the
# recorded host is stable between recording (real key) and offline replay (placeholder key).
_AZURE_REALTIME_ENDPOINT = 'https://pydantic-ai-realtime-dev.openai.azure.com/openai/v1'


class _SignalingModel(RealtimeModel):
    """A network-free model that records the resolved agent configuration sent to signaling methods."""

    def __init__(self, *, settings: RealtimeModelSettings | None = None) -> None:
        self.settings = settings
        self.calls: list[tuple[str | None, Sequence[ToolDefinition] | None, RealtimeModelSettings | None]] = []
        self.expires_after_seconds: int | None = None

    @property
    def model_name(self) -> str:
        return 'signaling-model'

    @property
    def system(self) -> str:
        return 'test'  # pragma: no cover - never opens a session, so provider identity is unused

    def connect(
        self,
        *,
        messages: Sequence[ModelMessage],
        model_settings: RealtimeModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AbstractAsyncContextManager[RealtimeConnection]:
        raise NotImplementedError

    async def answer_webrtc_offer(
        self,
        sdp_offer: str,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
    ) -> WebRTCAnswer:
        self.calls.append((instructions, tools, model_settings))
        return WebRTCAnswer(sdp=sdp_offer, session=WebRTCSession(provider_name='test', session_id='rtc_test'))

    async def create_client_secret(
        self,
        *,
        instructions: str | None = None,
        tools: Sequence[ToolDefinition] | None = None,
        model_settings: RealtimeModelSettings | None = None,
        expires_after_seconds: int | None = None,
    ) -> RealtimeClientSecret:
        self.calls.append((instructions, tools, model_settings))
        self.expires_after_seconds = expires_after_seconds
        return RealtimeClientSecret(value='ek_test', expires_at=datetime.now(timezone.utc))


async def test_agent_realtime_signaling_resolves_bound_configuration() -> None:
    model = _SignalingModel(settings=RealtimeModelSettings(max_tokens=100))
    agent = Agent(instructions='Literal instructions.')

    @agent.instructions
    def dynamic_instructions() -> str:
        return 'Dynamic instructions.'

    @agent.tool_plain
    def agent_tool(value: str) -> str:
        return value  # pragma: no cover - registered so its definition resolves; never executed by signaling

    toolset = FunctionToolset(instructions='Toolset instructions.')

    @toolset.tool_plain
    def accessor_tool(value: int) -> int:
        return value  # pragma: no cover - registered so its definition resolves; never executed by signaling

    realtime = agent.realtime(
        model,
        model_settings=RealtimeModelSettings(output_modality='text'),
        toolsets=[toolset],
    )
    answer = await realtime.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    secret = await realtime.create_client_secret(expires_after_seconds=45)

    assert answer.sdp == SAMPLE_SDP_OFFER
    assert secret.value == 'ek_test'
    assert model.expires_after_seconds == 45
    assert len(model.calls) == 2
    for instructions, tools, settings in model.calls:
        # Static parts (the literal and the toolset's own instructions) sort ahead of the dynamic
        # `@agent.instructions` function, exactly as a graph run's request does.
        assert instructions == 'Literal instructions.\n\nToolset instructions.\n\nDynamic instructions.'
        assert tools is not None
        assert [tool.name for tool in tools] == ['agent_tool', 'accessor_tool']
        assert settings == RealtimeModelSettings(max_tokens=100, output_modality='text')


async def test_agent_realtime_signaling_resolves_bound_run_identity() -> None:
    """Signaling resolves under the bound `run_id` and `usage_limits`.

    Dynamic instructions and capability/toolset hooks then see the same run identity a later
    `session()` on the same binding uses, so both push identical configuration.
    """
    model = _SignalingModel()
    seen: list[tuple[str | None, int | None]] = []
    agent = Agent(deps_type=type(None))

    @agent.instructions
    def record_run_identity(ctx: RunContext[None]) -> str:
        assert ctx.usage_limits is not None
        seen.append((ctx.run_id, ctx.usage_limits.tool_calls_limit))
        return ''

    realtime = agent.realtime(model, run_id='run-bound', usage_limits=UsageLimits(tool_calls_limit=3))
    await realtime.create_client_secret()
    await realtime.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    assert seen == [('run-bound', 3), ('run-bound', 3)]


async def test_agent_realtime_signaling_unsupported_model() -> None:
    class _UnsupportedModel(_SignalingModel):
        answer_webrtc_offer = RealtimeModel.answer_webrtc_offer
        create_client_secret = RealtimeModel.create_client_secret

    realtime = Agent().realtime(_UnsupportedModel())
    with pytest.raises(
        UserError, match=r"Realtime model 'signaling-model' does not support WebRTC.*connect over WebSockets"
    ):
        await realtime.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    with pytest.raises(
        UserError, match=r"Realtime model 'signaling-model' does not support WebRTC.*connect over WebSockets"
    ):
        await realtime.create_client_secret()


async def test_wrapper_agent_realtime_signaling_delegates() -> None:
    model = _SignalingModel()
    realtime = WrapperAgent(Agent(instructions='Wrapped instructions.')).realtime(model)
    await realtime.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    assert model.calls[0][0] == 'Wrapped instructions.'


def _mock_provider(handler: Any, *, api_key: str = 'sk-test') -> Any:
    """An `OpenAIProvider` whose HTTP calls are served by `handler` instead of the network."""
    return OpenAIProvider(api_key=api_key, http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))


def _unused_handler(request: httpx2.Request) -> httpx2.Response:
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
    # The defensive guards pass non-body, empty, non-JSON, and non-object bodies through untouched.
    assert _scrub_ephemeral_secret({}) == {}
    assert _scrub_ephemeral_secret({'body': {'string': b''}})['body']['string'] == b''
    assert _scrub_ephemeral_secret({'body': {'string': b'not json'}})['body']['string'] == b'not json'
    assert _scrub_ephemeral_secret({'body': {'string': b'[1, 2]'}})['body']['string'] == b'[1, 2]'


def test_zero_sdp_addresses_blanks_offer_addresses() -> None:
    """The VCR `before_record_request` hook keeps the recorder's own addresses out of a cassette.

    A unit test for the same reason as the one above: the hook only runs while recording. It matters
    for the sideband audio cassette, whose offer comes from a live `aiortc` peer rather than the
    hand-zeroed constants, so every recording of it would otherwise commit the recorder's machine
    addresses.
    """

    class _Request:
        def __init__(self, body: Any) -> None:
            self.body = body

    offer = (
        b'--boundary\r\nContent-Type: application/sdp\r\n\r\n'
        b'v=0\r\nc=IN IP4 192.168.1.5\r\n'
        b'a=candidate:1 1 udp 2130706431 192.168.1.5 46294 typ host\r\n'
        b'a=candidate:2 1 udp 2130706431 fd7a:115c:a1e0::1 57945 typ host\r\n'
        b'c=IN IP6 fd7a:115c:a1e0::1\r\na=ice-ufrag:creB\r\n'
    )
    assert _zero_sdp_addresses(_Request(offer)).body == (
        b'--boundary\r\nContent-Type: application/sdp\r\n\r\n'
        b'v=0\r\nc=IN IP4 0.0.0.0\r\n'
        b'a=candidate:1 1 udp 2130706431 0.0.0.0 46294 typ host\r\n'
        b'a=candidate:2 1 udp 2130706431 :: 57945 typ host\r\n'
        b'c=IN IP6 ::\r\na=ice-ufrag:creB\r\n'
    )
    # An SDP whose only address is the `c=` connection line (no `a=candidate:` lines) is still zeroed:
    # the substitution is gated on the body being bytes, not on an ICE candidate being present.
    assert _zero_sdp_addresses(_Request(b'v=0\r\nc=IN IP4 192.168.1.5\r\na=ice-ufrag:creB\r\n')).body == (
        b'v=0\r\nc=IN IP4 0.0.0.0\r\na=ice-ufrag:creB\r\n'
    )
    # Bodies with no address fields at all — every other recorded request — are unchanged.
    assert _zero_sdp_addresses(_Request(b'{"model": "gpt-realtime"}')).body == b'{"model": "gpt-realtime"}'
    assert _zero_sdp_addresses(_Request(None)).body is None


# --- client secret minting --------------------------------------------------------------------------


@pytest.mark.vcr
async def test_create_client_secret(openai_api_key: str, request: pytest.FixtureRequest) -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key=openai_api_key))
    secret = await model.create_client_secret(
        instructions='Be brief.',
        model_settings=OpenAIRealtimeModelSettings(openai_voice='marin'),
        expires_after_seconds=60,
    )

    assert secret.value
    assert secret.expires_at.tzinfo is not None
    # The secret expires shortly after recording, so only a live response can remain future-dated.
    recording = request.config.getoption('record_mode') == 'rewrite'
    assert not recording or secret.expires_at > datetime.now(timezone.utc)


@pytest.mark.vcr
async def test_agent_create_client_secret(openai_api_key: str, request: pytest.FixtureRequest) -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(instructions='Answer in two words.')

    @agent.tool_plain
    def get_temperature(city: str) -> str:
        return f'20 C in {city}'  # pragma: no cover - resolved into the request, never called

    secret = await agent.realtime(
        model, model_settings=OpenAIRealtimeModelSettings(openai_voice='marin')
    ).create_client_secret(expires_after_seconds=60)

    assert secret.value
    assert secret.expires_at.tzinfo is not None
    recording = request.config.getoption('record_mode') == 'rewrite'
    assert not recording or secret.expires_at > datetime.now(timezone.utc)


async def test_create_client_secret_missing_value() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={'expires_at': 1_700_000_060})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='did not include a `value`'):
        await model.create_client_secret()


async def test_create_client_secret_non_numeric_expires_at() -> None:
    # A `value` with a non-integer `expires_at` can't be turned into an expiry timestamp, so it's rejected.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={'value': 'ek_x', 'expires_at': 'soon'})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='numeric'):
        await model.create_client_secret()


async def test_create_client_secret_through_gateway() -> None:
    # A gateway-routed provider's base URL ends at `.../openai`; the gateway accepts the `/v1`-less
    # signaling path, so the client-secret URL is derived straight from that base without a `/v1` segment.
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured['url'] = str(request.url)
        return httpx2.Response(200, json={'value': 'ek_gw', 'expires_at': 1_700_000_060})

    provider = gateway_provider(
        'openai',
        api_key='gw-key',
        base_url='https://gateway.pydantic.dev/proxy',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    secret = await model.create_client_secret()

    assert captured['url'] == 'https://gateway.pydantic.dev/proxy/openai/realtime/client_secrets'
    assert secret.value == 'ek_gw'


async def test_create_client_secret_preserves_base_url_fragment() -> None:
    # A `#fragment` in the base URL is client-side URL state: the signaling path must land before it,
    # not inside it (which would leave the request going to the fragment-truncated base). Matches the
    # fragment handling in `realtime_websocket_url` / `with_realtime_query`.
    captured: dict[str, Any] = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured['url'] = str(request.url)
        return httpx2.Response(200, json={'value': 'ek_frag', 'expires_at': 1_700_000_060})

    provider = OpenAIProvider(
        base_url='https://example.com/v1#frag',
        api_key='sk-test',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    model = OpenAIRealtimeModel('gpt-realtime', provider=provider)
    secret = await model.create_client_secret()

    assert captured['url'] == 'https://example.com/v1/realtime/client_secrets#frag'
    assert secret.value == 'ek_frag'


async def test_create_client_secret_http_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, text='invalid api key')

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.create_client_secret()
    assert exc_info.value.status_code == 401
    assert exc_info.value.body == 'invalid api key'
    # `model_name` names the realtime model, not the provider, matching `ModelHTTPError` elsewhere.
    assert exc_info.value.model_name == 'gpt-realtime'


async def test_create_client_secret_out_of_range_expires_at() -> None:
    # A numeric-but-unrepresentable `expires_at` passes validation but overflows the platform's
    # timestamp range, so it's surfaced as unexpected output rather than a raw OverflowError/OSError.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={'value': 'ek_x', 'expires_at': 10**100})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='out of range'):
        await model.create_client_secret()


async def test_signaling_http_error_preserves_retry_after() -> None:
    # A 429 with a `Retry-After` header must carry the header through so callers can honor the delay.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(429, text='slow down', headers={'Retry-After': '30'})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.create_client_secret()
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 30.0


def test_client_secret_value_absent_from_repr() -> None:
    # Neither the live token nor the resolved session config (instructions/tools carried in
    # `provider_details`) must leak into logs via the dataclass repr.
    secret = RealtimeClientSecret(
        value='ek_live_secret',
        expires_at=datetime.now(timezone.utc),
        provider_details={'session': {'instructions': 'secret system prompt'}},
    )
    rendered = repr(secret)
    assert 'ek_live_secret' not in rendered
    assert 'secret system prompt' not in rendered


# --- WebRTC offer relay -----------------------------------------------------------------------------


@pytest.mark.vcr
async def test_answer_webrtc_offer(openai_api_key: str) -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key=openai_api_key))
    answer = await model.answer_webrtc_offer(
        REAL_SDP_OFFER,
        instructions='Answer in two words.',
        model_settings=OpenAIRealtimeModelSettings(openai_voice='cedar'),
    )

    assert answer.session.provider_name == 'openai'
    assert answer.session.call_id.startswith('rtc_')
    assert answer.sdp.startswith('v=0')


@pytest.mark.vcr
async def test_agent_answer_webrtc_offer(openai_api_key: str) -> None:
    model = OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key=openai_api_key))
    agent = Agent(instructions='Answer in two words.')

    @agent.tool_plain
    def get_temperature(city: str) -> str:
        return f'20 C in {city}'  # pragma: no cover - resolved into the request, never called

    answer = await agent.realtime(
        model, model_settings=OpenAIRealtimeModelSettings(openai_voice='cedar')
    ).answer_webrtc_offer(REAL_SDP_OFFER)

    assert answer.session.provider_name == 'openai'
    assert answer.session.call_id.startswith('rtc_')
    assert answer.sdp.startswith('v=0')


async def test_answer_webrtc_offer_missing_location() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(201, text=SAMPLE_SDP_ANSWER)  # no Location header

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(UnexpectedModelBehavior, match='did not return a parseable `call_id`'):
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)


async def test_answer_webrtc_offer_http_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(400, text='bad sdp')

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    assert exc_info.value.status_code == 400
    assert exc_info.value.body == 'bad sdp'
    assert exc_info.value.model_name == 'gpt-realtime'


async def test_answer_webrtc_offer_rejects_redirect() -> None:
    # A 3xx redirect is not a created call. Rejecting all non-2xx (not just 4xx/5xx) stops the redirect's
    # `Location` from being mistaken for a `call_id` and returned as a bogus answer.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(302, headers={'location': '/v1/realtime/calls/rtc_redirect'})

    model = OpenAIRealtimeModel('gpt-realtime', provider=_mock_provider(handler))
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    assert exc_info.value.status_code == 302


# --- Azure Microsoft Entra ID + endpoints -----------------------------------------------------------


def _azure_mock_provider(handler: Any) -> Any:
    return AzureProvider(
        azure_endpoint='https://resource.openai.azure.com/openai/v1/',
        api_key='azure-key',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
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

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured['url'] = str(request.url)
        captured['api_key'] = request.headers.get('api-key')
        captured['auth'] = request.headers.get('authorization')
        return httpx2.Response(200, json={'value': 'ek_az', 'expires_at': 1_700_000_060})

    credential = _FakeCredential()
    model = AzureRealtimeModel('gpt-realtime', provider=_azure_mock_provider(handler), credential=credential)
    secret = await model.create_client_secret(instructions='Hi.')

    # With an Entra credential, signaling uses a bearer token (data-plane scope) and never the api-key.
    assert credential.scopes == ('https://ai.azure.com/.default',)
    assert captured['url'] == 'https://resource.openai.azure.com/openai/v1/realtime/client_secrets'
    assert captured['auth'] == 'Bearer entra-token-xyz'
    assert captured['api_key'] is None
    assert secret.value == 'ek_az'


def test_azure_entra_credential_needs_no_resource_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An Entra-authenticated model constructs without `AZURE_OPENAI_API_KEY`.

    Regression: resolving the default `provider='azure'` went through `AzureProvider.for_realtime()`,
    which demanded a resource key unconditionally — so the one configuration the credential exists for,
    a resource locked to managed identity with no key at all, could not be constructed.
    """
    monkeypatch.delenv('AZURE_OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://my-resource.openai.azure.com')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    model = AzureRealtimeModel('gpt-realtime', credential=_FakeCredential())
    assert model._realtime_url() == (  # pyright: ignore[reportPrivateUsage]
        'wss://my-resource.openai.azure.com/openai/v1/realtime?model=gpt-realtime'
    )

    # The explicit form the docs show works the same way.
    explicit = AzureRealtimeModel(
        'gpt-realtime',
        provider=AzureProvider.for_realtime(
            azure_endpoint='https://my-resource.openai.azure.com', entra_authenticated=True
        ),
        credential=_FakeCredential(),
    )
    # The SDK's Entra placeholder is never handed out as a credential: asking still reports its absence.
    with pytest.raises(UserError, match='has no API key'):
        _ = explicit._azure_provider.api_key  # pyright: ignore[reportPrivateUsage]


def test_azure_entra_for_realtime_ignores_empty_resource_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An *empty* `AZURE_OPENAI_API_KEY` is treated as "no key", not as a key of value `''`.

    Regression: the guard was a membership test (`'AZURE_OPENAI_API_KEY' not in os.environ`), so an
    exported-but-empty variable skipped the Entra placeholder, resolved an empty key, and raised the
    key-required error on the very path that needs no key.
    """
    monkeypatch.setenv('AZURE_OPENAI_API_KEY', '')
    monkeypatch.setenv('AZURE_OPENAI_ENDPOINT', 'https://my-resource.openai.azure.com')
    monkeypatch.delenv('OPENAI_API_VERSION', raising=False)

    provider = AzureProvider.for_realtime(entra_authenticated=True)
    # No key resolved, and asking reports its absence rather than surfacing an empty string.
    with pytest.raises(UserError, match='has no API key'):
        _ = provider.api_key


def test_azure_entra_credential_survives_alongside_a_user_profile() -> None:
    """The Entra credential and the user `profile=` layer coexist on the hand-written `__init__`.

    `AzureRealtimeModel` is `@dataclass(init=False)` with a hand-written constructor, so `credential`
    is assigned explicitly rather than generated. Pinning it next to `profile=` keeps a future edit to
    that constructor from silently dropping either.
    """

    def _unused(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError('no request is made')  # pragma: no cover

    credential = _FakeCredential()
    model = AzureRealtimeModel(
        'gpt-realtime',
        provider=_azure_mock_provider(_unused),
        credential=credential,
        profile={'supports_webrtc': False},
    )

    assert model.credential is credential
    # The user layer wins over the provider's, and the rest of the resolved profile is untouched.
    assert model.profile.get('supports_webrtc') is False
    assert model.profile.get('supports_text_output') is True
    # Defaults still apply when neither is given.
    plain = AzureRealtimeModel('gpt-realtime', provider=_azure_mock_provider(_unused))
    assert plain.credential is None
    assert plain.profile.get('supports_webrtc') is True


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
    with pytest.raises(UserError, match=r"Realtime model 'ws-only' does not support WebRTC.*connect over WebSockets"):
        await model.answer_webrtc_offer(SAMPLE_SDP_OFFER)
    with pytest.raises(UserError, match=r"Realtime model 'ws-only' does not support WebRTC.*connect over WebSockets"):
        await model.create_client_secret()
    with pytest.raises(UserError, match=r"Realtime model 'ws-only' does not support WebRTC.*connect over WebSockets"):
        async with model.connect_webrtc(
            WebRTCSession(provider_name='ws-only', session_id='x'),
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        ):
            pass  # pragma: no cover - raises before yielding


def test_openai_family_webrtc_profiles() -> None:
    models = {
        'openai': OpenAIRealtimeModel('gpt-realtime', provider=OpenAIProvider(api_key='test')),
        'azure': AzureRealtimeModel(
            'gpt-realtime',
            provider=AzureProvider(
                azure_endpoint='https://example.openai.azure.com',
                api_version='2025-04-01-preview',
                api_key='test',
            ),
        ),
    }
    assert {provider: model.profile.get('supports_webrtc') for provider, model in models.items()} == {
        'openai': True,
        'azure': True,
    }
