from __future__ import annotations as _annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
from pydantic import SecretStr
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode

from pydantic_ai import Agent
from pydantic_ai.auth.codex import CodexAuth, CodexCredentials, CodexLoginRequiredError, CodexRefreshError
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ThinkingPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers import Provider
from pydantic_ai.providers.codex import CodexProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage, RunUsage

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import IsDatetime, IsStr, try_import
from .mock_openai import MockOpenAIResponses, get_mock_responses_kwargs

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types import responses as resp
    from openai.types.responses.response_output_message import ResponseOutputMessage

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed'),
]


class StaticCodexCredentialSource:
    async def get_credentials(
        self, *, force_refresh: bool = False, rejected_revision: str | None = None
    ) -> CodexCredentials:
        return CodexCredentials(
            access_token=SecretStr('cassette-access-token'),
            refresh_token=SecretStr('cassette-refresh-token'),
            id_token=SecretStr('cassette-id-token'),
            expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            account_id=SecretStr('cassette-account-id'),
            revision='cassette-revision',
        )


def codex_model(vcr: Cassette, model_name: str = 'gpt-5.5') -> OpenAIResponsesModel:
    credential_source = StaticCodexCredentialSource() if vcr.record_mode == RecordMode.NONE else CodexAuth()
    return OpenAIResponsesModel(model_name, provider=CodexProvider(credential_source=credential_source))


class MockCodexProvider(Provider[AsyncOpenAI]):
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return 'codex'

    @property
    def base_url(self) -> str:
        return 'https://chatgpt.com/backend-api/codex'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    model_profile = staticmethod(CodexProvider.model_profile)


async def test_codex_profile_streams_ordinary_requests_and_preserves_provider_identity(
    allow_model_requests: None,
) -> None:
    """A caller-set `openai_store=True` is overridden rather than passed through.

    Mocked rather than recorded because the assertion is about a setting the backend never sees: a
    cassette of a successful run cannot distinguish "the caller set `openai_store=True` and it was
    overridden" from "the caller set nothing". The recorded siblings in this file pin the resulting
    `store=False`/`stream=True` body for the ordinary case.
    """
    base_response = resp.Response(
        id='resp_001',
        model='gpt-5.5',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    stream: list[resp.ResponseStreamEvent] = [
        resp.ResponseCreatedEvent(response=base_response, type='response.created', sequence_number=0),
        resp.ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id='msg_001',
                content=[],
                role='assistant',
                status='in_progress',
                type='message',
            ),
            output_index=0,
            type='response.output_item.added',
            sequence_number=1,
        ),
        resp.ResponseTextDeltaEvent(
            item_id='msg_001',
            output_index=0,
            content_index=0,
            delta='ok',
            logprobs=[],
            type='response.output_text.delta',
            sequence_number=2,
        ),
        resp.ResponseCompletedEvent(
            response=base_response.model_copy(update={'status': 'completed'}),
            type='response.completed',
            sequence_number=3,
        ),
    ]
    mock_client = MockOpenAIResponses.create_mock_stream(stream)
    model = OpenAIResponsesModel('gpt-5.5', provider=MockCodexProvider(mock_client))

    response = await model.request(
        [ModelRequest.user_text_prompt('hello')],
        OpenAIResponsesModelSettings(openai_store=True),
        ModelRequestParameters(),
    )

    assert response.provider_name == 'codex'
    assert response.text == 'ok'
    assert model.system == 'codex'
    assert model.model_id == 'codex:gpt-5.5'
    assert get_mock_responses_kwargs(mock_client)[0]['store'] is False


async def test_codex_login_required_error_survives_the_sdk_transport_wrapper(
    allow_model_requests: None,
) -> None:
    """The OpenAI SDK turns everything raised inside `httpx.AsyncClient.send` into an
    `APIConnectionError`, which would otherwise reduce the actionable "run `clai auth login codex`"
    to `ModelAPIError: Connection error.` at the only boundary a user sees.
    """

    class LoggedOutCredentialSource:
        async def get_credentials(
            self, *, force_refresh: bool = False, rejected_revision: str | None = None
        ) -> CodexCredentials:
            raise CodexLoginRequiredError('Codex subscription login is required. Run `clai auth login codex`.')

    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError('no request should be sent without credentials')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = CodexProvider(credential_source=LoggedOutCredentialSource(), http_client=client)
        model = OpenAIResponsesModel('gpt-5.5', provider=provider)

        with pytest.raises(CodexLoginRequiredError, match='clai auth login codex'):
            await model.request([ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters())


async def test_codex_refresh_failure_still_falls_over_in_fallback_model(allow_model_requests: None) -> None:
    """The unwrap above has to stay narrow, or `FallbackModel` stops routing around network failures.

    An unreachable `auth.openai.com` reaches this boundary as a `CodexRefreshError` (pinned by
    `test_refresh_network_error_does_not_retain_refresh_token`), which is a `UserError`. Unwrapping
    every `UserError` out of the SDK's `APIConnectionError` presented that transport failure as an
    error `FallbackModel` does not fall over on — for every OpenAI-compatible model, not just Codex.
    """

    class UnreachableCredentialSource:
        async def get_credentials(
            self, *, force_refresh: bool = False, rejected_revision: str | None = None
        ) -> CodexCredentials:
            raise CodexRefreshError('Unable to reach the Codex authentication service.')

    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError('no request should be sent without credentials')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = CodexProvider(credential_source=UnreachableCredentialSource(), http_client=client)
        model = OpenAIResponsesModel('gpt-5.5', provider=provider)

        result = await Agent(FallbackModel(model, TestModel())).run('hello')

    assert result.output == snapshot('success (no tool calls)')


async def test_codex_count_tokens_replays_once_after_unauthorized(allow_model_requests: None) -> None:
    """`count_tokens` posts to `responses/input_tokens`, a second Codex request flavor that
    `UsageLimits(count_tokens_before_request=True)` reaches, so it needs the same 401 replay.
    """
    rotations: list[tuple[bool, str | None]] = []
    paths: list[str] = []

    def credentials(revision: str) -> CodexCredentials:
        return CodexCredentials(
            access_token=SecretStr(f'access-{revision}'),
            refresh_token=SecretStr(f'refresh-{revision}'),
            id_token=SecretStr(f'id-{revision}'),
            expires_at=datetime(2099, 1, 1, tzinfo=timezone.utc),
            account_id=SecretStr('account-id'),
            revision=revision,
        )

    class RotatingCredentialSource:
        async def get_credentials(
            self, *, force_refresh: bool = False, rejected_revision: str | None = None
        ) -> CodexCredentials:
            rotations.append((force_refresh, rejected_revision))
            return credentials('rotated' if force_refresh else 'initial')

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.headers['authorization'] == 'Bearer access-initial':
            return httpx.Response(401, json={'error': {'message': 'expired'}})
        return httpx.Response(200, json={'input_tokens': 12})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = CodexProvider(credential_source=RotatingCredentialSource(), http_client=client)
        model = OpenAIResponsesModel('gpt-5.5', provider=provider)

        usage = await model.count_tokens([ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters())

    assert usage.input_tokens == 12
    assert paths == ['/backend-api/codex/responses/input_tokens'] * 2
    assert rotations == [(False, None), (True, 'initial')]


def test_codex_is_not_an_embeddings_provider() -> None:
    """The public `OpenAIEmbeddingsCompatibleProvider` alias omits `codex`, so a `codex:` embedding
    name only arrives as a plain string, and the runtime guard is what a user actually hits.
    """
    with pytest.raises(UserError, match='does not provide an embeddings endpoint'):
        infer_embedding_model('codex:text-embedding-3-small')


@pytest.mark.vcr
async def test_codex_agent_run(allow_model_requests: None, vcr: Cassette) -> None:
    """The headline behavior: a non-streaming `run()` is served by a forced stream.

    The aggregation is what the message snapshot pins — the reasoning part, the text, the finish
    reason and the usage all have to be reassembled from deltas, so asserting only the output would
    leave everything the collapse produces unverified.
    """
    agent = Agent(codex_model(vcr))

    result = await agent.run('Reply with exactly "codex-live-ok" and nothing else.')

    assert result.output == snapshot('codex-live-ok')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Reply with exactly "codex-live-ok" and nothing else.', timestamp=IsDatetime()
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        id='rs_0256c040a8370cbd016a5251b75b34819bac06d66e16ad946b',
                        signature=IsStr(),
                        provider_name='codex',
                    ),
                    TextPart(
                        content='codex-live-ok',
                        id='msg_0256c040a8370cbd016a5251b78c60819bafb5c6baad91a37d',
                        provider_name='codex',
                        provider_details={'phase': 'final_answer'},
                    ),
                ],
                usage=RequestUsage(
                    details={'reasoning_tokens': 9}, input_tokens=19, output_reasoning_tokens=9, output_tokens=19
                ),
                model_name='gpt-5.5',
                timestamp=IsDatetime(),
                provider_name='codex',
                provider_url='https://chatgpt.com/backend-api/codex/',
                provider_details={'timestamp': IsDatetime(), 'finish_reason': 'completed'},
                provider_response_id='resp_0256c040a8370cbd016a5251b6c148819b9cd65318ab92aef0',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.usage == snapshot(
        RunUsage(
            details={'reasoning_tokens': 9}, requests=1, input_tokens=19, output_reasoning_tokens=9, output_tokens=19
        )
    )
    assert single_request_body(vcr)['stream'] is True
    assert single_request_body(vcr)['store'] is False


@pytest.mark.vcr
async def test_codex_drops_generic_settings_the_backend_rejects(allow_model_requests: None, vcr: Cassette) -> None:
    """The Codex backend answers `400 Unsupported parameter` for `max_output_tokens`, `temperature` and
    `top_p`, so the profile drops them rather than let a portable `ModelSettings` break every request.
    The recording is the proof: a run that sets all three still succeeds, and none reach the wire.

    `gpt-5.4` reasons off by default, so its sampling params are not already dropped as unsupported
    under reasoning and would otherwise be sent — this is the plain case where a caller sets nothing
    Codex-specific and every request fails.
    """
    agent = Agent(
        codex_model(vcr, 'gpt-5.4'),
        model_settings=ModelSettings(max_tokens=1024, temperature=0.5, top_p=0.9),
    )

    result = await agent.run('Reply with exactly "codex-settings-ok" and nothing else.')

    assert result.output == snapshot('codex-settings-ok')
    assert single_request_body(vcr) == snapshot(
        {
            'include': ['reasoning.encrypted_content'],
            'input': [{'role': 'user', 'content': 'Reply with exactly "codex-settings-ok" and nothing else.'}],
            'model': 'gpt-5.4',
            'reasoning': {'context': 'all_turns'},
            'store': False,
            'stream': True,
        }
    )


@pytest.mark.vcr
async def test_codex_tool_call_round_trip(allow_model_requests: None, vcr: Cassette) -> None:
    """The two cells the single-turn text recordings leave untested.

    The forced stream has to reassemble a tool call out of deltas rather than read it off a complete
    response body, and the forced `store=False` means the second turn cannot resume by response id:
    the encrypted reasoning has to travel back inside the request.
    """
    agent = Agent(
        codex_model(vcr),
        instructions='Use the `city_temperature` tool, then answer.',
        model_settings=OpenAIResponsesModelSettings(openai_reasoning_effort='high'),
    )

    @agent.tool_plain
    def city_temperature(city: str) -> str:
        return f'{city}: 21 degrees Celsius'

    result = await agent.run(
        'Look up Lisbon, then tell me how many degrees Fahrenheit warmer it is than a city at 5 degrees '
        'Celsius. Answer with just the number.'
    )

    assert result.output == snapshot('28.8')

    recorded_requests = vcr.requests  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    first_body, second_body = (
        json.loads(request.body)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        for request in recorded_requests  # pyright: ignore[reportUnknownVariableType]
    )
    assert [item.get('type', item.get('role')) for item in first_body['input']] == snapshot(['user'])
    assert 'previous_response_id' not in second_body
    assert [item.get('type', item.get('role')) for item in second_body['input']] == snapshot(
        ['user', 'reasoning', 'function_call', 'function_call_output']
    )
    reasoning_item = next(item for item in second_body['input'] if item.get('type') == 'reasoning')
    assert reasoning_item['encrypted_content']


@pytest.mark.vcr
async def test_codex_agent_run_stream(allow_model_requests: None, vcr: Cassette) -> None:
    agent = Agent(codex_model(vcr))

    async with agent.run_stream('Reply with exactly "codex-stream-ok" and nothing else.') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True, debounce_by=None)]

    assert ''.join(chunks) == snapshot('codex-stream-ok')
    assert len(chunks) > 1
    assert single_request_body(vcr)['stream'] is True
    assert single_request_body(vcr)['store'] is False
