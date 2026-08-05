from __future__ import annotations as _annotations

import json
from datetime import datetime, timezone

import httpx
import pytest
from pydantic import JsonValue, SecretStr
from vcr.cassette import Cassette
from vcr.record_mode import RecordMode

from pydantic_ai import Agent
from pydantic_ai.auth.codex import CodexAuth, CodexCredentials, CodexLoginRequiredError, CodexRefreshError
from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
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


async def test_codex_forced_stream_rejects_a_response_that_never_completed(allow_model_requests: None) -> None:
    """A forced stream that stops early must not be handed back as a finished response.

    `request()` has no way to signal a partial result, so the collapse would return the text that
    happened to arrive plus zeroed usage, indistinguishable from a genuine short answer. Mocked
    because the fault is a stream that ends without its terminal event, which a recording of a
    healthy backend cannot produce.
    """
    base_response = resp.Response(
        id='resp_002',
        model='gpt-5.5',
        object='response',
        created_at=1704067200,
        output=[],
        parallel_tool_calls=True,
        tool_choice='auto',
        tools=[],
    )
    truncated_stream: list[resp.ResponseStreamEvent] = [
        # `in_progress` is what a real stream reports until its terminal event flips it, so ending
        # here is exactly the truncation being guarded against.
        resp.ResponseCreatedEvent(
            response=base_response.model_copy(update={'status': 'in_progress'}),
            type='response.created',
            sequence_number=0,
        ),
        resp.ResponseOutputItemAddedEvent(
            item=ResponseOutputMessage(
                id='msg_002',
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
            item_id='msg_002',
            output_index=0,
            content_index=0,
            delta='half an ans',
            logprobs=[],
            type='response.output_text.delta',
            sequence_number=2,
        ),
    ]
    model = OpenAIResponsesModel(
        'gpt-5.5', provider=MockCodexProvider(MockOpenAIResponses.create_mock_stream(truncated_stream))
    )

    with pytest.raises(UnexpectedModelBehavior, match='Streamed response ended before it was complete'):
        await model.request([ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters())


@pytest.mark.parametrize('entry_point', ['request', 'compact_messages'])
async def test_codex_login_required_error_survives_the_sdk_transport_wrapper(
    allow_model_requests: None, entry_point: str
) -> None:
    """The OpenAI SDK turns everything raised inside `httpx.AsyncClient.send` into an
    `APIConnectionError`, which would otherwise reduce the actionable "run `clai auth login codex`"
    to `ModelAPIError: Connection error.` at the only boundary a user sees.

    Both entry points that reach the backend are covered: `compact_messages` mapped SDK errors on its
    own and so did not unwrap the sign-in error the way `request` does, reporting the identical
    not-signed-in condition as `ModelAPIError` depending only on which method the caller used.
    """

    class LoggedOutCredentialSource:
        async def get_credentials(
            self, *, force_refresh: bool = False, rejected_revision: str | None = None
        ) -> CodexCredentials:
            raise CodexLoginRequiredError('Codex subscription login is required. Run `clai auth login codex`.')

    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError('no request should be sent without credentials')

    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('hello')]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = CodexProvider(credential_source=LoggedOutCredentialSource(), http_client=client)
        model = OpenAIResponsesModel('gpt-5.5', provider=provider)

        with pytest.raises(CodexLoginRequiredError, match='clai auth login codex'):
            if entry_point == 'request':
                await model.request(messages, None, ModelRequestParameters())
            else:
                await model.compact_messages(
                    ModelRequestContext(
                        model=model,
                        messages=messages,
                        model_settings=None,
                        model_request_parameters=ModelRequestParameters(),
                    )
                )


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


async def test_codex_count_tokens_reports_the_endpoint_is_not_served(allow_model_requests: None) -> None:
    """The Codex backend does not route `/responses/input_tokens`.

    Its edge answers that path with the same challenge page any unknown path gets, so without the
    profile flag `count_tokens` fails as a `ModelHTTPError` carrying an HTML body. The guard runs
    before anything is built, which is what the unsent request asserts; it is a unit test because a
    cassette of a challenge page would pin the edge's error rendering rather than our behavior.
    """

    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError('count_tokens must not reach the wire on a backend without the endpoint')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = CodexProvider(credential_source=StaticCodexCredentialSource(), http_client=client)
        model = OpenAIResponsesModel('gpt-5.5', provider=provider)

        with pytest.raises(UserError, match='does not serve the `/responses/input_tokens` endpoint'):
            await model.count_tokens([ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters())


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


async def test_codex_compaction_takes_neither_profile_flag(allow_model_requests: None) -> None:
    """`/responses/compact` is the one Responses sink neither Codex profile flag governs.

    Their absence from the compact body is the behavior under test, not an oversight: the backend
    serves this path but answers `400 Unknown parameter` for `store` and `stream`, so forwarding
    either flag's wire effect here would break every compaction.

    Mocked rather than recorded as an interim: recording it needs credentials in Pydantic AI's own
    store, and seeding that from the Codex CLI's would put a rotating refresh token in a second file.
    """
    bodies: list[dict[str, JsonValue]] = []
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'id': 'resp_compact_001',
                'created_at': 1704067200,
                'object': 'response.compaction',
                'output': [{'id': 'cpt_001', 'type': 'compaction', 'encrypted_content': 'encrypted-compaction-blob'}],
                'usage': {'input_tokens': 41, 'output_tokens': 7, 'total_tokens': 48},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = CodexProvider(credential_source=StaticCodexCredentialSource(), http_client=client)
        model = OpenAIResponsesModel('gpt-5.5', provider=provider)

        compacted = await model.compact_messages(
            ModelRequestContext(
                model=model,
                messages=[ModelRequest.user_text_prompt('hello')],
                model_settings=None,
                model_request_parameters=ModelRequestParameters(),
            )
        )

    assert paths == snapshot(['/backend-api/codex/responses/compact'])
    assert 'store' not in bodies[0]
    assert 'stream' not in bodies[0]

    assert isinstance(compacted.parts[0], CompactionPart)
    assert compacted.parts[0].provider_name == snapshot('codex')
    assert compacted.provider_name == snapshot('codex')
    assert compacted.provider_details == snapshot({'compaction': True})
    assert compacted.usage.input_tokens == snapshot(41)


@pytest.mark.vcr
async def test_codex_agent_run_stream(allow_model_requests: None, vcr: Cassette) -> None:
    """Streaming keeps the `codex` provider identity, never `openai`.

    The identity is what routes response ids, reasoning and telemetry, so it is asserted on the
    assembled response rather than left implicit in the recording: `codex` and `openai` are two
    different backends behind one SDK, and a run that silently reported `openai` would attribute
    Codex traffic to the Platform account.
    """
    agent = Agent(codex_model(vcr))

    async with agent.run_stream('Reply with exactly "codex-stream-ok" and nothing else.') as result:
        chunks = [chunk async for chunk in result.stream_text(delta=True, debounce_by=None)]

    assert ''.join(chunks) == snapshot('codex-stream-ok')
    assert len(chunks) > 1

    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.provider_name == snapshot('codex')
    assert response.provider_url == snapshot('https://chatgpt.com/backend-api/codex/')
    assert response.model_name == snapshot('gpt-5.5')
    assert response.finish_reason == snapshot('stop')

    assert single_request_body(vcr)['stream'] is True
    assert single_request_body(vcr)['store'] is False
