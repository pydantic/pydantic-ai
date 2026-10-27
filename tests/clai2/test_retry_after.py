"""Bound excessive Claude Code backoff without replacing the SDK's retry or fallback machinery."""

import io
import json
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import anyio
import httpx2
import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicBedrock, RateLimitError
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.exceptions import ModelHTTPError, ModelRateLimitError
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models import Model, override_allow_model_requests
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_ai_harness.subagents import DelegationTaskEvent, DelegationTasks, SubAgent, SubAgents
from pydantic_clai2 import chat
from pydantic_clai2.cli.headless import run_headless
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.errors import error_message
from pydantic_clai2.models import retry_after
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.plugins import HostEvent, ModelProvider, TurnEnd
from pydantic_clai2.runtime.forks import Forks
from pydantic_clai2.runtime.session import Session
from tests.clai2.test_app_edges import inputs


class SubscriptionAuth(httpx2.Auth):
    """Keep the plugin's bearer auth and its separate one-time 401 refresh across SDK retries."""

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        request.headers.pop('x-api-key', None)
        request.headers['Authorization'] = 'Bearer subscription'
        response = yield request
        if response.status_code == 401:
            await response.aread()
            await response.aclose()
            request.headers['Authorization'] = 'Bearer refreshed'
            yield request


def claude_model(client: httpx2.AsyncClient) -> AnthropicModel:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='unused', http_client=client))


def plugin_resolver(*, model: Model, store: SettingsStore | None = None, prefix: str = 'claude-code') -> ModelResolver:
    plugin = ModelProvider(prefix=prefix, resolve=lambda name: model, resolve_profile=lambda name, profile: model)
    backup = ModelProvider(prefix='backup', resolve=lambda name: TestModel(call_tools=[], custom_output_text='backup'))
    return ModelResolver(
        console=Console(file=io.StringIO()), plugins=lambda: {prefix: plugin, 'backup': backup}, store=store
    )


def success(*, stream: bool = False, tool: str | None = None) -> httpx2.Response:
    content: list[JsonValue] = (
        [{'type': 'tool_use', 'id': 'toolu_1', 'name': tool, 'input': {}}] if tool else [{'type': 'text', 'text': 'ok'}]
    )
    message: dict[str, JsonValue] = {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': content,
        'stop_reason': 'tool_use' if tool else 'end_turn',
        'stop_sequence': None,
        'usage': {'input_tokens': 1, 'output_tokens': 2},
    }
    if not stream:
        return httpx2.Response(200, json=message)
    events: list[dict[str, JsonValue]] = [
        {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}},
        {'type': 'content_block_start', 'index': 0, 'content_block': content[0]},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'message_delta', 'delta': {'stop_reason': message['stop_reason']}, 'usage': {'output_tokens': 2}},
        {'type': 'message_stop'},
    ]
    body = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
    return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, text=body)


def rejection(*, status: int = 429, headers: dict[str, str] | None = None) -> httpx2.Response:
    return httpx2.Response(
        status,
        headers=headers,
        json={'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'subscription exhausted'}},
    )


async def consume(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in events:
        pass


@pytest.fixture
def sdk_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    waits: list[float] = []

    async def sleep(delay: float) -> None:
        waits.append(delay)

    # Replace only the SDK's module reference, not anyio.sleep process-wide.
    monkeypatch.setattr('anthropic._base_client.anyio', SimpleNamespace(sleep=sleep))
    monkeypatch.setattr(
        'pydantic_clai2.models.retry_after.time.time', lambda: datetime(2026, 10, 23, tzinfo=UTC).timestamp()
    )
    return waits


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize(
    ('headers', 'blocked', 'delay'),
    [
        ({'retry-after': '10511', 'x-should-retry': 'true', 'request-id': 'kept'}, True, None),
        ({'retry-after': '60'}, False, 60.0),
        ({'retry-after': '60.1'}, True, None),
        ({'retry-after-ms': '60001'}, True, None),
        ({'retry-after-ms': '1000', 'retry-after': '10511'}, False, 1.0),
        ({'retry-after-ms': 'bad', 'retry-after': '10511'}, True, None),
        ({'retry-after': 'Fri, 23 Oct 2026 00:02:00 GMT'}, True, None),
        ({'retry-after': 'Thu, 22 Oct 2026 00:00:00 GMT'}, False, None),
        ({'retry-after': 'bad'}, False, None),
        ({}, False, None),
        ({'retry-after': '-1'}, False, None),
        ({'retry-after': '0'}, False, None),
        ({'retry-after': 'nan'}, False, None),
        ({'retry-after': 'inf'}, True, None),
        ({'retry-after-ms': 'nan', 'retry-after': '10511'}, False, None),
        ({'retry-after-ms': '-1', 'retry-after': '10511'}, False, None),
    ],
)
async def test_claude_code_retry_budget(
    headers: dict[str, str], blocked: bool, delay: float | None, stream: bool, sdk_sleeps: list[float]
) -> None:
    calls: list[str | None] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        calls.append(request.headers.get('authorization'))
        response = rejection(headers=headers) if len(calls) == 1 else success(stream=stream)
        return response

    observed: list[int] = []

    async def existing_hook(response: httpx2.Response) -> None:
        observed.append(response.status_code)

    auth = SubscriptionAuth()
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(respond), auth=auth, event_hooks={'response': [existing_hook]}
    ) as client:
        model = claude_model(client)
        resolver = plugin_resolver(model=model)
        resolved = await resolver.resolve('claude-code:claude-sonnet-4-5')
        assert resolved is model
        assert await resolver.resolve('claude-code@other:claude-sonnet-4-5') is model
        assert len(client.event_hooks['response']) == 2
        assert client.event_hooks['response'][0] is existing_hook
        assert client.auth is auth
        assert model.client.max_retries == 2
        with override_allow_model_requests(True):
            if blocked:
                with pytest.raises(ModelRateLimitError) as caught:
                    await Agent(model).run(
                        'go',
                        model_settings={'max_tokens': 20, 'max_retries': 2},
                        event_stream_handler=consume if stream else None,
                    )
                error = caught.value
                assert isinstance(error, ModelHTTPError)
                assert error.status_code == 429
                assert error.headers is not None
                assert all(error.headers[key] == value for key, value in headers.items() if key != 'x-should-retry')
                assert error.headers['x-should-retry'] == 'false'
                assert 'Claude Code rate limit (HTTP 429)' in error_message(error)
                assert '/model' in error_message(error)
                assert sdk_sleeps == []
            else:
                result = await Agent(model).run(
                    'go',
                    model_settings={'max_tokens': 20, 'max_retries': 2},
                    event_stream_handler=consume if stream else None,
                )
                assert result.output == 'ok'
                assert len(sdk_sleeps) == 1
                if delay is not None:
                    assert sdk_sleeps == [delay]
                else:
                    assert 0 < sdk_sleeps[0] <= 0.5
        assert calls == ['Bearer subscription'] * (1 if blocked else 2)
        assert observed == ([429] if blocked else [429, 200])
        assert not client.is_closed


@pytest.mark.parametrize('status', [401, 408, 409, 500, 503])
async def test_claude_code_preserves_other_retries(status: int, sdk_sleeps: list[float]) -> None:
    authorizations: list[str | None] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        authorizations.append(request.headers.get('authorization'))
        return rejection(status=status) if len(authorizations) == 1 else success()

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond), auth=SubscriptionAuth()) as client:
        model = await plugin_resolver(model=claude_model(client)).resolve('claude-code:claude-sonnet-4-5')
        with override_allow_model_requests(True):
            assert (await Agent(model).run('go', model_settings={'max_tokens': 20})).output == 'ok'
    assert authorizations == ['Bearer subscription', 'Bearer refreshed' if status == 401 else 'Bearer subscription']
    assert len(sdk_sleeps) == (0 if status == 401 else 1)


async def test_claude_code_preserves_connection_retries(sdk_sleeps: list[float]) -> None:
    calls = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx2.ConnectError('disconnected')
        return success()

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond), auth=SubscriptionAuth()) as client:
        model = await plugin_resolver(model=claude_model(client)).resolve('claude-code:claude-sonnet-4-5')
        with override_allow_model_requests(True):
            assert (await Agent(model).run('go', model_settings={'max_tokens': 20})).output == 'ok'
    assert calls == 2
    assert len(sdk_sleeps) == 1


@pytest.mark.parametrize('retries', [0, 2, 4])
async def test_short_rejection_keeps_sdk_retry_count(retries: int, sdk_sleeps: list[float]) -> None:
    calls = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return rejection(headers={'retry-after': '1'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = await plugin_resolver(model=claude_model(client)).resolve('claude-code:claude-sonnet-4-5')
        with override_allow_model_requests(True), pytest.raises(ModelRateLimitError):
            await Agent(model).run('go', model_settings={'max_tokens': 20, 'max_retries': retries})
    assert calls == retries + 1
    assert sdk_sleeps == [1.0] * retries


@pytest.mark.parametrize('prefix', ['claude-code', 'other-plugin'])
async def test_server_retry_veto_is_unchanged(prefix: str, sdk_sleeps: list[float]) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: rejection(headers={'retry-after': '10511', 'x-should-retry': 'false'})
        )
    ) as client:
        model = await plugin_resolver(model=claude_model(client), prefix=prefix).resolve(f'{prefix}:claude-sonnet-4-5')
        with override_allow_model_requests(True), pytest.raises(ModelRateLimitError) as caught:
            await Agent(model).run('go', model_settings={'max_tokens': 20})
    assert sdk_sleeps == []
    assert 'CLAI skipped' not in error_message(caught.value)


async def test_other_models_keep_their_retry_policy(sdk_sleeps: list[float]) -> None:
    calls = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return rejection(headers={'retry-after': '10511'}) if calls == 1 else success()

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = await plugin_resolver(model=claude_model(client), prefix='other-plugin').resolve(
            'other-plugin:claude-sonnet-4-5'
        )
        with override_allow_model_requests(True):
            assert (await Agent(model).run('go', model_settings={'max_tokens': 20})).output == 'ok'
        assert client.event_hooks['response'] == []
    assert sdk_sleeps == [10511.0]
    ordinary = TestModel()
    assert await plugin_resolver(model=ordinary).resolve('claude-code:any') is ordinary


@pytest.mark.parametrize(
    ('method', 'path'),
    [('GET', '/api/oauth/usage'), ('POST', '/api/oauth/usage'), ('GET', '/v1/messages'), ('POST', '/v1/models')],
)
async def test_shared_client_non_model_requests_are_unchanged(method: str, path: str, sdk_sleeps: list[float]) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(
            lambda request: rejection(headers={'retry-after': '10511', 'x-should-retry': 'true'})
        )
    ) as client:
        model = claude_model(client)
        await plugin_resolver(model=model).resolve('claude-code:claude-sonnet-4-5')
        assert isinstance(model.client, AsyncAnthropic)
        sdk = model.client.with_options(max_retries=0)
        request = sdk.get if method == 'GET' else sdk.post
        with pytest.raises(RateLimitError) as caught:
            await request(path, cast_to=object)
        assert caught.value.response.headers['x-should-retry'] == 'true'
        assert 'clai2.rejected_retry_after' not in caught.value.response.extensions
    assert sdk_sleeps == []


@pytest.mark.parametrize('guarded', [False, True])
async def test_token_count_rejections_share_the_model_retry_budget(guarded: bool, sdk_sleeps: list[float]) -> None:
    calls = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        assert request.method == 'POST' and request.url.path == '/v1/messages/count_tokens'
        return (
            rejection(headers={'retry-after': '10511'})
            if calls == 1
            else httpx2.Response(200, json={'input_tokens': 1})
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = claude_model(client)
        if guarded:
            await plugin_resolver(model=model).resolve('claude-code:claude-sonnet-4-5')
        assert isinstance(model.client, AsyncAnthropic)
        if guarded:
            with pytest.raises(RateLimitError) as caught:
                await model.client.beta.messages.count_tokens(
                    model='claude-sonnet-4-5', messages=[{'role': 'user', 'content': 'go'}]
                )
            assert caught.value.response.headers['x-should-retry'] == 'false'
        else:
            result = await model.client.beta.messages.count_tokens(
                model='claude-sonnet-4-5', messages=[{'role': 'user', 'content': 'go'}]
            )
            assert result.input_tokens == 1
    assert calls == (1 if guarded else 2)
    assert sdk_sleeps == ([] if guarded else [10511.0])


async def test_nonstandard_anthropic_client_is_not_decorated() -> None:
    async with httpx2.AsyncClient() as client:
        sdk = AsyncAnthropicBedrock(
            aws_access_key='test', aws_secret_key='test', aws_region='us-east-1', http_client=client
        )
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=sdk))
        retry_after.limit_retry_after(model=model)
        assert client.event_hooks['response'] == []


@pytest.mark.parametrize('stream', [False, True])
async def test_configured_chain_advances_after_long_rejection(
    tmp_path: Path, stream: bool, sdk_sleeps: list[float]
) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_chain(name='chosen', models=['claude-code:claude-sonnet-4-5', 'backup:test'])
    calls = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return rejection(headers={'retry-after': '10511'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = await plugin_resolver(model=claude_model(client), store=store).resolve('chain:chosen')
        with override_allow_model_requests(True):
            result = await Agent(model).run('go', event_stream_handler=consume if stream else None)
    assert result.output == 'backup'
    assert calls == 1
    assert sdk_sleeps == []


@pytest.mark.parametrize('chain', [False, True])
async def test_long_rejection_does_not_replay_completed_tools(
    tmp_path: Path, chain: bool, sdk_sleeps: list[float]
) -> None:
    calls = 0
    effects: list[str] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return success(stream=True, tool='effect') if calls == 1 else rejection(headers={'retry-after': '10511'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        resolver = plugin_resolver(model=claude_model(client), store=SettingsStore(tmp_path / 'config.db'))
        assert resolver.store is not None
        resolver.store.save_chain(name='chosen', models=['claude-code:claude-sonnet-4-5', 'backup:test'])
        model = await resolver.resolve('chain:chosen' if chain else 'claude-code:claude-sonnet-4-5')
        agent = Agent(model, model_settings={'max_tokens': 20})

        @agent.tool_plain
        def effect() -> str:
            effects.append('done')
            return 'completed effect'

        session = Session(
            agent,
            deps=None,
            workspace=tmp_path,
            conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        )
        with override_allow_model_requests(True):
            if chain:
                assert (await session.prompt('go')).output == 'backup'
            else:
                with pytest.raises(ModelRateLimitError):
                    await session.prompt('go')
        assert session.conversations is not None
        saved = await session.conversations.get(conversation_id=session.summary.id)
        assert saved.summary.outcome == ('completed' if chain else 'failed')
        returns = [
            part
            for message in saved.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        assert len(returns) == 1 and returns[0].content == 'completed effect'
    assert effects == ['done']
    assert calls == 2
    assert sdk_sleeps == []


@pytest.mark.parametrize('mode', ['interactive', 'headless'])
async def test_long_retry_is_visible_and_history_retained(
    tmp_path: Path,
    mode: str,
    sdk_sleeps: list[float],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: rejection(headers={'retry-after': '10511'}))
    ) as client:
        model = await plugin_resolver(model=claude_model(client)).resolve('claude-code:claude-sonnet-4-5')
        agent = Agent(model)
        store = SettingsStore(tmp_path / 'config.db')
        settings = Settings(model=None, code_mode=False, session_namer=False)
        with override_allow_model_requests(True):
            if mode == 'headless':
                assert (
                    await run_headless(
                        text='keep this prompt', agent=agent, settings=settings, store=store, project=ProjectSettings()
                    )
                    == 1
                )
                captured = capsys.readouterr()
                assert captured.out == ''
                output = captured.err
            else:
                inputs(monkeypatch, ['keep this prompt', '/exit'])
                console = io.StringIO()
                await chat(
                    agent,
                    deps=None,
                    settings=settings,
                    store=store,
                    load_plugins=False,
                    console=Console(file=console, width=200),
                )
                output = console.getvalue()
        output = ' '.join(output.split())
        assert 'Claude Code rate limit (HTTP 429)' in output
        assert 'Retry after 10511s' in output
        assert 'CLAI skipped a wait over 60s' in output
        assert '/model' in output
        summaries = await SqliteConversationStore(database=tmp_path / 'sessions.db').listing()
        assert len(summaries) == 1 and summaries[0].outcome == 'failed'
    assert sdk_sleeps == []


@pytest.mark.parametrize('background', [False, True])
async def test_delegated_rejection_settles(tmp_path: Path, background: bool, sdk_sleeps: list[float]) -> None:
    finished = anyio.Event()

    async def observe(event: DelegationTaskEvent) -> None:
        if event.task.status == 'finished':
            finished.set()

    async def parent_stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {
                0: DeltaToolCall(
                    name='delegate_task',
                    json_args=json.dumps({'agent_name': 'child', 'task': 'go', 'background': background}),
                    tool_call_id='delegate',
                )
            }
        else:
            yield 'parent done'

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: rejection(headers={'retry-after': '10511'}))
    ) as client:
        model = await plugin_resolver(model=claude_model(client)).resolve('claude-code:claude-sonnet-4-5')
        child = Agent(model, name='child')
        owner = DelegationTasks(directory=tmp_path / 'tasks', observer=observe)
        parent = Agent(
            FunctionModel(stream_function=parent_stream),
            capabilities=[SubAgents(agents=[SubAgent(child)], agent_folders=None)],
        )
        with override_allow_model_requests(True), anyio.fail_after(10):
            async with owner.opened():
                with owner.bind():
                    await parent.run('delegate', event_stream_handler=consume)
                    await finished.wait()
                    (record,) = owner.records.values()
                    assert record.status == 'finished' and record.outcome == 'failed'
                    assert '429' in record.output
    assert sdk_sleeps == []


async def test_fork_resolves_a_guarded_client(tmp_path: Path, sdk_sleeps: list[float]) -> None:
    finished = anyio.Event()
    output = io.StringIO()

    async def fire(event: HostEvent) -> None:
        if isinstance(event, TurnEnd):
            assert event.outcome == 'failed'
            finished.set()

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: rejection(headers={'retry-after': '10511'}))
    ) as client:
        resolver = plugin_resolver(model=claude_model(client))

        def spawn(model: str | None, history: Sequence[ModelMessage]) -> Session[None, str]:
            session = Session(Agent(TestModel()), deps=None, workspace=tmp_path, message_history=history)
            session.model = model
            session.resolve_model = resolver.resolve
            return session

        forks = Forks(console=Console(file=output, width=200), history=lambda: (), spawn=spawn, fire=fire)
        with override_allow_model_requests(True), anyio.fail_after(10):
            await forks.start('go', model='claude-code:claude-sonnet-4-5')
            await finished.wait()
            await forks.records[0].task
        assert forks.records[0].status == 'failed'
        assert 'Retry after 10511s' in output.getvalue()
    assert sdk_sleeps == []


@pytest.mark.parametrize('guarded', [False, True])
@pytest.mark.parametrize(
    ('header', 'failure'),
    [
        ('Fri, 23 Oct 10000 00:02:00', ValueError),
        ('Fri, 23 Oct 999999999999999999999999 00:02:00', OverflowError),
    ],
)
async def test_unrepresentable_retry_date_preserves_sdk_failure(
    header: str, failure: type[Exception], guarded: bool, sdk_sleeps: list[float]
) -> None:
    calls = 0

    def respond(request: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return rejection(headers={'retry-after': header})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = claude_model(client)
        if guarded:
            await plugin_resolver(model=model).resolve('claude-code:claude-sonnet-4-5')
        with override_allow_model_requests(True), pytest.raises(failure) as caught:
            await Agent(model).run('go', model_settings={'max_tokens': 20})
        # The SDK itself cannot convert these dates: preserve its raw error rather than
        # raising from the HTTP hook and causing SDK connection-error retries instead.
        assert 'CLAI skipped' not in error_message(caught.value)
    assert calls == 1
    assert sdk_sleeps == []


def test_unrelated_rate_limit_errors_keep_their_message() -> None:
    error = ModelRateLimitError(model_name='test', message='another provider rate limit')
    assert error_message(error) == str(error)


async def test_cancel_during_an_allowed_sdk_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = anyio.Event()

    async def wait(delay: float) -> None:
        assert delay == 1.0
        entered.set()
        await anyio.Event().wait()

    monkeypatch.setattr('anthropic._base_client.anyio', SimpleNamespace(sleep=wait))
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: rejection(headers={'retry-after': '1'}))
    ) as client:
        model = await plugin_resolver(model=claude_model(client)).resolve('claude-code:claude-sonnet-4-5')
        session = Session(
            Agent(model),
            deps=None,
            workspace=tmp_path,
            conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        )
        with override_allow_model_requests(True), anyio.fail_after(10):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(session.prompt, 'go')
                await entered.wait()
                tasks.cancel_scope.cancel()
        assert session.summary.outcome == 'cancelled'
        assert session.messages
