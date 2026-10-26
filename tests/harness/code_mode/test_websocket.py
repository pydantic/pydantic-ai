"""End-to-end tests for CodeMode's remote Monty transport."""

from __future__ import annotations

import asyncio
import os
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from http import HTTPStatus
from pathlib import Path
from typing import Any, Never

import pytest
import websockets
from pydantic_monty import AsyncMontyWebsocket, MountDir
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import CodeMode, HarnessDeprecationWarning
from pydantic_ai_harness.code_mode import CodeModeToolset, MontyServer


def _parts(messages: list[ModelMessage], part_type: type[Any]) -> list[Any]:
    return [part for message in messages for part in message.parts if isinstance(part, part_type)]


def _retries(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for part in _parts(messages, ToolReturnPart) if part.outcome == 'retried']


def _snippets_model(*snippets: str) -> FunctionModel:
    """A model that calls `run_code` with each snippet in turn, then says 'done'."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        done = len(_parts(messages, ToolReturnPart))
        if done < len(snippets):
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': snippets[done]})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


@pytest.mark.skip(reason='Hangs intermittently in CI: https://github.com/pydantic/pydantic-ai/issues/8824')
async def test_code_mode_runs_over_websocket(websocket_relay_url: str, tmp_path: Path) -> None:
    """Remote feeds keep REPL state while tools, prints, mounts, `gather`, and barriers stay host-side."""
    (tmp_path / 'input.txt').write_text('mounted data')
    agent = Agent(
        _snippets_model(
            'value = await add(a=2, b=3)\nprint("remote tool result", value)',
            'from pathlib import Path\nprint(Path("/workspace/input.txt").read_text())\nvalue * 10',
            'import asyncio\nfirst, second = await asyncio.gather(add(a=1, b=1), add(a=2, b=2))\n[first, second, barrier()]',
        ),
        capabilities=[
            CodeMode(
                monty_server=MontyServer(websocket_relay_url),
                mount=MountDir(virtual_path='/workspace', host_path=tmp_path),
            )
        ],
    )

    @agent.tool_plain
    async def add(a: int, b: int) -> int:
        return a + b

    @agent.tool_plain(sequential=True)
    def barrier() -> str:
        return 'barrier'

    result = await agent.run('exercise the remote sandbox')

    assert result.output == 'done'
    assert [part.content for part in _parts(result.all_messages(), ToolReturnPart)] == [
        {'output': 'remote tool result 5\n'},
        {'output': 'mounted data\n', 'result': 50},
        [2, 4, 'barrier'],
    ]


@pytest.mark.parametrize('url', ['ws://monty-server:8000', 'wss://sandbox.example.com/monty'])
async def test_websocket_server_url_accepted(url: str) -> None:
    """Any `ws://` or `wss://` URL passes validation; workers dial lazily, not at enter."""
    agent = Agent(_snippets_model(), capabilities=[CodeMode(monty_server=MontyServer(url))])
    result = await agent.run('no run_code call, so nothing dials')
    assert result.output == 'done'


def test_non_websocket_server_url_rejected() -> None:
    """A URL Monty's WebSocket client cannot dial fails where the server is written, not at a later run."""
    with pytest.raises(UserError, match="not scheme 'https'"):
        MontyServer('https://sandbox.example.com/monty')


def test_deprecated_sandbox_url_becomes_monty_server() -> None:
    """`monty_sandbox_url` still works, warns once, and turns into a `MontyServer`."""
    with pytest.warns(HarnessDeprecationWarning, match=r'renamed to `CodeMode\(monty_server=\.\.\.\)`') as warned:
        capability = CodeMode(monty_sandbox_url='wss://sandbox.example.com/monty')
    assert warned[0].filename == __file__  # the caller's line, as users write it
    server = capability.monty_server
    assert not callable(server) and isinstance(server, MontyServer)
    assert server.url == 'wss://sandbox.example.com/monty'
    assert capability.monty_sandbox_url is None  # moved into `monty_server`, like other renamed fields
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert replace(capability).monty_server == capability.monty_server

    with pytest.warns(HarnessDeprecationWarning, match='monty_sandbox_url') as warned:
        toolset = CodeModeToolset(FunctionToolset(), monty_sandbox_url='wss://sandbox.example.com/monty')
    assert warned[0].filename == __file__
    assert toolset.monty_server == capability.monty_server


def test_url_becomes_monty_server() -> None:
    """A URL is shorthand for `MontyServer(url)`; an agent spec can also give the server as a mapping."""
    server = MontyServer[None]('wss://sandbox.example.com/monty')
    assert CodeMode[None](monty_server='wss://sandbox.example.com/monty').monty_server == server
    assert (
        CodeModeToolset[None](FunctionToolset(), monty_server='wss://sandbox.example.com/monty').monty_server == server
    )
    capability = CodeMode[None].from_spec(monty_server='wss://sandbox.example.com/monty')
    assert isinstance(capability, CodeMode)
    assert capability.monty_server == server
    capability = CodeMode[None].from_spec(monty_server={'url': 'wss://sandbox.example.com/monty', 'auth': 'k-123'})
    assert isinstance(capability, CodeMode)
    assert capability.monty_server == MontyServer('wss://sandbox.example.com/monty', auth='k-123')
    with pytest.raises(UserError, match="not scheme 'https'"):
        CodeMode[None](monty_server='https://sandbox.example.com/monty')


def test_deprecated_sandbox_url_and_monty_server_rejected() -> None:
    """Passing both forms is ambiguous, so it raises rather than picking one."""
    with pytest.raises(UserError, match='Pass `monty_server` only'):
        CodeMode[None](monty_server=MontyServer('wss://a.example.com'), monty_sandbox_url='wss://b.example.com')


@pytest.mark.parametrize(
    ('resource_limits', 'request_timeout'),
    [(None, 40.0), ({'max_duration_secs': 90}, 100.0), ('unlimited', None)],
)
async def test_remote_turn_deadline_follows_max_duration(
    monkeypatch: pytest.MonkeyPatch, resource_limits: Any, request_timeout: float | None
) -> None:
    """The transport's per-turn deadline leaves room for `max_duration_secs` to fire first."""
    seen: list[float | None] = []

    def recording_pool(url: str, *, request_timeout: float | None, connect_headers: Any) -> Never:
        seen.append(request_timeout)
        raise RuntimeError('not dialing')

    monkeypatch.setattr('pydantic_ai_harness._monty_exec.AsyncMontyWebsocket', recording_pool)
    agent = Agent(
        _snippets_model('1'),
        capabilities=[
            CodeMode(monty_server=MontyServer('wss://sandbox.example.com/monty'), resource_limits=resource_limits)
        ],
    )
    await agent.run('record the deadline')
    assert seen == [request_timeout]


async def test_dial_failure_redacts_server_url() -> None:
    """A failed dial's retry message must not leak the URL, which may carry credentials."""
    agent = Agent(
        _snippets_model('1 + 1'),
        capabilities=[CodeMode(monty_server=MontyServer('ws://127.0.0.1:1/?token=hunter2', auth='k-123'))],
    )

    result = await agent.run('dial a dead worker')

    (retry,) = _retries(result.all_messages())
    assert 'hunter2' not in str(retry.content)
    assert '<MontyServer.url>' in str(retry.content)


async def test_dial_failure_redacts_pool_factory_url() -> None:
    """A pool built by a function has no URL to match, so any WebSocket URL is kept out of the retry message."""
    agent = Agent(
        _snippets_model('1 + 1'),
        capabilities=[CodeMode(monty_server=lambda: AsyncMontyWebsocket('ws://127.0.0.1:1/?token=hunter2'))],
    )

    result = await agent.run('dial a dead worker')

    (retry,) = _retries(result.all_messages())
    assert 'hunter2' not in str(retry.content)
    assert '<Monty server URL>' in str(retry.content)


def test_repr_hides_server_url_and_key() -> None:
    """The URL and key are credentials, so neither the server, the capability, nor its toolset shows them."""
    server = MontyServer[None]('wss://alice:s3cr3t@sandbox.example.com/monty', auth='k-123')
    capability = CodeMode[None](monty_server=server)
    toolset = capability.get_wrapper_toolset(FunctionToolset[None]())

    for shown in (repr(server), repr(capability), repr(toolset)):
        assert 'sandbox.example.com' not in shown
        assert 'k-123' not in shown


@dataclass
class _Deps:
    monty_key: str


@asynccontextmanager
async def _refusing_server(seen: list[Request]) -> AsyncGenerator[str]:
    """A WebSocket endpoint that records each upgrade request and refuses it with 401."""

    def refuse(connection: ServerConnection, request: Request) -> Response:
        seen.append(request)
        return connection.respond(HTTPStatus.UNAUTHORIZED, 'unauthorized')

    async def never_called(connection: ServerConnection) -> None:  # pragma: no cover
        pass

    async with websockets.serve(never_called, '127.0.0.1', 0, process_request=refuse) as server:
        port: int = next(iter(server.sockets)).getsockname()[1]
        yield f'ws://127.0.0.1:{port}/monty-ws/'


async def test_key_reaches_the_server() -> None:
    """The key reaches the server; a refusal stays a retry, without the key in it."""
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'),
            deps_type=_Deps,
            capabilities=[CodeMode(monty_server=MontyServer(url, auth=lambda ctx: ctx.deps.monty_key))],
        )
        result = await agent.run('dial with a bad key', deps=_Deps(monty_key='k-run'))

    (retry,) = _retries(result.all_messages())
    assert 'k-run' not in str(retry.content)
    (request,) = seen
    assert request.headers['Authorization'] == 'Bearer k-run'


@pytest.mark.skipif(not os.environ.get('MONTY_SANDBOX_TEST_URL'), reason='needs a hosted Monty server (CI on main)')
async def test_hosted_monty_server() -> None:  # pragma: no cover - remote mode only
    """The shipped `monty_server` path, URL plus `LOGFIRE_API_KEY`, runs code and tools on a real server."""
    agent = Agent(
        _snippets_model('await add(a=20, b=22)'),
        capabilities=[CodeMode(monty_server=os.environ['MONTY_SANDBOX_TEST_URL'])],
    )

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    result = await agent.run('add on the server')
    assert [part.content for part in _parts(result.all_messages(), ToolReturnPart)] == [42]


@pytest.mark.skipif(not os.environ.get('MONTY_SANDBOX_TEST_URL'), reason='needs a hosted Monty server (CI on main)')
async def test_hosted_monty_pool_factory() -> None:  # pragma: no cover - remote mode only
    """A function that builds the pool, as `logfire.agent_sandboxes` does, runs code and tools on a real server."""
    headers = {'Authorization': f'Bearer {os.environ["LOGFIRE_API_KEY"]}'}
    agent = Agent(
        _snippets_model('await add(a=20, b=22)'),
        capabilities=[
            CodeMode(
                monty_server=lambda: AsyncMontyWebsocket(
                    os.environ['MONTY_SANDBOX_TEST_URL'], connect_headers=lambda: headers
                )
            )
        ],
    )

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    result = await agent.run('add on the server')
    assert [part.content for part in _parts(result.all_messages(), ToolReturnPart)] == [42]


def test_pool_instead_of_factory_rejected() -> None:
    """Passing the pool itself, such as a factory called by mistake, fails where it is written."""
    with pytest.raises(UserError, match='not the pool itself'):
        CodeMode(monty_server=AsyncMontyWebsocket('wss://sandbox.example.com/monty'))  # pyright: ignore[reportArgumentType]


async def test_failing_pool_factory_is_a_retry() -> None:
    """A factory that raises becomes a retry, and the next call builds the pool again."""
    calls: list[None] = []

    def build_pool() -> AsyncMontyWebsocket:
        calls.append(None)
        raise RuntimeError('no sandbox for you')

    agent = Agent(_snippets_model('1 + 1', '2 + 2'), capabilities=[CodeMode(monty_server=build_pool)])
    result = await agent.run('dial')

    assert ['no sandbox for you' in str(retry.content) for retry in _retries(result.all_messages())] == [True, True]
    assert len(calls) == 2


async def test_pool_factory_builds_each_run_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A function passed as `monty_server` builds a fresh pool per run, dialed with its own headers rather than `LOGFIRE_API_KEY`."""
    monkeypatch.setenv('LOGFIRE_API_KEY', 'k-env')
    seen: list[Request] = []
    pools: list[AsyncMontyWebsocket] = []
    async with _refusing_server(seen) as url:

        def build_pool() -> AsyncMontyWebsocket:
            pools.append(AsyncMontyWebsocket(url, connect_headers=lambda: {'Authorization': 'Bearer k-pool'}))
            return pools[-1]

        agent = Agent(_snippets_model('1 + 1'), capabilities=[CodeMode(monty_server=build_pool)])
        for _ in range(2):
            await agent.run('dial')

    assert len(pools) == 2 and pools[0] is not pools[1]
    assert [request.headers['Authorization'] for request in seen] == ['Bearer k-pool', 'Bearer k-pool']


def _no_key(ctx: RunContext[None]) -> None:
    return None


@pytest.mark.parametrize(
    ('env_key', 'auth', 'sent'),
    [
        ('k-env', None, 'Bearer k-env'),
        ('k-env', '', 'Bearer k-env'),
        ('k-env', _no_key, None),
        (None, None, None),
    ],
)
async def test_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str | None,
    auth: Any,
    sent: str | None,
) -> None:
    """Unset or `''`, `auth` sends `LOGFIRE_API_KEY` if set, as model providers do; a function returning `None` sends no key."""
    if env_key is None:
        monkeypatch.delenv('LOGFIRE_API_KEY', raising=False)
    else:
        monkeypatch.setenv('LOGFIRE_API_KEY', env_key)
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'),
            capabilities=[CodeMode(monty_server=MontyServer(url, auth=auth))],
        )
        await agent.run('dial')

    (request,) = seen
    assert request.headers.get('Authorization') == sent


async def test_auth_function_with_entered_agent() -> None:
    """`async with agent` enters a `toolsets=` toolset outside any run; an async `auth` still gives each run its key."""
    seen: list[Request] = []

    async def key_for(ctx: RunContext[_Deps]) -> str:
        return ctx.deps.monty_key

    async with _refusing_server(seen) as url:
        toolset = CodeModeToolset[_Deps](FunctionToolset(), monty_server=MontyServer(url, auth=key_for))
        agent = Agent(_snippets_model('1 + 1'), deps_type=_Deps, toolsets=[toolset])
        async with agent:
            for key in ('k-first', 'k-second'):
                await agent.run('dial', deps=_Deps(monty_key=key))

    assert [request.headers['Authorization'] for request in seen] == ['Bearer k-first', 'Bearer k-second']


async def test_deprecated_sandbox_url_sends_no_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`monty_sandbox_url` never sent a key, so it keeps ignoring `LOGFIRE_API_KEY`."""
    monkeypatch.setenv('LOGFIRE_API_KEY', 'k-env')
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        with pytest.warns(HarnessDeprecationWarning):
            capability = CodeMode(monty_sandbox_url=url)
        agent = Agent(_snippets_model('1 + 1'), capabilities=[capability])
        await agent.run('dial')

    (request,) = seen
    assert 'Authorization' not in request.headers


@pytest.mark.skip(reason='Hangs intermittently in CI: https://github.com/pydantic/pydantic-ai/issues/8824')
async def test_disconnect_mid_snippet_reports_started_calls(websocket_relay_url: str) -> None:
    """A dropped worker connection resets the session and lists the calls that already started."""
    connections: list[websockets.ServerConnection] = []

    async def proxy(client: websockets.ServerConnection) -> None:
        connections.append(client)
        async with websockets.connect(websocket_relay_url, max_size=None) as upstream:

            async def pump(source: Any, sink: Any) -> None:
                async for message in source:
                    await sink.send(message)

            await asyncio.gather(pump(client, upstream), pump(upstream, client), return_exceptions=True)

    async with websockets.serve(proxy, '127.0.0.1', 0, max_size=None) as server:
        port: int = next(iter(server.sockets)).getsockname()[1]
        agent = Agent(
            _snippets_model('await drop_connection()'),
            capabilities=[CodeMode(monty_server=MontyServer(f'ws://127.0.0.1:{port}'))],
        )

        @agent.tool_plain
        async def drop_connection() -> None:
            # One statement: the harness cancels this while the close is in flight, so a separate
            # wait line would only sometimes run. The wait never ends on its own.
            await asyncio.gather(*(connection.close() for connection in connections), asyncio.Event().wait())

        result = await agent.run('lose the worker mid-snippet')

    (retry,) = _retries(result.all_messages())
    assert 'MontyDisconnectError' in str(retry.content)
    assert 'drop_connection({}) did not finish' in str(retry.content)
