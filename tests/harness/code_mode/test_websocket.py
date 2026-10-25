"""End-to-end tests for CodeMode's remote Monty transport."""

from __future__ import annotations

import asyncio
import os
import re
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from http import HTTPStatus
from pathlib import Path
from typing import Any, Never

import pytest
import websockets
from pydantic_monty import MountDir
from websockets.asyncio.server import ServerConnection
from websockets.http11 import Request, Response

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai_harness import CodeMode, HarnessDeprecationWarning
from pydantic_ai_harness.code_mode import CodeModeToolset, MontyServer

from .test_code_mode import build_ctx, close_direct_toolsets


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


def test_auth_and_authorization_header_rejected() -> None:
    """The key goes in one place, so a conflicting `Authorization` header is refused."""
    with pytest.raises(UserError, match='not both'):
        MontyServer('wss://sandbox.example.com/monty', auth='k-123', headers={'authorization': 'Bearer other'})


def test_deprecated_sandbox_url_becomes_monty_server() -> None:
    """`monty_sandbox_url` still works, warns once, and turns into a `MontyServer`."""
    with pytest.warns(HarnessDeprecationWarning, match=r'renamed to `CodeMode\(monty_server=\.\.\.\)`') as warned:
        capability = CodeMode(monty_sandbox_url='wss://sandbox.example.com/monty')
    assert warned[0].filename == __file__  # the caller's line, as users write it
    assert capability.monty_server == MontyServer('wss://sandbox.example.com/monty')
    assert capability.monty_sandbox_url == 'wss://sandbox.example.com/monty'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert replace(capability).monty_server == capability.monty_server

    with pytest.warns(HarnessDeprecationWarning, match='CodeModeToolset') as warned:
        toolset = CodeModeToolset(FunctionToolset(), monty_sandbox_url='wss://sandbox.example.com/monty')
    assert warned[0].filename == __file__
    assert toolset.monty_server == MontyServer('wss://sandbox.example.com/monty')


def test_url_becomes_monty_server() -> None:
    """A URL is shorthand for `MontyServer(url)`, in code and in an agent spec."""
    server = MontyServer[None]('wss://sandbox.example.com/monty')
    assert CodeMode[None](monty_server='wss://sandbox.example.com/monty').monty_server == server
    assert (
        CodeModeToolset[None](FunctionToolset(), monty_server='wss://sandbox.example.com/monty').monty_server == server
    )
    capability = CodeMode[None].from_spec(monty_server='wss://sandbox.example.com/monty')
    assert isinstance(capability, CodeMode)
    assert capability.monty_server == server
    with pytest.raises(UserError, match="not scheme 'https'"):
        CodeMode[None](monty_server='https://sandbox.example.com/monty')


def test_spec_mapping_becomes_monty_server() -> None:
    """An agent spec passes `monty_server` as a mapping of the server's fields."""
    capability = CodeMode[None].from_spec(monty_server={'url': 'wss://sandbox.example.com/monty', 'auth': 'k-123'})
    assert isinstance(capability, CodeMode)
    assert capability.monty_server == MontyServer('wss://sandbox.example.com/monty', auth='k-123')


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


def test_repr_hides_server_url_and_key() -> None:
    """The URL and key are credentials, so neither the server, the capability, nor its toolset shows them."""
    server = MontyServer[None]('wss://alice:s3cr3t@sandbox.example.com/monty', auth='k-123', headers={'X-Key': 'h-1'})
    capability = CodeMode[None](monty_server=server)
    toolset = capability.get_wrapper_toolset(FunctionToolset[None]())

    for shown in (repr(server), repr(capability), repr(toolset)):
        assert 'sandbox.example.com' not in shown
        assert 'k-123' not in shown
        assert 'h-1' not in shown


@dataclass
class _Deps:
    monty_key: str


@asynccontextmanager
async def _refusing_server(seen: list[Request], status: HTTPStatus = HTTPStatus.UNAUTHORIZED) -> AsyncGenerator[str]:
    """A WebSocket endpoint that records each upgrade request and refuses it with `status`."""

    def refuse(connection: ServerConnection, request: Request) -> Response:
        seen.append(request)
        return connection.respond(status, status.phrase)

    async def never_called(connection: ServerConnection) -> None:  # pragma: no cover
        pass

    async with websockets.serve(never_called, '127.0.0.1', 0, process_request=refuse) as server:
        port: int = next(iter(server.sockets)).getsockname()[1]
        yield f'ws://127.0.0.1:{port}/monty-ws/'


async def test_refused_key_is_a_user_error() -> None:
    """The key and headers reach the server, and a 401 is for the developer, not a retry for the model."""
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'),
            deps_type=_Deps,
            capabilities=[
                CodeMode(monty_server=MontyServer(url, auth=lambda ctx: ctx.deps.monty_key, headers={'X-Proxy': 'p-1'}))
            ],
        )
        with pytest.raises(UserError, match=r'refused the connection \(401\)') as raised:
            await agent.run('dial with a bad key', deps=_Deps(monty_key='k-run'))

    assert 'k-run' not in str(raised.value)
    assert 'check the key `MontyServer.auth` gives' in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    (request,) = seen
    assert request.headers['Authorization'] == 'Bearer k-run'
    assert request.headers['X-Proxy'] == 'p-1'


@pytest.mark.skipif(not os.environ.get('MONTY_SANDBOX_TEST_URL'), reason='needs a hosted Monty server (CI on main)')
async def test_hosted_monty_server() -> None:  # pragma: no cover - remote mode only
    """The shipped `monty_server` path, URL plus `MONTY_EXECUTION_KEY`, runs code and tools on a real server."""
    agent = Agent(
        _snippets_model('await add(a=20, b=22)'),
        capabilities=[CodeMode(monty_server=os.environ['MONTY_SANDBOX_TEST_URL'])],
    )

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    result = await agent.run('add on the server')
    assert [part.content for part in _parts(result.all_messages(), ToolReturnPart)] == [42]


@pytest.mark.parametrize(
    ('env_key', 'auth', 'headers', 'sent', 'advice'),
    [
        ('k-env', None, None, 'Bearer k-env', 'check the key in the `MONTY_EXECUTION_KEY`'),
        ('k-env', '', None, 'Bearer k-env', 'check the key in the `MONTY_EXECUTION_KEY`'),
        ('k-env', None, {'Authorization': 'Basic b-1'}, 'Basic b-1', 'check the `Authorization` header'),
        (None, None, None, None, 'no key was sent: set `MONTY_EXECUTION_KEY`'),
    ],
)
async def test_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
    env_key: str | None,
    auth: str | None,
    headers: dict[str, str] | None,
    sent: str | None,
    advice: str,
) -> None:
    """Without `auth` (or with `''`), `MONTY_EXECUTION_KEY` is sent; a refusal names where the key came from."""
    if env_key is None:
        monkeypatch.delenv('MONTY_EXECUTION_KEY', raising=False)
    else:
        monkeypatch.setenv('MONTY_EXECUTION_KEY', env_key)
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'),
            capabilities=[CodeMode(monty_server=MontyServer(url, auth=auth, headers=headers))],
        )
        with pytest.raises(UserError, match=re.escape(advice)):
            await agent.run('dial')

    (request,) = seen
    assert request.headers.get('Authorization') == sent


async def test_direct_toolset_resolves_auth_from_call_context() -> None:
    """Used without an agent, the toolset calls `auth` with the context `run_code` is called with."""
    seen: list[Request] = []
    async with _refusing_server(seen, HTTPStatus.FORBIDDEN) as url:
        toolset = CodeModeToolset[_Deps](
            FunctionToolset(), monty_server=MontyServer(url, auth=lambda ctx: ctx.deps.monty_key)
        )
        ctx = await build_ctx(_Deps(monty_key='k-direct'), toolset)
        tools = await toolset.get_tools(ctx)
        with pytest.raises(UserError, match=r'refused the connection \(403\)'):
            await toolset.call_tool('run_code', {'code': '1'}, ctx, tools['run_code'])
        await close_direct_toolsets()

    (request,) = seen
    assert request.headers['Authorization'] == 'Bearer k-direct'


async def test_auth_function_not_called_without_code() -> None:
    """A run that never runs code never dials, so it never asks for a key."""
    calls: list[object] = []

    def key_for(ctx: Any) -> str:  # pragma: no cover - asserting it is never called
        calls.append(ctx)
        return 'k-unused'

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('no code needed')])

    agent = Agent(
        FunctionModel(answer),
        capabilities=[CodeMode(monty_server=MontyServer('wss://unused.example.com', auth=key_for))],
    )
    assert (await agent.run('hi')).output == 'no code needed'
    assert calls == []


async def test_auth_function_with_entered_agent() -> None:
    """`async with agent` enters a `toolsets=` toolset outside any run; each run still sends its own key."""
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        toolset = CodeModeToolset[_Deps](
            FunctionToolset(), monty_server=MontyServer(url, auth=lambda ctx: ctx.deps.monty_key)
        )
        agent = Agent(_snippets_model('1 + 1'), deps_type=_Deps, toolsets=[toolset])
        async with agent:
            for key in ('k-first', 'k-second'):
                with pytest.raises(UserError, match=r'refused the connection \(401\)'):
                    await agent.run('dial', deps=_Deps(monty_key=key))

    assert [request.headers['Authorization'] for request in seen] == ['Bearer k-first', 'Bearer k-second']


async def test_async_auth_resolved_per_run() -> None:
    """An async `auth` function is awaited at the start of each run, so each run sends its own key."""
    seen: list[Request] = []

    async def key_for(ctx: Any) -> str:
        return ctx.deps.monty_key

    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'),
            deps_type=_Deps,
            capabilities=[CodeMode(monty_server=MontyServer(url, auth=key_for))],
        )
        for key in ('k-first', 'k-second'):
            with pytest.raises(UserError):
                await agent.run('dial', deps=_Deps(monty_key=key))

    assert [request.headers['Authorization'] for request in seen] == ['Bearer k-first', 'Bearer k-second']


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
