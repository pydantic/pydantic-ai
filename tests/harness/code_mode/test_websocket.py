"""End-to-end tests for CodeMode's remote Monty transport."""

from __future__ import annotations

import asyncio
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
                executor=MontyServer(websocket_relay_url),
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
    agent = Agent(_snippets_model(), capabilities=[CodeMode(executor=MontyServer(url))])
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


def test_deprecated_sandbox_url_becomes_executor() -> None:
    """`monty_sandbox_url` still works, warns once, and turns into a `MontyServer`."""
    with pytest.warns(HarnessDeprecationWarning, match='monty_sandbox_url'):
        capability = CodeMode[None](monty_sandbox_url='wss://sandbox.example.com/monty')
    assert capability.executor == MontyServer('wss://sandbox.example.com/monty')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert replace(capability).executor == capability.executor

    with pytest.warns(HarnessDeprecationWarning, match='CodeModeToolset'):
        toolset = CodeModeToolset[None](FunctionToolset(), monty_sandbox_url='wss://sandbox.example.com/monty')
    assert toolset.executor == MontyServer('wss://sandbox.example.com/monty')


def test_spec_mapping_becomes_executor() -> None:
    """An agent spec passes `executor` as a mapping of the server's fields."""
    capability = CodeMode[None].from_spec(executor={'url': 'wss://sandbox.example.com/monty', 'auth': 'k-123'})
    assert isinstance(capability, CodeMode)
    assert capability.executor == MontyServer('wss://sandbox.example.com/monty', auth='k-123')


async def test_auth_function_needs_a_run() -> None:
    """Entered without `for_run`, a toolset has no run context to call `auth` with, so it says so."""
    toolset = CodeModeToolset[None](
        FunctionToolset(), executor=MontyServer('wss://sandbox.example.com/monty', auth=lambda ctx: 'k-123')
    )
    with pytest.raises(UserError, match='needs a run context'):
        async with toolset:
            pass  # pragma: no cover


def test_deprecated_sandbox_url_and_executor_rejected() -> None:
    """Passing both forms is ambiguous, so it raises rather than picking one."""
    with pytest.raises(UserError, match='Pass `executor` only'):
        CodeMode[None](executor=MontyServer('wss://a.example.com'), monty_sandbox_url='wss://b.example.com')


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
            CodeMode(executor=MontyServer('wss://sandbox.example.com/monty'), resource_limits=resource_limits)
        ],
    )
    await agent.run('record the deadline')
    assert seen == [request_timeout]


async def test_dial_failure_redacts_server_url() -> None:
    """A failed dial's retry message must not leak the URL, which may carry credentials."""
    agent = Agent(
        _snippets_model('1 + 1'),
        capabilities=[CodeMode(executor=MontyServer('ws://127.0.0.1:1/?token=hunter2', auth='k-123'))],
    )

    result = await agent.run('dial a dead worker')

    (retry,) = _retries(result.all_messages())
    assert 'hunter2' not in str(retry.content)
    assert '<MontyServer.url>' in str(retry.content)


def test_repr_hides_server_url_and_key() -> None:
    """The URL and key are credentials, so neither the server, the capability, nor its toolset shows them."""
    server = MontyServer[None]('wss://alice:s3cr3t@sandbox.example.com/monty', auth='k-123', headers={'X-Key': 'h-1'})
    capability = CodeMode[None](executor=server)
    toolset = capability.get_wrapper_toolset(FunctionToolset[None]())

    for shown in (repr(server), repr(capability), repr(toolset)):
        assert 'sandbox.example.com' not in shown
        assert 'k-123' not in shown
        assert 'h-1' not in shown


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


async def test_refused_key_is_a_user_error() -> None:
    """The key and headers reach the server, and a 401 is for the developer, not a retry for the model."""
    seen: list[Request] = []
    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'),
            deps_type=_Deps,
            capabilities=[
                CodeMode(executor=MontyServer(url, auth=lambda ctx: ctx.deps.monty_key, headers={'X-Proxy': 'p-1'}))
            ],
        )
        with pytest.raises(UserError, match=r'refused the connection \(401\)') as raised:
            await agent.run('dial with a bad key', deps=_Deps(monty_key='k-run'))

    assert 'k-run' not in str(raised.value)
    assert raised.value.__cause__ is None and raised.value.__suppress_context__
    (request,) = seen
    assert request.headers['Authorization'] == 'Bearer k-run'
    assert request.headers['X-Proxy'] == 'p-1'


async def test_async_auth_resolved_per_run() -> None:
    """An async `auth` function is awaited at the start of each run, so each run sends its own key."""
    seen: list[Request] = []

    async def key_for(ctx: Any) -> str:
        return ctx.deps.monty_key

    async with _refusing_server(seen) as url:
        agent = Agent(
            _snippets_model('1 + 1'), deps_type=_Deps, capabilities=[CodeMode(executor=MontyServer(url, auth=key_for))]
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
            capabilities=[CodeMode(executor=MontyServer(f'ws://127.0.0.1:{port}'))],
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
