"""End-to-end tests for CodeMode's remote Monty transport."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import websockets
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_monty import MountDir
from typing_extensions import Never

from pydantic_ai_harness import CodeMode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _parts(messages: list[ModelMessage], part_type: type[Any]) -> list[Any]:
    return [part for message in messages for part in message.parts if isinstance(part, part_type)]


def _snippets_model(*snippets: str) -> FunctionModel:
    """A model that calls `run_code` with each snippet in turn, then says 'done'."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        done = len(_parts(messages, ToolReturnPart)) + len(_parts(messages, RetryPromptPart))
        if done < len(snippets):
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': snippets[done]})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(model)


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
                monty_sandbox_url=websocket_relay_url,
                mount=MountDir(virtual_path='/workspace', host_path=tmp_path),
            )
        ],
    )

    @agent.tool_plain
    async def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return a + b

    @agent.tool_plain(sequential=True)
    def barrier() -> str:  # pyright: ignore[reportUnusedFunction]
        return 'barrier'

    result = await agent.run('exercise the remote sandbox')

    assert result.output == 'done'
    assert [part.content for part in _parts(result.all_messages(), ToolReturnPart)] == [
        {'output': 'remote tool result 5\n'},
        {'output': 'mounted data\n', 'result': 50},
        [2, 4, 'barrier'],
    ]


@pytest.mark.parametrize('url', ['ws://monty-server:8000', 'wss://sandbox.example.com/monty'])
async def test_websocket_sandbox_url_accepted(url: str) -> None:
    """Any `ws://` or `wss://` URL passes validation; workers dial lazily, not at enter."""
    agent = Agent(_snippets_model(), capabilities=[CodeMode(monty_sandbox_url=url)])
    result = await agent.run('no run_code call, so nothing dials')
    assert result.output == 'done'


async def test_non_websocket_sandbox_url_rejected() -> None:
    """A URL Monty's WebSocket client cannot dial fails when the run starts, not at every `run_code`."""
    agent = Agent(_snippets_model(), capabilities=[CodeMode(monty_sandbox_url='https://sandbox.example.com/monty')])
    with pytest.raises(UserError, match="not scheme 'https'"):
        await agent.run('never dials')


@pytest.mark.parametrize(
    ('resource_limits', 'request_timeout'),
    [(None, 40.0), ({'max_duration_secs': 90}, 100.0), ('unlimited', None)],
)
async def test_remote_turn_deadline_follows_max_duration(
    monkeypatch: pytest.MonkeyPatch, resource_limits: Any, request_timeout: float | None
) -> None:
    """The transport's per-turn deadline leaves room for `max_duration_secs` to fire first."""
    seen: list[float | None] = []

    def recording_pool(url: str, *, request_timeout: float | None) -> Never:
        seen.append(request_timeout)
        raise RuntimeError('not dialing')

    monkeypatch.setattr('pydantic_ai_harness._monty_exec.AsyncMontyWebsocket', recording_pool)
    agent = Agent(
        _snippets_model('1'),
        capabilities=[CodeMode(monty_sandbox_url='wss://sandbox.example.com/monty', resource_limits=resource_limits)],
    )
    await agent.run('record the deadline')
    assert seen == [request_timeout]


async def test_dial_failure_redacts_sandbox_url() -> None:
    """A failed dial's retry message must not leak the URL, which may carry credentials."""
    agent = Agent(
        _snippets_model('1 + 1'), capabilities=[CodeMode(monty_sandbox_url='ws://127.0.0.1:1/?token=hunter2')]
    )

    result = await agent.run('dial a dead worker')

    (retry,) = _parts(result.all_messages(), RetryPromptPart)
    assert 'hunter2' not in str(retry.content)
    assert '<monty_sandbox_url>' in str(retry.content)


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
            capabilities=[CodeMode(monty_sandbox_url=f'ws://127.0.0.1:{port}')],
        )

        @agent.tool_plain
        async def drop_connection() -> None:  # pyright: ignore[reportUnusedFunction]
            # One statement: the harness cancels this while the close is in flight, so a separate
            # wait line would only sometimes run. The wait never ends on its own.
            await asyncio.gather(*(connection.close() for connection in connections), asyncio.Event().wait())

        result = await agent.run('lose the worker mid-snippet')

    (retry,) = _parts(result.all_messages(), RetryPromptPart)
    assert 'MontyDisconnectError' in str(retry.content)
    assert 'drop_connection({}) did not finish' in str(retry.content)
