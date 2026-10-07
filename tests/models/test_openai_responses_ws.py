"""Wire-level session ownership tests using the real SDK against a local scripted peer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, ModelHTTPError, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types import responses
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.exceptions import ConnectionClosed

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import response_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI or websockets not installed')
READINESS_WAIT_TIMEOUT = 10


def text_frames(response_id: str, text: str = 'Hello') -> list[dict[str, Any]]:
    response = response_message(
        [],
        usage=responses.ResponseUsage(
            input_tokens=3,
            output_tokens=2,
            total_tokens=5,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ),
    ).model_copy(update={'id': response_id, 'status': 'completed'})
    return [
        {'type': 'response.created', 'sequence_number': 0, 'response': response.model_dump()},
        {
            'type': 'response.output_text.delta',
            'sequence_number': 1,
            'item_id': f'msg_{response_id}',
            'content_index': 0,
            'output_index': 0,
            'delta': text,
            'logprobs': [],
        },
        {'type': 'response.completed', 'sequence_number': 2, 'response': response.model_dump()},
    ]


@dataclass
class Peer:
    scripts: list[list[dict[str, Any]] | None] = field(default_factory=list[list[dict[str, Any]] | None])
    requests: list[tuple[int, dict[str, Any]]] = field(default_factory=list[tuple[int, dict[str, Any]]])
    connections: list[ServerConnection] = field(default_factory=list[ServerConnection])
    closed: list[asyncio.Event] = field(default_factory=list[asyncio.Event])
    received: asyncio.Queue[None] = field(default_factory=asyncio.Queue[None])
    url: str = ''

    async def handle(self, connection: ServerConnection) -> None:
        connection_id = len(self.connections)
        self.connections.append(connection)
        finished = asyncio.Event()
        self.closed.append(finished)
        try:
            async for raw in connection:
                self.requests.append((connection_id, json.loads(raw)))
                self.received.put_nowait(None)
                script = self.scripts.pop(0) if self.scripts else text_frames(f'resp_{len(self.requests)}')
                if script is None:
                    await connection.close()
                    return
                for frame in script:
                    await connection.send(json.dumps(frame))
        except ConnectionClosed:
            pass
        finally:
            finished.set()

    def model(self) -> OpenAIResponsesModel:
        return OpenAIResponsesModel(
            'gpt-4o', provider=OpenAIProvider(base_url=self.url, api_key='test'), transport='websocket'
        )


@pytest.fixture
async def peer(allow_model_requests: None) -> AsyncIterator[Peer]:
    peer = Peer()
    async with serve(peer.handle, '127.0.0.1', 0) as server:
        peer.url = f'http://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/v1'
        yield peer


async def test_session_reuses_websocket_across_runs(peer: Peer):
    agent = Agent(peer.model())
    async with agent.session() as session:
        first = await session.run('one')
        async with session.run_stream('two') as stream:
            assert await stream.get_output() == 'Hello'
        assert first.output == 'Hello'
        assert len(peer.connections) == 1
        assert not peer.closed[0].is_set()
        assert len(session.conversation.messages) == 4
        assert session.conversation.usage.requests == 2
        assert session.conversation.usage.input_tokens == 6
        assert len(first.all_messages()) == 2
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await peer.closed[0].wait()
    assert all(
        body['type'] == 'response.create' and 'stream' not in body and 'background' not in body
        for _, body in peer.requests
    )


async def test_sessions_isolate_shared_model_and_legacy_runs(peer: Peer):
    agent = Agent(peer.model())
    async with agent.session() as first, agent.session() as second:
        await first.run('first')
        await second.run('second')
        await first.run('third')
        assert [connection_id for connection_id, _ in peer.requests] == [0, 1, 0]
        assert len(second.conversation.messages) == 2
    await agent.run('legacy one')
    await agent.run('legacy two')
    assert [connection_id for connection_id, _ in peer.requests] == [0, 1, 0, 2, 3]


async def test_websocket_tool_roundtrip_uses_same_connection(peer: Peer):
    call = responses.ResponseFunctionToolCall(
        type='function_call', name='lookup', arguments='{}', call_id='call_1', id='fc_1'
    )
    response = response_message([call]).model_copy(update={'id': 'resp_tool', 'status': 'completed'})
    peer.scripts = [
        [
            {'type': 'response.created', 'sequence_number': 0, 'response': response.model_dump()},
            {'type': 'response.output_item.added', 'sequence_number': 1, 'output_index': 0, 'item': call.model_dump()},
            {'type': 'response.output_item.done', 'sequence_number': 2, 'output_index': 0, 'item': call.model_dump()},
            {'type': 'response.completed', 'sequence_number': 3, 'response': response.model_dump()},
        ],
        text_frames('resp_answer'),
    ]
    agent = Agent(peer.model())
    calls = 0

    @agent.tool_plain
    def lookup() -> str:
        nonlocal calls
        calls += 1
        return 'found'

    async with agent.session() as session:
        result = await session.run('look up')
        assert result.output == 'Hello'
        assert calls == 1
        assert [connection_id for connection_id, _ in peer.requests] == [0, 0]
        assert any(
            item.get('type') == 'function_call_output' and item['output'] == 'found'
            for item in peer.requests[1][1]['input']
        )


@pytest.mark.parametrize('cancel', [False, True])
async def test_aborted_request_discards_socket_not_conversation(peer: Peer, cancel: bool):
    peer.scripts = [text_frames('resp_partial')[:2], text_frames('resp_next')]
    agent = Agent(peer.model())
    async with agent.session() as session:
        if cancel:
            async with anyio.create_task_group() as group:
                group.start_soon(session.run, 'one')
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await peer.received.get()
                group.cancel_scope.cancel()
        else:
            async with session.run_stream('one'):
                pass
        assert (await session.run('two')).output == 'Hello'
        assert len(peer.connections) == 2
        assert len(peer.requests) == 2


async def test_websocket_error_no_automatic_replay(peer: Peer):
    peer.scripts = [
        [{'type': 'error', 'status': 429, 'error': {'code': 'rate_limit', 'message': 'slow down'}}],
        text_frames('resp_next'),
    ]
    async with Agent(peer.model()).session() as session:
        with pytest.raises(ModelHTTPError, match='slow down') as exc:
            await session.run('one')
        assert exc.value.status_code == 429
        assert len(peer.requests) == 1
        assert (await session.run('two')).output == 'Hello'
        assert len(peer.connections) == 2


async def test_direct_model_websocket_request_owns_temporary_connection(peer: Peer):
    model = peer.model()
    result = await model.request([ModelRequest(parts=[UserPromptPart('one')])], None, ModelRequestParameters())
    assert isinstance(result, ModelResponse)
    assert result.text == 'Hello'
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await peer.closed[0].wait()
