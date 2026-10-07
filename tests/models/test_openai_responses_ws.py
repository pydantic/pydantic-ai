"""Wire-level session ownership tests using the real SDK against a local scripted peer."""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import Mock

import anyio
import pytest

from pydantic_ai import Agent, ModelAPIError, ModelHTTPError, ModelRequest, ModelResponse, UserError, UserPromptPart
from pydantic_ai._steering import SteeringController
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from httpx import Timeout
    from openai.resources.responses.responses import AsyncResponsesConnection
    from openai.types import responses
    from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.exceptions import ConnectionClosed
    from websockets.http11 import Request, Response

    from pydantic_ai.models._openai_responses_ws import ResponsesWebSocket
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
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
    scripts: list[Sequence[dict[str, Any] | None] | None] = field(
        default_factory=list[Sequence[dict[str, Any] | None] | None]
    )
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
                    if frame is None:
                        await connection.close()
                        return
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


@pytest.mark.parametrize('wrap_in_hook', [False, True])
async def test_session_reuses_websocket_across_runs(peer: Peer, wrap_in_hook: bool):
    class WrapModel(AbstractCapability[None]):
        async def before_model_request(self, ctx: RunContext[None], request_context: ModelRequestContext):
            request_context.model = WrapperModel(request_context.model)
            return request_context

    agent = Agent(peer.model(), deps_type=type(None), capabilities=[WrapModel()] if wrap_in_hook else [])
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
    agent = Agent(peer.model(), model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'))
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
        assert 'previous_response_id' not in peer.requests[1][1]


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


async def test_incremental_history_resets_after_header_change_and_restore(peer: Peer):
    agent = Agent(peer.model(), model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'))
    async with agent.session() as session:
        await session.run('one')
        await session.run('two')
        await session.run('three', model_settings={'extra_headers': {'x-session-test': 'changed'}})
        await session.run('four')
        conversation = session.conversation
    async with agent.session(conversation=conversation) as restored:
        await restored.run('five')
    assert [connection_id for connection_id, _ in peer.requests] == [0, 0, 1, 2, 3]
    bodies = [body for _, body in peer.requests]
    assert [body.get('previous_response_id') for body in bodies] == [None, 'resp_1', None, None, None]
    assert bodies[1]['input'] == [{'role': 'user', 'content': 'two'}]
    assert [item['content'] for item in bodies[-1]['input'] if item.get('role') == 'user'] == [
        'one',
        'two',
        'three',
        'four',
        'five',
    ]
    request = peer.connections[1].request
    assert request is not None
    assert request.headers['x-session-test'] == 'changed'


@pytest.mark.parametrize('after_created', [False, True])
async def test_disconnect_never_replays_uncertain_request(peer: Peer, after_created: bool):
    peer.scripts = [[text_frames('resp_lost')[0], None] if after_created else None, text_frames('resp_next')]
    async with Agent(peer.model()).session() as session:
        with pytest.raises(ModelAPIError, match='WebSocket connection failed') as exc:
            await session.run('one')
        assert isinstance(exc.value.__cause__, ConnectionClosed)
        assert len(peer.requests) == 1
        assert (await session.run('two')).output == 'Hello'
        assert [connection_id for connection_id, _ in peer.requests] == [0, 1]


async def test_handshake_error_preserves_status_body_and_headers(allow_model_requests: None):
    peer = Peer()

    def reject(connection: ServerConnection, request: Request) -> Response:
        response = connection.respond(429, 'slow down')
        response.headers['Retry-After'] = '7'
        return response

    async with serve(peer.handle, '127.0.0.1', 0, process_request=reject) as server:
        peer.url = f'http://127.0.0.1:{next(iter(server.sockets)).getsockname()[1]}/v1'
        with pytest.raises(ModelHTTPError) as exc:
            await Agent(peer.model()).run('one')
        assert exc.value.status_code == 429
        assert exc.value.body == 'slow down'
        assert exc.value.headers is not None
        assert exc.value.headers['retry-after'] == '7'
        assert not peer.requests


async def test_read_timeout_releases_connection_for_next_run(peer: Peer):
    peer.scripts = [[], text_frames('resp_next')]
    async with Agent(peer.model()).session() as session:
        with pytest.raises(ModelAPIError, match='timed out'):
            # Only the deliberately silent peer is subject to the short read deadline.
            await session.run('one', model_settings={'timeout': Timeout(10, read=0.05)})
        assert (await session.run('two')).output == 'Hello'
        assert [connection_id for connection_id, _ in peer.requests] == [0, 1]


async def test_write_timeout_discards_connection(peer: Peer, monkeypatch: pytest.MonkeyPatch):
    # A loopback socket cannot reliably fill its write buffer. Block the SDK send boundary
    # while retaining its real handshake and close behavior.
    async def blocked_send(*args: object, **kwargs: object) -> None:
        await anyio.sleep_forever()

    async with Agent(peer.model()).session() as session:
        with monkeypatch.context() as patch:
            patch.setattr(AsyncResponsesConnection, 'send', blocked_send)
            with pytest.raises(ModelAPIError, match='timed out'):
                await session.run('one', model_settings={'timeout': Timeout(10, write=0)})
        assert not peer.requests
        assert (await session.run('two')).output == 'Hello'
        assert len(peer.connections) == 2


@pytest.mark.parametrize('extra_body', [['invalid'], {'stream': True}, {'background': False}])
async def test_invalid_websocket_options_do_not_open_connection(peer: Peer, extra_body: Any):
    async with Agent(peer.model()).session() as session:
        with pytest.raises(UserError, match='Responses WebSocket'):
            await session.run('one', model_settings={'extra_body': extra_body})
        assert not peer.connections
        assert (await session.run('two')).output == 'Hello'


async def test_bound_model_rejects_overlapping_and_closed_requests(peer: Peer):
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('one')])]
    parameters = ModelRequestParameters()
    async with peer.model().open_session() as bound:
        async with bound.request_stream(messages, None, parameters) as stream:
            with pytest.raises(UserError, match='one request at a time'):
                await bound.request(messages, None, parameters)
            async for _ in stream:
                pass
        assert (await bound.request(messages, None, parameters)).text == 'Hello'
    with pytest.raises(UserError, match='session has closed'):
        await bound.request(messages, None, parameters)
    assert len(peer.requests) == 2


@pytest.mark.parametrize('stream', [False, True])
async def test_direct_request_prepares_once(peer: Peer, monkeypatch: pytest.MonkeyPatch, stream: bool):
    model = peer.model()
    prepare = Mock(wraps=model.prepare_request)
    monkeypatch.setattr(model, 'prepare_request', prepare)
    messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('one')])]
    if stream:
        async with model.request_stream(messages, None, ModelRequestParameters()) as result:
            async for _ in result:
                pass
    else:
        await model.request(messages, None, ModelRequestParameters())
    assert prepare.call_count == 1


async def test_missing_previous_response_is_not_replayed(peer: Peer):
    peer.scripts = [
        text_frames('resp_1'),
        [
            {
                'type': 'error',
                'status': 400,
                'error': {'code': 'previous_response_not_found', 'message': 'cache expired'},
            }
        ],
        text_frames('resp_3'),
    ]
    agent = Agent(peer.model(), model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'))
    async with agent.session() as session:
        await session.run('one')
        with pytest.raises(ModelHTTPError, match='previous_response_not_found'):
            await session.run('two')
        assert len(peer.requests) == 2
        await session.run('three')
    assert [connection_id for connection_id, _ in peer.requests] == [0, 0, 1]
    assert peer.requests[1][1]['previous_response_id'] == 'resp_1'
    assert 'previous_response_id' not in peer.requests[2][1]


async def test_steering_enabled_request_without_event_handler(peer: Peer):
    result = await Agent(peer.model(), model_settings=OpenAIResponsesModelSettings(openai_steering=True)).run('one')
    assert result.output == 'Hello'
    assert len(peer.requests) == 1


async def test_direct_steering_requires_agent_context(peer: Peer):
    with pytest.raises(UserError, match='inside an agent run'):
        await peer.model().request(
            [ModelRequest([UserPromptPart('one')])],
            OpenAIResponsesModelSettings(openai_steering=True),
            ModelRequestParameters(),
        )
    assert peer.connections == []


async def test_durable_context_cannot_enable_native_transport(peer: Peer):
    # Model adapters are also called directly by durable activities, outside Agent's entry guard.
    async with peer.model().open_session() as bound:
        ctx = RunContext(deps=None, model=bound, usage=RunUsage())
        controller = SteeringController({}, 'run', list)
        controller.blocked = True
        ctx._steering = controller  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(UserError, match='durable execution units'):
            async with bound.request_stream(
                [ModelRequest([UserPromptPart('one')])],
                OpenAIResponsesModelSettings(openai_steering=True),
                ModelRequestParameters(),
                ctx,
            ):
                assert False, 'durable native transport was accepted'
    assert peer.connections == []


async def test_background_mode_is_rejected_before_websocket_connect(peer: Peer):
    with pytest.raises(UserError, match='openai_background=True'):
        await Agent(peer.model(), model_settings=OpenAIResponsesModelSettings(openai_background=True)).run('one')
    assert peer.connections == []


async def test_missing_websocket_dependency_has_install_guidance(peer: Peer, monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, 'websockets.exceptions', None)
        with pytest.raises(ImportError, match=r'Install `openai\[realtime\]`'):
            await Agent(peer.model()).run('one')
    assert peer.connections == []


async def test_native_transport_rejects_unbound_handles(peer: Peer):
    # Transport admission is independent of the controller guard; no provider exchange is needed.
    model = peer.model()
    transport = ResponsesWebSocket(model.client, model.model_name)
    with pytest.raises(UserError, match='original idle WebSocket'):
        transport.receive(None)
    with pytest.raises(UserError, match='active WebSocket response'):
        await transport.steer('input', 'parent')
    await transport.close()
    assert peer.connections == []
