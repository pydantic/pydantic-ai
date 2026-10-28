"""Temporal lifecycle experiment. Provider I/O is scripted; Temporal is a real local server."""

from __future__ import annotations

import pytest

pytest.importorskip('temporalio', reason='temporal not installed', exc_type=ImportError)
pytest.importorskip('openai')
pytest.importorskip('websockets')
import json
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

import httpx2
from openai import AsyncOpenAI
from temporalio import activity, workflow
from temporalio.client import Client, WorkflowHistory
from temporalio.common import RetryPolicy
from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker
from websockets.exceptions import ConnectionClosedError

from pydantic_ai import Agent
from pydantic_ai.durable_exec._operation_backend import durable_unit_scope
from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from tests.models.test_openai_responses_ws import FRAME_ADAPTER, Frame, ScriptedSocket, completed_response, text_events

_agent: Agent[None, str]


@workflow.defn
class TwoTurns:
    def __init__(self) -> None:
        self.first_done = False
        self.resume = False

    @workflow.run
    async def run(self, paused: bool, streamed: bool) -> str:
        first = await _agent.run('remember the word sapphire')
        self.first_done = True
        if paused:
            await workflow.wait_condition(lambda: self.resume)
        if streamed:
            async with _agent.run_stream('recall the word', message_history=first.all_messages()) as second:
                await second.get_output()
                return second.all_messages_json().decode()
        second = await _agent.run('recall the word', message_history=first.all_messages())
        return second.all_messages_json().decode()

    @workflow.update
    async def wait_first(self) -> None:
        await workflow.wait_condition(lambda: self.first_done)

    @workflow.signal
    async def continue_run(self) -> None:
        self.resume = True


@workflow.defn
class OneTurn:
    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await _agent.run(prompt)).output


@dataclass
class ObservedSocket(ScriptedSocket):
    attempts: list[int] = field(default_factory=list[int])
    disconnect_after_send: int | None = None

    async def send(self, data: str | bytes) -> None:
        self.attempts.append(activity.info().attempt if activity.in_activity() else 0)
        await super().send(data)
        if self.disconnect_after_send == len(self.sent):
            self.push(ConnectionClosedError(None, None))


@dataclass
class Network:
    pending: deque[ObservedSocket] = field(default_factory=deque[ObservedSocket])
    opened: list[ObservedSocket] = field(default_factory=list[ObservedSocket])
    http_frames: list[Frame] = field(default_factory=list[Frame])

    async def connect(self, uri: str, *, additional_headers: Mapping[str, str], **options: object) -> ObservedSocket:
        assert uri == 'wss://api.openai.com/v1/responses'
        assert self.pending, 'unexpected socket creation'
        socket = self.pending.popleft()
        self.opened.append(socket)
        return socket

    def http(self, request: httpx2.Request) -> httpx2.Response:
        frame = FRAME_ADAPTER.validate_json(request.content)
        self.http_frames.append(frame)
        output: list[dict[str, object]] = [
            {
                'type': 'message',
                'id': 'msg_http',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'sapphire', 'annotations': []}],
            }
        ]
        response = completed_response(f'resp_http_{len(self.http_frames)}', store=False)
        response['output'] = FRAME_ADAPTER.validate_python({'output': output})['output']
        return httpx2.Response(200, content=FRAME_ADAPTER.dump_json(response))


def create_model(network: Network) -> tuple[OpenAIResponsesModel, httpx2.AsyncClient]:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(network.http))
    client = AsyncOpenAI(api_key='prototype-only', http_client=http, max_retries=0)
    settings: OpenAIResponsesModelSettings = {'openai_transport': 'websocket', 'openai_store': False}
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=client), settings=settings)
    return (model, http)


def configure_agent(model: OpenAIResponsesModel, *, retry: bool = False) -> Agent[None, str]:
    global _agent
    durability = TemporalDurability(
        activity_config={
            'start_to_close_timeout': timedelta(seconds=15),
            'retry_policy': RetryPolicy(
                maximum_attempts=2 if retry else 1,
                initial_interval=timedelta(milliseconds=1),
                maximum_interval=timedelta(milliseconds=1),
            ),
        }
    )
    _agent = Agent(model, name='websocket_lifetime_probe', capabilities=[durability])
    return _agent


def worker(client: Client, queue: str, agent: Agent[None, str]) -> Worker:
    return Worker(
        client,
        task_queue=queue,
        workflows=[TwoTurns, OneTurn],
        plugins=[AgentPlugin(agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    )


def frame_summary(socket: ObservedSocket) -> list[dict[str, object]]:
    return [
        {
            'previous_response_id': frame.get('previous_response_id'),
            'input': frame.get('input'),
            'stream_id': frame.get('stream_id'),
            'activity_attempt': attempt,
        }
        for frame, attempt in zip(socket.sent, socket.attempts, strict=True)
    ]


async def replay(client: Client, agent: Agent[None, str], history: WorkflowHistory, network: Network) -> None:
    before = (len(network.opened), sum(len(s.sent) for s in network.opened), len(network.http_frames))
    with ThreadPoolExecutor() as executor:
        result = await Replayer(
            workflows=[TwoTurns, OneTurn],
            plugins=[AgentPlugin(agent)],
            workflow_runner=UnsandboxedWorkflowRunner(),
            workflow_task_executor=executor,
            data_converter=client.data_converter,
        ).replay_workflow(history)
    assert result.replay_failure is None
    after = (len(network.opened), sum(len(s.sent) for s in network.opened), len(network.http_frames))
    assert after == before, (before, after)


def assert_closed(model: OpenAIResponsesModel, sockets: list[ObservedSocket]) -> None:
    assert model._websocket.states == {}  # pyright: ignore[reportPrivateUsage]
    assert all(socket.close_count == 1 for socket in sockets)


async def warm_reuse(client: Client, streamed: bool) -> None:
    socket = ObservedSocket(
        responses=deque(
            [text_events('sapphire', 'resp_warm_1', store=False), text_events('sapphire', 'resp_warm_2', store=False)]
        )
    )
    network = Network(pending=deque([socket]))
    model, http = create_model(network)
    agent = configure_agent(model)
    queue = f'warm-{uuid4()}'
    with patch('openai.lib._websocket._WebSocketConnect', network.connect):
        async with http, model, worker(client, queue, agent):
            handle = await client.start_workflow(
                TwoTurns.run, args=[False, streamed], id=f'warm-{uuid4()}', task_queue=queue
            )
            messages = ModelMessagesTypeAdapter.validate_json(await handle.result())
            history = await handle.fetch_history()
            assert socket.close_count == 0
        assert_closed(model, [socket])
        assert len(socket.sent) == 2 and (not network.http_frames)
        assert socket.sent[1]['previous_response_id'] == 'resp_warm_1'
        assert socket.sent[1]['input'] == [{'role': 'user', 'content': 'recall the word'}]
        assert isinstance(messages[-1], ModelResponse)
        await replay(client, agent, history, network)


async def cold_worker(client: Client) -> None:
    first_socket = ObservedSocket(responses=deque([text_events('sapphire', 'resp_cold_1', store=False)]))
    second_socket = ObservedSocket(responses=deque([text_events('sapphire', 'resp_cold_2', store=False)]))
    network = Network(pending=deque([first_socket, second_socket]))
    first_model, first_http = create_model(network)
    first_agent = configure_agent(first_model)
    queue = f'cold-{uuid4()}'
    with patch('openai.lib._websocket._WebSocketConnect', network.connect):
        async with first_http, first_model, worker(client, queue, first_agent):
            handle = await client.start_workflow(
                TwoTurns.run, args=[True, False], id=f'cold-{uuid4()}', task_queue=queue
            )
            await handle.execute_update(TwoTurns.wait_first)  # pyright: ignore[reportUnknownMemberType]
        assert_closed(first_model, [first_socket])
        second_model, second_http = create_model(network)
        second_agent = configure_agent(second_model)
        assert second_model is not first_model
        async with second_http, second_model, worker(client, queue, second_agent):
            await handle.signal(TwoTurns.continue_run)
            messages = ModelMessagesTypeAdapter.validate_json(await handle.result())
            history = await handle.fetch_history()
        assert_closed(second_model, [second_socket])
        assert len(first_socket.sent) == len(second_socket.sent) == 1
        frame = second_socket.sent[0]
        assert frame.get('previous_response_id') is None
        assert isinstance(frame['input'], list) and len(frame['input']) == 3
        assert 'remember the word sapphire' in json.dumps(frame['input'])
        assert 'recall the word' in json.dumps(frame['input'])
        assert isinstance(messages[-1], ModelResponse)
        await replay(client, second_agent, history, network)


async def without_external_owner(client: Client) -> None:
    network = Network()
    model, http = create_model(network)
    agent = configure_agent(model)
    queue = f'unowned-{uuid4()}'
    with patch('openai.lib._websocket._WebSocketConnect', network.connect):
        async with http, worker(client, queue, agent):
            await client.execute_workflow(OneTurn.run, 'hello', id=f'unowned-{uuid4()}', task_queue=queue)
        assert len(network.http_frames) == 1 and (not network.opened)
    network2 = Network()
    model2, http2 = create_model(network2)
    with patch('openai.lib._websocket._WebSocketConnect', network2.connect), durable_unit_scope():
        async with http2, model2:
            await model2.request([ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters())
        assert len(network2.http_frames) == 1 and (not network2.opened)
    assert_closed(model, [])
    assert_closed(model2, [])


async def overflow(client: Client) -> None:
    socket = ObservedSocket(responses=deque([]))
    network = Network(pending=deque([socket]))
    model, http = create_model(network)
    agent = configure_agent(model)
    queue = f'overflow-{uuid4()}'
    with (
        patch('openai.lib._websocket._WebSocketConnect', network.connect),
        patch('pydantic_ai.models._openai_responses_websocket.ACTIVE_RESPONSE_LIMIT', 1),
    ):
        async with http, model, worker(client, queue, agent):
            first = await client.start_workflow(OneTurn.run, 'first', id=f'overflow-a-{uuid4()}', task_queue=queue)
            await socket.waiting_for_input.wait()
            second = await client.start_workflow(OneTurn.run, 'second', id=f'overflow-b-{uuid4()}', task_queue=queue)
            assert await second.result() == 'sapphire'
            assert len(network.http_frames) == 1 and len(socket.sent) == 1
            socket.push(*text_events('sapphire', 'resp_overflow', store=False))
            assert await first.result() == 'sapphire'
        assert_closed(model, [socket])


async def rejection_retry(client: Client) -> None:
    rejection: Frame = {
        'type': 'error',
        'status': 429,
        'error': {'type': 'rate_limit_error', 'code': 'rate_limit_exceeded', 'message': 'scripted rejection'},
    }
    socket = ObservedSocket(
        responses=deque(
            [
                text_events('sapphire', 'resp_retry_1', store=False),
                [rejection],
                text_events('sapphire', 'resp_retry_2', store=False),
            ]
        )
    )
    network = Network(pending=deque([socket]))
    model, http = create_model(network)
    agent = configure_agent(model, retry=True)
    queue = f'retry-{uuid4()}'
    with patch('openai.lib._websocket._WebSocketConnect', network.connect):
        async with http, model, worker(client, queue, agent):
            handle = await client.start_workflow(
                TwoTurns.run, args=[False, False], id=f'retry-{uuid4()}', task_queue=queue
            )
            await handle.result()
            history = await handle.fetch_history()
        assert_closed(model, [socket])
        assert socket.attempts == [1, 1, 2]
        assert socket.sent[1]['previous_response_id'] == 'resp_retry_1'
        assert socket.sent[2].get('previous_response_id') is None
        assert isinstance(socket.sent[2]['input'], list) and len(socket.sent[2]['input']) == 3
        await replay(client, agent, history, network)


async def interrupted_response(client: Client) -> None:
    first_socket = ObservedSocket(
        responses=deque(
            [
                text_events('sapphire', 'resp_disconnect_1', store=False),
                text_events('unfinished', 'resp_disconnect_partial', store=False)[:-1],
            ]
        ),
        disconnect_after_send=2,
    )
    second_socket = ObservedSocket(responses=deque([text_events('sapphire', 'resp_disconnect_2', store=False)]))
    network = Network(pending=deque([first_socket, second_socket]))
    model, http = create_model(network)
    agent = configure_agent(model, retry=True)
    queue = f'disconnect-{uuid4()}'
    with patch('openai.lib._websocket._WebSocketConnect', network.connect):
        async with http, model, worker(client, queue, agent):
            handle = await client.start_workflow(
                TwoTurns.run, args=[False, True], id=f'disconnect-{uuid4()}', task_queue=queue
            )
            messages = ModelMessagesTypeAdapter.validate_json(await handle.result())
            history = await handle.fetch_history()
        assert_closed(model, [first_socket, second_socket])
        assert first_socket.attempts == [1, 1]
        assert second_socket.attempts == [2]
        assert len(first_socket.sent) == 2 and len(second_socket.sent) == 1
        assert second_socket.sent[0].get('previous_response_id') is None
        assert isinstance(second_socket.sent[0]['input'], list) and len(second_socket.sent[0]['input']) == 3
        assert 'unfinished' not in ModelMessagesTypeAdapter.dump_json(messages).decode()
        await replay(client, agent, history, network)


async def test_temporal_worker_owns_responses_websocket(client: Client, allow_model_requests: None):
    """Activities borrow a worker-owned socket, and replay/cold recovery require no live socket state."""
    await warm_reuse(client, False)
    await warm_reuse(client, True)
    await cold_worker(client)
    await without_external_owner(client)
    await overflow(client)
    await rejection_retry(client)
    await interrupted_response(client)
