"""Recorded Responses WebSocket integration through the public session API."""

from __future__ import annotations

import os
from collections.abc import AsyncIterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, ModelAPIError, RunContext, UserError
from pydantic_ai.messages import AgentStreamEvent, PartStartEvent, TextPart, UserPromptPart

from ..conftest import try_import
from ..realtime.ws_cassettes import (
    CassetteMessage,
    RealtimeCassette,
    RecordingWebSocket,
    ReplayWebSocket,
    realtime_cassette_plan,
)

with try_import() as imports_successful:
    from openai.lib import _websocket

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI or websockets not installed')


@dataclass
class RecordedConnection:
    cassette: RealtimeCassette
    sockets: list[RecordingWebSocket | ReplayWebSocket] = field(
        default_factory=list[RecordingWebSocket | ReplayWebSocket]
    )
    closed: int = 0


@pytest.fixture
def responses_ws_recording(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[RecordedConnection]:
    # Pytest does not type FixtureRequest.node; this is a function-scoped fixture.
    name = request.node.name  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    path = Path(__file__).parent / 'cassettes' / 'test_openai_responses_live_ws' / f'{name}.yaml'
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=request.config.getoption('record_mode'))
    if plan == 'error_missing':
        raise RuntimeError(f'Missing Responses WebSocket cassette: {path}')
    recording = RecordedConnection(RealtimeCassette.load(path) if plan == 'replay' else RealtimeCassette())
    real_connect = _websocket._WebSocketConnect

    async def connect(*args: Any, **kwargs: Any) -> RecordingWebSocket | ReplayWebSocket:
        if plan == 'replay':
            socket = ReplayWebSocket(recording.cassette)
        else:
            base_url = os.getenv('RESPONSES_WS_BASE_URL', 'https://api.openai.com/v1')
            assert args[0] == base_url.replace('https:', 'wss:').replace('http:', 'ws:').rstrip('/') + '/responses'
            socket = RecordingWebSocket(await real_connect(*args, **kwargs), recording.cassette)
        real_close = socket.close

        async def close(*args: Any, **kwargs: Any) -> None:
            await real_close(*args, **kwargs)
            recording.closed += 1

        monkeypatch.setattr(socket, 'close', close)
        recording.sockets.append(socket)
        return socket

    monkeypatch.setattr(_websocket, '_WebSocketConnect', connect)
    try:
        yield recording
    finally:
        if plan == 'record' and recording.cassette.interactions:
            _scrub_account_metadata(recording.cassette)
            recording.cassette.dump(path)


def _scrub_account_metadata(cassette: RealtimeCassette) -> None:
    # Compatible gateways may include account quotas and opaque turn tokens as extra events.
    # Keep their unknown event tags (the adapter must ignore them), not their private payloads.
    for event in cassette.interactions:
        if isinstance(event, CassetteMessage) and event.direction == 'received':
            if event.data.get('type') in ('codex.rate_limits', 'codex.response.metadata'):
                event.data = {'type': event.data['type']}
            if isinstance(response := event.data.get('response'), dict):
                for key in ('safety_identifier', 'prompt_cache_key'):
                    if key in response:
                        response[key] = '<scrubbed>'


def test_recording_scrubs_private_gateway_metadata():
    cassette = RealtimeCassette(
        interactions=[
            CassetteMessage(direction='received', data={'type': 'codex.rate_limits', 'credits': {'balance': '123'}}),
            CassetteMessage(
                direction='received', data={'type': 'codex.response.metadata', 'headers': {'token': 'private'}}
            ),
            CassetteMessage(
                direction='received',
                data={
                    'type': 'response.completed',
                    'response': {'id': 'resp_1', 'safety_identifier': 'user-private', 'prompt_cache_key': 'private'},
                },
            ),
            CassetteMessage(direction='sent', data={'type': 'response.create', 'previous_response_id': 'resp_1'}),
        ]
    )
    _scrub_account_metadata(cassette)
    assert cassette.interactions == [
        CassetteMessage(direction='received', data={'type': 'codex.rate_limits'}),
        CassetteMessage(direction='received', data={'type': 'codex.response.metadata'}),
        CassetteMessage(
            direction='received',
            data={
                'type': 'response.completed',
                'response': {'id': 'resp_1', 'safety_identifier': '<scrubbed>', 'prompt_cache_key': '<scrubbed>'},
            },
        ),
        CassetteMessage(direction='sent', data={'type': 'response.create', 'previous_response_id': 'resp_1'}),
    ]


async def test_responses_ws_session_live(
    allow_model_requests: None, openai_api_key: str, responses_ws_recording: RecordedConnection
):
    """Two runs retain context on one socket with store=False and incremental input.

    Record with OPENAI_API_KEY and optionally RESPONSES_WS_BASE_URL for a compatible gateway.
    The shared test configuration intentionally clears OPENAI_BASE_URL.
    The cassette records frames only, never handshake credentials or the gateway address.
    """
    model = OpenAIResponsesModel(
        'gpt-6-astra',
        provider=OpenAIProvider(api_key=openai_api_key, base_url=os.getenv('RESPONSES_WS_BASE_URL')),
        transport='websocket',
    )
    agent = Agent(
        model,
        model_settings=OpenAIResponsesModelSettings(
            openai_store=False, openai_previous_response_id='auto', thinking='low', max_tokens=256, timeout=30
        ),
    )
    with anyio.fail_after(90):
        async with agent.session() as session:
            async with session.run_stream('Remember the word persimmon for my next message. Reply only OK.') as first:
                assert (await first.get_output()).strip() == 'OK'
            second = await session.run('What word did I ask you to remember? Reply with only that word.')
            assert second.output.strip().lower() == 'persimmon'
            assert len(session.conversation.messages) == 4
            assert session.conversation.usage.requests == 2
            assert session.conversation.usage.input_tokens > 0
            assert session.state.active_run_id is None
        assert len(responses_ws_recording.sockets) == 1
        assert responses_ws_recording.closed == 1
    requests = [
        event.data
        for event in responses_ws_recording.cassette.interactions
        if isinstance(event, CassetteMessage) and event.direction == 'sent'
    ]
    assert len(requests) == 2
    assert all(request['type'] == 'response.create' and request['store'] is False for request in requests)
    assert requests[1]['previous_response_id'] == first.response.provider_response_id
    assert len(requests[1]['input']) == 1


async def test_responses_ws_gateway_rejects_steering_live(
    allow_model_requests: None, openai_api_key: str, responses_ws_recording: RecordedConnection
):
    """The configured gateway rejects steering; preserve uncertainty rather than inventing a commit.

    Recorded against a compatible gateway, not the official OpenAI API. This is evidence of
    safe failure handling, NOT a successful live native-steering verification.
    """
    model = OpenAIResponsesModel(
        'gpt-6-astra',
        provider=OpenAIProvider(api_key=openai_api_key, base_url=os.getenv('RESPONSES_WS_BASE_URL')),
        transport='websocket',
    )
    agent = Agent(
        model,
        deps_type=type(None),
        model_settings=OpenAIResponsesModelSettings(
            openai_store=False, openai_steering=True, thinking='low', max_tokens=512, timeout=30
        ),
    )
    deliveries: list[str] = []
    correction = 'Stop the list. Instead reply with only the word persimmon.'

    async def handle(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart) and not deliveries:
                deliveries.append(await ctx.steer(correction))

    with anyio.fail_after(90):
        async with agent.session() as session:
            with pytest.raises(ModelAPIError, match='concurrent_turn_not_supported'):
                async with session.run_stream(
                    'List the integers from 1 to 200, spelling each out in English on its own line.',
                    event_stream_handler=handle,
                ) as result:
                    await result.get_output()
            state = session.state
            assert len(deliveries) == 1
            delivery = state.steering[0]
            assert delivery.delivery_id == deliveries[0]
            assert delivery.status == 'uncertain'
            part = delivery.messages[0].parts[0]
            assert isinstance(part, UserPromptPart) and part.content == [correction]
            assert delivery.successor_response_id is None
            assert state.active_run_id is None
            with pytest.raises(UserError, match=r'Steering delivery .* is unresolved'):
                await session.run('Do not silently replay uncommitted input')
        assert len(responses_ws_recording.sockets) == 1
        assert responses_ws_recording.closed == 1
    requests = [
        event.data
        for event in responses_ws_recording.cassette.interactions
        if isinstance(event, CassetteMessage) and event.direction == 'sent'
    ]
    assert [request['type'] for request in requests] == ['response.create', 'response.steer']
    assert requests[1]['previous_response_id'] == delivery.parent_response_id
    assert requests[1]['input'] == [
        {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': correction}]}
    ]
