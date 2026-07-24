"""Direct unit tests for the realtime WebSocket cassette engine."""

from __future__ import annotations as _annotations

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from ..conftest import try_import
from .ws_cassettes import (
    CassetteClose,
    CassetteMessage,
    CassettePlan,
    RealtimeCassette,
    RecordingWebSocket,
    ReplayWebSocket,
    realtime_cassette_plan,
    ws_cassettes_available,
)

with try_import() as imports_successful:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    from websockets.frames import Close

pytestmark = pytest.mark.skipif(
    not imports_successful() or not ws_cassettes_available(), reason='PyYAML / websockets not installed'
)


class _FakeWebSocket:
    marker = 'wrapped'  # only reachable through `RecordingWebSocket.__getattr__`

    def __init__(self, received: Iterable[str | bytes | BaseException] | None = None) -> None:
        self.received = list(received or ())
        self.sent: list[str | bytes] = []
        self.closed_with: tuple[tuple[object, ...], dict[str, object]] | None = None

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def recv(self, **kwargs: object) -> str | bytes:
        del kwargs
        item = self.received.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    async def close(self, *args: object, **kwargs: object) -> None:
        self.closed_with = (args, kwargs)


@pytest.mark.anyio
async def test_recording_scrubs_secrets_and_internal_config(tmp_path: Path) -> None:
    """Unit test: recording safety must be pinned without putting real credentials on the wire."""
    sent_frame = {
        'type': 'client.config',
        'headers': {
            'Authorization': 'Bearer fake-bearer-token',
            'api-key': 'sk-fake_api_key_123456',
            'google-api-key': 'AIzaFakeApiKey123456',
        },
    }
    received_frame = {
        'type': 'session.updated',
        'session': {
            'xvad_settings': {'threshold': 0.5},
            'asr_classifier': ['internal-asr'],
            'response_patient_starter_config': {'enabled': True},
            'model_address': 'internal.service:443',
            'xvad_model_name': 'internal-model',
        },
    }
    fake_ws = _FakeWebSocket([json.dumps(received_frame)])
    cassette = RealtimeCassette()
    recording = RecordingWebSocket(fake_ws, cassette)

    await recording.send(json.dumps(sent_frame))
    await recording.recv()
    path = tmp_path / 'cassette.yaml'
    cassette.dump(path)

    persisted = path.read_text(encoding='utf-8')
    for secret in (
        'fake-bearer-token',
        'sk-fake_api_key_123456',
        'AIzaFakeApiKey123456',
        'internal.service:443',
        'internal-model',
    ):
        assert secret not in persisted
    assert RealtimeCassette.load(path).interactions == [
        CassetteMessage(
            direction='sent',
            data={
                'type': 'client.config',
                'headers': {
                    'Authorization': '<scrubbed>',
                    'api-key': '<scrubbed>',
                    'google-api-key': '<scrubbed>',
                },
            },
        ),
        CassetteMessage(
            direction='received',
            data={
                'type': 'session.updated',
                'session': {
                    'xvad_settings': '<scrubbed>',
                    'asr_classifier': '<scrubbed>',
                    'response_patient_starter_config': '<scrubbed>',
                    'model_address': '<scrubbed>',
                    'xvad_model_name': '<scrubbed>',
                },
            },
        ),
    ]


@pytest.mark.anyio
async def test_recording_normalizes_client_ids() -> None:
    """Unit test: generated outbound IDs need deterministic matching without a provider session."""
    first_id = '0123456789abcdef01234567'
    second_id = '89abcdef0123456789abcdef'
    frame = {
        'id': first_id,
        'items': [{'item_id': first_id}, {'previous_item_id': second_id}],
        'response_id': second_id,
        'metadata': {'id': 'ABCDEF0123456789ABCDEF01'},
    }
    fake_ws = _FakeWebSocket()
    cassette = RealtimeCassette()
    recording = RecordingWebSocket(fake_ws, cassette)

    await recording.send(json.dumps(frame))
    await recording.send(json.dumps({'item_id': second_id, 'previous_item_id': first_id}))

    assert cassette.interactions == [
        CassetteMessage(
            direction='sent',
            data={
                'id': '<client-id-1>',
                'items': [{'item_id': '<client-id-1>'}, {'previous_item_id': '<client-id-2>'}],
                'response_id': second_id,
                'metadata': {'id': 'ABCDEF0123456789ABCDEF01'},
            },
        ),
        CassetteMessage(direction='sent', data={'item_id': '<client-id-2>', 'previous_item_id': '<client-id-1>'}),
    ]


@pytest.mark.parametrize(
    ('record_mode', 'missing_plan', 'existing_plan'),
    [
        (None, 'error_missing', 'replay'),
        ('none', 'error_missing', 'replay'),
        ('once', 'record', 'replay'),
        ('rewrite', 'record', 'record'),
        ('all', 'record', 'record'),
    ],
)
def test_realtime_cassette_plan(
    record_mode: str | None, missing_plan: CassettePlan, existing_plan: CassettePlan
) -> None:
    """Unit test: local record/replay selection is deterministic and does not need provider traffic."""
    assert realtime_cassette_plan(cassette_exists=False, record_mode=record_mode) == missing_plan
    assert realtime_cassette_plan(cassette_exists=True, record_mode=record_mode) == existing_plan


@pytest.mark.anyio
async def test_record_dump_load_replays_frames_byte_identically(tmp_path: Path) -> None:
    """Unit test: the raw-frame persistence round-trip can be verified without a live WebSocket."""
    sent_frames = [json.dumps({'type': 'client.one'}), json.dumps({'type': 'client.two', 'value': 'café'})]
    received_frames = [
        json.dumps({'type': 'server.one', 'value': [1, 2]}),
        json.dumps({'type': 'server.two', 'done': True}),
    ]
    fake_ws = _FakeWebSocket(received_frames.copy())
    cassette = RealtimeCassette()
    recording = RecordingWebSocket(fake_ws, cassette)

    for sent, received in zip(sent_frames, received_frames):
        await recording.send(sent)
        assert await recording.recv() == received

    path = tmp_path / 'nested' / 'cassette.yaml'
    cassette.dump(path)
    replay = ReplayWebSocket(RealtimeCassette.load(path))
    for sent, received in zip(sent_frames, received_frames):
        await replay.send(sent)
        assert await replay.recv(decode=False) == received.encode()


@pytest.mark.anyio
async def test_replay_waits_for_send_and_replays_close() -> None:
    """Unit test: full-duplex ordering and close handling require controlled task scheduling."""
    cassette = RealtimeCassette(
        interactions=[
            CassetteMessage(direction='sent', data={'id': '<client-id-1>', 'type': 'client.event'}),
            CassetteMessage(direction='received', data={'type': 'server.event'}),
            CassetteClose(code=1011, reason='provider failure', ok=False),
        ]
    )
    replay = ReplayWebSocket(cassette)

    receive_task = asyncio.create_task(replay.recv())
    await asyncio.sleep(0)
    assert not receive_task.done()
    await replay.send(json.dumps({'id': '0123456789abcdef01234567', 'type': 'client.event'}))
    assert await receive_task == json.dumps({'type': 'server.event'})

    with pytest.raises(ConnectionClosedError) as exc_info:
        await replay.recv()
    assert exc_info.value.rcvd is not None
    assert exc_info.value.rcvd.code == 1011
    assert exc_info.value.rcvd.reason == 'provider failure'


@pytest.mark.anyio
async def test_empty_replay_closes_cleanly_and_disconnect_requires_binding() -> None:
    cassette = RealtimeCassette()
    with pytest.raises(RuntimeError, match='no active WebSocket'):
        await cassette.disconnect()

    replay = ReplayWebSocket(cassette)
    assert [message async for message in replay] == []
    assert replay._peek() is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
async def test_recording_truncates_inbound_audio() -> None:
    """Unit test: inbound audio is truncated so cassettes stay small — both provider shapes."""
    long_audio = 'A' * 400  # far longer than the retained byte budget
    openai_frame = {'type': 'response.output_audio.delta', 'delta': long_audio}
    gemini_frame = {'serverContent': {'modelTurn': {'parts': [{'inlineData': {'data': long_audio}}]}}}
    # `inlineData` present but without string `data` (e.g. metadata-only) is walked through untouched.
    gemini_no_data = {'serverContent': {'modelTurn': {'parts': [{'inlineData': {'mimeType': 'audio/pcm'}}]}}}
    fake_ws = _FakeWebSocket([json.dumps(openai_frame), json.dumps(gemini_frame), json.dumps(gemini_no_data)])
    cassette = RealtimeCassette()
    recording = RecordingWebSocket(fake_ws, cassette)

    await recording.recv()
    await recording.recv()
    await recording.recv()

    openai_stored, gemini_stored, no_data_stored = cassette.interactions
    assert isinstance(openai_stored, CassetteMessage) and isinstance(gemini_stored, CassetteMessage)
    assert 0 < len(openai_stored.data['delta']) < len(long_audio)
    stored_gemini = gemini_stored.data['serverContent']['modelTurn']['parts'][0]['inlineData']['data']
    assert 0 < len(stored_gemini) < len(long_audio)
    assert isinstance(no_data_stored, CassetteMessage)
    assert no_data_stored.data == gemini_no_data  # unchanged: nothing to truncate


@pytest.mark.anyio
async def test_recording_records_clean_close_while_iterating() -> None:
    """Unit test: async iteration records inbound frames and persists a clean terminal close."""
    frame = json.dumps({'type': 'server.event'})
    fake_ws = _FakeWebSocket([frame, ConnectionClosedOK(Close(1000, 'bye'), None)])
    cassette = RealtimeCassette()
    recording = RecordingWebSocket(fake_ws, cassette)

    assert [message async for message in recording] == [frame]
    assert cassette.interactions == [
        CassetteMessage(direction='received', data={'type': 'server.event'}),
        CassetteClose(code=1000, reason='bye', ok=True),
    ]


@pytest.mark.anyio
async def test_recording_records_error_close_and_delegates_passthrough() -> None:
    """Unit test: an abnormal disconnect records a non-ok close; `close()` and unknown attrs delegate."""
    fake_ws = _FakeWebSocket([ConnectionClosedError(Close(1011, 'boom'), None)])
    cassette = RealtimeCassette()
    recording = RecordingWebSocket(fake_ws, cassette)

    with pytest.raises(ConnectionClosedError):
        await recording.recv()
    assert cassette.interactions == [CassetteClose(code=1011, reason='boom', ok=False)]

    await recording.close(1000, 'done')
    assert fake_ws.closed_with == ((1000, 'done'), {})
    assert recording.marker == 'wrapped'  # unknown attribute falls through to the wrapped socket


def test_load_round_trips_close_frame(tmp_path: Path) -> None:
    """Unit test: a recorded terminal close survives the YAML dump/load round-trip."""
    cassette = RealtimeCassette(
        interactions=[
            CassetteMessage(direction='received', data={'type': 'server.hi'}),
            CassetteClose(code=1000, reason='bye', ok=True),
        ]
    )
    path = tmp_path / 'cassette.yaml'
    cassette.dump(path)
    assert RealtimeCassette.load(path).interactions == cassette.interactions


@pytest.mark.anyio
async def test_replay_rejects_unexpected_outbound_frame() -> None:
    """Unit test: replay asserts outbound frames match the recording, catching silent wire drift."""
    # No recorded send at this position (the next interaction is inbound) → the send is unexpected.
    no_send = RealtimeCassette(interactions=[CassetteMessage(direction='received', data={'type': 'server.event'})])
    with pytest.raises(AssertionError, match='no matching recorded send'):
        await ReplayWebSocket(no_send).send(json.dumps({'type': 'client.unexpected'}))
    # A recorded send at this position, but with different content → a content mismatch.
    wrong_content = RealtimeCassette(interactions=[CassetteMessage(direction='sent', data={'type': 'client.expected'})])
    with pytest.raises(AssertionError, match='did not match cassette'):
        await ReplayWebSocket(wrong_content).send(json.dumps({'type': 'client.unexpected'}))


@pytest.mark.anyio
async def test_replay_close_is_noop() -> None:
    """Unit test: replay's `close()` accepts the websockets signature and does nothing."""
    replay = ReplayWebSocket(RealtimeCassette())
    await replay.close(1000, 'done')
