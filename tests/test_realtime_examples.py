"""Unit tests for the realtime examples' own helpers.

`test_examples.py` runs the documented snippets; these cover the parts of the runnable examples that
no snippet reaches — the voice assistant's barge-in accounting, the camera server's origin check, and
the Twilio bridge's call handling — because they are load-bearing and none is exercised by simply
importing the module.
"""

from __future__ import annotations

import asyncio
import base64
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from tests._timeouts import READINESS_WAIT_TIMEOUT

from .conftest import try_import
from .realtime_connections import InferredConnection

with try_import() as imports_successful:
    # These tests exercise example helpers, not application-wide telemetry setup.
    # Importing the runnable apps must not configure global exporters during collection.
    with patch('logfire.configure'), patch('logfire.instrument_pydantic_ai'):
        from examples.pydantic_ai_examples import realtime_voice
        from examples.pydantic_ai_examples.realtime_camera import app as realtime_camera
        from examples.pydantic_ai_examples.realtime_twilio import app as realtime_twilio
    from fastapi.testclient import TestClient
    from starlette.testclient import WebSocketTestSession
    from starlette.websockets import WebSocketDisconnect
    from twilio.request_validator import RequestValidator

    from examples.pydantic_ai_examples.realtime_twilio.audio import mulaw_to_pcm16, pcm16_to_mulaw
    from pydantic_ai.messages import (
        BinaryAudio,
        ModelMessage,
        RealtimeInputSpeechStartEvent,
        RealtimeResponseInterruptedEvent,
    )
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.realtime._utils import session_instructions
    from pydantic_ai.realtime.codec import (
        AudioDelta,
        CancelResponse,
        RealtimeCodecEvent,
        RealtimeConnection,
        RealtimeInput,
        ResponseDone,
        TruncateOutput,
    )
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel

    class ScriptedConnection(InferredConnection):
        """Replays an assistant turn and a user barge-in, recording what the session sends back.

        Defined inside the `try_import` block because its base class is one of the guarded imports:
        at module level it would raise `NameError` in environments without the realtime extras.
        """

        def __init__(
            self,
            *,
            chunk: bytes,
            wait_for: asyncio.Event,
            barge_in: bool,
            completed_turn: tuple[bytes, asyncio.Event] | None = None,
        ) -> None:
            self._chunk = chunk
            self._wait_for = wait_for
            self._barge_in = barge_in
            self._completed_turn = completed_turn
            self.sent: list[RealtimeInput] = []
            self.response_cancelled = asyncio.Event()

        async def send_input(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, CancelResponse):
                self.response_cancelled.set()

        async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
            if self._completed_turn is not None:
                # A full earlier turn, spoken and played to the end before the next turn starts.
                prior_chunk, prior_played = self._completed_turn
                yield AudioDelta(data=prior_chunk)
                yield ResponseDone()
                await prior_played.wait()
            yield AudioDelta(data=self._chunk)
            # Only report the user speaking once the chunk reached the speaker, so what the example
            # handles is playback state, not a race with its own audio delivery.
            await self._wait_for.wait()
            if self._barge_in:
                yield RealtimeInputSpeechStartEvent()
                # A real provider settles the cancelled response before speaking again; the reply
                # is a new part, so it must flow to the same, still-subscribed playback stream.
                await self.response_cancelled.wait()
                yield ResponseDone(interrupted=True)
                yield AudioDelta(data=FAST_CHUNK)
            else:
                yield RealtimeInputSpeechStartEvent()
            yield ResponseDone()

    class TwilioCallConnection(InferredConnection):
        """Speaks two chunks of a reply, then reports the caller talking once their audio arrives.

        `interruption` is how: the speech start most providers report, or the interruption Gemini and
        ElevenLabs Agents report. The session runs on the app's event loop, in `TestClient`'s thread,
        so the test waits for the cancel on a `threading.Event`.
        """

        def __init__(self, interruption: RealtimeCodecEvent | None = None, *, replies: int = 1) -> None:
            self.interruption = interruption or RealtimeInputSpeechStartEvent()
            self.replies = replies
            self.sent: list[RealtimeInput] = []
            self.instructions: str | None = None
            self.cancelled = threading.Event()
            self._caller_spoke = asyncio.Event()
            self._reply_cancelled = asyncio.Event()

        async def send_input(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, BinaryAudio):
                self._caller_spoke.set()
            elif isinstance(content, CancelResponse):
                self.cancelled.set()
                self._reply_cancelled.set()

        async def codec_events(self) -> AsyncIterator[RealtimeCodecEvent]:
            for reply in range(self.replies):
                if reply:
                    # As a provider does, end the reply spoken over once the session cancels it, then start the next.
                    await self._reply_cancelled.wait()
                    self._reply_cancelled.clear()
                    yield ResponseDone(interrupted=True)
                yield AudioDelta(data=REPLY_CHUNK)
                yield AudioDelta(data=REPLY_CHUNK)
                await self._caller_spoke.wait()
                self._caller_spoke.clear()
                yield self.interruption
            await asyncio.Event().wait()


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='extras not installed'),
]

MIC_CHUNK = b'\xaa' * 4
# 100 ms each at gpt-realtime's 24 kHz mono PCM16, so misattributed playback shows up in `played_ms`.
SLOW_CHUNK = b'\x01' * 4800  # playback of this chunk never completes
FAST_CHUNK = b'\x02' * 4800  # playback of this chunk completes immediately


class FakeMicrophone:
    """A `listentome.InputStream` stand-in that captures one block and then stays live."""

    def __init__(self, **kwargs: Any) -> None:
        self._captured = False

    async def __aenter__(self) -> FakeMicrophone:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    def __aiter__(self) -> FakeMicrophone:
        return self

    async def __anext__(self) -> bytes:
        if not self._captured:
            self._captured = True
            return MIC_CHUNK
        await asyncio.Event().wait()  # keep capturing until the conversation is cancelled
        raise StopAsyncIteration  # pragma: no cover

    async def read(self) -> bytes:  # pragma: no cover - part of the stream interface
        return await self.__anext__()


class FakeSpeaker:
    """A `listentome.OutputStream` stand-in whose `write()` mimics device pacing.

    `SLOW_CHUNK` plays until `resume` fires — like a real speaker mid-chunk when the user barges
    in — while any other chunk is consumed immediately. `received_slow` fires when the slow chunk
    reaches the device, `played` when any chunk finishes.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.written: list[bytes] = []
        self.received_slow = asyncio.Event()
        self.played = asyncio.Event()
        self.resume = asyncio.Event()

    async def __aenter__(self) -> FakeSpeaker:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def write(self, data: bytes) -> None:
        if data == SLOW_CHUNK:
            self.received_slow.set()
            await self.resume.wait()
        self.written.append(data)
        self.played.set()


def _fake_audio_io(monkeypatch: pytest.MonkeyPatch) -> FakeSpeaker:
    """Route the example's audio I/O through the fakes and return the speaker."""
    speaker = FakeSpeaker()

    def output_stream(**kwargs: Any) -> FakeSpeaker:
        return speaker

    monkeypatch.setattr('listentome.InputStream', FakeMicrophone)
    monkeypatch.setattr('listentome.OutputStream', output_stream)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    return speaker


def _script_connection(mocker: MockerFixture, connection: ScriptedConnection) -> None:
    @asynccontextmanager
    async def connect(self: OpenAIRealtimeModel, **kwargs: Any) -> AsyncGenerator[RealtimeConnection]:
        yield connection

    mocker.patch.object(OpenAIRealtimeModel, 'connect', new=connect)


async def test_voice_assistant_barge_in_drops_unheard_audio(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Barge-in mid-chunk truncates at 0 ms, and the reply reaches the same playback stream.

    The model's chunk is still playing when the user speaks, so `interrupt(played_bytes=0)` must
    truncate at 0 ms — and because the session flushes and suppresses the cancelled audio itself,
    the one playback task keeps running and the model's next reply comes out of the same stream.
    """
    speaker = _fake_audio_io(monkeypatch)
    connection = ScriptedConnection(chunk=SLOW_CHUNK, wait_for=speaker.received_slow, barge_in=True)
    speaker.resume = connection.response_cancelled  # the device finishes the chunk mid-cancel
    _script_connection(mocker, connection)

    await realtime_voice.main()

    assert TruncateOutput(audio_end_ms=0) in connection.sent
    assert CancelResponse() in connection.sent
    assert speaker.written == [SLOW_CHUNK, FAST_CHUNK]
    # The microphone block was forwarded through `send_audio(mic)`.
    assert BinaryAudio(data=MIC_CHUNK, media_type='audio/pcm') in connection.sent


async def test_voice_assistant_barge_in_excludes_earlier_turns_from_played_ms(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The truncation point covers the interrupted turn only, not earlier completed turns.

    A whole first turn plays to the end (100 ms of audio); the second turn's chunk is still playing
    when the user speaks. The session must attribute the cumulative device position to the second
    turn and truncate it at 0 ms, not misreport the first turn's 100 ms as heard second-turn audio.
    """
    speaker = _fake_audio_io(monkeypatch)
    connection = ScriptedConnection(
        chunk=SLOW_CHUNK,
        wait_for=speaker.received_slow,
        barge_in=True,
        completed_turn=(FAST_CHUNK, speaker.played),
    )
    speaker.resume = connection.response_cancelled
    _script_connection(mocker, connection)

    await realtime_voice.main()

    assert TruncateOutput(audio_end_ms=0) in connection.sent
    assert speaker.written == [FAST_CHUNK, SLOW_CHUNK, FAST_CHUNK]


async def test_voice_assistant_no_interrupt_when_turn_was_heard_in_full(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A speech-start event after playback finished must not interrupt.

    The speech-start event also fires on an ordinary turn where the user heard the whole previous
    reply; reporting an interruption then would make the provider discard part of a completed turn.
    """
    speaker = _fake_audio_io(monkeypatch)
    connection = ScriptedConnection(chunk=FAST_CHUNK, wait_for=speaker.played, barge_in=False)
    _script_connection(mocker, connection)

    await realtime_voice.main()

    assert speaker.written == [FAST_CHUNK]
    assert not any(isinstance(frame, (TruncateOutput, CancelResponse)) for frame in connection.sent)


def test_camera_websocket_origin_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Direct loopback and proxied (forwarded-host) origins connect; cross-site and DNS-rebinding don't."""
    same_origin = realtime_camera._same_origin  # pyright: ignore[reportPrivateUsage]

    assert same_origin(Mock(headers={'origin': 'http://localhost:8000', 'host': 'localhost:8000'}))
    # DNS rebinding: an attacker domain resolving to 127.0.0.1 matches `Host` with its own origin,
    # so a bare same-origin comparison is not enough — non-loopback origins need a proxy or allowlist.
    assert not same_origin(Mock(headers={'origin': 'http://attacker.example:8000', 'host': 'attacker.example:8000'}))
    # A reverse proxy that rewrites `Host` forwards the browser-facing host; browsers cannot send
    # `X-Forwarded-Host`, so it proves a proxy hop.
    assert same_origin(
        Mock(
            headers={
                'origin': 'https://app.proxy.example',
                'host': '127.0.0.1:8000',
                'x-forwarded-host': 'app.proxy.example',
            }
        )
    )
    assert not same_origin(
        Mock(
            headers={
                'origin': 'https://evil.example',
                'host': '127.0.0.1:8000',
                'x-forwarded-host': 'app.proxy.example',
            }
        )
    )
    assert not same_origin(Mock(headers={'host': '127.0.0.1:8000'}))
    # Proxies that forward neither `Host` nor `X-Forwarded-Host` are covered by the explicit allowlist.
    monkeypatch.setenv('CAMERA_ALLOWED_ORIGINS', 'https://tunnel.example, https://other.example')
    assert same_origin(Mock(headers={'origin': 'https://tunnel.example', 'host': '127.0.0.1:8000'}))


async def test_camera_defaults_are_safe_to_embed_in_script(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(realtime_camera, 'VOICE', '</script><script>alert(1)</script>')

    response = await realtime_camera.index()
    html = bytes(response.body).decode()

    assert '</script><script>alert(1)</script>' not in html
    assert r'\u003c/script\u003e\u003cscript\u003ealert(1)\u003c/script\u003e' in html


# 100 ms of the model's speech at gpt-realtime's 24 kHz.
REPLY_CHUNK = b'\x10\x00' * 2400
STREAM_SID = 'MZ1'
TWILIO_TOKEN = 'test-auth-token'


def _twilio_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('TWILIO_AUTH_TOKEN', TWILIO_TOKEN)
    monkeypatch.setattr(realtime_twilio, 'public_url', 'https://bridge.example')
    monkeypatch.setattr(realtime_twilio, 'answered_calls', set[str]())
    return TestClient(realtime_twilio.app)


def _answer(client: TestClient, call_sid: str) -> str:
    """Call the webhook as Twilio does for an incoming call, signed, returning the TwiML."""
    params = {'CallSid': call_sid, 'From': '+15551234567'}
    signature = RequestValidator(TWILIO_TOKEN).compute_signature('https://bridge.example/twiml', params)  # pyright: ignore[reportUnknownMemberType]
    response = client.post('/twiml', data=params, headers={'X-Twilio-Signature': signature})
    assert response.status_code == 200
    return response.text


def _connect_to(mocker: MockerFixture, connection: TwilioCallConnection) -> None:
    @asynccontextmanager
    async def connect(
        self: OpenAIRealtimeModel,
        *,
        messages: Sequence[ModelMessage],
        model_request_parameters: ModelRequestParameters,
        **kwargs: Any,
    ) -> AsyncGenerator[RealtimeConnection]:
        connection.instructions = session_instructions(messages, model_request_parameters)
        yield connection

    mocker.patch.object(OpenAIRealtimeModel, 'connect', new=connect)


def _start(call_sid: str) -> dict[str, Any]:
    return {
        'event': 'start',
        'sequenceNumber': '1',
        'streamSid': STREAM_SID,
        'start': {
            'accountSid': 'AC1',
            'callSid': call_sid,
            'streamSid': STREAM_SID,
            'tracks': ['inbound'],
            'mediaFormat': {'encoding': 'audio/x-mulaw', 'sampleRate': 8000, 'channels': 1},
            'customParameters': {'caller': '+15551234567'},
        },
    }


def _caller_audio() -> dict[str, Any]:
    # 20 ms of the caller, as Twilio sends it.
    payload = base64.b64encode(pcm16_to_mulaw(b'\x00\x01' * 160, rate=8000)).decode()
    return {'event': 'media', 'streamSid': STREAM_SID, 'media': {'track': 'inbound', 'payload': payload}}


def _until_closed(twilio: WebSocketTestSession) -> list[str]:
    """The events the bridge sends until it closes the socket."""
    events: list[str] = []
    with pytest.raises(WebSocketDisconnect):
        while True:
            events.append(twilio.receive_json()['event'])
    return events


def test_twilio_webhook_connects_only_the_calls_it_answers(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """A signed webhook gets TwiML that streams the call, with per-call context as a `<Parameter>`, as the
    stream URL takes no query string. An unsigned webhook is refused, and so is a stream for a call the
    webhook didn't answer."""
    client = _twilio_client(monkeypatch)
    connection = TwilioCallConnection()
    _connect_to(mocker, connection)

    assert _answer(client, 'CA1') == snapshot(
        '<?xml version="1.0" encoding="UTF-8"?><Response><Connect><Stream url="wss://bridge.example/media-stream"><Parameter name="caller" value="+15551234567"/></Stream></Connect></Response>'
    )
    assert client.post('/twiml', data={'CallSid': 'CA2', 'From': '+1'}).status_code == 403

    with client.websocket_connect('/media-stream') as twilio:
        twilio.send_json({'event': 'connected', 'protocol': 'Call', 'version': '1.0.0'})
        twilio.send_json(_start('CA2'))
        with pytest.raises(WebSocketDisconnect) as closed:
            twilio.receive_json()
    assert closed.value.code == 1008
    assert connection.instructions is None


def test_twilio_webhook_needs_the_auth_token(monkeypatch: pytest.MonkeyPatch):
    """Anyone can sign with an empty key, so without the token the webhook answers no call."""
    client = _twilio_client(monkeypatch)
    monkeypatch.setenv('TWILIO_AUTH_TOKEN', '')
    params = {'CallSid': 'CA1', 'From': '+1'}
    signature = RequestValidator('').compute_signature('https://bridge.example/twiml', params)  # pyright: ignore[reportUnknownMemberType]
    with pytest.raises(RuntimeError, match='Set `TWILIO_AUTH_TOKEN`'):
        client.post('/twiml', data=params, headers={'X-Twilio-Signature': signature})
    assert realtime_twilio.answered_calls == set()


def test_twilio_bridge_interrupts_at_what_the_caller_heard(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """When the caller talks over the reply, the bridge clears Twilio's buffer and interrupts the session at
    the audio Twilio confirmed playing: the first chunk's mark came back before the caller spoke, the
    second's only as `clear` dropped it."""
    client = _twilio_client(monkeypatch)
    connection = TwilioCallConnection()
    _connect_to(mocker, connection)
    _answer(client, 'CA1')

    with client.websocket_connect('/media-stream') as twilio:
        twilio.send_json({'event': 'connected', 'protocol': 'Call', 'version': '1.0.0'})
        twilio.send_json(_start('CA1'))
        reply = [twilio.receive_json() for _ in range(4)]
        assert [message['event'] for message in reply] == ['media', 'mark', 'media', 'mark']
        # 100 ms at the line's 8 kHz, in mu-law's byte per sample.
        assert len(base64.b64decode(reply[0]['media']['payload'])) == 800
        first_mark, second_mark = reply[1]['mark'], reply[3]['mark']
        twilio.send_json({'event': 'mark', 'streamSid': STREAM_SID, 'mark': first_mark})
        twilio.send_json(_caller_audio())

        assert twilio.receive_json() == {'event': 'clear', 'streamSid': STREAM_SID}
        twilio.send_json({'event': 'mark', 'streamSid': STREAM_SID, 'mark': second_mark})
        assert connection.cancelled.wait(READINESS_WAIT_TIMEOUT)
        twilio.send_json({'event': 'stop', 'streamSid': STREAM_SID})
        assert _until_closed(twilio) == []

    assert [content for content in connection.sent if isinstance(content, (TruncateOutput, CancelResponse))] == [
        TruncateOutput(audio_end_ms=100),
        CancelResponse(),
    ]
    assert (
        BinaryAudio(
            data=mulaw_to_pcm16(pcm16_to_mulaw(b'\x00\x01' * 160, rate=8000), rate=24000), media_type='audio/pcm'
        )
        in connection.sent
    )


def test_twilio_bridge_interrupts_each_reply_at_what_the_caller_heard(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    """The audio a `clear` dropped still counts toward the next interruption's position: the session handed it
    over, so the second interruption, like the first, lands after the one chunk the caller heard of its reply."""
    client = _twilio_client(monkeypatch)
    connection = TwilioCallConnection(replies=2)
    _connect_to(mocker, connection)
    _answer(client, 'CA1')

    with client.websocket_connect('/media-stream') as twilio:
        twilio.send_json({'event': 'connected', 'protocol': 'Call', 'version': '1.0.0'})
        twilio.send_json(_start('CA1'))
        for _ in range(2):
            reply = [twilio.receive_json() for _ in range(4)]
            assert [message['event'] for message in reply] == ['media', 'mark', 'media', 'mark']
            twilio.send_json({'event': 'mark', 'streamSid': STREAM_SID, 'mark': reply[1]['mark']})
            twilio.send_json(_caller_audio())
            assert twilio.receive_json() == {'event': 'clear', 'streamSid': STREAM_SID}
            twilio.send_json({'event': 'mark', 'streamSid': STREAM_SID, 'mark': reply[3]['mark']})
            assert connection.cancelled.wait(READINESS_WAIT_TIMEOUT)
            connection.cancelled.clear()
        twilio.send_json({'event': 'stop', 'streamSid': STREAM_SID})
        assert _until_closed(twilio) == []

    assert [content for content in connection.sent if isinstance(content, (TruncateOutput, CancelResponse))] == [
        TruncateOutput(audio_end_ms=100),
        CancelResponse(),
        TruncateOutput(audio_end_ms=100),
        CancelResponse(),
    ]


def test_twilio_bridge_passes_the_call_context_and_key_presses(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """The caller's number reaches the agent's instructions through the stream's custom parameters, a key
    press reaches the model as text, and `stop` ends the session and closes the socket."""
    client = _twilio_client(monkeypatch)
    connection = TwilioCallConnection()
    _connect_to(mocker, connection)
    _answer(client, 'CA1')

    with client.websocket_connect('/media-stream') as twilio:
        twilio.send_json({'event': 'connected', 'protocol': 'Call', 'version': '1.0.0'})
        twilio.send_json(_start('CA1'))
        assert [twilio.receive_json()['event'] for _ in range(4)] == ['media', 'mark', 'media', 'mark']
        twilio.send_json({'event': 'dtmf', 'streamSid': STREAM_SID, 'dtmf': {'track': 'inbound', 'digit': '5'}})
        twilio.send_json({'event': 'stop', 'streamSid': STREAM_SID})
        assert _until_closed(twilio) == []

    assert connection.instructions is not None
    assert 'The caller is calling from +15551234567.' in connection.instructions
    assert 'The caller pressed 5 on their keypad.' in connection.sent


def test_twilio_bridge_clears_twilio_when_the_model_interrupts_itself(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    """Gemini and ElevenLabs Agents cut their own reply short and say so: the bridge still clears the audio
    Twilio holds, and has nothing to interrupt."""
    client = _twilio_client(monkeypatch)
    connection = TwilioCallConnection(RealtimeResponseInterruptedEvent())
    _connect_to(mocker, connection)
    _answer(client, 'CA1')

    with client.websocket_connect('/media-stream') as twilio:
        twilio.send_json({'event': 'connected', 'protocol': 'Call', 'version': '1.0.0'})
        twilio.send_json(_start('CA1'))
        assert [twilio.receive_json()['event'] for _ in range(4)] == ['media', 'mark', 'media', 'mark']
        twilio.send_json(_caller_audio())
        assert twilio.receive_json() == {'event': 'clear', 'streamSid': STREAM_SID}
        twilio.send_json({'event': 'stop', 'streamSid': STREAM_SID})
        assert _until_closed(twilio) == []

    assert not any(isinstance(content, (TruncateOutput, CancelResponse)) for content in connection.sent)


def test_twilio_bridge_clears_without_interrupting_a_model_that_cant_be(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
):
    """A model that reports the caller talking but whose profile has no interruption still gets Twilio's buffer
    cleared, and is not interrupted, which it would refuse."""
    client = _twilio_client(monkeypatch)
    monkeypatch.setattr(
        realtime_twilio, 'MODEL', OpenAIRealtimeModel('gpt-realtime', profile={'supports_interruption': False})
    )
    connection = TwilioCallConnection()
    _connect_to(mocker, connection)
    _answer(client, 'CA1')

    with client.websocket_connect('/media-stream') as twilio:
        twilio.send_json({'event': 'connected', 'protocol': 'Call', 'version': '1.0.0'})
        twilio.send_json(_start('CA1'))
        assert [twilio.receive_json()['event'] for _ in range(4)] == ['media', 'mark', 'media', 'mark']
        twilio.send_json(_caller_audio())
        assert twilio.receive_json() == {'event': 'clear', 'streamSid': STREAM_SID}
        twilio.send_json({'event': 'stop', 'streamSid': STREAM_SID})
        assert _until_closed(twilio) == []

    assert not any(isinstance(content, (TruncateOutput, CancelResponse)) for content in connection.sent)


def test_twilio_marks_echoed_by_clear_are_not_played():
    """`clear` makes Twilio echo every pending mark, for audio the caller never heard: the caller heard up to
    the played chunk, and the dropped one counts only toward later positions."""
    marks = realtime_twilio.PlaybackMarks()
    first, second = marks.sent(4800), marks.sent(4800)
    marks.played(first)
    assert marks.clear() == 4800
    marks.played(second)
    assert marks.played_bytes == 9600


def test_twilio_audio_converts_between_the_line_and_the_session():
    """Each mu-law code at the line's 8 kHz becomes PCM16 at the session's rate and back.

    Both zero codes decode to silence, which encodes as the positive one.
    """
    for rate in (16000, 24000):
        for code in range(256):
            pcm = mulaw_to_pcm16(bytes([code]), rate=rate)
            assert len(pcm) == rate // 8000 * 2
            assert pcm16_to_mulaw(pcm, rate=rate) == (b'\xff' if code == 0x7F else bytes([code]))
    with pytest.raises(ValueError, match='11025 Hz is not a whole multiple of 8000 Hz'):
        mulaw_to_pcm16(b'\xff', rate=11025)
