"""Unit tests for the realtime examples' own helpers.

`test_examples.py` runs the documented snippets; these cover the parts of the runnable examples that
no snippet reaches — the voice assistant's barge-in accounting and the camera server's origin check —
because both are load-bearing and neither is exercised by simply importing the module.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from .conftest import try_import

with try_import() as imports_successful:
    from examples.pydantic_ai_examples import realtime_voice
    from examples.pydantic_ai_examples.realtime_camera import app as realtime_camera
    from pydantic_ai.messages import BinaryAudio, RealtimeInputSpeechStartEvent
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

    class ScriptedConnection(RealtimeConnection):
        """Replays an assistant turn and a user barge-in, recording what the session sends back.

        Defined inside the `try_import` block because its base class is one of the guarded imports:
        at module level it would raise `NameError` in environments without the realtime extras.
        """

        def __init__(
            self,
            *,
            chunk: bytes,
            wait_for_playback: asyncio.Event | None = None,
            completed_turn: tuple[bytes, asyncio.Event] | None = None,
        ) -> None:
            self._chunk = chunk
            self._wait_for_playback = wait_for_playback
            self._completed_turn = completed_turn
            self.sent: list[RealtimeInput] = []
            self._response_cancelled = asyncio.Event()

        async def send(self, content: RealtimeInput) -> None:
            self.sent.append(content)
            if isinstance(content, CancelResponse):
                self._response_cancelled.set()

        async def __aiter__(self) -> AsyncIterator[RealtimeCodecEvent]:
            if self._completed_turn is not None:
                # A full earlier turn, spoken and played to the end before the next turn starts.
                prior_chunk, prior_played = self._completed_turn
                yield AudioDelta(data=prior_chunk)
                yield ResponseDone()
                await prior_played.wait()
            yield AudioDelta(data=self._chunk)
            if self._wait_for_playback is not None:
                # Only report the user speaking once the speaker finished the chunk, so the turn was
                # heard in full by the time the example handles the speech-start event.
                await self._wait_for_playback.wait()
                yield RealtimeInputSpeechStartEvent()
            else:
                yield RealtimeInputSpeechStartEvent()
                # A real provider only speaks again after the barge-in cancelled the response; the
                # reply must reach the freshly subscribed playback task, not the cancelled one.
                await self._response_cancelled.wait()
                yield AudioDelta(data=FAST_CHUNK)
            yield ResponseDone()


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

    `SLOW_CHUNK` never finishes playing — like a real speaker mid-chunk when the user barges in —
    while any other chunk is consumed immediately. `played` fires when a chunk finishes.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.written: list[bytes] = []
        self.played = asyncio.Event()

    async def __aenter__(self) -> FakeSpeaker:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def write(self, data: bytes) -> None:
        if data == SLOW_CHUNK:
            await asyncio.Event().wait()  # cancelled when barge-in replaces the playback task
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
    """Barge-in mid-chunk truncates to what was played and replaces the playback subscription.

    The model's chunk never finishes playing, so when the user speaks the example must report 0 ms
    played, and the model's next reply must come out of the speaker — proving the replacement
    playback task subscribed to live audio after the stale subscription was cancelled.
    """
    speaker = _fake_audio_io(monkeypatch)
    connection = ScriptedConnection(chunk=SLOW_CHUNK)
    _script_connection(mocker, connection)

    await realtime_voice.main()

    assert TruncateOutput(audio_end_ms=0) in connection.sent
    assert CancelResponse() in connection.sent
    assert speaker.written == [FAST_CHUNK]
    # The microphone block was forwarded through `send_audio(mic)`.
    assert BinaryAudio(data=MIC_CHUNK, media_type='audio/pcm') in connection.sent


async def test_voice_assistant_barge_in_excludes_earlier_turns_from_played_ms(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`played_ms` reports playback of the interrupted turn only, not of earlier completed turns.

    A whole first turn plays to the end (100 ms of audio); the second turn's chunk never finishes
    playing. Interrupting the second turn must truncate it at 0 ms — a per-session counter without
    the turn-start watermark would misreport the first turn's 100 ms as heard second-turn audio.
    """
    speaker = _fake_audio_io(monkeypatch)
    connection = ScriptedConnection(chunk=SLOW_CHUNK, completed_turn=(FAST_CHUNK, speaker.played))
    _script_connection(mocker, connection)

    await realtime_voice.main()

    assert TruncateOutput(audio_end_ms=0) in connection.sent
    assert speaker.written == [FAST_CHUNK, FAST_CHUNK]


async def test_voice_assistant_no_interrupt_when_turn_was_heard_in_full(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A speech-start event after playback finished must not interrupt.

    The speech-start event also fires on an ordinary turn where the user heard the whole previous
    reply; reporting an interruption then would make the provider discard part of a completed turn.
    """
    speaker = _fake_audio_io(monkeypatch)
    connection = ScriptedConnection(chunk=FAST_CHUNK, wait_for_playback=speaker.played)
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
