"""A minimal voice assistant built on a realtime speech-to-speech model.

This opens a realtime session with OpenAI's `gpt-realtime` model, streams your microphone
audio to it, and plays the model's spoken replies back through your speakers. The agent
exposes a single `get_weather` tool the model can call mid-conversation.

Talk to it — and try interrupting while it's speaking: the model stops and listens (barge-in).

It needs the `sounddevice` package for microphone and speaker access
(`pip install sounddevice`), and an OpenAI API key set via `OPENAI_API_KEY`.

Run with:

    uv run -m pydantic_ai_examples.realtime_voice
"""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from contextlib import suppress
from functools import partial

import logfire

from pydantic_ai import (
    Agent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SpeechPart,
    SpeechPartDelta,
)
from pydantic_ai.realtime import (
    RealtimeEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeSession,
)

try:
    import sounddevice
except (ImportError, OSError) as e:  # pragma: no cover
    # `sounddevice` needs the PortAudio system library, which raises `OSError` (not `ImportError`)
    # when missing — e.g. on headless CI. Defer the failure to `main()` so the module still imports.
    sounddevice = None
    _sounddevice_error: Exception | None = e
else:
    _sounddevice_error = None

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# OpenAI's realtime models speak and listen in 24 kHz mono PCM16 audio.
SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK_SIZE = 2400  # 100 ms per audio block
MIC_QUEUE_BLOCKS = 10
PLAYBACK_BUFFER_SECONDS = 5

agent = Agent(
    instructions='You are a friendly voice assistant. Keep your replies short and conversational.'
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f'It is currently 21 degrees and sunny in {city}.'


def capture_mic(
    loop: asyncio.AbstractEventLoop,
    mic_queue: asyncio.Queue[bytes],
    indata: object,
    *_: object,
) -> None:
    """Microphone callback (PortAudio thread): hand captured audio to the event loop safely."""
    loop.call_soon_threadsafe(enqueue_latest, mic_queue, bytes(indata))


def enqueue_latest(audio_queue: asyncio.Queue[bytes], chunk: bytes) -> None:
    """Keep microphone latency bounded by dropping the oldest block on overflow."""
    if audio_queue.full():
        try:
            audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    audio_queue.put_nowait(chunk)


class PlaybackBuffer:
    """Thread-safe, bounded model-audio buffer with playback accounting."""

    def __init__(self, max_bytes: int):
        self._max_bytes = max_bytes
        self._chunks: deque[bytes] = deque()
        self._carry = bytearray()
        self._buffered_bytes = 0
        self._played_bytes = 0
        self._lock = threading.Lock()

    def start_turn(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._carry.clear()
            self._buffered_bytes = 0
            self._played_bytes = 0

    def add(self, chunk: bytes) -> None:
        with self._lock:
            # If the speaker falls far enough behind that the model is seconds ahead of what the
            # caller hears, drop the oldest audio rather than raise: a glitch is recoverable, and
            # ending a live call because one machine stuttered is not. (After a drop the
            # played-duration accounting is approximate, which is fine for a glitch.)
            # A chunk longer than the whole window keeps its tail.
            chunk = chunk[-self._max_bytes :]
            while (over := self._buffered_bytes + len(chunk) - self._max_bytes) > 0:
                # Oldest first: whatever `fill` already staged for the speaker, then queued chunks.
                if self._carry:
                    drop = min(len(self._carry), over)
                    del self._carry[:drop]
                else:
                    drop = len(self._chunks.popleft())
                self._buffered_bytes -= drop
            self._chunks.append(chunk)
            self._buffered_bytes += len(chunk)

    def fill(self, outdata: bytearray) -> None:
        """Fill one speaker block, padding an underrun with silence."""
        with self._lock:
            want = len(outdata)
            while len(self._carry) < want and self._chunks:
                self._carry.extend(self._chunks.popleft())
            played = min(want, len(self._carry))
            outdata[:] = bytes(self._carry[:played]).ljust(want, b'\x00')
            del self._carry[:played]
            self._buffered_bytes -= played
            self._played_bytes += played

    def interrupt(self) -> int | None:
        """Drop unheard audio; return milliseconds played, or `None` if nothing was left unheard.

        A turn the user heard in full needs no truncation — reporting one anyway would only make
        the provider discard part of a completed turn — so an interruption is only reported when
        unplayed audio was actually dropped.
        """
        with self._lock:
            if not self._chunks and not self._carry:
                return None
            self._chunks.clear()
            self._carry.clear()
            self._buffered_bytes = 0
            played_ms = self._played_bytes * 1000 // (SAMPLE_RATE * CHANNELS * 2)
            self._played_bytes = 0
            return played_ms


def fill_speaker(playback: PlaybackBuffer, outdata: bytearray, *_: object) -> None:
    """Speaker callback (PortAudio thread)."""
    playback.fill(outdata)


async def handle_event(
    session: RealtimeSession,
    event: RealtimeEvent,
    playback: PlaybackBuffer,
) -> None:
    """Handle one session event."""
    match event:
        case PartDeltaEvent(delta=SpeechPartDelta(audio_chunk=chunk)) if chunk:
            playback.add(chunk)
        case RealtimeInputSpeechStartEvent():
            # The provider stops the model on its own when the user speaks; what it can't know is how
            # much of its audio actually reached the speaker. Drop what didn't, and report the rest so
            # the provider doesn't record a turn the user never heard. The event fires whenever the
            # user starts speaking — including when nothing is playing — so only interrupt when
            # unheard audio was actually dropped.
            if (played_ms := playback.interrupt()) is not None:
                await session.interrupt(played_ms=played_ms)
        case PartStartEvent(part=SpeechPart(speaker='assistant')):
            playback.start_turn()
        case PartEndEvent(part=SpeechPart(speaker='user', transcript=transcript)):
            print(f'you: {transcript}')
        case PartEndEvent(part=SpeechPart(speaker='assistant', transcript=transcript)):
            print(f'assistant: {transcript}')
        case FunctionToolCallEvent(part=call):
            print(f'[calling {call.tool_name}]')
        case FunctionToolResultEvent(part=result):
            print(f'[{result.tool_name} returned: {result.content}]')


async def stream_mic(session: RealtimeSession, mic_queue: asyncio.Queue[bytes]) -> None:
    while True:
        await session.send_audio(await mic_queue.get())


async def main():
    if sounddevice is None:  # pragma: no cover
        raise ImportError(
            'This example needs the `sounddevice` package for microphone and speaker access. '
            'Install it with `pip install sounddevice`. '
            'On Linux you also need the PortAudio system library (`apt install libportaudio2`).'
        ) from _sounddevice_error

    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=MIC_QUEUE_BLOCKS)
    playback = PlaybackBuffer(
        max_bytes=SAMPLE_RATE * CHANNELS * 2 * PLAYBACK_BUFFER_SECONDS
    )

    stream_kwargs = dict(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=BLOCK_SIZE
    )
    mic = sounddevice.RawInputStream(
        callback=partial(capture_mic, loop, mic_queue), **stream_kwargs
    )
    speaker = sounddevice.RawOutputStream(
        callback=partial(fill_speaker, playback), **stream_kwargs
    )

    # The session opens before the microphone starts capturing, so no audio from before the
    # conversation began is queued up and sent to the model as stale input.
    async with agent.realtime('openai:gpt-realtime').session() as session:
        with mic, speaker:
            pump = asyncio.create_task(stream_mic(session, mic_queue))
            print('Listening — start talking (Ctrl-C to quit).')
            try:
                async for event in session:
                    await handle_event(session, event, playback)
            finally:
                pump.cancel()
                with suppress(asyncio.CancelledError):
                    await pump


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
