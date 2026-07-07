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
import queue
from contextlib import suppress
from functools import partial

import logfire

from pydantic_ai import Agent
from pydantic_ai.realtime import (
    AudioWithTranscriptPart,
    AudioWithTranscriptPartDelta,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    RealtimeSession,
    RealtimeSessionEvent,
    SessionError,
    SpeechStarted,
)
from pydantic_ai.realtime.openai import OpenAIRealtimeModel

try:
    import sounddevice
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'This example needs the `sounddevice` package for microphone and speaker access. '
        'Install it with `pip install sounddevice`.'
    ) from e

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# OpenAI's realtime models speak and listen in 24 kHz mono PCM16 audio.
SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK_SIZE = 2400  # 100 ms per audio block

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
    loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))


def fill_speaker(
    play_queue: queue.Queue[bytes], carry: bytearray, outdata: bytearray, *_: object
) -> None:
    """Speaker callback: fill the block from buffered model audio, padding with silence on underrun."""
    want = len(outdata)
    while len(carry) < want:
        try:
            carry.extend(play_queue.get_nowait())
        except queue.Empty:
            break
    outdata[:] = bytes(carry[:want]).ljust(want, b'\x00')
    del carry[:want]


def drain(play_queue: queue.Queue[bytes]) -> None:
    """Drop any buffered playback audio (used for barge-in)."""
    while True:
        try:
            play_queue.get_nowait()
        except queue.Empty:
            break


async def handle_event(
    session: RealtimeSession,
    event: RealtimeSessionEvent,
    play_queue: queue.Queue[bytes],
) -> bool:
    """Handle one session event; return `True` to stop the session."""
    match event:
        case PartDeltaEvent(delta=AudioWithTranscriptPartDelta(audio_chunk=chunk)) if (
            chunk
        ):
            play_queue.put_nowait(chunk)
        case SpeechStarted():
            # Barge-in: drop buffered audio locally and cancel the model's turn.
            drain(play_queue)
            await session.interrupt()
        case PartEndEvent(
            part=AudioWithTranscriptPart(speaker='user', transcript=transcript)
        ):
            print(f'you: {transcript}')
        case PartEndEvent(
            part=AudioWithTranscriptPart(speaker='assistant', transcript=transcript)
        ):
            print(f'assistant: {transcript}')
        case FunctionToolCallEvent(part=call):
            print(f'[calling {call.tool_name}]')
        case FunctionToolResultEvent(part=result):
            print(f'[{result.tool_name} returned: {result.content}]')
        case SessionError(message=message, recoverable=recoverable):
            print(f'error: {message}')
            return not recoverable
    return False


async def stream_mic(session: RealtimeSession, mic_queue: asyncio.Queue[bytes]) -> None:
    while True:
        await session.send_audio(await mic_queue.get())


async def main():
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    play_queue: queue.Queue[bytes] = queue.Queue()
    # Partial audio block held between speaker callbacks (touched by the callback thread only).
    carry = bytearray()

    stream_kwargs = dict(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=BLOCK_SIZE
    )
    mic = sounddevice.RawInputStream(
        callback=partial(capture_mic, loop, mic_queue), **stream_kwargs
    )
    speaker = sounddevice.RawOutputStream(
        callback=partial(fill_speaker, play_queue, carry), **stream_kwargs
    )

    with mic, speaker:
        async with agent.realtime_session(
            model=OpenAIRealtimeModel('gpt-realtime')
        ) as session:
            pump = asyncio.create_task(stream_mic(session, mic_queue))
            print('Listening — start talking (Ctrl-C to quit).')
            try:
                async for event in session:
                    if await handle_event(session, event, play_queue):
                        break
            finally:
                pump.cancel()
                with suppress(asyncio.CancelledError):
                    await pump


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
