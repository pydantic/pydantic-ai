"""A minimal voice assistant built on a realtime speech-to-speech model.

This opens a realtime session with OpenAI's `gpt-realtime` model, streams your microphone
audio to it, and plays the model's spoken replies back through your speakers. The agent
exposes a single `get_weather` tool the model can call mid-conversation.

Talk to it — and try interrupting while it's speaking: the model stops and listens (barge-in).

It needs the `listentome` package for microphone and speaker access
(`pip install listentome`), which requires the PortAudio system library
(`brew install portaudio` on macOS, `apt install libportaudio2` on Debian/Ubuntu),
and an OpenAI API key set via `OPENAI_API_KEY`.

Run with:

    uv run -m pydantic_ai_examples.realtime_voice
"""

from __future__ import annotations

import anyio
import listentome
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
from pydantic_ai.realtime import RealtimeInputSpeechStartEvent, RealtimeSession

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent(
    instructions='You are a friendly voice assistant. Keep your replies short and conversational.'
)


@agent.tool_plain
def get_weather(city: str) -> str:
    """Look up the current weather in a city."""
    return f'It is currently 21 degrees and sunny in {city}.'


async def conversation(session: RealtimeSession) -> None:
    """Wire the microphone and speaker to the session and run the conversation."""
    # Capture and play at the rates this model expects; they can differ per direction.
    mic = listentome.InputStream(
        samplerate=session.audio_input_sample_rate,
        channels=1,
        dtype='int16',
        blocksize=session.audio_input_sample_rate // 10,  # 100 ms per block
    )
    speaker = listentome.OutputStream(
        samplerate=session.audio_output_sample_rate, channels=1, dtype='int16'
    )
    byte_rate = session.audio_output_sample_rate * 2  # bytes per second of mono PCM16

    async with mic, speaker, anyio.create_task_group() as tg:
        # The microphone is an async iterator of PCM blocks; `send_audio` forwards them
        # all. If the network falls behind, the stream drops its oldest blocks rather
        # than letting latency grow without bound.
        tg.start_soon(session.send_audio, mic)

        # `write()` returns once the device has consumed a chunk, so playback advances at
        # speaker pace while the model runs ahead; `stream_audio()`'s own buffer bounds
        # the backlog, dropping its oldest chunks if playback falls too far behind, so a
        # machine that stutters glitches instead of ending the call. Received and played
        # byte counts run for the whole session — a new turn can start while the previous
        # turn's tail is still playing (after a quick tool call, say), so per-turn resets
        # would misattribute that tail. `turn_start` marks where the current turn begins
        # in the received count, letting barge-in report how much of it was really heard.
        received = played = turn_start = 0

        async def play_audio(scope: anyio.CancelScope) -> None:
            nonlocal played
            with scope:
                async for chunk in session.stream_audio():
                    await speaker.write(chunk)
                    played += len(chunk)

        playback = anyio.CancelScope()
        tg.start_soon(play_audio, playback)

        print('Listening — start talking (Ctrl-C to quit).')
        async for event in session:
            match event:
                case PartDeltaEvent(delta=SpeechPartDelta(audio_chunk=bytes(chunk))):
                    received += len(chunk)
                case RealtimeInputSpeechStartEvent():
                    # The provider stops the model on its own when the user speaks; what
                    # it can't know is how much of its audio actually reached the
                    # speaker. Drop what didn't — by replacing the playback task with one
                    # subscribed to a fresh `stream_audio()`, which only carries live
                    # audio — and report the rest, so the provider doesn't record a turn
                    # the user never heard. The event fires whenever the user starts
                    # speaking — including when nothing is playing — so only interrupt
                    # when unheard audio was actually dropped.
                    if received > played:
                        playback.cancel()
                        playback = anyio.CancelScope()
                        tg.start_soon(play_audio, playback)
                        # If the previous turn's tail was still playing, none of this
                        # turn was heard yet, so its played duration clamps to zero.
                        played_ms = max(0, played - turn_start) * 1000 // byte_rate
                        await session.interrupt(played_ms=played_ms)
                        played = received  # the dropped audio is settled; stay in sync
                case PartStartEvent(part=SpeechPart(speaker='assistant')):
                    turn_start = received  # older audio belongs to earlier turns
                case PartEndEvent(part=SpeechPart() as part) if part.transcript:
                    print(f'{part.speaker}: {part.transcript}')
                case FunctionToolCallEvent(part=call):
                    print(f'[calling {call.tool_name}]')
                case FunctionToolResultEvent(part=result):
                    print(f'[{result.tool_name} returned: {result.content}]')
                case _:
                    pass
        tg.cancel_scope.cancel()


async def main():
    # The session opens before the microphone starts capturing, so no audio from before
    # the conversation began is queued up and sent to the model as stale input.
    async with agent.realtime('openai:gpt-realtime').session() as session:
        await conversation(session)


if __name__ == '__main__':
    try:
        anyio.run(main)
    except KeyboardInterrupt:
        pass
