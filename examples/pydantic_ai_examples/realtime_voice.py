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
    PartEndEvent,
    SpeechPart,
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

    async with mic, speaker, anyio.create_task_group() as tg:
        # The microphone is an async iterator of PCM blocks; `send_audio` forwards them
        # all. If the network falls behind, the stream drops its oldest blocks rather
        # than letting latency grow without bound.
        tg.start_soon(session.send_audio, mic)

        # `write()` returns once the device has consumed a chunk, so playback advances at
        # speaker pace while the model runs ahead; `stream_audio()`'s own buffer bounds
        # the backlog, dropping its oldest chunks if playback falls too far behind, so a
        # machine that stutters glitches instead of ending the call. The one thing the
        # session can't see is how much audio the device actually consumed, so count it.
        played = 0

        async def play_audio() -> None:
            nonlocal played
            async for chunk in session.stream_audio():
                await speaker.write(chunk)
                played += len(chunk)

        tg.start_soon(play_audio)

        print('Listening — start talking (Ctrl-C to quit).')
        async for event in session:
            match event:
                case RealtimeInputSpeechStartEvent():
                    # The provider stops the model on its own when the user speaks; what
                    # it can't know is how much of its audio actually reached the
                    # speaker. Given the device position, the session drops the audio the
                    # user will never hear and truncates the turn's transcript to what
                    # was really heard — or does nothing when the turn was heard in full,
                    # since the event also fires when nothing is playing.
                    await session.interrupt(played_bytes=played)
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
