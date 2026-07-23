"""Type a text prompt, hear the model speak the reply — the simplest realtime session.

A realtime speech-to-speech model doesn't only listen to a microphone: you can send it plain
text and have it reply with speech. This is the text-in / audio-out path of a realtime session,
and the smallest possible way to try one out — no audio hardware required.

This opens a realtime session with OpenAI's `gpt-realtime` model, sends your text prompt, streams
back the spoken reply while printing the transcript as it arrives, and saves the audio to a `.wav`
file you can play afterwards.

Sending text into an OpenAI realtime session immediately asks the model to respond, so there's no
microphone, voice-activity detection, or manual turn-taking to manage — just `send()` and iterate.

It needs an OpenAI API key set via `OPENAI_API_KEY`.

Run with:

    uv run -m pydantic_ai_examples.realtime_text_to_audio "Tell me a fun fact about octopuses."
"""

from __future__ import annotations

import asyncio
import sys
import wave

import logfire

from pydantic_ai import Agent
from pydantic_ai.realtime import PartDeltaEvent, SpeechPartDelta, TurnCompleteEvent
from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

# OpenAI's realtime models speak in 24 kHz mono PCM16 audio.
SAMPLE_RATE = 24000

DEFAULT_PROMPT = 'Tell me a fun fact about octopuses.'
OUTPUT_PATH = 'realtime-response.wav'

agent = Agent(
    instructions='You are a friendly voice assistant. Keep your replies short and conversational.'
)


def save_wav(path: str, audio: bytes) -> None:
    """Wrap the streamed raw PCM16 audio in a WAV container so it can be played back."""
    with wave.open(path, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit samples
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio)


async def main(prompt: str, output_path: str) -> None:
    audio = bytearray()

    async with agent.realtime_session(
        model=OpenAIRealtimeModel('gpt-realtime'),
        model_settings=OpenAIRealtimeModelSettings(voice='marin'),
    ) as session:
        # Sending text (rather than audio) into an OpenAI realtime session asks the model to respond
        # right away — with speech, since a session's default output modality is audio.
        await session.send(prompt)

        print(f'you: {prompt}')
        print('assistant: ', end='', flush=True)
        async for event in session:
            match event:
                case PartDeltaEvent(delta=SpeechPartDelta() as delta):
                    # Deltas carry raw PCM16 audio for playback and/or incremental transcript text.
                    if delta.audio_chunk:
                        audio.extend(delta.audio_chunk)
                    if delta.transcript_delta:
                        print(delta.transcript_delta, end='', flush=True)
                case TurnCompleteEvent():
                    # The model finished speaking; this was a one-shot request, so we're done.
                    break
                case _:
                    pass
        print()

    save_wav(output_path, bytes(audio))
    print(f'\nSaved {len(audio)} bytes of audio to {output_path}')


if __name__ == '__main__':
    prompt = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPT
    asyncio.run(main(prompt, OUTPUT_PATH))
