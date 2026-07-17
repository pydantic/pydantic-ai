from __future__ import annotations

import asyncio

from pydantic_ai.realtime import AudioDelta, Transcript
from pydantic_ai.realtime.bedrock import BedrockRealtimeModel


async def main() -> None:
    model = BedrockRealtimeModel()
    transcript: list[str] = []
    audio_bytes = 0
    async with model.connect(instructions='Reply briefly.') as connection:
        await connection.send_text('Say the word hello', 'USER')
        async for event in connection:
            if isinstance(event, Transcript):
                transcript.append(event.text)
            elif isinstance(event, AudioDelta):
                audio_bytes += len(event.data)
            if transcript and audio_bytes:
                break
    print({'transcript': ''.join(transcript), 'audio_bytes': audio_bytes})


if __name__ == '__main__':
    asyncio.run(main())
