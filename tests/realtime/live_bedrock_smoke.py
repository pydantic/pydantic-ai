from __future__ import annotations

import asyncio

from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.realtime.bedrock import BedrockRealtimeModel
from pydantic_ai.realtime.codec import AudioDelta, Transcript


async def main() -> None:
    model = BedrockRealtimeModel()
    transcript: list[str] = []
    audio_bytes = 0
    async with model.connect(
        messages=[ModelRequest(parts=[], instructions='Reply briefly.')],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    ) as connection:
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
