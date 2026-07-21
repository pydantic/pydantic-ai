"""Cassette-backed end-to-end test for the Azure AI Voice Live realtime provider."""

from __future__ import annotations as _annotations

from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, SpeechPart, UserPromptPart
from pydantic_ai.realtime import TurnCompleteEvent

from ..conftest import IsDatetime, try_import
from .ws_cassettes import RealtimeCassette
from .ws_helpers import collapse_event_types, sent_frames_containing

with try_import() as imports_successful:
    from pydantic_ai.providers.azure_voicelive import AzureVoiceLiveProvider
    from pydantic_ai.realtime.azure import AzureRealtimeModel

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='websockets not installed')]


async def test_text_in_audio_out_turn(
    azure_ws_cassette: tuple[AzureVoiceLiveProvider, RealtimeCassette],
) -> None:
    """A text turn produces live audio/transcript events and standard message history."""
    provider, cassette = azure_ws_cassette
    model = AzureRealtimeModel('gpt-realtime', provider=provider)
    agent = Agent(instructions='Answer in two or three words.')

    events: list[Any] = []
    async with agent.realtime_session(model=model, audio_retention='output') as session:
        await session.send('Say a short greeting.')
        with anyio.fail_after(30):
            async for event in session:  # pragma: no branch - breaks on the recorded terminal event
                events.append(event)
                if isinstance(event, TurnCompleteEvent):
                    break

    assert sent_frames_containing(cassette, 'Answer in two or three words.') == snapshot(
        [
            {
                'type': 'session.update',
                'session': {
                    'instructions': 'Answer in two or three words.',
                    'modalities': ['text', 'audio'],
                    'input_audio_format': 'pcm16',
                    'output_audio_format': 'pcm16',
                    'input_audio_sampling_rate': 24000,
                    'turn_detection': {
                        'type': 'server_vad',
                        'create_response': True,
                        'interrupt_response': True,
                    },
                    'input_audio_transcription': {'model': 'whisper-1'},
                },
            }
        ]
    )

    assert collapse_event_types(events) == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent', 'TurnCompleteEvent']
    )
    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    assert messages[0] == ModelRequest(parts=[UserPromptPart(content='Say a short greeting.', timestamp=IsDatetime())])
    response = messages[1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == 'gpt-realtime-global-standard'
    part = response.parts[0]
    assert isinstance(part, SpeechPart)
    assert part.speaker == 'assistant'
    assert part.transcript == snapshot('Hello there! Great to meet you.')
    assert isinstance(part.audio, BinaryContent)
    assert part.audio.media_type == 'audio/wav'
    assert len(part.audio.data) > 0
