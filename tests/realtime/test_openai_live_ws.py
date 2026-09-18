"""Cassette-backed GPT-Live WebSocket tests.

Live's wire shape differs from the Realtime API's in ways that only a real conversation exercises:
work is delegated to a Responses backend rather than tool-called directly, neither the user's turn
nor the model's reply has a terminal frame, and output audio is a continuous track rather than a
per-response stream. These record the real frames so the default suite runs offline.
"""

from __future__ import annotations as _annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    SpeechPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.providers import Provider
from pydantic_ai.realtime import RealtimeTurnCompleteEvent

from ..conftest import try_import
from .ws_cassettes import RealtimeCassette

with try_import() as imports_successful:
    from pydantic_ai.realtime.openai_live import OpenAILiveModel, OpenAILiveModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]

# Live has no end-of-turn frame, so the adapter waits out a quiet stretch. Replay delivers the
# recorded frames back to back, so only the wait after the last one costs real time; keeping the
# default would add two seconds to every test here.
_FAST_TURN = OpenAILiveModelSettings(openai_live_turn_silence_ms=1000)


# Live's session timeline advances with the audio it receives, so a clip that simply stops leaves the
# model with nothing to react to. A real microphone keeps sending silence; these tests send a fixed
# amount of it, which is also what keeps the outbound frame count deterministic for cassette replay.
_TRAILING_SILENCE_FRAMES = 120


async def _stream(session: Any, pcm: bytes) -> None:
    """Feed a clip, then a fixed tail of silence, in the ~100 ms frames a microphone would produce."""
    for start in range(0, len(pcm), 4800):
        await session.send_audio(pcm[start : start + 4800])
    for _ in range(_TRAILING_SILENCE_FRAMES):
        await session.send_audio(b'\x00' * 4800)


async def test_audio_in_delegated_tool_round(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """A spoken request is delegated to the Responses backend, which calls the agent's tool.

    This is the whole point of the adapter: the Live model runs the conversation, but the tool
    executes locally through the ordinary `ToolManager`, so the round lands in history in exactly the
    four-message shape a standard run produces.
    """
    provider, _ = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(instructions='You answer weather questions. Use the `lookup_forecast` tool.')

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()
    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert [
        type(event).__name__ for event in events if isinstance(event, (FunctionToolCallEvent, FunctionToolResultEvent))
    ] == snapshot(['FunctionToolCallEvent', 'FunctionToolResultEvent'])

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    # The user's spoken turn is recorded even though Live never says a transcript is finished.
    user_speech = [part for part in messages[0].parts if isinstance(part, SpeechPart)]
    assert len(user_speech) == 1
    assert user_speech[0].speaker == 'user'
    assert 'Amsterdam' in (user_speech[0].transcript or '')
    # The delegated call and its result are ordinary parts, not provider-specific ones.
    assert any(isinstance(part, ToolCallPart) for part in messages[1].parts)
    assert any(isinstance(part, ToolReturnPart) for part in messages[2].parts)
    # The delegated backend's tokens are recorded. Live's own audio seconds are not, because it
    # reports them on a timer and this exchange finishes before the first tick: see the caveat on
    # `docs/realtime/openai-live.md`.
    assert session.usage.input_tokens > 0
    assert session.usage.output_tokens > 0
    assert 'billable_audio_seconds' not in session.usage.details


async def test_text_reaches_the_model_as_context(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """`send(text)` steers a Live session even though Live has no user-message event.

    Text is injected as speakable context, and lands on the session's audio timeline, so it only
    takes effect while audio is flowing — hence the silence on both sides of it. Nobody speaks here:
    the reply is entirely the result of the injected text.
    """
    provider, _ = openai_live_ws_cassette
    # Relaying injected context takes the model a beat longer than answering, and a turn boundary
    # inferred from silence will cut in if it is too eager — the tradeoff the setting exists for.
    model = OpenAILiveModel(
        'gpt-live-1', provider=provider, settings=OpenAILiveModelSettings(openai_live_turn_silence_ms=2500)
    )
    agent = Agent(instructions='Relay what you are told, in one short sentence.')

    async def silence(frames: int) -> None:
        for _ in range(frames):
            await session.send_audio(b'\x00' * 4800)

    async with agent.realtime(model).session() as session:
        await silence(20)
        await session.send('Tell the user their package arrives on Friday.')
        await silence(_TRAILING_SILENCE_FRAMES)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    spoken = ' '.join(
        part.transcript or ''
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'Friday' in spoken


async def test_history_seeding(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """Prior text history is seeded into the session and is still present in `all_messages()`."""
    provider, _ = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(instructions='Answer in a few words.')
    history = [
        ModelRequest(parts=[UserPromptPart(content='My favorite color is orange.')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Good to know!')]),
    ]

    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()
    async with agent.realtime(model, message_history=history).session() as session:
        await _stream(session, pcm)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    assert session.all_messages()[:2] == history
