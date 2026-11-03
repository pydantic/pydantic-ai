"""Cassette-backed GPT-Live WebSocket tests.

Live's wire shape differs from the Realtime API's in ways that only a real conversation exercises:
work is delegated to a Responses backend rather than tool-called directly, neither the user's turn
nor the model's reply has a terminal frame, and output audio is a continuous track rather than a
per-response stream. These record the real frames so the default suite runs offline.
"""

from __future__ import annotations as _annotations

import asyncio
import fractions
import importlib
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest
from genai_prices import calc_price
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    BinaryImage,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartStartEvent,
    SpeechPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.providers import Provider
from pydantic_ai.realtime import RealtimeSession, RealtimeSessionReconnectEvent, RealtimeTurnCompleteEvent

from ..conftest import try_import
from .conftest import REAL_SDP_OFFER
from .ws_cassettes import CassetteMessage, RealtimeCassette

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.realtime.openai_live import OpenAILiveModel, OpenAILiveModelSettings

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]

# Live has no end-of-turn frame, so the adapter waits out a quiet stretch. Replay delivers the
# recorded frames back to back, so only the wait after the last one costs real time; keeping the
# default would add two seconds to every test here.
_FAST_TURN = OpenAILiveModelSettings(openai_live_turn_silence_ms=1000)

# The agent's own model is what the Live session delegates to, so naming one here pins the backend these
# recordings were made with rather than whatever `'auto'` resolves to today.
_BACKEND = 'openai:gpt-5.6-sol'


# Live's session timeline advances with the audio it receives, and the model can only speak onto that
# timeline. A clip that simply stops leaves it nothing to speak into; a real microphone keeps sending
# silence, so these tests send a fixed amount of it, which also keeps the outbound frame count
# deterministic for cassette replay. It has to cover the delegated answer, which Live speaks several
# seconds after the question.
_TRAILING_SILENCE_FRAMES = 150


async def _stream(
    session: Any, pcm: bytes, cassette: RealtimeCassette, *, paced: bool, frame_bytes: int = 4800
) -> None:
    """Feed a clip, then a fixed tail of silence, in the ~100 ms frames a microphone would produce.

    `frame_bytes` is 100 ms of audio at the session's rate: the default is 24 kHz PCM16.

    `paced` sends them at a microphone's pace. Sent in one burst, the whole tail lands at once, Live's
    timeline runs ahead of the conversation and then stops, and the model never gets to voice a
    delegated answer — so a recording made that way captures no reply at all. Replay sends as fast as
    it can instead, letting each frame wait for its recorded turn among the session's own sends.
    """
    frames = [pcm[start : start + frame_bytes] for start in range(0, len(pcm), frame_bytes)]
    frames += [b'\x00' * frame_bytes] * _TRAILING_SILENCE_FRAMES
    for frame in frames:
        await cassette.before_audio_send()
        await session.send_audio(frame)
        if paced:  # pragma: no branch
            await anyio.sleep(0.1)  # pragma: no cover  # only while recording


async def test_audio_in_delegated_tool_round(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """A spoken request is delegated to the Responses backend, which calls the agent's tool.

    This is the whole point of the adapter: the Live model runs the conversation, but the tool
    executes locally through the ordinary `ToolManager`, so the round lands in history in exactly the
    four-message shape a standard run produces.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='You answer weather questions. Use the `lookup_forecast` tool.')

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()
    events: list[Any] = []
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
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
    # Both meters are recorded: the delegated backend's tokens, and Live's own audio seconds, which it
    # reports on a timer (see the caveat on `docs/realtime/openai.md`).
    assert session.usage.input_tokens > 0
    assert session.usage.output_tokens > 0
    assert session.usage.audio_seconds > 0
    # Live reports how full its own context is; the backend's tokens don't measure it.
    assert session.context_window_used == snapshot(0.0106484375)
    # Each backend response's tokens land on the `ModelResponse` it produced, as in a standard run: the
    # one that asked for the tool on the tool-call response, the continuation on the spoken answer.
    tool_call_response, spoken_reply = messages[1], messages[3]
    assert isinstance(tool_call_response, ModelResponse) and isinstance(spoken_reply, ModelResponse)
    assert tool_call_response.usage.input_tokens > 0
    assert spoken_reply.usage.input_tokens > 0
    assert tool_call_response.usage.input_tokens + spoken_reply.usage.input_tokens == session.usage.input_tokens
    # Both carry Live's name, so each also records the backend model that spent its tokens, which is
    # what lets the cost be recalculated from `usage` later without charging it at Live's rate.
    assert tool_call_response.model_name == spoken_reply.model_name == snapshot('gpt-live-1')
    assert tool_call_response.provider_details == snapshot(
        {
            'delegated_model': 'gpt-5.6-sol',
            'delegated_response_id': 'resp_02967ef4d60b72e3006abc4434f40487d194be2a7080d73332',
        }
    )
    assert spoken_reply.provider_details == snapshot(
        {
            'delegated_model': 'gpt-5.6-sol',
            'delegated_response_id': 'resp_02967ef4d60b72e3006abc4436367887d18f682cb59deccc5d',
        }
    )
    for response in (tool_call_response, spoken_reply):
        assert response.provider_details is not None
        repriced = calc_price(response.usage, response.provider_details['delegated_model'], provider_id='openai')
        assert response.usage.cost == repriced.total_price
    # Live goes on speaking after the backend asked for the tool, and after its result went back (" now." comes
    # after the backend's terminal): that is the reply that called it, ahead of the result, not the answer. The call
    # lands mid-sentence, so the speech is one part, ahead of it, rather than cut in two at it.
    assert [
        part.transcript if isinstance(part, SpeechPart) else type(part).__name__ for part in tool_call_response.parts
    ] == snapshot(['Okay, checking the forecast now.', 'ToolCallPart'])
    # And Live speaks the backend's answer, which is the point of delegating.
    answer = spoken_reply.parts[0]
    assert isinstance(answer, SpeechPart) and answer.speaker == 'assistant'
    assert answer.transcript == snapshot(
        ' It\u2019s looking like about 14 degrees Celsius with some light rain. You might want a jacket.'
    )


async def test_thinking_sets_the_backends_reasoning_effort(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """The shared `thinking` setting reaches the delegated backend, which is the model that reasons.

    Left at its default effort (`medium`, as the backend echoes it), the backend spent 75 reasoning tokens
    on this request in a control recording made alongside this one; `thinking=False` is sent as `'none'`
    and it spends none.
    `parallel_tool_calls` reaches the backend the same way. The recording pins both in the session
    config, and the backend echoes both on every response.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(thinking=False, parallel_tool_calls=False, openai_live_turn_silence_ms=1000),
    )
    agent = Agent(
        _BACKEND,
        instructions=(
            'You answer weather questions. Use the `lookup_forecast` tool, then say whether the temperature '
            'is above the yearly average for that city, reasoning it out from what you know.'
        ),
    )

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    # Both backend responses (the tool call and the answer after it) reported their usage.
    assert session.usage.requests == 2
    assert session.usage.tool_calls == 1
    assert session.usage.details == {'reasoning_tokens': 0}


async def test_typed_question_runs_the_agents_tool(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], realtime_recording: bool
) -> None:
    """`send(text)` is a user turn: the backend is run on it, calls the agent's tool, and Live speaks the answer.

    Live has no user-message event, so the text goes to the backend as the user's message, with `response.create`
    to run it, and the wait for its reply ends once Live has spoken the answer, not at what it says while the
    backend works.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(openai_live_idle_audio=True, openai_live_turn_silence_ms=2500),
    )
    agent = Agent(
        _BACKEND, instructions='Always use the `get_weather` tool for weather questions. Answer in one sentence.'
    )

    @agent.tool_plain
    async def get_weather(city: str) -> str:
        """Look up the current weather for a city."""
        return f'It is foggy and 12 degrees Celsius in {city}.'

    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in London?')
        with anyio.fail_after(60):
            await session.wait_for_reply()
        if realtime_recording:  # pragma: no cover
            # Replay's clock moves with the recorded frames: keep the ones that carry it past the turn's end.
            await anyio.sleep(0.5)
        messages = session.all_messages()

    assert [
        [
            f'{part.part_kind}: {part.content}'
            if isinstance(part, UserPromptPart)
            else f'{part.part_kind}: {part.tool_name} {part.args}'
            if isinstance(part, ToolCallPart)
            else f'{part.part_kind}: {part.content}'
            if isinstance(part, ToolReturnPart)
            else f'{part.part_kind}: {part.transcript}'
            if isinstance(part, SpeechPart)
            else part.part_kind
            for part in message.parts
        ]
        for message in messages
    ] == snapshot(
        [
            ['user-prompt: What is the weather in London?'],
            ['speech:  Let me check the current weather in London.', 'tool-call: get_weather {"city":"London"}'],
            ['tool-return: It is foggy and 12 degrees Celsius in London.'],
            ['speech:  It\u2019s foggy and 12 degrees Celsius.'],
        ]
    )
    # Closing asked Live to end the session, which it confirmed.
    assert [
        interaction.data['type']
        for interaction in cassette.interactions
        if isinstance(interaction, CassetteMessage) and interaction.data.get('type', '').startswith('session.clos')
    ] == ['session.close', 'session.closed']


async def test_typed_small_talk_is_answered_in_one_reply(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], realtime_recording: bool
) -> None:
    """Text that needs no tool runs the backend too, and Live speaks its answer as the one reply to the text.

    With `openai_live_idle_audio`, there is no microphone at all: the connection streams silence in its place, which
    Live's timeline needs to advance for the model to speak.
    """
    provider, _ = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(openai_live_idle_audio=True, openai_live_turn_silence_ms=2500),
    )
    agent = Agent(_BACKEND, instructions='Keep replies short.')

    async with agent.realtime(model).session() as session:
        await session.send("Thanks, that's all I needed. Have a nice day!")
        with anyio.fail_after(60):
            await session.wait_for_reply()
        if realtime_recording:  # pragma: no cover
            # Replay's clock moves with the recorded frames: keep the ones that carry it past the turn's end.
            await anyio.sleep(0.5)
        messages = session.all_messages()

    assert [
        [part.content for part in message.parts if isinstance(part, UserPromptPart)]
        if isinstance(message, ModelRequest)
        else [part.transcript for part in message.parts if isinstance(part, SpeechPart)]
        for message in messages
    ] == snapshot(
        [
            ["Thanks, that's all I needed. Have a nice day!"],
            [" Oh, okay, let me just, check on that. You're welcome! Have a great day!"],
        ]
    )


async def test_history_seeding(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """Prior text history is seeded into the session and is still present in `all_messages()`."""
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='Answer in a few words.')
    history = [
        ModelRequest(parts=[UserPromptPart(content='My favorite color is orange.')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Good to know!')]),
    ]

    pcm = assets_path.joinpath('marcelo_24khz.pcm').read_bytes()
    async with agent.realtime(model, message_history=history).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    assert messages[:2] == history
    # The live exchange follows the seeded one: the user's words, then a spoken reply to them.
    assert [type(message).__name__ for message in messages[2:]] == snapshot(['ModelRequest', 'ModelResponse'])
    user_speech, reply = messages[2].parts[0], messages[3].parts[0]
    assert isinstance(user_speech, SpeechPart) and 'Marcelo' in (user_speech.transcript or '')
    assert isinstance(reply, SpeechPart) and reply.speaker == 'assistant' and reply.transcript


async def test_an_image_is_described_by_the_backend(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette],
    image_content: BinaryImage,
    realtime_recording: bool,
) -> None:
    """Live's voice model sees no images, but the backend it delegates to does, and Live speaks for it.

    With `respond=True` the image goes to the backend as ordinary Responses input and the backend runs
    on it straight away; the microphone keeps streaming meanwhile, since that is what moves Live's
    timeline along.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='Say what is in images the user shares, in one short sentence.')

    async def silence(frames: int) -> None:
        for _ in range(frames):
            await cassette.before_audio_send()
            await session.send_audio(b'\x00' * 4800)
            if realtime_recording:  # pragma: no branch
                await anyio.sleep(0.1)  # pragma: no cover  # only while recording

    async with agent.realtime(model).session() as session:
        await silence(10)
        await session.send(image_content, respond=True)
        await silence(_TRAILING_SILENCE_FRAMES)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break
        # The reply the image asked for is over: nothing is owed any more.
        core = session._core  # pyright: ignore[reportPrivateUsage]
        assert core is not None and not core.wait_tokens()

    spoken = ' '.join(
        part.transcript or ''
        for message in session.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'kiwi' in spoken.lower()


@pytest.mark.vcr
async def test_the_backend_searches_the_web(
    openai_live_ws_and_http_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
    allow_model_requests: None,
) -> None:
    """`WebSearchTool` runs on the delegated backend, and its searches land in history as native parts.

    The backend searches while Live keeps the conversation going, then Live speaks the answer; the
    searches are recorded on the spoken reply they informed, ahead of the speech, as a standard run
    records them ahead of its text. The history carries on in a standard run on a Responses model (the
    HTTP cassette), which OpenAI accepts only because each search is recorded with the reasoning that led
    to it.
    """
    provider, cassette = openai_live_ws_and_http_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(
        _BACKEND,
        instructions='Always search the web before answering, and answer in one short sentence.',
        capabilities=[WebSearch(native=WebSearchTool(search_context_size='low', allowed_domains=['wikipedia.org']))],
    )

    pcm = assets_path.joinpath('amsterdam_population_question_24khz.pcm').read_bytes()
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(['ModelRequest', 'ModelResponse'])
    reply = messages[1]
    assert isinstance(reply, ModelResponse)
    calls = [part for part in reply.parts if isinstance(part, NativeToolCallPart)]
    returns = [part for part in reply.parts if isinstance(part, NativeToolReturnPart)]
    assert {part.tool_name for part in [*calls, *returns]} == {'web_search'}
    assert [part.tool_call_id for part in calls] == [part.tool_call_id for part in returns]
    assert all(part.provider_name == 'openai' for part in calls)
    # Each search records what the backend searched for, as on a direct Responses call.
    assert [part.args for part in calls] == snapshot(
        [
            {
                'type': 'search',
                'queries': ['site:wikipedia.org Amsterdam population 2025 municipality'],
                'query': 'site:wikipedia.org Amsterdam population 2025 municipality',
            }
        ]
    )
    assert [part.content for part in returns] == snapshot([{'status': 'completed'}])
    # The searches come first, then what Live said with their results.
    speech = reply.parts[-1]
    assert isinstance(speech, SpeechPart) and speech.speaker == 'assistant'
    assert all(not isinstance(part, SpeechPart) for part in reply.parts[:-1])
    assert 'thirty-six thousand' in (speech.transcript or '')
    # Each search follows the backend reasoning that led to it, as a direct Responses run records it.
    assert [type(part).__name__ for part in reply.parts] == snapshot(
        [
            'ThinkingPart',
            'NativeToolCallPart',
            'NativeToolReturnPart',
            'SpeechPart',
        ]
    )
    thinking = [part for part in reply.parts if isinstance(part, ThinkingPart)]
    assert thinking and all(part.id and part.signature and part.provider_name == 'openai' for part in thinking)

    # A text agent on the backend's model picks the conversation up where the call left it.
    follow_up = await Agent(OpenAIResponsesModel('gpt-5.6-sol', provider=provider)).run(
        'What number did you just give me? Answer with digits only.', message_history=messages
    )
    assert '936' in follow_up.output


async def test_the_backend_summarizes_its_reasoning(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """`reasoning_summary` reaches the backend, which accepts it and summarizes its reasoning.

    The reasoning recorded ahead of each search then reads as the summary, as on a direct Responses run
    that requests one.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(
            openai_live_delegation={'reasoning_summary': 'auto'}, openai_live_turn_silence_ms=1000
        ),
    )
    agent = Agent(
        _BACKEND,
        instructions='Always search the web before answering, and answer in one short sentence.',
        capabilities=[WebSearch(native=WebSearchTool(search_context_size='low', allowed_domains=['wikipedia.org']))],
    )

    pcm = assets_path.joinpath('amsterdam_population_question_24khz.pcm').read_bytes()
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    reply = session.all_messages()[-1]
    assert isinstance(reply, ModelResponse)
    # The summary is requested next to the backend's other settings.
    assert _session_starts(cassette)[0]['delegation']['responses']['reasoning'] == snapshot({'summary': 'auto'})
    # Only reasoning the backend did summarize carries text: here, the reasoning before the first search.
    # Curly apostrophes are straightened, since pre-commit would rewrite them in the snapshot below.
    thinking = [part.content.replace('\u2019', "'") for part in reply.parts if isinstance(part, ThinkingPart)]
    assert thinking == snapshot(
        [
            """\
**Searching for Amsterdam population**

I need to check Wikipedia specifically for the current population of Amsterdam. There's a bit of ambiguity about whether I'm looking for the city or the municipality, which makes it a little tricky. I'm thinking the population is likely around 933,680 as of January 2025, but I need to confirm that by searching. Let's go ahead and get that information to be sure!\
""",
            '',
            '',
            '',
        ]
    )


async def test_an_image_a_tool_returns_reaches_the_backend(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    image_content: BinaryImage,
    realtime_recording: bool,
) -> None:
    """An image a tool returns follows its output to the delegated backend, and Live speaks for it.

    The session runs at 16 kHz, set through the profile, because that is the rate the recorded question is in.
    """
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=_FAST_TURN,
        profile={'audio_input_sample_rate': 16000, 'audio_output_sample_rate': 16000},
    )
    agent = Agent(
        _BACKEND,
        instructions=(
            "The user's image is only available through the `get_image` tool. Call it when they ask about the "
            'image, then say what is in it in one short sentence.'
        ),
    )

    @agent.tool_plain
    async def get_image() -> BinaryImage:
        """Get the image the user is asking about."""
        return image_content

    pcm = assets_path.joinpath('what_fruit_is_in_the_image_16khz.pcm').read_bytes()
    async with agent.realtime(model).session() as session:
        await _stream(session, pcm, cassette, paced=realtime_recording, frame_bytes=3200)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break

    messages = session.all_messages()
    # The image is recorded on the tool's return, as in a standard run.
    tool_return = next(part for message in messages for part in message.parts if isinstance(part, ToolReturnPart))
    assert tool_return.tool_name == 'get_image'
    assert tool_return.content == image_content
    spoken = ' '.join(
        part.transcript or ''
        for message in messages
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'kiwi' in spoken.lower()


# `(offer_sdp, connect(answer_sdp), speak())`: the browser side of a WebRTC call.
_BrowserPeer = tuple[str, Callable[[str], Awaitable[None]], Callable[[], None]]


@asynccontextmanager
async def _speaking_browser(pcm: bytes, *, rate: int = 24000) -> AsyncGenerator[_BrowserPeer]:  # pragma: no cover
    """Negotiate a real WebRTC call with `aiortc`, standing in for a browser whose user asks a question.

    Recording only. Live's timeline moves with the audio it receives, and on a WebRTC call that audio
    comes from the browser, so a canned offer that never connects records a session in which nothing
    happens. The track sends silence until `speak()`, then the clip, then silence again, so the question
    is asked only once the sideband is attached to hear the answer. `aiortc` is not a project dependency:
    record with `uv run --with aiortc --env-file .env pytest ... --record-mode=rewrite`.
    """
    aiortc = importlib.import_module('aiortc')
    av = importlib.import_module('av')
    frame_samples = rate // 50  # 20 ms frames of the clip
    state = {'position': -1}

    class _Microphone(aiortc.MediaStreamTrack):
        kind = 'audio'

        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self._start: float | None = None
            self._timestamp = 0

        async def recv(self) -> Any:
            if self._start is None:
                self._start = time.monotonic()
            else:
                await asyncio.sleep(max(0.0, self._start + self._timestamp / rate - time.monotonic()))
            chunk = b''
            if state['position'] >= 0:
                chunk = pcm[state['position'] : state['position'] + frame_samples * 2]
                state['position'] += frame_samples * 2
            frame = av.AudioFrame(format='s16', layout='mono', samples=frame_samples)
            frame.planes[0].update(chunk.ljust(frame_samples * 2, b'\x00'))
            frame.sample_rate = rate
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, rate)
            self._timestamp += frame_samples
            return frame

    # No STUN server, as in `test_openai_ws.py`: host candidates reach OpenAI's ICE-lite endpoint.
    pc = aiortc.RTCPeerConnection(aiortc.RTCConfiguration(iceServers=[]))
    pc.addTrack(_Microphone())

    @pc.on('track')
    def _drain_inbound_audio(track: Any) -> None:
        async def pump() -> None:
            while True:
                try:
                    await track.recv()
                except Exception:
                    return

        asyncio.ensure_future(pump())

    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != 'complete':
        await anyio.sleep(0.1)

    async def connect(answer_sdp: str) -> None:
        await pc.setRemoteDescription(aiortc.RTCSessionDescription(sdp=answer_sdp, type='answer'))

    def speak() -> None:
        state['position'] = 0

    try:
        yield pc.localDescription.sdp, connect, speak
    finally:
        await pc.close()


async def _no_browser_to_connect(answer_sdp: str) -> None:
    """Replay has no browser: the recorded sideband frames are the whole call."""


@pytest.mark.vcr
async def test_webrtc_sideband_runs_the_delegated_tool_round(
    openai_live_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """A browser WebRTC call, negotiated by the server, with the agent run over a sideband.

    The server starts the session from the browser's offer, with the agent's instructions and tools
    already on the backend; the browser holds the audio; the sideband runs the backend's tool call and
    records the conversation. The offer is an HTTP VCR cassette, the sideband a WebSocket cassette.
    """
    provider, _ = openai_live_ws_sideband_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='You answer weather questions. Use the `lookup_forecast` tool.')

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        return f'{city}: 14 degrees Celsius, light rain.'

    realtime = agent.realtime(model)
    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()

    @asynccontextmanager
    async def browser() -> AsyncGenerator[_BrowserPeer]:
        if realtime_recording:  # pragma: no cover
            async with _speaking_browser(pcm) as peer:
                yield peer
            return
        yield REAL_SDP_OFFER, _no_browser_to_connect, lambda: None

    async with browser() as (offer, connect, speak):
        answer = await realtime.answer_webrtc_offer(offer)
        assert answer.sdp.startswith('v=0')
        assert answer.session.provider_name == 'openai'
        assert answer.session.session_id.startswith('live_')
        await connect(answer.sdp)

        async with realtime.session(provider_session=answer.session) as session:
            with pytest.raises(UserError, match='does not own the audio transport'):
                await session.send_audio(b'\x00\x00')
            speak()
            events: list[Any] = []
            with anyio.fail_after(60):
                async for event in session:  # pragma: no branch
                    events.append(event)
                    if isinstance(event, RealtimeTurnCompleteEvent) and any(
                        isinstance(e, FunctionToolResultEvent) for e in events
                    ):
                        break

    messages = session.all_messages()
    assert [type(message).__name__ for message in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    user_speech = [part for part in messages[0].parts if isinstance(part, SpeechPart)]
    assert 'Amsterdam' in (user_speech[0].transcript or '')
    assert any(isinstance(part, ToolCallPart) for part in messages[1].parts)
    assert any(isinstance(part, ToolReturnPart) for part in messages[2].parts)
    answer_part = messages[3].parts[-1]
    assert isinstance(answer_part, SpeechPart)
    assert '14' in (answer_part.transcript or '') or 'fourteen' in (answer_part.transcript or '').lower()
    # The browser plays the audio, so the sideband records the reply without its bytes.
    assert answer_part.audio is None
    assert session.usage.input_tokens > 0


@pytest.mark.vcr
async def test_webrtc_sideband_reattaches_after_a_drop(
    openai_live_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """With a `reconnect` policy, a dropped sideband re-attaches to the call's session, which carried on without it.

    The sideband drops while the backend waits on a tool. The session is the same one, so the tool's result
    still goes out, on the new sideband, and the backend answers with it. Live sends the re-attached sideband
    the last few seconds of events again, which are skipped: the history is the one a call without a drop records.
    """
    provider, cassette = openai_live_ws_sideband_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(
            openai_live_turn_silence_ms=1000, reconnect={'base_delay': 0.0, 'jitter': False}
        ),
    )
    agent = Agent(_BACKEND, instructions='You answer weather questions. Use the `lookup_forecast` tool.')
    dropped = anyio.Event()

    @agent.tool_plain
    async def lookup_forecast(city: str) -> str:
        """Look up tomorrow's forecast for a city."""
        await dropped.wait()
        return f'{city}: 14 degrees Celsius, light rain.'

    realtime = agent.realtime(model)
    pcm = assets_path.joinpath('weather_question_24khz.pcm').read_bytes()

    @asynccontextmanager
    async def browser() -> AsyncGenerator[_BrowserPeer]:
        if realtime_recording:  # pragma: no cover
            async with _speaking_browser(pcm) as peer:
                yield peer
            return
        yield REAL_SDP_OFFER, _no_browser_to_connect, lambda: None

    events: list[Any] = []
    async with browser() as (offer, connect, speak):
        answer = await realtime.answer_webrtc_offer(offer)
        await connect(answer.sdp)

        async with realtime.session(provider_session=answer.session) as session:
            speak()
            with anyio.fail_after(60):
                async for event in session:  # pragma: no branch
                    events.append(event)
                    if isinstance(event, FunctionToolCallEvent):
                        await cassette.disconnect()
                    elif isinstance(event, RealtimeSessionReconnectEvent):
                        dropped.set()
                    elif isinstance(event, RealtimeTurnCompleteEvent) and any(
                        isinstance(e, FunctionToolResultEvent) for e in events
                    ):
                        break

    reconnects = [event for event in events if isinstance(event, RealtimeSessionReconnectEvent)]
    assert reconnects == [RealtimeSessionReconnectEvent(state_restored=True)]
    messages = session.all_messages()
    parts = [part for message in messages for part in message.parts]

    def spoken(speaker: str) -> str:
        transcripts = (
            part.transcript or '' for part in parts if isinstance(part, SpeechPart) and part.speaker == speaker
        )
        return ' '.join(' '.join(transcripts).split())

    # Every word once: what Live sent again after the re-attach was skipped. (The user is still speaking when the
    # model starts to answer here, so how many turns their words fall into depends on timing.)
    assert spoken('user') == snapshot('What will the weather be in Amsterdam tomorrow Please look it up')
    # The call asked for before the drop was answered after it, once.
    assert [part.tool_name for part in parts if isinstance(part, ToolCallPart)] == ['lookup_forecast']
    assert [part.outcome for part in parts if isinstance(part, ToolReturnPart)] == ['success']
    answer = messages[-1]
    assert isinstance(answer, ModelResponse)
    assert spoken('assistant').count("I'll look it up.") == 1
    last = answer.parts[-1]
    assert isinstance(last, SpeechPart) and '14' in (last.transcript or '')


@pytest.mark.vcr
async def test_webrtc_call_continues_the_bound_history(
    openai_live_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """History bound with `agent.realtime(...)` is seeded when the offer starts the Live session.

    Live takes history only then, so the sideband opened from the same object seeds nothing, and the model
    answers from the history it was started with.
    """
    provider, cassette = openai_live_ws_sideband_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    agent = Agent(_BACKEND, instructions='Answer in a few words.')
    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Ada.')]),
        ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Nice to meet you, Ada.')]),
    ]
    realtime = agent.realtime(model, message_history=history)
    pcm = assets_path.joinpath('remind_me_my_name_16khz.pcm').read_bytes()

    @asynccontextmanager
    async def browser() -> AsyncGenerator[_BrowserPeer]:
        if realtime_recording:  # pragma: no cover
            async with _speaking_browser(pcm, rate=16000) as peer:
                yield peer
            return
        yield REAL_SDP_OFFER, _no_browser_to_connect, lambda: None

    async with browser() as (offer, connect, speak):
        answer = await realtime.answer_webrtc_offer(offer)
        await connect(answer.sdp)

        async with realtime.session(provider_session=answer.session) as session:
            speak()
            with anyio.fail_after(60):
                async for event in session:  # pragma: no branch
                    if isinstance(event, RealtimeTurnCompleteEvent):
                        break

    # The sideband seeded nothing: Live replayed the history the offer started the session with. All it sent
    # is the empty update that reports the session, however late the sideband attaches.
    assert [i.data for i in cassette.interactions if isinstance(i, CassetteMessage) and i.direction == 'sent'] == [
        {'type': 'session.update', 'session': {}}
    ]
    messages = session.all_messages()
    assert messages[:2] == history
    spoken = ' '.join(
        part.transcript or ''
        for message in messages[2:]
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, SpeechPart)
    )
    assert 'Ada' in spoken


@pytest.mark.vcr
async def test_webrtc_hang_up_ends_the_call(
    openai_live_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """`hang_up()` on a sideband ends the browser's call; `close()` alone would only detach from it.

    Live ends the session when the call is hung up, and asking again finds no call to end, which is not an
    error. The offer and both hangups are an HTTP VCR cassette, the sideband a WebSocket cassette.
    """
    provider, _ = openai_live_ws_sideband_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    realtime = Agent(_BACKEND, instructions='Answer in a few words.').realtime(model)

    answer = await realtime.answer_webrtc_offer(REAL_SDP_OFFER)
    async with realtime.session(provider_session=answer.session) as session:
        await session.hang_up()
    assert session.closed

    # The call is gone now: hanging it up again, without a sideband, finds nothing to end.
    await realtime.hang_up(answer.session)


@pytest.mark.vcr
async def test_webrtc_browser_hang_up_ends_the_session(
    openai_live_ws_sideband_cassette: tuple[Provider[Any], RealtimeCassette],
    realtime_recording: bool,
) -> None:
    """A browser that hangs up ends the server's sideband session as any hangup does, without an error.

    Live reports the browser's lost connection as `session.closed` with `connection_lost`, about half a minute
    later, then drops the sideband without a closing handshake.
    """
    provider, _ = openai_live_ws_sideband_cassette
    model = OpenAILiveModel('gpt-live-1', provider=provider, settings=_FAST_TURN)
    realtime = Agent(_BACKEND, instructions='Answer in a few words.').realtime(model)

    @asynccontextmanager
    async def browser() -> AsyncGenerator[_BrowserPeer]:
        if realtime_recording:  # pragma: no cover
            async with _speaking_browser(b'') as peer:
                yield peer
            return
        yield REAL_SDP_OFFER, _no_browser_to_connect, lambda: None

    async with AsyncExitStack() as browser_call:
        offer, connect, _ = await browser_call.enter_async_context(browser())
        answer = await realtime.answer_webrtc_offer(offer)
        await connect(answer.sdp)
        async with realtime.session(provider_session=answer.session) as session:
            if realtime_recording:  # pragma: no cover
                await anyio.sleep(2)
            # The browser hangs up.
            await browser_call.aclose()
            with anyio.fail_after(90):
                # The browser said nothing, so the call ends without an event, and above all without an error.
                events = [event async for event in session]

    assert session.closed
    assert events == []


_FAVORITE_COLOR = [
    ModelRequest(parts=[UserPromptPart(content='My favorite color is turquoise.')]),
    ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Got it, turquoise.')]),
]


async def _ask_after_a_drop(
    model: OpenAILiveModel, cassette: RealtimeCassette, pcm: bytes, *, paced: bool
) -> tuple[list[Any], str]:
    """Drop the connection as soon as the session is up, then ask about the history from before the drop."""
    agent = Agent(_BACKEND, instructions='Answer in a few words.')
    events: list[Any] = []
    async with agent.realtime(model, message_history=_FAVORITE_COLOR).session() as session:
        await cassette.disconnect()
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                # The drop is the first thing that happens, so the reconnect is the first event.
                if isinstance(event, RealtimeSessionReconnectEvent):  # pragma: no branch
                    break
        await _stream(session, pcm, cassette, paced=paced)
        with anyio.fail_after(60):
            async for event in session:  # pragma: no branch
                events.append(event)
                if isinstance(event, RealtimeTurnCompleteEvent):
                    break
    reply = session.all_messages()[-1]
    assert isinstance(reply, ModelResponse)
    return events, ' '.join(part.transcript or '' for part in reply.parts if isinstance(part, SpeechPart))


def _session_starts(cassette: RealtimeCassette) -> list[dict[str, Any]]:
    return [
        interaction.data['session']
        for interaction in cassette.interactions
        if isinstance(interaction, CassetteMessage)
        and interaction.direction == 'sent'
        and interaction.data.get('type') == 'session.start'
    ]


async def test_a_stored_session_reconnects_by_forking(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """With `openai_live_store`, a dropped session is forked: the new one has the conversation, server-side."""
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(
            openai_live_store=True, openai_live_turn_silence_ms=1000, reconnect={'base_delay': 0.0, 'jitter': False}
        ),
    )
    pcm = assets_path.joinpath('favorite_color_question_24khz.pcm').read_bytes()

    events, answer = await _ask_after_a_drop(model, cassette, pcm, paced=realtime_recording)

    reconnects = [event for event in events if isinstance(event, RealtimeSessionReconnectEvent)]
    assert reconnects == [RealtimeSessionReconnectEvent(state_restored=True)]
    # The fork starts with no configuration of its own: it inherits the stored session's, history included.
    first, fork = _session_starts(cassette)
    assert first['store'] is True and len(first['input']) == 2
    assert fork == {}
    assert 'turquoise' in answer.lower()


async def test_a_session_that_is_not_stored_reconnects_by_replaying_its_history(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette], assets_path: Path, realtime_recording: bool
) -> None:
    """Without storage, the replacement session is seeded with the history so far, and nothing is stored."""
    provider, cassette = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(
            openai_live_turn_silence_ms=1000, reconnect={'base_delay': 0.0, 'jitter': False}
        ),
    )
    pcm = assets_path.joinpath('favorite_color_question_24khz.pcm').read_bytes()

    events, answer = await _ask_after_a_drop(model, cassette, pcm, paced=realtime_recording)

    reconnects = [event for event in events if isinstance(event, RealtimeSessionReconnectEvent)]
    assert reconnects == [RealtimeSessionReconnectEvent(state_restored=True)]
    first, replacement = _session_starts(cassette)
    assert 'store' not in first and 'store' not in replacement
    assert replacement['input'] == first['input']
    assert replacement['delegation'] == first['delegation']
    assert 'turquoise' in answer.lower()


@pytest.mark.realtime_ws_hold_open
async def test_text_sent_mid_reply_is_answered_by_that_reply(
    openai_live_ws_cassette: tuple[Provider[Any], RealtimeCassette],
) -> None:
    """Live speaks to text sent mid-reply in that same reply, not in one of its own.

    Live opens a delegation for the text (`session.delegation.created`, under the `response.create`'s id), and the
    reply under way goes on to answer it: that reply settles the text's reply obligation, so the wait for it ends
    with that reply instead of waiting for another that never comes.
    """
    provider, _ = openai_live_ws_cassette
    model = OpenAILiveModel(
        'gpt-live-1',
        provider=provider,
        settings=OpenAILiveModelSettings(openai_live_idle_audio=True, openai_live_turn_silence_ms=2500),
    )
    agent = Agent(_BACKEND, instructions='Keep replies short.')

    messages: list[ModelMessage] = []
    speaking = anyio.Event()

    async def watch(session: RealtimeSession) -> None:
        async for event in session:  # pragma: no branch
            if isinstance(event, PartStartEvent) and isinstance(event.part, SpeechPart):
                speaking.set()

    async with agent.realtime(model).session() as session, anyio.create_task_group() as tasks:
        tasks.start_soon(watch, session)
        await session.send('Tell me a three-sentence story about a cat.')
        with anyio.fail_after(60):
            await speaking.wait()
            await session.send('Actually, what is 2 plus 2?')
            await session.wait_for_reply()
        messages = session.all_messages()
        tasks.cancel_scope.cancel()

    assert [
        [part.transcript for part in message.parts if isinstance(part, SpeechPart)]
        if isinstance(message, ModelResponse)
        else [part.content for part in message.parts if isinstance(part, UserPromptPart)]
        for message in messages
        if isinstance(message, (ModelRequest, ModelResponse))
    ] == snapshot(
        [
            ['Tell me a three-sentence story about a cat.'],
            [
                ' Sure, I can give you a three sentence cat story. Milo the cat found a tiny golden key beneath the moonlit porch. It opened a secret garden filled with singing flowers. By dawn, Milo was home again, with one silver petal tucked behind his ear. 2 plus 2 is 4.'
            ],
            ['Actually, what is 2 plus 2?'],
        ]
    )
