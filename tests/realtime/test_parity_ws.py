"""Canonical cross-provider parity matrix for the realtime abstraction.

Each case is a provider route and concrete model generation. The same public-API scenarios run
unchanged for every case: a tool round and a history-seeded follow-up. WebSocket cassettes preserve
the real provider conversations while keeping the default suite offline.

Provider-specific wire shapes belong in the provider cassette tests. This matrix deliberately asserts
only the normalized event, message, part, usage, and profile contracts users can rely on.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequest,
    ModelResponse,
    RealtimeSessionErrorEvent,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeModel, RealtimeTurnCompleteEvent

from ..conftest import try_import
from .ws_cassettes import RealtimeCassette

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings
    from pydantic_ai.realtime.openai_live import OpenAILiveModel, OpenAILiveModelSettings
    from pydantic_ai.realtime.xai import XaiRealtimeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]

_Route = Literal['openai', 'openai-live', 'azure', 'xai', 'google', 'gateway-openai', 'gateway-google']
_ModelKind = Literal['openai', 'openai-live', 'azure', 'xai', 'google']


@dataclass(frozen=True)
class RealtimeParityCase:
    """One concrete route/model entry in the canonical realtime parity matrix."""

    id: str
    model_kind: _ModelKind
    model_name: str
    route: _Route
    supports_image_input: bool
    supports_manual_turn_control: bool
    supports_interruption: bool
    supports_native_tools: bool
    supports_text_output: bool = True
    audio_input_sample_rate: int = 24000
    drives_turns_with_text: bool = True
    """Whether a text turn on its own can drive a whole exchange.

    GPT-Live has no user-message event: text reaches it as context placed on an audio timeline that
    only advances while audio flows, so text alone cannot start a turn. It runs the spoken scenario
    instead of the text one.
    """
    synthesizes_turn_boundary: bool = False
    """Whether the adapter infers the end of a turn rather than reading it off the wire.

    Where it does, the spoken scenario keeps a silent microphone running so the clock can run out,
    the way a real call would.
    """


# Adding a supported model generation is one row. Gateway routes intentionally have their own rows:
# they exercise distinct production routes even though their normalized behavior must match direct
# OpenAI and Gemini connections.
REALTIME_PARITY_CASES = [
    RealtimeParityCase(
        id='openai-current',
        model_kind='openai',
        model_name='gpt-realtime-2.1',
        route='openai',
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
    ),
    RealtimeParityCase(
        id='openai-previous',
        model_kind='openai',
        model_name='gpt-realtime',
        route='openai',
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
    ),
    RealtimeParityCase(
        id='openai-live',
        model_kind='openai-live',
        model_name='gpt-live-1',
        route='openai-live',
        supports_image_input=False,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_native_tools=False,
        supports_text_output=False,
        drives_turns_with_text=False,
        synthesizes_turn_boundary=True,
    ),
    RealtimeParityCase(
        id='azure-current',
        model_kind='azure',
        model_name='gpt-realtime',
        route='azure',
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
    ),
    RealtimeParityCase(
        id='xai-current',
        model_kind='xai',
        model_name='grok-voice-latest',
        route='xai',
        supports_image_input=False,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
        supports_text_output=False,  # Grok Voice always speaks
    ),
    RealtimeParityCase(
        id='xai-pinned',
        model_kind='xai',
        model_name='grok-voice-think-fast-1.0',
        route='xai',
        supports_image_input=False,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
        supports_text_output=False,  # Grok Voice always speaks
    ),
    RealtimeParityCase(
        id='google-current',
        model_kind='google',
        model_name='gemini-3.1-flash-live-preview',
        route='google',
        supports_image_input=True,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_native_tools=True,
        supports_text_output=False,  # every Gemini Live model rejects a TEXT response modality
        audio_input_sample_rate=16000,
    ),
    RealtimeParityCase(
        id='google-previous',
        model_kind='google',
        model_name='gemini-2.5-flash-native-audio-latest',
        route='google',
        supports_image_input=True,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_native_tools=True,
        supports_text_output=False,  # every Gemini Live model rejects a TEXT response modality
        audio_input_sample_rate=16000,
    ),
    RealtimeParityCase(
        id='gateway-openai',
        model_kind='openai',
        model_name='gpt-realtime',
        route='gateway-openai',
        supports_image_input=True,
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
    ),
    RealtimeParityCase(
        id='gateway-google',
        model_kind='google',
        model_name='gemini-live-2.5-flash',
        route='gateway-google',
        supports_image_input=True,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_native_tools=True,
        supports_text_output=False,  # every Gemini Live model rejects a TEXT response modality
        audio_input_sample_rate=16000,
    ),
]

# A real microphone never stops. Server VAD only needs a beat of silence to hear the end of speech,
# and giving it much more makes it open further (empty) user turns. A model whose turn boundary is
# inferred instead needs the timeline to keep running long enough for the whole reply to arrive.
_TRAILING_SILENCE_FRAMES = 10
_INFERRED_BOUNDARY_SILENCE_FRAMES = 120

_CASES = [pytest.param((case, case.route), id=case.id) for case in REALTIME_PARITY_CASES]
_TEXT_CASES = [
    pytest.param((case, case.route), id=case.id) for case in REALTIME_PARITY_CASES if case.drives_turns_with_text
]

# Our Azure realtime resource answers 401, so the spoken scenario could not be recorded for it. The
# row is skipped rather than dropped, so the hole stays visible: record it (and delete this mark)
# once the Azure key works again. Azure's text scenario still runs from its existing recording.
_AUDIO_CASES = [
    pytest.param(
        (case, case.route),
        id=case.id,
        marks=(
            pytest.mark.skip(reason='Azure realtime credentials return 401; cassette cannot be recorded')
            if case.route == 'azure'
            else ()
        ),
    )
    for case in REALTIME_PARITY_CASES
]


def _model(
    case: RealtimeParityCase,
    provider: Provider[Any],
    *,
    text_output: bool = False,
) -> RealtimeModel:
    # A text turn is only asked for where the model can produce one: `output_modality='text'` on a
    # model whose profile reports `supports_text_output=False` is a `UserError`, not a silent no-op,
    # so the table row decides — and `test_text_tool_round_parity` asserts the row matches the profile.
    settings = (
        OpenAIRealtimeModelSettings(output_modality='text') if text_output and case.supports_text_output else None
    )
    if case.model_kind == 'openai-live':
        # Live never produces text, so it never takes the `text_output` branch above.
        return OpenAILiveModel(
            case.model_name, provider=provider, settings=OpenAILiveModelSettings(openai_live_turn_silence_ms=1000)
        )
    if case.model_kind == 'openai':
        return OpenAIRealtimeModel(case.model_name, provider=provider, settings=settings)
    if case.model_kind == 'azure':
        assert isinstance(provider, AzureProvider)
        return AzureRealtimeModel(case.model_name, provider=provider, settings=settings)
    if case.model_kind == 'xai':
        assert isinstance(provider, XaiProvider)
        return XaiRealtimeModel(case.model_name, provider=provider, settings=settings)
    return GoogleRealtimeModel(case.model_name, provider=provider, settings=settings)


async def _collect_complete_turn(session: Any, *, after_tool_result: bool = False) -> list[Any]:
    events: list[Any] = []
    tool_result_seen = not after_tool_result
    with anyio.fail_after(45):
        async for event in session:  # pragma: no branch
            events.append(event)
            if isinstance(event, FunctionToolResultEvent):
                tool_result_seen = True
            elif (
                isinstance(event, RealtimeTurnCompleteEvent)
                and tool_result_seen
                and isinstance(session.all_messages()[-1], ModelResponse)
                and session.all_messages()[-1].parts
            ):
                # xAI can finish the mixed speech/tool-call response before it emits the tool result.
                # Newer OpenAI and Gemini models can also emit a completion marker for the tool-call
                # response after its local result. The portable boundary is a completed response that
                # has produced a normalized part, not a provider-specific count of completion events.
                break
    return events


@pytest.mark.parametrize('parity_ws_cassette', _TEXT_CASES, indirect=True)
async def test_text_tool_round_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """A text turn executes a local tool and records the same normalized four-message round."""
    case, provider, _ = parity_ws_cassette
    model = _model(case, provider, text_output=True)
    profile = model.profile
    assert profile.get('supports_image_input', False) is case.supports_image_input
    assert profile.get('supports_manual_turn_control', False) is case.supports_manual_turn_control
    assert profile.get('supports_interruption', False) is case.supports_interruption
    assert bool(profile.get('supported_native_tools', frozenset())) is case.supports_native_tools
    assert profile.get('supports_text_output', True) is case.supports_text_output
    assert profile.get('supports_session_seeding', False)
    assert profile.get('audio_input_sample_rate', 24000) == case.audio_input_sample_rate
    assert profile.get('audio_output_sample_rate', 24000) == 24000

    agent = Agent(instructions='Always call get_weather for a weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in London?')
        events = await _collect_complete_turn(session, after_tool_result=True)

    assert not any(isinstance(event, RealtimeSessionErrorEvent) for event in events)
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1
    messages = session.all_messages()
    assert [type(message) for message in messages[:3]] == [ModelRequest, ModelResponse, ModelRequest]
    # One tool round is exactly four messages on every provider and route. Gemini closes the turn once
    # when it takes the tool result and again when it has spoken; the first boundary carries usage but
    # no output, and is folded into the answer rather than recorded as an empty response.
    answer_responses = messages[3:]
    assert len(answer_responses) == 1
    final = answer_responses[-1]
    assert isinstance(final, ModelResponse) and final.parts
    assert isinstance(messages[1].parts[-1], ToolCallPart)
    assert isinstance(messages[2].parts[0], ToolReturnPart)
    final_part = final.parts[-1]
    assert isinstance(final_part, (SpeechPart, TextPart))
    answer = final_part.transcript if isinstance(final_part, SpeechPart) else final_part.content
    assert answer is not None and 'fog' in answer.lower()
    assert session.usage.requests >= 1
    assert session.usage.input_tokens >= 0
    assert session.usage.output_tokens >= 0


@pytest.mark.parametrize('parity_ws_cassette', _TEXT_CASES, indirect=True)
async def test_history_seeding_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """Seeded user/assistant text precedes the live turn and affects every provider's answer.

    Driven by a text turn, so it covers the same routes the text tool round does. GPT-Live seeds
    history too, but has to be asked out loud; `test_openai_live_ws.py::test_history_seeding` covers it.
    """
    case, provider, _ = parity_ws_cassette
    model = _model(case, provider, text_output=True)
    history = [
        ModelRequest(parts=[UserPromptPart(content='My name is Alice and my favorite color is teal.')]),
        ModelResponse(parts=[TextPart(content='Nice to meet you, Alice!')]),
    ]
    agent = Agent(instructions='Answer in one short sentence.')

    async with agent.realtime(model, message_history=history).session() as session:
        await session.send('What is my name and favorite color?')
        events = await _collect_complete_turn(session)

    assert not any(isinstance(event, RealtimeSessionErrorEvent) for event in events)
    messages = session.all_messages()
    assert messages[:2] == history
    assert [type(message) for message in messages[2:]] == [ModelRequest, ModelResponse]
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    part = response.parts[-1]
    assert isinstance(part, (SpeechPart, TextPart))
    answer = part.transcript if isinstance(part, SpeechPart) else part.content
    assert answer is not None
    assert 'alice' in answer.lower() and 'teal' in answer.lower()


@pytest.mark.parametrize('parity_ws_cassette', _AUDIO_CASES, indirect=True)
async def test_audio_tool_round_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette], assets_path: Path
) -> None:
    """A spoken turn executes a local tool and records the same normalized four-message round.

    The text scenario is the portable one for every provider that takes a user text turn. This is the
    portable scenario for *voice*, which is the whole point of the surface, and it is the only one a
    model like GPT-Live can run at all. Both must produce the same history.
    """
    case, provider, _ = parity_ws_cassette
    model = _model(case, provider)
    profile = model.profile
    assert profile.get('synthesizes_turn_boundary', False) is case.synthesizes_turn_boundary
    rate = profile.get('audio_input_sample_rate', 24000)
    assert rate == case.audio_input_sample_rate

    agent = Agent(instructions='Always call get_weather for a weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    pcm = assets_path.joinpath(f'weather_question_{rate // 1000}khz.pcm').read_bytes()
    frame = rate // 10 * 2  # 100 ms of 16-bit mono audio
    async with agent.realtime(model).session() as session:
        for start in range(0, len(pcm), frame):
            await session.send_audio(pcm[start : start + frame])
        silence_frames = (
            _INFERRED_BOUNDARY_SILENCE_FRAMES if case.synthesizes_turn_boundary else _TRAILING_SILENCE_FRAMES
        )
        for _ in range(silence_frames):
            await session.send_audio(b'\x00' * frame)
        events = await _collect_complete_turn(session, after_tool_result=True)

    assert not any(isinstance(event, RealtimeSessionErrorEvent) for event in events)
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1

    messages = session.all_messages()
    # Unlike a text turn, a spoken one is segmented by the provider's own voice-activity detection, so
    # how many user turns a single utterance becomes is a provider (and pause) detail, not a contract.
    # What every provider must agree on is the round itself: the user spoke, a tool ran on the result,
    # and the model answered.
    assert isinstance(messages[0], ModelRequest)
    assert any(isinstance(part, SpeechPart) and part.speaker == 'user' for part in messages[0].parts)
    tool_calls = [part for message in messages for part in message.parts if isinstance(part, ToolCallPart)]
    tool_returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    assert len(tool_calls) == 1
    assert len(tool_returns) == 1 and tool_returns[0].tool_name == 'get_weather'
    # The answer comes after the tool result, and carries content.
    final = messages[-1]
    assert isinstance(final, ModelResponse) and final.parts
    assert isinstance(messages[-2], ModelRequest)
    assert isinstance(messages[-2].parts[0], ToolReturnPart)
