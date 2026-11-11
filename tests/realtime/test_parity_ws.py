"""Canonical cross-provider parity matrix for the realtime abstraction.

Each case is a provider route and concrete model generation. The same public-API scenarios run
unchanged for every case: a text and a spoken tool round, a history-seeded follow-up, and spoken
multi-turn conversations (over a microphone that never stops streaming, with push-to-talk, with a
barge-in, and without input transcription) checked against the same history invariants. WebSocket cassettes preserve
the real provider conversations while keeping the default suite offline.

Provider-specific wire shapes belong in the provider cassette tests. This matrix deliberately asserts
only the normalized event, message, part, usage, and profile contracts users can rely on.
"""

from __future__ import annotations as _annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    InstructionDeltaPart,
    ModelRequest,
    ModelResponse,
    RealtimeSessionErrorEvent,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime import RealtimeModel, RealtimeModelSettings, RealtimeSession, RealtimeTurnCompleteEvent

from ..conftest import try_import
from .conversation import Utterance, assert_conversation_invariants, load_utterance, speak_continuously
from .ws_cassettes import RealtimeCassette

with try_import() as imports_successful:
    from pydantic_ai.providers import Provider
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.elevenlabs import ElevenLabsProvider
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.elevenlabs import ElevenLabsRealtimeModel
    from pydantic_ai.realtime.elevenlabs_agents import (
        ElevenLabsAgentsRealtimeModel,
        ElevenLabsAgentsRealtimeModelSettings,
    )
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings
    from pydantic_ai.realtime.openai_live import OpenAILiveModel, OpenAILiveModelSettings
    from pydantic_ai.realtime.xai import XaiRealtimeModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]

_Route = Literal[
    'openai',
    'openai-live',
    'azure',
    'xai',
    'elevenlabs',
    'google',
    'gateway-openai',
    'gateway-google',
    'elevenlabs-agents',
]
_ModelKind = Literal['openai', 'openai-live', 'azure', 'xai', 'elevenlabs', 'google', 'elevenlabs-agents']

# ElevenLabs Agents wraps a hosted agent, so its "model name" is the id of the agent the cassettes were recorded
# against (the id is not a secret and pins the recorded REST paths). To re-record, create the agents with
# `scripts/setup_elevenlabs_agents_test_workspace.py` and pass the environment it prints.
_ELEVENLABS_AGENT_ID = os.environ.get('ELEVENLABS_TEST_AGENT_ID', 'agent_7901m52mec0veknvz7vfnm56aaym')


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
    supports_session_seeding: bool = True
    """Whether prior history can be seeded into the session before its first turn.

    A hosted-agent platform whose conversation always starts fresh (ElevenLabs Agents: `contextual_update`
    injects context, not history) has nothing for the seeded-history scenario to run against.
    """
    audio_output_sample_rate: int = 24000
    """The PCM rate the model speaks at. Every model here speaks 24 kHz except an ElevenLabs agent on
    its default formats, which is 16 kHz both ways."""
    labels_phase: bool = False
    """Whether the model labels each assistant message `commentary` or `final_answer` (its `phase`)."""


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
        labels_phase=True,
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
        # Web search, run by the delegated backend.
        supports_native_tools=True,
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
        id='elevenlabs-current',
        model_kind='elevenlabs',
        model_name='eleven_realtime_v1_mini',
        route='elevenlabs',
        supports_image_input=False,  # `input_image` parts are dropped from the item
        supports_manual_turn_control=True,
        supports_interruption=True,
        supports_native_tools=False,
        supports_text_output=False,  # text output ignores audio input
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
        supports_text_output=False,  # the Developer API Live models reject a TEXT response modality
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
        supports_text_output=False,  # the Developer API Live models reject a TEXT response modality
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
        # The Vertex half-cascade model is the one Gemini Live model that answers in text.
        audio_input_sample_rate=16000,
    ),
    RealtimeParityCase(
        id='elevenlabs-agents',
        model_kind='elevenlabs-agents',
        model_name=_ELEVENLABS_AGENT_ID,
        route='elevenlabs-agents',
        supports_image_input=False,
        supports_manual_turn_control=False,
        supports_interruption=False,
        supports_native_tools=False,
        supports_text_output=True,  # the toggle-gated `text_only` override
        audio_input_sample_rate=16000,
        supports_session_seeding=False,
        audio_output_sample_rate=16000,
    ),
]

# A real microphone never stops. Server VAD only needs a beat of silence to hear the end of speech,
# and giving it much more makes it open further (empty) user turns. A model whose turn boundary is
# inferred instead needs the timeline to keep running long enough for the whole reply to arrive.
_TRAILING_SILENCE_FRAMES = 10
_INFERRED_BOUNDARY_SILENCE_FRAMES = 120


def _marks(case: RealtimeParityCase, *, unrecorded_on_azure: bool = False) -> tuple[pytest.MarkDecorator, ...]:
    """The marks a row needs: ElevenLabs' REST preflight records through HTTP VCR next to the WebSocket cassette.

    Our Azure realtime resource answers 401, so scenarios recorded after that (the spoken ones, and the raising-tool
    one) have no Azure recording. Their row is skipped rather than dropped, so the hole stays visible: record it (and
    delete this mark) once the Azure key works again. Azure's text scenario still runs from its existing recording.
    """
    marks: list[pytest.MarkDecorator] = []
    if case.route == 'elevenlabs-agents':
        marks.append(pytest.mark.vcr)
    if unrecorded_on_azure and case.route == 'azure':
        marks.append(pytest.mark.skip(reason='Azure realtime credentials return 401; cassette cannot be recorded'))
    return tuple(marks)


_CASES = [pytest.param((case, case.route), id=case.id, marks=_marks(case)) for case in REALTIME_PARITY_CASES]
_TEXT_CASES = [
    pytest.param((case, case.route), id=case.id, marks=_marks(case))
    for case in REALTIME_PARITY_CASES
    if case.drives_turns_with_text
]
_SEEDING_CASES = [
    pytest.param((case, case.route), id=case.id, marks=_marks(case))
    for case in REALTIME_PARITY_CASES
    if case.drives_turns_with_text and case.supports_session_seeding
]
_INSTRUCTION_CHANGE_CASES = [
    pytest.param((case, case.route), id=case.id, marks=_marks(case, unrecorded_on_azure=True))
    for case in REALTIME_PARITY_CASES
    if case.drives_turns_with_text
]
_AUDIO_CASES = [
    pytest.param((case, case.route), id=case.id, marks=_marks(case, unrecorded_on_azure=True))
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
        # Named after the `+`, the backend stays the one this row was recorded with.
        return OpenAILiveModel(
            f'{case.model_name}+gpt-5.6-sol',
            provider=provider,
            settings=OpenAILiveModelSettings(openai_live_turn_silence_ms=1000),
        )
    if case.model_kind == 'openai':
        return OpenAIRealtimeModel(case.model_name, provider=provider, settings=settings)
    if case.model_kind == 'azure':
        assert isinstance(provider, AzureProvider)
        return AzureRealtimeModel(case.model_name, provider=provider, settings=settings)
    if case.model_kind == 'xai':
        assert isinstance(provider, XaiProvider)
        return XaiRealtimeModel(case.model_name, provider=provider, settings=settings)
    if case.model_kind == 'elevenlabs':
        assert isinstance(provider, ElevenLabsProvider)
        return ElevenLabsRealtimeModel(case.model_name, provider=provider, settings=settings)
    if case.model_kind == 'elevenlabs-agents':
        # The hosted agent's tools are workspace state, so one dev agent (with `get_weather` attached)
        # serves every scenario, the tool-free ones included, with the connect-time tool comparison
        # off. That comparison is the provider's own contract, pinned by its cassette tests.
        elevenlabs_settings = ElevenLabsAgentsRealtimeModelSettings(elevenlabs_agents_tool_sync='off')
        if text_output and case.supports_text_output:
            elevenlabs_settings['output_modality'] = 'text'
        return ElevenLabsAgentsRealtimeModel(case.model_name, provider=provider, settings=elevenlabs_settings)
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
    assert profile.get('supports_session_seeding', False) is case.supports_session_seeding
    assert profile.get('audio_input_sample_rate', 24000) == case.audio_input_sample_rate
    assert profile.get('audio_output_sample_rate', 24000) == case.audio_output_sample_rate

    agent = Agent(instructions='Always call get_weather for a weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in London?')
        events = await _collect_complete_turn(session, after_tool_result=True)

    assert not any(isinstance(event, RealtimeSessionErrorEvent) for event in events)
    assert sum(isinstance(event, RealtimeTurnCompleteEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolCallEvent) for event in events) == 1
    assert sum(isinstance(event, FunctionToolResultEvent) for event in events) == 1
    messages = session.all_messages()
    assert [type(message) for message in messages[:3]] == [ModelRequest, ModelResponse, ModelRequest]
    # One tool round is exactly four messages, and one turn boundary, on every provider and route.
    # Vertex's `gemini-live-2.5-flash` closes the turn once when the tool-call generation ends and again
    # when it has spoken; the first boundary carries usage but no output, and is folded into the answer
    # rather than recorded as an empty response or reported as the end of the exchange.
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
    if case.labels_phase:
        # Kept on the part, under the key the Responses API reads it from when the history is handed off.
        assert [part.provider_details for part in messages[1].parts if isinstance(part, TextPart)] == [
            {'phase': 'commentary'}
        ]
        assert final_part.provider_details == {'phase': 'final_answer'}
        assert final_part.provider_name == 'openai'
    assert session.usage.requests >= 1
    assert session.usage.input_tokens >= 0
    assert session.usage.output_tokens >= 0


@pytest.mark.parametrize(
    'parity_ws_cassette',
    [
        pytest.param((case, case.route), id=case.id, marks=_marks(case, unrecorded_on_azure=True))
        for case in REALTIME_PARITY_CASES
        if case.drives_turns_with_text
    ],
    indirect=True,
)
async def test_raising_tool_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """A tool that raises is answered with the generic failure, so the model speaks to it and the session goes on.

    The exception still reaches the consumer, the turn ends once with the model's answer to the failure, and a
    later turn is answered as usual.
    """
    case, provider, _ = parity_ws_cassette
    model = _model(case, provider, text_output=True)
    agent = Agent(instructions='Always call get_weather for a weather question, then answer in one short sentence.')

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        raise RuntimeError(f'weather service unavailable for {city}')

    turns: list[list[Any]] = [[]]
    failing_turn = 0
    errors: list[BaseException] = []
    async with agent.realtime(model).session() as session:
        await session.send('What is the weather in London?')
        with anyio.fail_after(60):
            while True:
                try:
                    async for event in session:  # pragma: no branch
                        turns[-1].append(event)
                        last = session.all_messages()[-1]
                        if not (
                            isinstance(event, RealtimeTurnCompleteEvent)
                            and errors
                            and isinstance(last, ModelResponse)
                            and last.parts
                        ):
                            continue
                        if len(turns) == 2:
                            break
                        failing_turn = len(session.all_messages())
                        turns.append([])
                        await session.send('What is two plus two? Answer with just the number.')
                    break
                except RuntimeError as e:
                    # Raised from `async for`; iterating again goes on receiving.
                    errors.append(e)

    # The model may try the tool again; every attempt fails the same way.
    assert errors and all(str(error) == 'weather service unavailable for London' for error in errors)
    assert not any(isinstance(event, RealtimeSessionErrorEvent) for turn in turns for event in turn)
    assert [sum(isinstance(event, RealtimeTurnCompleteEvent) for event in turn) for turn in turns] == [1, 1]
    messages = session.all_messages()
    # Each call is answered with the generic failure, and the turn ended on the model's answer to it.
    returned = [
        part for message in messages[:failing_turn] for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert [(part.content, part.outcome) for part in returned] == [
        ('The tool raised an unhandled error.', 'failed')
    ] * len(errors)
    failure_answer = messages[failing_turn - 1]
    assert isinstance(failure_answer, ModelResponse) and failure_answer.parts
    assert not any(isinstance(part, ToolCallPart) for part in failure_answer.parts)
    final = messages[-1]
    assert isinstance(final, ModelResponse)
    final_part = final.parts[-1]
    assert isinstance(final_part, (SpeechPart, TextPart))
    answer = final_part.transcript if isinstance(final_part, SpeechPart) else final_part.content
    assert answer is not None and ('4' in answer or 'four' in answer.lower())


@pytest.mark.parametrize('parity_ws_cassette', _SEEDING_CASES, indirect=True)
async def test_history_seeding_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """Seeded user/assistant text precedes the live turn and affects every provider's answer.

    Driven by a text turn, so it covers the routes the text tool round does, less the ones whose
    sessions cannot be seeded at all. GPT-Live seeds history too, but has to be asked out loud;
    `test_openai_live_ws.py::test_history_seeding` covers it.
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


@pytest.mark.parametrize('parity_ws_cassette', _TEXT_CASES, indirect=True)
async def test_system_prompt_note_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """A system prompt sent mid-session with `respond=False` gets no reply of its own, but shapes the next one.

    Each provider delivers it on its own channel for the operator's text (a `system` message, a contextual
    update, or `<system>`-tagged user text), and history records it as the `SystemPromptPart` it is. GPT-Live,
    which has to be asked out loud, delivers it as a note to the voice model; its cassette tests cover it.
    """
    case, provider, _ = parity_ws_cassette
    model = _model(case, provider, text_output=True)
    agent = Agent(instructions='You are Sam, a phone support agent for Acme Telecom. Answer in one short sentence.')
    note = SystemPromptPart(content='The caller has been verified as the account holder, Jamie. Address them by name.')

    async with agent.realtime(model).session() as session:
        await session.send(note, respond=False)
        await session.send('Hi, can you tell me whose account this is?')
        events = await _collect_complete_turn(session)

    assert not any(isinstance(event, RealtimeSessionErrorEvent) for event in events)
    messages = session.all_messages()
    assert [type(message) for message in messages] == [ModelRequest, ModelRequest, ModelResponse]
    assert messages[0].parts == [note]
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    part = response.parts[-1]
    assert isinstance(part, (SpeechPart, TextPart))
    answer = part.transcript if isinstance(part, SpeechPart) else part.content
    assert answer is not None
    assert 'jamie' in answer.lower()


@pytest.mark.parametrize('parity_ws_cassette', _INSTRUCTION_CHANGE_CASES, indirect=True)
async def test_instruction_change_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """A dynamic instruction that changes is re-resolved after the turn and delivered before the next one.

    The change goes out on each provider's channel for the operator's text, as `send(SystemPromptPart(...))` does,
    and history records it as the `InstructionDeltaPart` a standard run records. GPT-Live, whose agent instructions
    are its backend's, updates the backend instead; its cassette tests cover it.
    """
    case, provider, _ = parity_ws_cassette
    model = _model(case, provider, text_output=True)
    code_word = {'value': 'BLUE'}
    agent = Agent(instructions='You are a voice assistant. Answer in one short sentence.')

    @agent.instructions(name='code_word')
    def current_code_word() -> str:
        return f'The code word is {code_word["value"]}. Tell the user when they ask for it.'

    async with agent.realtime(model).session() as session:
        await session.send('What is the code word?')
        # Changed while the first reply is under way: the session re-resolves the instructions once it's over.
        code_word['value'] = 'ORANGE'
        await _collect_complete_turn(session)
        await session.send('What is the code word now?')
        events = await _collect_complete_turn(session)

    assert not any(isinstance(event, RealtimeSessionErrorEvent) for event in events)
    messages = session.all_messages()
    changes = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, InstructionDeltaPart)
    ]
    assert [change.content for change in changes] == ['The code word is ORANGE. Tell the user when they ask for it.']
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    part = response.parts[-1]
    assert isinstance(part, (SpeechPart, TextPart))
    answer = part.transcript if isinstance(part, SpeechPart) else part.content
    assert answer is not None
    assert 'orange' in answer.lower()


@pytest.mark.parametrize('parity_ws_cassette', _AUDIO_CASES, indirect=True)
async def test_audio_tool_round_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
    request: pytest.FixtureRequest,
) -> None:
    """A spoken turn executes a local tool and records the same normalized four-message round.

    The text scenario is the portable one for every provider that takes a user text turn. This is the
    portable scenario for *voice*, which is the whole point of the surface, and it is the only one a
    model like GPT-Live can run at all. Both must produce the same history.
    """
    case, provider, cassette = parity_ws_cassette
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
    silence_frames = _INFERRED_BOUNDARY_SILENCE_FRAMES if case.synthesizes_turn_boundary else _TRAILING_SILENCE_FRAMES
    frames = [pcm[start : start + frame] for start in range(0, len(pcm), frame)] + [b'\x00' * frame] * silence_frames
    async with agent.realtime(model).session() as session:
        for chunk in frames:
            await cassette.before_audio_send()
            await session.send_audio(chunk)
            if realtime_recording:  # pragma: no branch
                # At a microphone's pace while recording: see `realtime_recording`.
                await anyio.sleep(0.1)  # pragma: no cover  # only while recording
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


def _conversation_cases(*, include: Callable[[RealtimeParityCase], bool]) -> list[Any]:
    """The parity cases a spoken-conversation scenario runs on, as picked by `include`."""
    return [
        pytest.param((case, case.route), id=case.id, marks=_marks(case, unrecorded_on_azure=True))
        for case in REALTIME_PARITY_CASES
        if include(case)
    ]


# The push-to-talk, barge-in, and transcription-off scenarios aren't recorded for GPT-Live, which reports
# no speech boundaries and takes no manual turns.
_CONVERSATION_CASES = _conversation_cases(include=lambda case: not case.synthesizes_turn_boundary)


# GPT-Live ends a turn after a stretch of wall-clock silence. Replay delivers a recording's frames back
# to back, but its cassette records when each frame arrived, and replay runs Live's turn clock on that
# recorded time: see `ws_cassettes.ReplayWebSocket.now`.
_CONVERSATION = [
    Utterance('my_name_is_alice', keyword='alice'),
    Utterance('weather_in_paris', keyword='paris'),
    Utterance('remind_me_my_name', keyword='remind'),
]
# Long enough for the reply, and the tool round, to finish before the next utterance starts.
_SILENCE_BETWEEN_TURNS = 8.0


@pytest.mark.parametrize('parity_ws_cassette', _AUDIO_CASES, indirect=True)
async def test_continuous_microphone_conversation_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """Three spoken turns over an always-on microphone record one user request per utterance, in order.

    Real voice apps stream silence between utterances rather than stopping the microphone, so the
    session has to find turn boundaries while audio keeps flowing, including through a tool round.
    """
    case, provider, cassette = parity_ws_cassette
    model = _model(case, provider)
    rate = case.audio_input_sample_rate
    agent = Agent(
        instructions='You are a voice assistant. Always call get_weather for a weather question. '
        'Answer in one short sentence.'
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    utterances = [load_utterance(assets_path, utterance, rate) for utterance in _CONVERSATION]
    async with agent.realtime(model).session() as session:
        await speak_continuously(
            session,
            utterances,
            sample_rate=rate,
            silence_after=_SILENCE_BETWEEN_TURNS,
            before_send=cassette.before_audio_send,
            pace=realtime_recording,
        )
        with anyio.fail_after(30):
            await session.wait_for_reply()

    assert_conversation_invariants(session, [utterance.keyword for utterance in _CONVERSATION])
    tool_calls = [
        part for message in session.all_messages() for part in message.parts if isinstance(part, ToolCallPart)
    ]
    assert [call.tool_name for call in tool_calls] == ['get_weather']


async def _wait_for_user_turns(session: RealtimeSession, count: int) -> None:
    """Wait until `count` user turns are in history: a transcript can land well after the reply it prompted."""
    with anyio.fail_after(30):
        while (
            sum(
                isinstance(part, SpeechPart) and part.speaker == 'user'
                for message in session.all_messages()
                for part in message.parts
            )
            < count
        ):
            await anyio.sleep(0.05)


@pytest.mark.realtime_ws_hold_open
@pytest.mark.parametrize(
    'parity_ws_cassette',
    _conversation_cases(include=lambda case: case.supports_manual_turn_control),
    indirect=True,
)
async def test_push_to_talk_conversation_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
    assets_path: Path,
) -> None:
    """Three push-to-talk turns record one user request per utterance, each ahead of its answer.

    With manual turn-taking nothing reports speech boundaries, and a turn's transcript routinely lands
    after the answer to it: text answers, where the model offers them, come back fastest.
    """
    case, provider, cassette = parity_ws_cassette
    model = _model(case, provider, text_output=True)
    rate = case.audio_input_sample_rate
    agent = Agent(
        instructions='You are a voice assistant. Always call get_weather for a weather question. '
        'Answer in one short sentence.'
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    utterances = [load_utterance(assets_path, utterance, rate) for utterance in _CONVERSATION]
    settings = RealtimeModelSettings(turn_detection=False)
    if case.model_kind == 'openai':
        # `whisper-1` streams no partial transcripts: each turn's arrives whole, and only once it's
        # transcribed, which makes it the slowest to catch up with the answer.
        settings['input_transcription_model'] = 'whisper-1'
    async with agent.realtime(model, model_settings=settings).session() as session:
        for pcm in utterances:
            await speak_continuously(
                session, [pcm], sample_rate=rate, silence_after=0, before_send=cassette.before_audio_send, pace=False
            )
            await session.commit_audio()
            await session.create_response()
            with anyio.fail_after(30):
                await session.wait_for_reply()
        await _wait_for_user_turns(session, len(_CONVERSATION))

    assert_conversation_invariants(session, [utterance.keyword for utterance in _CONVERSATION])


_BARGE_IN_CONVERSATION = [
    Utterance('tell_me_a_long_story', keyword='story'),
    Utterance('stop_and_say_goodbye', keyword='goodbye'),
]
# Long enough for the model to be well into its story, short enough to be still telling it.
_SILENCE_BEFORE_BARGE_IN = 4.0


@pytest.mark.realtime_ws_hold_open
@pytest.mark.parametrize('parity_ws_cassette', _CONVERSATION_CASES, indirect=True)
async def test_barge_in_conversation_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """Speaking over the model's answer files the interrupting turn after the answer it cut short."""
    case, provider, cassette = parity_ws_cassette
    model = _model(case, provider)
    rate = case.audio_input_sample_rate
    agent = Agent(
        instructions='You are a voice assistant. Asked for a story, tell it at length straight away, without questions.'
    )

    story, goodbye = (load_utterance(assets_path, utterance, rate) for utterance in _BARGE_IN_CONVERSATION)
    async with agent.realtime(model).session() as session:
        await speak_continuously(
            session,
            [story],
            sample_rate=rate,
            silence_after=_SILENCE_BEFORE_BARGE_IN,
            before_send=cassette.before_audio_send,
            pace=realtime_recording,
        )
        await speak_continuously(
            session,
            [goodbye],
            sample_rate=rate,
            silence_after=_SILENCE_BETWEEN_TURNS,
            before_send=cassette.before_audio_send,
            pace=realtime_recording,
        )
        with anyio.fail_after(30):
            await session.wait_for_reply()

    assert_conversation_invariants(session, [utterance.keyword for utterance in _BARGE_IN_CONVERSATION])
    story_response = next(message for message in session.all_messages() if isinstance(message, ModelResponse))
    # Grok Voice generates the whole story long before it has been played, so the provider reports no
    # response cut short there: whether history marks it interrupted is up to the local barge-in handling.
    if case.model_kind != 'xai':
        assert story_response.state == 'interrupted'


# Only the OpenAI rows. Gemini Live reports no speech boundaries, so without transcripts nothing tells an
# utterance from the silence around it. Grok Voice still transcribes the user's audio with transcription
# off, so a transcript-free history can't be asserted there either.
@pytest.mark.realtime_ws_hold_open
@pytest.mark.parametrize(
    'parity_ws_cassette',
    _conversation_cases(include=lambda case: case.model_kind in ('openai', 'azure')),
    indirect=True,
)
async def test_untranscribed_continuous_microphone_conversation_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
    assets_path: Path,
    realtime_recording: bool,
) -> None:
    """With input transcription off, an always-on microphone records exactly one user turn per utterance.

    The silence it streams while the model answers, and after the last answer, is no turn of its own.
    """
    case, provider, cassette = parity_ws_cassette
    model = _model(case, provider)
    rate = case.audio_input_sample_rate
    agent = Agent(
        instructions='You are a voice assistant. Always call get_weather for a weather question. '
        'Answer in one short sentence.'
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Look up the weather for a city."""
        return f'It is foggy and 12 degrees in {city}.'

    utterances = [load_utterance(assets_path, utterance, rate) for utterance in _CONVERSATION]
    settings = RealtimeModelSettings(input_transcription_model=None)
    async with agent.realtime(model, model_settings=settings).session() as session:
        await speak_continuously(
            session,
            utterances,
            sample_rate=rate,
            silence_after=_SILENCE_BETWEEN_TURNS,
            before_send=cassette.before_audio_send,
            pace=realtime_recording,
        )
        with anyio.fail_after(30):
            await session.wait_for_reply()

    assert_conversation_invariants(session, [None] * len(_CONVERSATION))
