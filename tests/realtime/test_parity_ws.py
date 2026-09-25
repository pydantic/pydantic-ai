"""Canonical cross-provider parity matrix for the realtime abstraction.

Each case is a provider route and concrete model generation. The same public-API scenarios run
unchanged for every case: a tool round, a history-seeded follow-up, and spoken multi-turn
conversations (over a microphone that never stops streaming, with push-to-talk, with a barge-in, and
without input transcription) checked against the same history invariants. WebSocket cassettes preserve
the real provider conversations while keeping the default suite offline.

Provider-specific wire shapes belong in the provider cassette tests. This matrix deliberately asserts
only the normalized event, message, part, usage, and profile contracts users can rely on.
"""

from __future__ import annotations as _annotations

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
    ModelRequest,
    ModelResponse,
    RealtimeSessionErrorEvent,
    SpeechPart,
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
    from pydantic_ai.providers.xai import XaiProvider
    from pydantic_ai.realtime.azure import AzureRealtimeModel
    from pydantic_ai.realtime.google import GoogleRealtimeModel
    from pydantic_ai.realtime.openai import OpenAIRealtimeModel, OpenAIRealtimeModelSettings
    from pydantic_ai.realtime.xai import XaiRealtimeModel

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='realtime provider dependencies not installed'),
]

_Route = Literal['openai', 'azure', 'xai', 'google', 'gateway-openai', 'gateway-google']
_ModelKind = Literal['openai', 'azure', 'xai', 'google']


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

_CASES = [pytest.param((case, case.route), id=case.id) for case in REALTIME_PARITY_CASES]


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


@pytest.mark.parametrize('parity_ws_cassette', _CASES, indirect=True)
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


@pytest.mark.parametrize('parity_ws_cassette', _CASES, indirect=True)
async def test_history_seeding_parity(
    parity_ws_cassette: tuple[RealtimeParityCase, Provider[Any], RealtimeCassette],
) -> None:
    """Seeded user/assistant text precedes the live turn and affects every provider's answer."""
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


def _conversation_cases(*, include: Callable[[RealtimeParityCase], bool]) -> list[Any]:
    """The parity cases a spoken-conversation scenario runs on, as picked by `include`.

    Our Azure realtime resource answers 401, so these scenarios could not be recorded for it. Its row is
    skipped rather than dropped, so the hole stays visible: record it (and delete this mark) once the
    Azure key works again.
    """
    return [
        pytest.param(
            (case, case.route),
            id=case.id,
            marks=pytest.mark.skip(reason='Azure realtime credentials return 401; cassette cannot be recorded')
            if case.route == 'azure'
            else (),
        )
        for case in REALTIME_PARITY_CASES
        if include(case)
    ]


_CONVERSATION_CASES = _conversation_cases(include=lambda case: True)
_CONVERSATION = [
    Utterance('my_name_is_alice', keyword='alice'),
    Utterance('weather_in_paris', keyword='paris'),
    Utterance('remind_me_my_name', keyword='remind'),
]
# Long enough for the reply, and the tool round, to finish before the next utterance starts.
_SILENCE_BETWEEN_TURNS = 8.0


@pytest.mark.parametrize('parity_ws_cassette', _CONVERSATION_CASES, indirect=True)
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
