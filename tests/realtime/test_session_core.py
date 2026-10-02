"""The new realtime session core, fed events and commands directly.

The session runs it in shadow on every OpenAI-protocol test in this suite (see `conftest.py`), and the
simulator judges it by its own invariants; these pin what it makes of the orderings neither reaches on
demand: a response that ends while another is streaming, a turn that is discarded after joining, a wait
that follows a tool round through to its answer.
"""

from __future__ import annotations as _annotations

from decimal import Decimal
from typing import Any

from inline_snapshot import snapshot

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime._core import (
    AudioCleared,
    AudioSent,
    Closed,
    CoreInput,
    ExchangeAbandoned,
    InputSent,
    InputWithdrawn,
    Interrupted,
    Owed,
    ReceiveEnded,
    SessionCore,
    ToolCallRefused,
    ToolReturned,
)
from pydantic_ai.realtime._lifecycle import (
    InputAdded,
    InputLost,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    UserTurnDiscarded,
    UserTurnEnded,
    UserTurnStarted,
)
from pydantic_ai.realtime.codec import (
    AudioDelta,
    InputRejected,
    InputTranscript,
    OutputTranscript,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
)
from pydantic_ai.usage import RequestUsage


def core(**kwargs: Any) -> SessionCore:
    return SessionCore(
        model_name=lambda: 'gpt-realtime',
        provider_name='openai',
        provider_url=None,
        conversation_id=None,
        run_id=None,
        **kwargs,
    )


def feed(session_core: SessionCore, *items: CoreInput) -> SessionCore:
    for item in items:
        session_core.apply(item)
    return session_core


def summary(messages: list[ModelMessage]) -> list[str]:
    """Each message as its kind and what it says, compactly."""

    def part_text(part: Any) -> str:
        if isinstance(part, SpeechPart):
            audio = '+audio' if part.audio is not None else ''
            cut = f'@{part.interrupted_at_ms}' if part.interrupted_at_ms is not None else ''
            return f'{part.speaker}:{part.transcript}{audio}{cut}'
        if isinstance(part, TextPart):
            return f'text:{part.content}'
        if isinstance(part, ToolCallPart):
            return f'call:{part.tool_call_id}'
        if isinstance(part, ToolReturnPart):
            return f'return:{part.tool_call_id}'
        assert isinstance(part, UserPromptPart)
        return f'prompt:{part.content}'

    lines: list[str] = []
    for message in messages:
        parts = ', '.join(part_text(part) for part in message.parts)
        if isinstance(message, ModelResponse):
            lines.append(f'{message.provider_response_id} [{parts}] {message.state} {message.finish_reason}')
        else:
            lines.append(f'{{{parts}}}')
    return lines


def text_request(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=text)])


def started(response_id: str, *answers: int) -> ResponseStarted:
    return ResponseStarted(response_id=response_id, answers=answers)


def ended(response_id: str, status: Any = 'completed', **kwargs: Any) -> ResponseEnded:
    return ResponseEnded(response_id=response_id, status=status, **kwargs)


def said(response_id: str | None, text: str, item_id: str | None = None, **kwargs: Any) -> OutputTranscript:
    return OutputTranscript(text, response_id=response_id, item_id=item_id, **kwargs)


def test_a_response_assembles_its_parts_and_is_recorded_once_it_ends() -> None:
    session_core = feed(
        core(retain_output_audio=True),
        started('r1'),
        said('r1', 'Hello', 'item_1'),
        said('r1', 'Hello there.', 'item_1', is_final=True),
        AudioDelta(b'\x00\x00', response_id='r1', item_id='item_1'),
        said('r1', 'Second item.', 'item_2'),
        said('r1', 'As text.', output_text=True),
        AudioDelta(b'', response_id='r1', item_id='item_3'),
        AudioDelta(b'', response_id='unknown'),
        SessionUsage(
            RequestUsage(input_tokens=3, output_tokens=4), provider_response_id='r1', provider_details={'x': 1}
        ),
    )
    assert session_core.all_messages() == []
    feed(session_core, ended('r1', finish_reason='stop', provider_details={'status': 'completed'}))
    (message,) = session_core.all_messages()
    assert isinstance(message, ModelResponse)
    assert summary([message]) == snapshot(
        ['r1 [assistant:Hello there.+audio, assistant:Second item., text:As text., assistant:None] complete stop']
    )
    assert (message.provider_details, message.usage.input_tokens, session_core.usage.requests) == snapshot(
        ({'x': 1, 'status': 'completed'}, 3, 1)
    )


def test_a_response_ended_by_the_connection_with_nothing_said_is_not_recorded() -> None:
    session_core = feed(core(), started('r1'), ended('r1', 'lost'), started('r2'), said('r2', 'Cut'))
    feed(session_core, Interrupted(played_ms=120), ended('r2', 'cancelled'), Interrupted(played_ms=5))
    assert summary(session_core.all_messages()) == snapshot(['r2 [assistant:Cut@120] interrupted None'])


def test_usage_goes_to_its_own_response_or_to_the_session() -> None:
    session_core = feed(
        core(responses_are_requests=False),
        started('r1'),
        SessionUsage(RequestUsage(input_tokens=1), provider_response_id='r1'),
        SessionUsage(RequestUsage(input_tokens=2), response_scoped=False),
        SessionUsage(RequestUsage(input_tokens=4), provider_response_id=None),
        ended('r1', provider_details={'status': 'completed'}),
        SessionUsage(RequestUsage(input_tokens=8), provider_response_id='r1'),
    )
    (message,) = session_core.all_messages()
    assert isinstance(message, ModelResponse)
    assert (message.usage.input_tokens, session_core.usage.input_tokens, session_core.usage.requests) == snapshot(
        (1, 15, 3)
    )


def test_a_tool_round_is_waited_for_until_its_answer_ends() -> None:
    session_core = feed(core(), InputSent(input_id=0, request=text_request('Weather?'), solicits=True))
    wait = session_core.wait_tokens()
    feed(
        session_core,
        InputAdded(input_id=0),
        started('r1', 0),
        ToolCall('call_1', tool_name='weather', args='{}', response_id='r1'),
        ToolCall('call_2', tool_name='weather', args='{}', response_id='unknown'),
        ended('r1', finish_reason='tool_call'),
    )
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='call', key='call_1', epoch=0)}))
    result = ModelRequest(parts=[ToolReturnPart(tool_name='weather', content='sunny', tool_call_id='call_1')])
    feed(
        session_core,
        InputSent(input_id=1, solicits=True, tool_call_id='call_1'),
        ToolReturned(tool_call_id='call_1', request=result),
    )
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='input', key='1', epoch=0)}))
    feed(session_core, started('r2', 1), said('r2', 'Sunny.'))
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='response', key='r2', epoch=0)}))
    feed(session_core, ended('r2'))
    assert session_core.still_owed(wait) == snapshot(frozenset())
    assert summary(session_core.all_messages()) == snapshot(
        [
            '{prompt:Weather?}',
            'r1 [call:call_1] complete tool_call',
            '{return:call_1}',
            'r2 [assistant:Sunny.] complete stop',
        ]
    )


def test_a_cancelled_call_owes_nothing_and_a_cancelled_response_leads_nowhere() -> None:
    session_core = feed(
        core(),
        started('r1'),
        ToolCall('call_1', tool_name='weather', args='{}', response_id='r1'),
        ToolCall('call_2', tool_name='weather', args='{}', response_id='r1'),
    )
    assert session_core.wait_tokens() == snapshot(
        frozenset(
            {
                Owed(kind='call', key='call_1', epoch=0),
                Owed(kind='call', key='call_2', epoch=0),
                Owed(kind='response', key='r1', epoch=0),
            }
        )
    )
    feed(session_core, ToolCallCancelled(['call_1']), ended('r1', 'cancelled'))
    assert session_core.reply_outstanding() is False


def test_obligations_settle_by_answer_refusal_loss_or_withdrawal() -> None:
    session_core = feed(
        core(),
        *(InputSent(input_id=index, solicits=True) for index in range(4)),
        ResponseRequestRefused(input_ids=(0,)),
        InputLost(input_ids=(1,)),
        InputWithdrawn(input_ids=(2,)),
    )
    assert session_core.wait_tokens() == snapshot(frozenset({Owed(kind='input', key='3', epoch=0)}))
    feed(session_core, ReceiveEnded())
    assert session_core.reply_outstanding() is False


def test_an_abandoned_exchange_is_not_waited_for_again() -> None:
    session_core = feed(core(), InputSent(input_id=0, solicits=True), started('r1', 0))
    wait = session_core.wait_tokens()
    feed(session_core, ExchangeAbandoned())
    assert (session_core.still_owed(wait), session_core.wait_tokens()) == snapshot((frozenset(), frozenset()))
    feed(session_core, InputSent(input_id=1, solicits=True))
    assert session_core.wait_tokens() == snapshot(frozenset({Owed(kind='input', key='1', epoch=1)}))


def test_inputs_join_history_where_the_provider_placed_them() -> None:
    first, second, refused, withdrawn = (text_request(text) for text in ('First.', 'Second.', 'Refused.', 'Gone.'))
    session_core = feed(
        core(),
        InputSent(input_id=0, request=first),
        InputSent(input_id=1, request=second, solicits=True),
        InputSent(input_id=2, request=refused),
        InputSent(input_id=3, request=withdrawn),
        started('r1'),
        InputAdded(input_id=0),
        InputAdded(input_id=0),
        InputAdded(input_id=9),
        InputRejected(2, refused='content'),
        InputRejected(3, refused='response'),
        said('r1', 'Hi.'),
        ended('r1'),
        started('r2', 1),
        ended('r2'),
        InputAdded(input_id=3),
        InputWithdrawn(input_ids=(3,)),
    )
    assert summary(session_core.all_messages()) == snapshot(
        ['r1 [assistant:Hi.] complete stop', '{prompt:First.}', '{prompt:Second.}', 'r2 [] complete stop']
    )


def test_a_spoken_turn_joins_where_it_was_committed_once_it_is_transcribed() -> None:
    session_core = feed(
        core(retain_input_audio=True),
        AudioSent(data=b'\x00\x00'),
        RealtimeInputSpeechStartEvent(item_id='u1'),
        UserTurnStarted(turn_id='u1'),
        RealtimeInputSpeechEndEvent(item_id='u1'),
        UserTurnEnded(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
        started('r1'),
        said('r1', 'Hm.'),
        ended('r1'),
    )
    assert session_core.all_messages() == []
    feed(
        session_core,
        InputTranscript('Hello ', item_id='u1'),
        InputTranscript('there', item_id='u1', is_final=True),
        InputTranscript('late', item_id='u1', is_final=True),
        InputTranscript('nobody', item_id='unknown', is_final=True),
        InputTranscript('anonymous', is_final=True),
        RealtimeInputSpeechEndEvent(item_id='unknown'),
        RealtimeInputSpeechEndEvent(),
        RealtimeInputTranscriptionErrorEvent(message='?', item_id='unknown'),
    )
    assert summary(session_core.all_messages()) == snapshot(
        ['{user:Hello there+audio}', 'r1 [assistant:Hm.] complete stop']
    )


def test_spoken_turns_without_transcripts() -> None:
    session_core = feed(
        core(input_transcription_enabled=False, retain_input_audio=True),
        AudioSent(data=b'\x01\x00'),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
        AudioSent(data=b'\x02\x00'),
        AudioCleared(),
        UserTurnStarted(turn_id='u2'),
        UserTurnEnded(turn_id='u2'),
    )
    assert summary(session_core.all_messages()) == snapshot(['{user:None+audio}', '{user:None}'])


def test_a_turn_that_joins_while_it_is_still_spoken_ends_with_the_speech() -> None:
    """xAI adds a spoken turn's item at speech start: the audio the user says after that is still the turn's."""
    session_core = feed(
        core(input_transcription_enabled=False, retain_input_audio=True),
        AudioSent(data=b'\x01\x00'),
        UserTurnStarted(turn_id='u1'),
        RealtimeInputSpeechStartEvent(item_id='u1'),
        UserTurnEnded(turn_id='u1'),
        AudioSent(data=b'\x02\x00' * 4),
    )
    assert session_core.all_messages() == []
    feed(session_core, RealtimeInputSpeechEndEvent(item_id='u1'))
    [request] = session_core.all_messages()
    part = request.parts[0]
    assert isinstance(part, SpeechPart) and part.audio is not None
    assert len(part.audio.data) == 44 + 10  # a WAV header, and every byte sent before the speech ended

    # One cleared while it was spoken, and one still spoken at close, end there with what they have.
    feed(
        session_core,
        RealtimeInputSpeechStartEvent(item_id='u2'),
        UserTurnStarted(turn_id='u2'),
        UserTurnEnded(turn_id='u2'),
        UserTurnDiscarded(turn_id='u2'),
        RealtimeInputSpeechStartEvent(item_id='u3'),
        UserTurnStarted(turn_id='u3'),
        UserTurnEnded(turn_id='u3'),
        Closed(),
    )
    assert summary(session_core.all_messages()) == snapshot(['{user:None+audio}', '{user:None}', '{user:None}'])


def test_failed_and_discarded_turns() -> None:
    session_core = feed(
        core(),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
        RealtimeInputTranscriptionErrorEvent(message='?', item_id='u1'),
        UserTurnStarted(turn_id='u2'),
        UserTurnDiscarded(turn_id='u2'),
        UserTurnDiscarded(turn_id='never'),
        UserTurnStarted(turn_id='u3'),
        UserTurnEnded(turn_id='u3'),
        UserTurnDiscarded(turn_id='u3'),
    )
    assert summary(session_core.all_messages()) == snapshot(['{user:None}', '{user:None}'])


def test_closing_settles_what_is_still_open() -> None:
    session_core = feed(
        core(),
        InputSent(input_id=0, request=text_request('Unacknowledged.'), solicits=True),
        UserTurnStarted(turn_id='u1'),
        InputTranscript('Half a sent', item_id='u1'),
        UserTurnStarted(turn_id='u2'),
        UserTurnEnded(turn_id='u2'),
        started('r1'),
        said('r1', 'Cut off'),
        ResponseDone(),
    )
    feed(session_core, Closed(), Closed())
    assert summary(session_core.all_messages()) == snapshot(
        ['{user:None}', 'r1 [assistant:Cut off] interrupted None', '{user:Half a sent}', '{prompt:Unacknowledged.}']
    )
    assert (session_core.reply_outstanding(), session_core.new_messages() == session_core.all_messages()) == snapshot(
        (False, True)
    )


def test_seeded_history_leads() -> None:
    seeded = text_request('Earlier.')
    assert core(seeded=[seeded]).all_messages() == [seeded]


def test_content_naming_no_response_goes_to_the_only_one_open() -> None:
    session_core = feed(
        core(),
        said(None, 'Nobody.'),
        started('r1'),
        OutputTranscript('Mine.', output_text=True),
        started('r2'),
        OutputTranscript('Ambiguous.', output_text=True),
        ended('r1'),
        ended('r2'),
    )
    assert summary(session_core.all_messages()) == snapshot(['r1 [text:Mine.] complete stop', 'r2 [] complete stop'])


def test_more_orderings() -> None:
    """A part learning its item late, a provider-priced response, a transcript before the commit, and more."""
    session_core = feed(
        core(retain_input_audio=True),
        started('r1'),
        said('r1', 'First'),
        said('r1', ' part', 'item_1'),
        SessionUsage(RequestUsage(input_tokens=1, cost=Decimal('0.5')), provider_response_id='r1'),
        ended('r1'),
        AudioSent(data=b'\x00\x00'),
        UserTurnStarted(turn_id='u1'),
        RealtimeInputSpeechEndEvent(item_id='u1'),
        AudioSent(data=b'\x01\x00'),
        RealtimeInputSpeechEndEvent(item_id='u1'),
        InputTranscript('Said before the commit.', item_id='u1', is_final=True),
        UserTurnEnded(turn_id='u1'),
        InputWithdrawn(input_ids=(7,)),
    )
    assert summary(session_core.all_messages()) == snapshot(
        ['r1 [assistant:First part] complete stop', '{user:Said before the commit.+audio}']
    )
    (response, _) = session_core.all_messages()
    assert isinstance(response, ModelResponse)
    assert (response.usage.cost, session_core.usage.cost) == snapshot((Decimal('0.5'), Decimal('0.5')))


def test_a_call_that_settles_without_a_result_owes_nothing() -> None:
    session_core = feed(
        core(), started('r1'), ToolCall('call_1', tool_name='boom', args='{}', response_id='r1'), ended('r1')
    )
    wait = session_core.wait_tokens()
    failure = ModelRequest(parts=[ToolReturnPart(tool_name='boom', content='failed', tool_call_id='call_1')])
    feed(session_core, ToolReturned(tool_call_id='call_1', request=failure))
    assert session_core.still_owed(wait) == snapshot(frozenset())


def test_a_call_the_session_refused_is_left_out() -> None:
    session_core = feed(
        core(),
        started('r1'),
        ToolCall('call_1', tool_name='lookup', args='{}', response_id='r1'),
        ToolCall('call_2', tool_name='lookup', args='{}', response_id='r1'),
        ToolCallRefused(tool_call_id='call_2'),
        ToolCallRefused(tool_call_id='unknown'),
        ended('r1'),
        ToolCallRefused(tool_call_id='call_1'),
    )
    assert summary(session_core.all_messages()) == snapshot(['r1 [call:call_1] complete stop'])


def test_a_turn_whose_transcript_can_no_longer_be_read_ends_with_what_it_has() -> None:
    session_core = feed(
        core(),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
        InputTranscript('Good', item_id='u1'),
        UserTurnStarted(turn_id='u2'),
        started('r1'),
        said('r1', 'Hm.'),
        ended('r1'),
    )
    assert session_core.all_messages() == []
    feed(session_core, ReceiveEnded())
    assert summary(session_core.all_messages()) == snapshot(['{user:Good}', 'r1 [assistant:Hm.] complete stop'])
