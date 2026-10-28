"""The new realtime session core, fed events and commands directly.

The session runs it in shadow on every OpenAI-protocol test in this suite (see `conftest.py`), and the
simulator judges it by its own invariants; these pin what it makes of the orderings neither reaches on
demand: a response that ends while another is streaming, a turn that is discarded after joining, a wait
that follows a tool round through to its answer.
"""

from __future__ import annotations as _annotations

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    PartEndEvent,
    PartStartEvent,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeInputTranscriptionErrorEvent,
    RealtimeResponseInterruptedEvent,
    RealtimeSessionReconnectEvent,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.realtime._core import (
    MAX_CUTTABLE_OUTPUT_ITEMS,
    AudioCleared,
    AudioSent,
    Closed,
    CoreInput,
    ExchangeAbandoned,
    ExchangeOver,
    InputSent,
    InputWithdrawn,
    Interrupted,
    Owed,
    ReceiveEnded,
    ResponseRefused,
    SessionCore,
    ToolCallRefused,
    ToolCallsPruned,
    ToolReturned,
    ToolRoundsOver,
    TranscriptOverdue,
)
from pydantic_ai.realtime._inferred_lifecycle import InferredLifecycle
from pydantic_ai.realtime._lifecycle import (
    AnswersRevised,
    InputAdded,
    InputLost,
    LifecycleEvent,
    OutputItemCut,
    OutputItemDetails,
    ResponseEnded,
    ResponseRequestRefused,
    ResponseStarted,
    StaleEvent,
    ToolCallsAbandoned,
    UserTurnClosed,
    UserTurnEnded,
    UserTurnStarted,
)
from pydantic_ai.realtime._retained_audio import RetainedAudioBudget
from pydantic_ai.realtime.codec import (
    AudioDelta,
    ConversationItemCreated,
    InputRejected,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    SessionUsage,
    ToolCall,
    ToolCallCancelled,
    ToolResult,
)
from pydantic_ai.realtime.openai_live import _LiveLifecycle  # pyright: ignore[reportPrivateUsage]
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


def lifecycle_frame(tracker: InferredLifecycle, session_core: SessionCore, *codec: RealtimeCodecEvent) -> None:
    """Apply what the tracker makes of one provider message to the core."""
    for event in tracker.frame_for(list(codec)):
        assert not isinstance(event, StaleEvent)
        session_core.apply(event)


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


def test_a_barge_in_on_a_response_that_ended_records_it_as_cut_off() -> None:
    """Generation outruns playback, so the reply the user speaks over has usually ended already.

    Each item the session truncated is recorded as cut off where the listener stopped hearing it, in an
    interrupted copy of the message at the same position: a snapshot already handed out doesn't change.
    """
    session_core = feed(
        core(),
        started('r1'),
        said('r1', 'First item.', 'item_a'),
        said('r1', 'Second item.', 'item_b'),
        ToolCall(tool_name='t', args='{}', tool_call_id='call-1', response_id='r1'),
        ended('r1'),
        ToolReturned(tool_call_id='call-1', request=ModelRequest(parts=[ToolReturnPart('t', 'ok', 'call-1')])),
        started('r2'),
        said('r2', 'Third item.', 'item_c'),
        ended('r2'),
    )
    before = session_core.all_messages()
    assert summary(before) == snapshot(
        [
            'r1 [assistant:First item., assistant:Second item., call:call-1] complete stop',
            '{return:call-1}',
            'r2 [assistant:Third item.] complete stop',
        ]
    )
    feed(session_core, Interrupted(played_ms=None, cuts=(('item_b', 120), ('item_c', 0), ('unknown', 5))))
    after = session_core.all_messages()
    assert summary(after) == snapshot(
        [
            'r1 [assistant:First item., assistant:Second item.@120, call:call-1] interrupted stop',
            '{return:call-1}',
            'r2 [assistant:Third item.@0] interrupted stop',
        ]
    )
    assert after[1] is before[1]
    assert summary(before)[0] == 'r1 [assistant:First item., assistant:Second item., call:call-1] complete stop'
    assert session_core.replayable_messages() == after

    # A later cut of the same item can only take it back further.
    feed(session_core, Interrupted(played_ms=None, cuts=(('item_b', 200),)))
    assert session_core.all_messages()[0] is after[0]
    feed(session_core, Interrupted(played_ms=None, cuts=(('item_b', 80),)))
    assert summary(session_core.all_messages())[0] == snapshot(
        'r1 [assistant:First item., assistant:Second item.@80, call:call-1] interrupted stop'
    )


def test_a_cut_of_a_response_still_under_way_lands_once_it_is_recorded() -> None:
    """A truncated item of a response still being generated is recorded as cut off, however the response ends."""
    session_core = feed(
        core(),
        started('r1'),
        said('r1', 'First item.', 'item_a'),
        said('r1', 'Second item.', 'item_b'),
        Interrupted(played_ms=None, cuts=(('item_a', 40), ('item_b', 0))),
        Interrupted(played_ms=None, cuts=(('item_a', 60),)),
        ended('r1'),
    )
    assert summary(session_core.all_messages()) == snapshot(
        ['r1 [assistant:First item.@40, assistant:Second item.@0] interrupted stop']
    )


def test_a_provider_reported_cut_records_what_was_heard() -> None:
    """A provider that says which words were heard (ElevenLabs) cuts the transcript down to them.

    The provider's text stays on the part, and an item heard not at all keeps no transcript. A cut of a response
    still under way lands once it is recorded, the latest words reported and the shortest hearing winning.
    """
    session_core = feed(
        core(),
        started('r1'),
        said('r1', 'First item.', 'item_a'),
        said('r1', 'Second item.', 'item_b'),
        OutputItemCut(item_id='item_a', heard_transcript='First...', heard_ms=300),
        OutputItemCut(item_id='item_a', heard_transcript='Fir...', heard_ms=500),
        ended('r1'),
    )
    before = session_core.all_messages()
    feed(session_core, OutputItemCut(item_id='item_b', heard_transcript=''))
    after = session_core.all_messages()
    assert summary(before) == snapshot(['r1 [assistant:Fir...@300, assistant:Second item.] interrupted stop'])
    assert summary(after) == snapshot(['r1 [assistant:Fir...@300, assistant:None] interrupted stop'])
    assert [part.provider_details for part in after[0].parts if isinstance(part, SpeechPart)] == snapshot(
        [{'original_transcript': 'First item.'}, {'original_transcript': 'Second item.'}]
    )

    # Reporting the same again changes nothing.
    feed(session_core, OutputItemCut(item_id='item_b', heard_transcript=''), OutputItemCut(item_id='item_a'))
    assert session_core.all_messages()[0] is after[0]


def test_only_the_latest_output_items_can_be_cut_off() -> None:
    session_core = core()
    for index in range(MAX_CUTTABLE_OUTPUT_ITEMS + 1):
        feed(session_core, started(f'r{index}'), said(f'r{index}', 'Hi.', f'item_{index}'), ended(f'r{index}'))
    before = session_core.all_messages()
    feed(session_core, Interrupted(played_ms=None, cuts=(('item_0', 0), (f'item_{MAX_CUTTABLE_OUTPUT_ITEMS}', 0))))
    after = session_core.all_messages()
    assert after[0] is before[0]
    assert summary(after[-1:]) == snapshot([f'r{MAX_CUTTABLE_OUTPUT_ITEMS} [assistant:Hi.@0] interrupted stop'])


def test_a_cut_off_response_keeps_its_retained_audio_in_the_budget() -> None:
    """The cut copy holds the same audio, so the budget still counts it and eviction still takes it."""
    session_core = feed(_budget_core(0.2), *_answered_turn(0))
    feed(session_core, Interrupted(played_ms=None, cuts=(('a0', 50),)))
    [_, response] = session_core.all_messages()
    assert isinstance(response, ModelResponse) and response.state == 'interrupted'
    feed(session_core, *_answered_turn(1))
    assert _retained(session_core) == [
        ('user', 'Question 0.', False),
        ('assistant', 'Answer 0.', False),
        ('user', 'Question 1.', True),
        ('assistant', 'Answer 1.', True),
    ]
    assert summary(session_core.all_messages()[1:2]) == snapshot(['r0 [assistant:Answer 0.@50] interrupted stop'])


def test_usage_goes_to_its_own_response_or_to_the_session() -> None:
    session_core = feed(
        core(responses_are_requests=False),
        started('r1'),
        SessionUsage(RequestUsage(input_tokens=1), provider_response_id='r1'),
        SessionUsage(RequestUsage(input_tokens=2), response_scoped=False),
        # Naming no response: the one under way.
        SessionUsage(RequestUsage(input_tokens=4), provider_response_id=None),
        ended('r1', provider_details={'status': 'completed'}),
        SessionUsage(RequestUsage(input_tokens=8), provider_response_id='r1'),
    )
    (message,) = session_core.all_messages()
    assert isinstance(message, ModelResponse)
    assert (message.usage.input_tokens, session_core.usage.input_tokens, session_core.usage.requests) == snapshot(
        (5, 15, 3)
    )


def test_usage_naming_no_response_between_responses_is_the_next_ones() -> None:
    """Gemini's usage names no response: reported between two (a boundary that said nothing else), it is the next's."""
    session_core = feed(
        core(),
        SessionUsage(RequestUsage(input_tokens=2), finish_reason='stop', provider_details={'x': 1}),
        ResponseStarted(response_id='made_up', provider_id=False),
        said('made_up', 'Hi.'),
        SessionUsage(RequestUsage(input_tokens=3)),
        ended('made_up'),
    )
    (message,) = session_core.all_messages()
    assert isinstance(message, ModelResponse)
    assert (message.usage.input_tokens, message.provider_response_id, message.provider_details) == snapshot(
        (5, None, {'x': 1})
    )


def test_native_tool_parts_lead_the_response() -> None:
    call = NativeToolCallPart(tool_name='web_search', args={'query': 'weather'}, tool_call_id='native_1')
    session_core = feed(
        core(),
        started('r1'),
        said('r1', 'Sunny.'),
        PartStartEvent(index=0, part=call),
        PartEndEvent(index=0, part=call),
        ended('r1'),
    )
    (message,) = session_core.all_messages()
    assert [type(part).__name__ for part in message.parts] == snapshot(['NativeToolCallPart', 'SpeechPart'])
    # A native part with no response under way has nothing to belong to.
    feed(session_core, PartStartEvent(index=1, part=call))
    assert len(session_core.all_messages()) == 1


def test_a_wait_follows_a_response_to_the_one_that_carries_it_on() -> None:
    """Gemini's extended-thinking model ends a filler saying the exchange goes on: the wait goes on with it."""
    session_core = feed(core(), InputSent(input_id=0, request=text_request('Weather?'), solicits=True))
    wait = session_core.wait_tokens()
    feed(session_core, started('r1', 0), said('r1', 'Let me see.'), ended('r1'))
    feed(session_core, ResponseStarted(response_id='r2', continues='r1'), said('r2', 'Still looking.'))
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='response', key='r2', epoch=0)}))
    feed(session_core, ended('r2'))
    assert session_core.still_owed(wait) == snapshot(frozenset())


def test_a_turn_heard_after_its_reply_started_joins_ahead_of_it() -> None:
    """Gemini transcribes the user after the model starts answering them: the turn goes before the reply."""
    session_core = feed(
        core(),
        started('r1'),
        said('r1', 'Sure.'),
        UserTurnStarted(turn_id='turn_1'),
        UserTurnEnded(turn_id='turn_1', before_response='r1'),
        InputTranscript('Can you help?', is_final=True),
        ended('r1'),
        # A response already over can't be joined ahead of any more: the turn goes at the end.
        UserTurnStarted(turn_id='turn_2'),
        UserTurnEnded(turn_id='turn_2', before_response='r1'),
        InputTranscript('Thanks.', is_final=True),
    )
    assert summary(session_core.all_messages()) == snapshot(
        ['{user:Can you help?}', 'r1 [assistant:Sure.] complete stop', '{user:Thanks.}']
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


def frame(session_core: SessionCore, *items: CoreInput) -> list[Any]:
    """Apply one frame's events, returning whether the exchange it ended is over (`ExchangeOver`)."""
    feed(session_core, *items)
    session_core.end_frame()
    return [effect for effect in session_core.take_effects() if isinstance(effect, ExchangeOver)]


def test_an_input_a_cut_off_response_releases_is_owed_until_the_next_answers_it() -> None:
    """Gemini cuts a reply off for an input that reaches it mid-generation: the next reply answers that input.

    Released, the input stays where it joined the conversation: no message that was visible moves.
    """
    first, second = text_request('Tell me a story.'), text_request('What is 2 plus 2?')
    session_core = feed(
        core(),
        InputSent(input_id=0, request=first, solicits=True),
        InputSent(input_id=1, request=second, solicits=True),
    )
    wait = session_core.wait_tokens()
    feed(session_core, started('r1', 0, 1), said('r1', 'Once upon'))
    before = session_core.all_messages()
    assert frame(session_core, AnswersRevised(response_id='r1', released=(1,)), ended('r1', 'cancelled')) == []
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='input', key='1', epoch=0)}))
    after = session_core.all_messages()
    assert after[: len(before)] == before
    assert frame(session_core, started('r2', 1), said('r2', 'Four.'), ended('r2')) == [ExchangeOver(interrupted=False)]
    assert session_core.still_owed(wait) == snapshot(frozenset())
    messages = session_core.all_messages()
    assert messages[: len(after)] == after
    assert summary(messages) == snapshot(
        [
            '{prompt:Tell me a story.}',
            '{prompt:What is 2 plus 2?}',
            'r1 [assistant:Once upon] interrupted None',
            'r2 [assistant:Four.] complete stop',
        ]
    )


def test_a_tool_calling_response_can_release_an_input_after_it_ended() -> None:
    """Gemini cancels the calls of a turn an input cuts off: the response that made them answers only what started it."""
    session_core = feed(
        core(),
        InputSent(input_id=0, request=text_request('Order 4411?'), solicits=True),
        InputSent(input_id=1, request=text_request('Also, 2 plus 2?'), solicits=True),
        started('r1', 0, 1),
        ToolCall('call_1', tool_name='lookup', args='{}', response_id='r1'),
        ended('r1', finish_reason='tool_call'),
    )
    wait = session_core.wait_tokens()
    feed(session_core, AnswersRevised(response_id='r1', released=(1,)), ToolCallCancelled(['call_1']))
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='input', key='1', epoch=0)}))
    # (A response that ended claims nothing more.)
    with pytest.raises(AssertionError):
        feed(session_core, AnswersRevised(response_id='r1', claimed=(1,)))


def test_abandoned_tool_calls_owe_nothing_though_they_run_on() -> None:
    """The provider answers none of the abandoned calls' results, sent or still to come: the exchange can end."""
    result = ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='shipped', tool_call_id='call_1')])
    session_core = feed(
        core(),
        InputSent(input_id=0, request=text_request('Order 4411, and 4412?'), solicits=True),
        started('r1', 0),
        ToolCall('call_1', tool_name='lookup', args='{}', response_id='r1'),
        ToolCall('call_2', tool_name='lookup', args='{}', response_id='r1'),
        ended('r1', finish_reason='tool_call'),
        InputSent(input_id=1, solicits=True, tool_call_id='call_1'),
        ToolReturned(tool_call_id='call_1', request=result),
    )
    wait = session_core.wait_tokens()
    assert session_core.result_asks_reply('call_2')
    feed(session_core, ToolCallsAbandoned(tool_call_ids=('call_1', 'call_2')))
    assert session_core.still_owed(wait) == frozenset()
    assert not session_core.reply_outstanding()
    # The call still running settles later: its result asks for no reply, and is recorded all the same.
    assert not session_core.result_asks_reply('call_2')
    feed(session_core, InputSent(input_id=2, solicits=True, tool_call_id='call_2'))
    assert not session_core.reply_outstanding()
    assert session_core.requests_ahead() == 0


def test_a_wait_taken_once_the_tool_result_is_out_still_follows_a_released_input() -> None:
    """A waiter that holds only the result input still sees an input the calling response later released."""
    result = ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='shipped', tool_call_id='call_1')])
    session_core = feed(
        core(),
        InputSent(input_id=0, request=text_request('Order 4411?'), solicits=True),
        InputSent(input_id=1, request=text_request('Also, 2 plus 2?'), solicits=True),
        started('r1', 0, 1),
        ToolCall('call_1', tool_name='lookup', args='{}', response_id='r1'),
        ended('r1', finish_reason='tool_call'),
        InputSent(input_id=2, solicits=True, tool_call_id='call_1'),
        ToolReturned(tool_call_id='call_1', request=result),
    )
    wait = session_core.wait_tokens()
    feed(session_core, AnswersRevised(response_id='r1', released=(1,)), InputLost(input_ids=(2,)))
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='input', key='1', epoch=0)}))


def test_a_response_under_way_can_claim_an_input_sent_during_it() -> None:
    """GPT-Live speaks to text placed mid-reply in that reply: it settles the text, which joins history after it."""
    session_core = feed(
        core(),
        InputSent(input_id=0, request=text_request('Tell me a story.'), solicits=True),
        InputAdded(input_id=0),
        started('r1', 0),
        said('r1', 'Once upon a time'),
        InputSent(input_id=1, request=text_request('What is 2 plus 2?'), solicits=True),
    )
    wait = session_core.wait_tokens()
    before = session_core.all_messages()
    # Claiming an input already settled, or releasing one the response doesn't answer, changes nothing.
    feed(session_core, AnswersRevised(response_id='r1', claimed=(0,), released=(1,)))
    assert session_core.still_owed(wait) == snapshot(
        frozenset({Owed(kind='input', key='1', epoch=0), Owed(kind='response', key='r1', epoch=0)})
    )
    feed(session_core, AnswersRevised(response_id='r1', claimed=(1,)), said('r1', '... and two plus two is four.'))
    assert session_core.still_owed(wait) == snapshot(frozenset({Owed(kind='response', key='r1', epoch=0)}))
    assert frame(session_core, ended('r1'), InputAdded(input_id=1)) == [ExchangeOver(interrupted=False)]
    messages = session_core.all_messages()
    assert messages[: len(before)] == before
    assert summary(messages) == snapshot(
        [
            '{prompt:Tell me a story.}',
            'r1 [assistant:Once upon a time... and two plus two is four.] complete stop',
            '{prompt:What is 2 plus 2?}',
        ]
    )


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
        UserTurnEnded(turn_id='u1', still_speaking=True),
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
        UserTurnEnded(turn_id='u2', still_speaking=True),
        UserTurnClosed(turn_id='u2'),
        RealtimeInputSpeechStartEvent(item_id='u3'),
        UserTurnStarted(turn_id='u3'),
        UserTurnEnded(turn_id='u3', still_speaking=True),
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
        UserTurnClosed(turn_id='u2'),
        UserTurnClosed(turn_id='never'),
        UserTurnStarted(turn_id='u3'),
        UserTurnEnded(turn_id='u3'),
        UserTurnClosed(turn_id='u3'),
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
    """Unless its response is recorded already: then it gets an interrupted return."""
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
    assert summary(session_core.all_messages()) == snapshot(['r1 [call:call_1] complete stop', '{return:call_1}'])


@pytest.mark.parametrize(
    ('speech_spans_tool_calls', 'expected'),
    [
        (False, ['r1 [assistant:Checking the, call:c1, call:c2, assistant: weather now.] complete stop']),
        (True, ['r1 [assistant:Checking the weather now., call:c1, call:c2] complete stop']),
    ],
)
def test_speech_that_goes_on_through_tool_calls_is_one_part_ahead_of_them(
    speech_spans_tool_calls: bool, expected: list[str]
) -> None:
    """Where the model talks while its tools run (GPT-Live), a call lands wherever it is in a sentence.

    Ending the speech there would cut the sentence in two, so it is one part, placed where it started.
    """
    session_core = feed(
        core(speech_spans_tool_calls=speech_spans_tool_calls),
        started('r1'),
        said('r1', 'Checking the'),
        ToolCall('c1', tool_name='weather', args='{}', response_id='r1'),
        ToolCall('c2', tool_name='weather', args='{}', response_id='r1'),
        said('r1', ' weather now.'),
        ended('r1'),
    )
    assert summary(session_core.all_messages()) == expected


def test_speech_spanning_tool_calls_keeps_its_place_when_an_earlier_call_is_refused() -> None:
    session_core = feed(
        core(speech_spans_tool_calls=True),
        started('r1'),
        ToolCall('c1', tool_name='weather', args='{}', response_id='r1'),
        said('r1', 'Checking.'),
        ToolCall('c2', tool_name='weather', args='{}', response_id='r1'),
        ToolCallRefused(tool_call_id='c1'),
        ended('r1'),
    )
    assert summary(session_core.all_messages()) == snapshot(['r1 [assistant:Checking., call:c2] complete stop'])


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


def test_an_output_items_details_go_on_its_part() -> None:
    """OpenAI's `phase`, for one: commentary on the way to a tool call, and the final answer after."""
    session_core = feed(
        core(),
        started('r1'),
        OutputItemDetails(response_id='r1', item_id='i1', provider_details={'phase': 'commentary'}),
        OutputItemDetails(response_id='r1', item_id='i2', provider_details={'phase': 'final_answer'}),
        OutputItemDetails(response_id='unknown', item_id='i3', provider_details={'phase': 'commentary'}),
        said('r1', 'Let me check.', item_id='i1', output_text=True),
        said('r1', 'Sunny.'),
        said('r1', ' Warm, too.', item_id='i2'),
        said('r1', 'No details.', item_id='i4'),
        ended('r1'),
    )
    [response] = session_core.all_messages()
    assert isinstance(response, ModelResponse)
    assert [(part.provider_name, part.provider_details) for part in response.parts] == snapshot(
        [('openai', {'phase': 'commentary'}), ('openai', {'phase': 'final_answer'}), ('openai', None)]
    )


def test_a_turn_whose_transcript_is_overdue_is_recorded_with_what_it_has() -> None:
    session_core = feed(
        core(),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
        InputTranscript('Partly', item_id='u1'),
        started('r1'),
        said('r1', 'Hm.'),
        ended('r1'),
    )
    assert session_core.transcript_holding_history() == 'u1'
    assert session_core.all_messages() == []
    feed(session_core, TranscriptOverdue(turn_id='u1'), TranscriptOverdue(turn_id='unknown'))
    assert session_core.transcript_holding_history() is None
    # The transcript, if it does come after all, changes nothing recorded.
    feed(session_core, InputTranscript(' said.', item_id='u1', is_final=True))
    assert summary(session_core.all_messages()) == snapshot(['{user:Partly}', 'r1 [assistant:Hm.] complete stop'])


def test_only_a_turn_with_something_ready_after_it_holds_history() -> None:
    session_core = feed(core(), UserTurnStarted(turn_id='u1'), UserTurnEnded(turn_id='u1'), started('r1'))
    # Nothing after the turn is ready yet: there is nothing to hold back.
    assert session_core.transcript_holding_history() is None
    feed(session_core, ended('r1', 'lost'))
    assert session_core.transcript_holding_history() is None  # (lost before it said anything: not recorded)
    feed(session_core, InputSent(input_id=0, request=text_request('Hi.')), InputAdded(input_id=0))
    assert session_core.transcript_holding_history() == 'u1'
    # Nor does a response under way, or a turn still spoken, ahead of it.
    other = feed(core(), started('r1'), InputSent(input_id=0, request=text_request('Hi.')), InputAdded(input_id=0))
    assert other.transcript_holding_history() is None
    spoken = feed(
        core(),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1', still_speaking=True),
        InputSent(input_id=0, request=text_request('Hi.')),
        InputAdded(input_id=0),
    )
    assert spoken.transcript_holding_history() is None


def test_clearing_the_audio_drops_a_turn_that_had_not_joined() -> None:
    session_core = feed(
        core(input_transcription_enabled=False),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1', still_speaking=True),
        UserTurnStarted(turn_id='u2'),
        AudioCleared(),
        Closed(),
    )
    # The turn that joined stays; the one still being said is gone, so closing records nothing for it.
    assert summary(session_core.all_messages()) == snapshot(['{user:None}'])


def test_a_response_under_way_when_reading_ends_is_recorded_as_cut_off() -> None:
    session_core = feed(core(), started('r1'), said('r1', 'Partly'), ReceiveEnded())
    assert summary(session_core.all_messages()) == snapshot(['r1 [assistant:Partly] interrupted None'])


def test_a_withdrawn_input_is_let_go() -> None:
    """An evicted image, say: nothing keeps what it carried alive, placed or not."""
    session_core = feed(
        core(),
        InputSent(input_id=0, request=text_request('Placed.')),
        InputAdded(input_id=0),
        InputSent(input_id=1, request=text_request('Not yet.')),
        InputWithdrawn(input_ids=(0, 1, 2)),
        InputAdded(input_id=1),
    )
    assert session_core.all_messages() == []
    assert not session_core._inputs and not session_core._unplaced  # pyright: ignore[reportPrivateUsage]
    assert session_core._placed == []  # pyright: ignore[reportPrivateUsage]


def test_a_repeated_or_replayed_call_is_recorded_once() -> None:
    """A natively resumed conversation can repeat a call of a response still open, and replay items history has."""
    call = ToolCall('call_1', tool_name='lookup', args='{}', response_id='r1')
    session_core = feed(
        core(),
        started('r1'),
        call,
        call,
        ConversationItemCreated(item_id='item_old', tool_call_id='call_old', replayed=True),
        ConversationItemCreated(tool_call_id='call_older', replayed=True),
        ConversationItemCreated(item_id='item_new'),
        ToolCall('call_old', tool_name='lookup', args='{}', response_id='r1'),
        said('r1', 'Replayed.', item_id='item_old'),
        ended('r1'),
    )
    assert summary(session_core.all_messages()) == snapshot(['r1 [call:call_1] complete stop'])


def test_a_new_call_whose_id_matches_a_replayed_item_is_recorded() -> None:
    """Item ids and call ids are different kinds of id: one replayed as an item says nothing about a call."""
    session_core = feed(
        core(),
        ConversationItemCreated(item_id='shared', replayed=True),
        started('r1'),
        ToolCall('shared', tool_name='lookup', args='{}', response_id='r1'),
        ended('r1'),
    )
    assert summary(session_core.all_messages()) == snapshot(['r1 [call:shared] complete stop'])


def test_a_turn_still_spoken_when_the_connection_is_replaced_is_over() -> None:
    """xAI added it at speech start; the speech end was lost with the old connection, and the new one won't hear it."""
    session_core = feed(
        core(input_transcription_enabled=False, retain_input_audio=True),
        RealtimeInputSpeechStartEvent(item_id='u1'),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1', still_speaking=True),
        AudioSent(data=b'\x01\x00'),
        RealtimeSessionReconnectEvent(state_restored=True),
        RealtimeSessionReconnectEvent(state_restored=True),
        started('r1'),
        said('r1', 'Hm.'),
        ended('r1'),
        UserTurnStarted(turn_id='u2'),
        UserTurnEnded(turn_id='u2'),
    )
    # It keeps the audio sent for it, which the next turn doesn't inherit.
    assert summary(session_core.all_messages()) == snapshot(
        ['{user:None+audio}', 'r1 [assistant:Hm.] complete stop', '{user:None}']
    )


# --- retained audio, bounded by `retain_audio_max_seconds` (as `RealtimeSession` bounds it) ---------------


def _tenth_of_a_second(value: int) -> bytes:
    """A tenth of a second of PCM16 at 24 kHz, every byte `value`, so each turn's audio is recognizable."""
    return bytes([value]) * 4800


def _retained(session_core: SessionCore) -> list[tuple[str, str | None, bool]]:
    """Each recorded speech part as what was said, and whether its audio is still retained."""
    return [
        (part.speaker, part.transcript, part.audio is not None)
        for message in session_core.all_messages()
        for part in message.parts
        if isinstance(part, SpeechPart)
    ]


def _answered_turn(turn: int) -> list[CoreInput]:
    """A spoken question, transcribed at once, and its spoken answer: a tenth of a second of audio each."""
    turn_id, response_id = f'u{turn}', f'r{turn}'
    return [
        AudioSent(data=_tenth_of_a_second(turn)),
        RealtimeInputSpeechStartEvent(item_id=turn_id),
        UserTurnStarted(turn_id=turn_id),
        RealtimeInputSpeechEndEvent(item_id=turn_id),
        UserTurnEnded(turn_id=turn_id),
        InputTranscript(f'Question {turn}.', item_id=turn_id, is_final=True),
        started(response_id),
        AudioDelta(_tenth_of_a_second(100 + turn), response_id=response_id, item_id=f'a{turn}'),
        said(response_id, f'Answer {turn}.', item_id=f'a{turn}'),
        ended(response_id),
    ]


def _budget_core(max_seconds: float | None, **kwargs: Any) -> SessionCore:
    return core(retain_input_audio=True, retain_output_audio=True, retain_audio_max_seconds=max_seconds, **kwargs)


def test_retained_audio_keeps_the_latest_output_audio() -> None:
    session_core = feed(
        core(retain_output_audio=True, retain_audio_max_seconds=0.02),
        started('r1'),
        *(AudioDelta(bytes([index]) * 480, response_id='r1') for index in range(5)),
        said('r1', 'A long answer.'),
        ended('r1'),
    )
    [response] = session_core.all_messages()
    [part] = response.parts
    assert isinstance(part, SpeechPart) and part.audio is not None
    # 20 ms at 24 kHz is 960 bytes: the last two of the five deltas, behind a WAV header.
    assert part.audio.data[44:] == bytes([3]) * 480 + bytes([4]) * 480


def test_retained_audio_keeps_the_latest_audio_of_a_turn_still_coming_in() -> None:
    session_core = feed(
        core(input_transcription_enabled=False, retain_input_audio=True, retain_audio_max_seconds=0.01),
        *(AudioSent(data=bytes([index]) * 240) for index in range(5)),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
    )
    [request] = session_core.all_messages()
    [part] = request.parts
    assert isinstance(part, SpeechPart) and part.audio is not None
    assert part.audio.data[44:] == bytes([3]) * 240 + bytes([4]) * 240


def test_retained_audio_evicts_the_oldest_and_keeps_transcripts_and_snapshots() -> None:
    session_core = feed(_budget_core(0.3), *_answered_turn(0))
    snapshot_after_first = session_core.all_messages()
    feed(session_core, *_answered_turn(1), *_answered_turn(2))

    # Three tenths of a second fit: the latest three, whichever side said them.
    assert _retained(session_core) == [
        ('user', 'Question 0.', False),
        ('assistant', 'Answer 0.', False),
        ('user', 'Question 1.', False),
        ('assistant', 'Answer 1.', True),
        ('user', 'Question 2.', True),
        ('assistant', 'Answer 2.', True),
    ]
    # A snapshot taken before an eviction doesn't change.
    assert [
        part.audio is not None
        for message in snapshot_after_first
        for part in message.parts
        if isinstance(part, SpeechPart)
    ] == [True, True]


def test_retained_audio_counts_turns_waiting_for_their_transcript_and_evicts_the_oldest() -> None:
    def spoken(turn: int) -> list[CoreInput]:
        return [
            AudioSent(data=_tenth_of_a_second(turn)),
            RealtimeInputSpeechStartEvent(item_id=f'u{turn}'),
            UserTurnStarted(turn_id=f'u{turn}'),
            RealtimeInputSpeechEndEvent(item_id=f'u{turn}'),
            UserTurnEnded(turn_id=f'u{turn}'),
        ]

    session_core = feed(
        core(retain_input_audio=True, retain_audio_max_seconds=0.15),
        *spoken(0),
        *spoken(1),
        # A repeated speech end cuts nothing new, for the evicted turn or the other one.
        AudioSent(data=bytes(960)),
        RealtimeInputSpeechEndEvent(item_id='u0'),
        InputTranscript('First.', item_id='u0', is_final=True),
        InputTranscript('Second.', item_id='u1', is_final=True),
    )
    assert _retained(session_core) == [('user', 'First.', False), ('user', 'Second.', True)]


def test_retained_audio_drops_a_cleared_or_discarded_turns_waiting_audio() -> None:
    session_core = feed(
        core(retain_input_audio=True, retain_audio_max_seconds=0.15),
        AudioSent(data=_tenth_of_a_second(0)),
        RealtimeInputSpeechStartEvent(item_id='u0'),
        UserTurnStarted(turn_id='u0'),
        RealtimeInputSpeechEndEvent(item_id='u0'),
        UserTurnClosed(turn_id='u0'),
        AudioSent(data=_tenth_of_a_second(1)),
        RealtimeInputSpeechStartEvent(item_id='u1'),
        UserTurnStarted(turn_id='u1'),
        RealtimeInputSpeechEndEvent(item_id='u1'),
        AudioCleared(),
        # Neither counts any more, so this one fits beside nothing else.
        AudioSent(data=_tenth_of_a_second(2)),
        UserTurnStarted(turn_id='u2'),
        UserTurnEnded(turn_id='u2'),
        InputTranscript('Third.', item_id='u2', is_final=True),
    )
    assert _retained(session_core) == [('user', 'Third.', True)]


class _CountingTurns(OrderedDict[str, Any]):
    """Counts every walk over the turns waiting with audio, which checking the budget on each chunk must not need."""

    walks = 0

    def __iter__(self) -> Iterator[str]:
        type(self).walks += 1
        return super().__iter__()


def test_retained_audio_check_does_not_walk_waiting_turns() -> None:
    session_core = _budget_core(1)
    session_core._waiting_turn_audio = _CountingTurns()  # pyright: ignore[reportPrivateUsage]
    for turn in range(20):
        feed(
            session_core,
            AudioSent(data=_tenth_of_a_second(turn)),
            RealtimeInputSpeechStartEvent(item_id=f'u{turn}'),
            UserTurnStarted(turn_id=f'u{turn}'),
            RealtimeInputSpeechEndEvent(item_id=f'u{turn}'),
            UserTurnEnded(turn_id=f'u{turn}'),
        )
    feed(session_core, *(AudioSent(data=bytes([index]) * 480) for index in range(10)))
    feed(session_core, *(InputTranscript(f'Q{turn}.', item_id=f'u{turn}', is_final=True) for turn in range(20)))

    assert _CountingTurns.walks == 0
    _ = list(session_core._waiting_turn_audio)  # pyright: ignore[reportPrivateUsage]
    assert _CountingTurns.walks == 1, 'the counter sees a walk'
    # A second holds the last nine waiting turns beside the 0.1 s streamed after them.
    assert [audio for _, _, audio in _retained(session_core)] == [turn >= 11 for turn in range(20)]


def test_retained_audio_passes_over_history_once_across_many_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each eviction resumes where the last one stopped, so many short turns cost work linear in their number."""
    strips = 0
    strip = RetainedAudioBudget.strip

    def counting_strip(self: RetainedAudioBudget, message: ModelMessage, excess: int) -> tuple[ModelMessage, int]:
        nonlocal strips
        strips += 1
        return strip(self, message, excess)

    monkeypatch.setattr(RetainedAudioBudget, 'strip', counting_strip)
    session_core = feed(_budget_core(0.2), *(item for turn in range(40) for item in _answered_turn(turn)))

    messages = session_core.all_messages()
    assert len(messages) == 80
    assert [audio for _, _, audio in _retained(session_core)] == [False] * 78 + [True] * 2
    # Starting every pass from the beginning of history would be about 80 * 80.
    assert strips <= 3 * len(messages)


def test_retained_audio_comes_back_for_a_message_recorded_after_the_eviction_passed_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response under way isn't skipped for good: once it is recorded, its audio is evicted too."""
    strips = 0
    strip = RetainedAudioBudget.strip

    def counting_strip(self: RetainedAudioBudget, message: ModelMessage, excess: int) -> tuple[ModelMessage, int]:
        nonlocal strips
        strips += 1
        return strip(self, message, excess)

    session_core = feed(_budget_core(0.2), *_answered_turn(0))
    monkeypatch.setattr(RetainedAudioBudget, 'strip', counting_strip)
    feed(
        session_core,
        started('r1'),
        AudioDelta(_tenth_of_a_second(101) * 2, response_id='r1', item_id='a1a'),
        # A second part closes the first, which the budget then tracks while the response is still under way.
        AudioDelta(b'\x01\x00', response_id='r1', item_id='a1b'),
        *(AudioSent(data=bytes(480)) for _ in range(30)),
    )
    # The chunks streamed meanwhile don't search history again each time: each recorded message is looked at
    # about once (the last one evicted from again, once).
    assert strips <= 3
    feed(session_core, ended('r1'), *_answered_turn(2))
    assert [audio for _, _, audio in _retained(session_core)] == [False, False, False, False, True, True]


def test_retained_audio_still_evicts_after_an_earlier_message_leaves_history() -> None:
    """An input withdrawn from ahead of where eviction resumes doesn't make it skip the next answer's audio."""

    def answer(turn: int) -> list[CoreInput]:
        response_id = f'r{turn}'
        return [
            started(response_id),
            AudioDelta(_tenth_of_a_second(100 + turn), response_id=response_id, item_id=f'a{turn}'),
            ended(response_id),
        ]

    session_core = feed(
        core(retain_output_audio=True, retain_audio_max_seconds=0.1),
        InputSent(input_id=0, request=text_request('An image, say.')),
        InputAdded(input_id=0),
        *answer(0),
        *answer(1),
        InputWithdrawn(input_ids=(0,)),
        *answer(2),
    )
    assert [audio for _, _, audio in _retained(session_core)] == [False, False, True]


def test_retained_audio_unbounded_keeps_it_all() -> None:
    session_core = feed(_budget_core(None), *(item for turn in range(3) for item in _answered_turn(turn)))
    assert all(audio for _, _, audio in _retained(session_core))


def test_a_conversation_id_resolved_late_reaches_what_the_core_recorded() -> None:
    session_core = feed(
        core(),
        *_answered_turn(0),
        InputSent(input_id=0, request=text_request('Typed.')),
        InputAdded(input_id=0),
        started('r9'),
        ToolCall('call_1', tool_name='lookup', args='{}', response_id='r9'),
        ended('r9'),
        ToolReturned(
            tool_call_id='call_1',
            request=ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='ok', tool_call_id='call_1')]),
        ),
        # Still being said, so not recorded yet.
        UserTurnStarted(turn_id='u9'),
        UserTurnEnded(turn_id='u9'),
    )
    session_core.set_conversation_id('c1')
    feed(session_core, InputTranscript('Late.', item_id='u9', is_final=True), *_answered_turn(1))
    assert {message.conversation_id for message in session_core.all_messages()} == {'c1'}


def test_a_reconnect_replays_a_held_turns_audio_without_counting_it() -> None:
    """The replay builds the held turn's request with its audio, which isn't retained history, so isn't counted."""
    session_core = feed(
        core(retain_input_audio=True, retain_audio_max_seconds=0.1),
        AudioSent(data=_tenth_of_a_second(1)),
        RealtimeInputSpeechStartEvent(item_id='u1'),
        UserTurnStarted(turn_id='u1'),
        RealtimeInputSpeechEndEvent(item_id='u1'),
        UserTurnEnded(turn_id='u1'),
    )
    [request] = session_core.replayable_messages()
    [part] = request.parts
    assert isinstance(part, SpeechPart) and part.audio is not None
    assert session_core._audio_budget.tracked_parts == 0  # pyright: ignore[reportPrivateUsage]


def test_retained_audio_eviction_passes_over_what_is_not_recorded() -> None:
    """A response lost before it said anything records nothing, and a turn waiting for its transcript is come back to."""
    session_core = feed(
        _budget_core(0.2),
        started('r_lost'),
        ended('r_lost', 'lost'),
        AudioSent(data=_tenth_of_a_second(5)),
        RealtimeInputSpeechStartEvent(item_id='u5'),
        UserTurnStarted(turn_id='u5'),
        RealtimeInputSpeechEndEvent(item_id='u5'),
        UserTurnEnded(turn_id='u5'),
        started('r6'),
        AudioDelta(_tenth_of_a_second(106), response_id='r6', item_id='a6'),
        said('r6', 'Answer 6.', item_id='a6'),
        ended('r6'),
        # Over budget: the eviction passes the lost response, waits on the turn, and evicts the answer after it.
        AudioSent(data=_tenth_of_a_second(7)),
        InputTranscript('Question 5.', item_id='u5', is_final=True),
    )
    assert _retained(session_core) == [('user', 'Question 5.', True), ('assistant', 'Answer 6.', False)]
    # The turn, recorded now, is next in line: its audio is the oldest left.
    assert list(session_core._recorded_audio) == [session_core._turns['u5']]  # pyright: ignore[reportPrivateUsage]


def test_retained_audio_trims_the_oldest_audio_coming_in_first() -> None:
    """A response whose incoming audio was emptied counts as newest when more comes in for it."""
    session_core = feed(
        core(retain_output_audio=True, retain_audio_max_seconds=0.1),
        started('r1'),
        started('r2'),
        AudioDelta(_tenth_of_a_second(1), response_id='r1', item_id='a1'),
        AudioDelta(_tenth_of_a_second(2), response_id='r2', item_id='a2'),
        AudioDelta(_tenth_of_a_second(3), response_id='r1', item_id='a1'),
        said('r1', 'One.', item_id='a1'),
        said('r2', 'Two.', item_id='a2'),
        ended('r1'),
        ended('r2'),
    )
    by_transcript = {
        part.transcript: part.audio.data[44:] if part.audio is not None else None
        for message in session_core.all_messages()
        for part in message.parts
        if isinstance(part, SpeechPart)
    }
    # r1's first tenth went first, then r2's (older than r1's second): only r1's newest audio fits.
    assert by_transcript == {'One.': _tenth_of_a_second(3), 'Two.': None}


def test_retained_audio_evicted_is_let_go_by_the_core_too() -> None:
    """Evicted audio isn't held anywhere else in the core: not as a turn's PCM, nor in a response's parts."""
    session_core = feed(_budget_core(0.2), *(item for turn in range(5) for item in _answered_turn(turn)))
    assert [audio for _, _, audio in _retained(session_core)] == [False] * 8 + [True] * 2
    turns = session_core._turns.values()  # pyright: ignore[reportPrivateUsage]
    responses = session_core._responses.values()  # pyright: ignore[reportPrivateUsage]
    assert all(not turn.audio for turn in turns)
    assert all(not response.parts for response in responses)
    # What is left: the last question and answer, in history alone.
    assert session_core._audio_budget.tracked_parts == 2  # pyright: ignore[reportPrivateUsage]
    assert len(session_core._recorded_audio) == 2  # pyright: ignore[reportPrivateUsage]


def test_input_audio_that_went_out_after_its_turn_was_recorded_is_left_to_the_next() -> None:
    """Audio is reported once its send completed, so a chunk still on its way when a turn is recorded comes after it.

    The provider ended that turn without the chunk, which is the next turn's start, or, if its send fails, no
    turn's: the recorded turn keeps only the audio that went out ahead of it either way.
    """
    session_core = feed(
        core(input_transcription_enabled=False, retain_input_audio=True),
        AudioSent(data=b'\x01\x00'),
        UserTurnStarted(turn_id='u1'),
        UserTurnEnded(turn_id='u1'),
        AudioSent(data=b'\x02\x00'),
        UserTurnStarted(turn_id='u2'),
        UserTurnEnded(turn_id='u2'),
    )
    assert [
        part.audio.data[44:]
        for message in session_core.all_messages()
        for part in message.parts
        if isinstance(part, SpeechPart) and part.audio is not None
    ] == [b'\x01\x00', b'\x02\x00']


def test_retained_audio_eviction_does_not_walk_what_waits_or_has_no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    """A turn waiting for its transcript ahead of many recorded messages doesn't make each chunk walk them again.

    Counts the strips, so the work of passing over messages; taking from the front of the queue is constant
    work as it is a deque.
    """
    strips = 0
    strip = RetainedAudioBudget.strip

    def counting_strip(self: RetainedAudioBudget, message: ModelMessage, excess: int) -> tuple[ModelMessage, int]:
        nonlocal strips
        strips += 1
        return strip(self, message, excess)

    session_core = feed(
        _budget_core(0.3),
        AudioSent(data=_tenth_of_a_second(9)),
        RealtimeInputSpeechStartEvent(item_id='held'),
        UserTurnStarted(turn_id='held'),
        RealtimeInputSpeechEndEvent(item_id='held'),
        UserTurnEnded(turn_id='held'),
        *(item for turn in range(20) for item in _answered_turn(turn)),
    )
    monkeypatch.setattr(RetainedAudioBudget, 'strip', counting_strip)
    feed(session_core, *(AudioSent(data=bytes(480)) for _ in range(50)))
    # Each chunk strips a message that still has audio, at most: it never passes over those that don't.
    assert strips <= 50


def test_retained_audio_eviction_keeps_a_message_that_still_has_audio_in_line() -> None:
    """A message with several parts that loses only its oldest part's audio stays first in line for the next eviction."""
    session_core = feed(
        _budget_core(0.2),
        started('r1'),
        AudioDelta(_tenth_of_a_second(1), response_id='r1', item_id='a1'),
        said('r1', 'One.', item_id='a1'),
        AudioDelta(_tenth_of_a_second(2), response_id='r1', item_id='a2'),
        said('r1', 'Two.', item_id='a2'),
        ended('r1'),
        AudioSent(data=_tenth_of_a_second(3)),
    )
    assert [(transcript, audio) for _, transcript, audio in _retained(session_core)] == [
        ('One.', False),
        ('Two.', True),
    ]
    assert list(session_core._recorded_audio) == [session_core._responses['r1']]  # pyright: ignore[reportPrivateUsage]


def test_a_turn_recorded_after_the_reply_to_it_is_evicted_before_that_reply() -> None:
    session_core = feed(
        _budget_core(1),
        AudioSent(data=_tenth_of_a_second(1)),
        RealtimeInputSpeechStartEvent(item_id='u1'),
        UserTurnStarted(turn_id='u1'),
        RealtimeInputSpeechEndEvent(item_id='u1'),
        UserTurnEnded(turn_id='u1'),
        started('r1'),
        AudioDelta(_tenth_of_a_second(101), response_id='r1', item_id='a1'),
        said('r1', 'Answer.', item_id='a1'),
        ended('r1'),
        InputTranscript('Question.', item_id='u1', is_final=True),
    )
    turn, response = session_core._turns['u1'], session_core._responses['r1']  # pyright: ignore[reportPrivateUsage]
    assert list(session_core._recorded_audio) == [turn, response]  # pyright: ignore[reportPrivateUsage]


def test_a_held_filler_keeps_the_wait_through_a_transcript_until_the_model_carries_on() -> None:
    """Gemini's extended-thinking model holds its filler open (`IN_PROGRESS`): the user's words then don't end the wait."""
    tracker = InferredLifecycle(transcribes=True)
    session_core = core()

    session_core.apply(InputSent(input_id=0, request=text_request('Weather?'), solicits=True))
    tracker.input_sent('Weather?')
    lifecycle_frame(tracker, session_core, OutputTranscript('Let me check.'))
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=1)), ResponseDone(more_expected=True))
    wait = session_core.wait_tokens()
    lifecycle_frame(tracker, session_core, InputTranscript('Hm'))
    assert session_core.still_owed(wait) != frozenset()
    lifecycle_frame(tracker, session_core, OutputTranscript('Still checking.'))
    assert session_core.still_owed(wait) != frozenset()
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=2)), ResponseDone())
    assert session_core.still_owed(wait) == frozenset()
    assert summary(session_core.all_messages()) == snapshot(
        [
            '{prompt:Weather?}',
            'None [assistant:Let me check.] complete stop',
            '{user:Hm}',
            'None [assistant:Still checking.] complete stop',
        ]
    )


def test_a_reply_the_provider_will_never_give_is_lost_once() -> None:
    """GPT-Live drops a tool result whose backend gave up: its reply is lost, unless a response already took it."""
    tracker = InferredLifecycle(transcribes=True)
    tracker.input_sent(ToolResult('call_1', output='sunny'))
    tracker.input_unanswerable(0)
    tracker.input_unanswerable(0)
    assert tracker.take_pending() == snapshot([InputAdded(input_id=0), InputLost(input_ids=(0,))])


def test_what_the_user_says_as_they_cut_in_is_a_turn_of_its_own() -> None:
    """Gemini reports the words that cut a reply off ahead of the cut, in one message: they are new speech."""
    tracker = InferredLifecycle(transcribes=True)
    session_core = core()

    lifecycle_frame(tracker, session_core, InputTranscript('First question'))
    lifecycle_frame(tracker, session_core, OutputTranscript('Let me check.'))
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=1)), ResponseDone(more_expected=True))
    lifecycle_frame(tracker, session_core, InputTranscript('Second question'), RealtimeResponseInterruptedEvent())
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=2)), ResponseDone(interrupted=True))
    lifecycle_frame(tracker, session_core, OutputTranscript('Sure.'))
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=3)), ResponseDone())
    assert summary(session_core.all_messages()) == snapshot(
        [
            '{user:First question}',
            'None [assistant:Let me check.] interrupted None',
            '{user:Second question}',
            'None [assistant:Sure.] complete stop',
        ]
    )
    # Likewise over a reply the model started on its own, before any transcript of what it answers came.
    lifecycle_frame(tracker, session_core, OutputTranscript('By the way...'))
    lifecycle_frame(tracker, session_core, InputTranscript('Stop'), RealtimeResponseInterruptedEvent())
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=4)), ResponseDone(interrupted=True))
    lifecycle_frame(tracker, session_core, OutputTranscript('Okay.'))
    lifecycle_frame(tracker, session_core, SessionUsage(RequestUsage(input_tokens=5)), ResponseDone())
    assert summary(session_core.all_messages()[4:]) == snapshot(
        ['None [assistant:By the way...] interrupted None', '{user:Stop}', 'None [assistant:Okay.] complete stop']
    )


def test_a_reply_taken_with_audio_starts_ahead_of_it() -> None:
    """GPT-Live's audio track runs on between replies, so its connection says where the model takes the turn."""
    tracker = _LiveLifecycle()
    assert [type(event).__name__ for event in tracker.frame_for([AudioDelta(b'\x00\x00')])] == snapshot(['AudioDelta'])
    tracker.turn_taken = True
    assert [type(event).__name__ for event in tracker.frame_for([AudioDelta(b'\x00\x10')])] == (
        snapshot(['ResponseStarted', 'AudioDelta'])
    )


def test_usage_naming_an_unknown_response_with_none_under_way_counts_toward_the_session_only() -> None:
    """A report naming no response the core knows, between responses, has no message to add its usage to."""
    session_core = feed(core(), SessionUsage(RequestUsage(input_tokens=3), provider_response_id='resp_unknown'))
    assert session_core.usage.input_tokens == 3
    assert session_core.all_messages() == []


def test_a_tool_result_the_backend_will_never_take_doesnt_join_the_conversation() -> None:
    """Held behind the reply under way, a result that goes nowhere stays out of the conversation when that reply ends."""
    tracker = InferredLifecycle(transcribes=True)
    tracker.frame_for([OutputTranscript('Let me check.')])
    input_id = tracker.input_sent(ToolResult('call_1', output='sunny'))
    tracker.input_unanswerable(input_id)
    assert [type(event).__name__ for event in tracker.frame_for([ResponseDone()])] == snapshot(
        ['InputLost', 'ResponseDone', 'ResponseEnded']
    )


@pytest.mark.parametrize('responses_are_requests', [True, False])
def test_a_refused_response_recorded_within_its_frame_is_taken_back(responses_are_requests: bool) -> None:
    """The session refuses a response as it handles the frame that started it, which can have ended it, audio and all."""
    session_core = feed(
        core(retain_output_audio=True, responses_are_requests=responses_are_requests),
        started('r1'),
        AudioDelta(b'\x00\x00', response_id='r1'),
        said('r1', 'Hi.', 'item_1'),
        ended('r1'),
        ResponseRefused(response_id='r1'),
        # A barge-in naming its item has nothing left to cut.
        Interrupted(played_ms=None, cuts=(('item_1', 0),)),
    )
    assert session_core.all_messages() == []
    assert session_core.usage.requests == 0


def test_a_refused_response_that_made_tool_calls_stays_for_its_calls_to_be_refused() -> None:
    session_core = feed(
        core(),
        started('r1'),
        ToolCall(tool_call_id='c1', tool_name='lookup', args='{}', response_id='r1'),
        ended('r1'),
        ResponseRefused(response_id='r1'),
    )
    assert summary(session_core.all_messages()) == snapshot(['r1 [call:c1] complete stop'])


def test_a_turn_still_transcribing_when_the_next_starts_holds_no_history_back() -> None:
    """Words naming no turn are the latest one's, so a joined turn the next one replaces has all of its words."""
    tracker = InferredLifecycle(transcribes=True)
    session_core = core()
    lifecycle_frame(tracker, session_core, RealtimeInputSpeechStartEvent())
    lifecycle_frame(tracker, session_core, InputTranscript('Hello'))
    lifecycle_frame(tracker, session_core, RealtimeInputSpeechEndEvent())
    lifecycle_frame(tracker, session_core, RealtimeInputSpeechStartEvent())
    lifecycle_frame(tracker, session_core, InputTranscript('again', is_final=True))
    lifecycle_frame(tracker, session_core, RealtimeInputSpeechEndEvent())
    lifecycle_frame(tracker, session_core, OutputTranscript('Hi.'), ResponseDone())
    assert session_core.transcript_holding_history() is None
    assert summary(session_core.all_messages()) == snapshot(
        ['{user:Hello}', '{user:again}', 'None [assistant:Hi.] complete stop']
    )


def test_a_lifecycle_event_the_core_does_not_know_is_refused() -> None:
    """`LifecycleEvent` is a base class a connection could subclass: the core says so rather than ignore it."""

    @dataclass(frozen=True, kw_only=True)
    class _Unknown(LifecycleEvent):
        pass

    with pytest.raises(TypeError, match='no handling for the lifecycle event'):
        core().apply(_Unknown())


def test_a_turn_dropped_after_its_transcript_started_streaming_ends_its_part() -> None:
    """A turn that never joined the conversation isn't recorded, but the part it streamed still ends."""
    session_core = feed(core(), UserTurnStarted(turn_id='u1'), InputTranscript('Hel', item_id='u1'), AudioCleared())
    assert [type(effect).__name__ for effect in session_core.take_effects()] == snapshot(
        ['PartStartEvent', 'PartDeltaEvent', 'PartEndEvent']
    )
    assert session_core.all_messages() == []


def test_tools_see_what_a_response_under_way_said_and_called() -> None:
    """A tool runs while the response that called it can still be under way: what it said and called is in view."""
    session_core = feed(
        core(),
        started('r1'),
        said('r1', 'Let me check.', 'item_1'),
        ToolCall(tool_call_id='c1', tool_name='lookup', args='{}', response_id='r1'),
        said('r1', 'Still checking.', 'item_2'),
    )
    assert summary(session_core.conversation_so_far()) == snapshot(
        ['r1 [assistant:Let me check., call:c1, assistant:Still checking.] complete None']
    )
    assert session_core.all_messages() == []


def test_a_native_part_ends_under_the_index_it_started_with_after_its_response_ended() -> None:
    """A connection can end a native part in a later frame than its response: it still closes the part it opened."""
    call = NativeToolCallPart(tool_name='web_search', args={}, tool_call_id='n1')
    session_core = feed(
        core(), started('r1'), PartStartEvent(index=7, part=call), ended('r1'), PartEndEvent(index=7, part=call)
    )
    indexes = [
        (type(effect).__name__, effect.index)
        for effect in session_core.take_effects()
        if isinstance(effect, (PartStartEvent, PartEndEvent))
    ]
    assert indexes == snapshot([('PartStartEvent', 0), ('PartEndEvent', 0)])


def test_stripping_a_message_without_tracked_audio_frees_nothing() -> None:
    message = ModelRequest(parts=[SpeechPart(speaker='user', transcript='Hi.')])
    budget = RetainedAudioBudget(10, input_sample_rate=24000, output_sample_rate=24000)
    assert budget.strip(message) == (message, 0)


def test_a_response_that_calls_no_tool_ends_the_tool_rounds_before_it() -> None:
    """Once a response completes without calling a tool, the answered calls before it are named, once; pruned ones
    stay in history, and leave the replay with the content their result carried."""
    returned = ModelRequest(
        parts=[
            ToolReturnPart(tool_name='photo', content='Taken.', tool_call_id='c1'),
            UserPromptPart(content='(photo)'),
        ]
    )
    session_core = feed(
        core(tracks_tool_rounds=True),
        started('r1'),
        said('r1', 'Let me look.', 'item_1'),
        ToolCall(tool_call_id='c1', tool_name='photo', args='{}', response_id='r1'),
        ended('r1'),
        InputSent(input_id=0, solicits=True, tool_call_id='c1'),
        ToolReturned(tool_call_id='c1', request=returned),
        started('r2', 0),
        said('r2', 'A kiwi.'),
    )
    session_core.take_effects()
    feed(session_core, ended('r2'), started('r3'), said('r3', 'Anything else?'), ended('r3'))
    rounds = [effect for effect in session_core.take_effects() if isinstance(effect, ToolRoundsOver)]
    assert [[call.tool_call_id for call in effect.calls] for effect in rounds] == [['c1']]

    feed(session_core, ToolCallsPruned(tool_call_ids=('c1',)))
    assert summary(session_core.replayable_messages()) == snapshot(
        [
            'r1 [assistant:Let me look.] complete stop',
            'r2 [assistant:A kiwi.] complete stop',
            'r3 [assistant:Anything else?] complete stop',
        ]
    )
    assert summary(session_core.all_messages()) == snapshot(
        [
            'r1 [assistant:Let me look., call:c1] complete stop',
            '{return:c1, prompt:(photo)}',
            'r2 [assistant:A kiwi.] complete stop',
            'r3 [assistant:Anything else?] complete stop',
        ]
    )


def test_a_pruned_round_with_nothing_else_said_leaves_the_replay_entirely() -> None:
    returned = ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='Found.', tool_call_id='c1')])
    session_core = feed(
        core(tracks_tool_rounds=True),
        started('r1'),
        ToolCall(tool_call_id='c1', tool_name='lookup', args='{}', response_id='r1'),
        ended('r1'),
        InputSent(input_id=0, solicits=True, tool_call_id='c1'),
        ToolReturned(tool_call_id='c1', request=returned),
        started('r2', 0),
        said('r2', 'Found it.'),
        ended('r2'),
        ToolCallsPruned(tool_call_ids=('c1',)),
    )
    assert summary(session_core.replayable_messages()) == snapshot(['r2 [assistant:Found it.] complete stop'])


def test_a_round_still_running_or_unanswered_waits_and_one_without_a_result_is_never_named() -> None:
    def returned(call_id: str) -> ModelRequest:
        return ModelRequest(parts=[ToolReturnPart(tool_name='t', content='ok', tool_call_id=call_id)])

    session_core = feed(
        core(tracks_tool_rounds=True),
        started('r1'),
        ToolCall(tool_call_id='c1', tool_name='t', args='{}', response_id='r1'),
        ToolCall(tool_call_id='c2', tool_name='t', args='{}', response_id='r1'),
        ToolCall(tool_call_id='c3', tool_name='t', args='{}', response_id='r1'),
        ended('r1'),
        InputSent(input_id=0, solicits=True, tool_call_id='c1'),
        ToolReturned(tool_call_id='c1', request=returned('c1')),
        # c3 settles without its result going out (a usage limit stopped it, say).
        ToolReturned(tool_call_id='c3', request=returned('c3')),
        started('r2', 0),
        ended('r2'),
    )

    def named() -> list[list[str]]:
        return [
            [call.tool_call_id for call in effect.calls]
            for effect in session_core.take_effects()
            if isinstance(effect, ToolRoundsOver)
        ]

    assert named() == [['c1']]
    feed(
        session_core,
        InputSent(input_id=1, solicits=True, tool_call_id='c2'),
        ToolReturned(tool_call_id='c2', request=returned('c2')),
        started('r3'),
        ended('r3'),
    )
    assert named() == []  # (c2's result isn't answered yet)
    feed(session_core, started('r4', 1), ended('r4'))
    assert named() == [['c2']]


def test_a_native_part_with_no_response_to_join_streams_neither_its_start_nor_its_end() -> None:
    """A native part the core drops (no single response under way) doesn't leave an end for a start never streamed."""
    call = NativeToolCallPart(tool_name='web_search', args={}, tool_call_id='n1')
    session_core = feed(core(), PartStartEvent(index=7, part=call), PartEndEvent(index=7, part=call))
    assert [type(effect).__name__ for effect in session_core.take_effects()] == []


class _CountingPlaced(list[Any]):
    """Counts the conversation entries the core looks at, however it reaches them."""

    visits = 0

    def __iter__(self) -> Iterator[Any]:
        for entry in super().__iter__():
            type(self).visits += 1
            yield entry

    def __getitem__(self, index: Any) -> Any:
        item = super().__getitem__(index)
        type(self).visits += len(item) if isinstance(index, slice) else 1
        return item


class _CountingResponses(dict[str, Any]):
    """Counts the responses the core looks at by walking them, rather than by id."""

    visits = 0

    def values(self) -> Any:
        for response in super().values():
            type(self).visits += 1
            yield response


def test_a_frame_costs_the_same_however_long_the_conversation_got() -> None:
    """What the session reads off the core after every frame doesn't walk the turns and responses that finished."""
    session_core = core()
    session_core._placed = _CountingPlaced()  # pyright: ignore[reportPrivateUsage]
    session_core._responses = _CountingResponses()  # pyright: ignore[reportPrivateUsage]
    costs: list[int] = []
    for turn in range(30):
        *answering, end = _answered_turn(turn)
        feed(session_core, *answering)
        _CountingPlaced.visits = _CountingResponses.visits = 0
        # A frame with content naming no response, and what the session reads off the core after each frame.
        feed(session_core, said(None, ' More.'))
        assert session_core.transcript_holding_history() is None
        assert session_core.response_holding_turns() is None
        assert session_core.speaking_part() is not None
        costs.append(_CountingPlaced.visits + _CountingResponses.visits)
        feed(session_core, end)

    assert len(session_core.all_messages()) == 60
    # Past the first, each frame also passes the answer the turn before finished, once, and nothing older.
    assert costs == [5] + [6] * 29
    _CountingPlaced.visits = _CountingResponses.visits = 0
    _ = list(session_core._placed), list(session_core._responses.values())  # pyright: ignore[reportPrivateUsage]
    assert (_CountingPlaced.visits, _CountingResponses.visits) == (60, 30), 'the counters see a walk'
