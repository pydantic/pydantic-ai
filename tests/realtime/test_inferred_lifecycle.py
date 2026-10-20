"""`InferredLifecycle`'s corners: the lifecycle it works out from what a connection sends and reads, in order.

The session-level behavior this makes up is covered in `test_session.py` and by the simulator; these pin the
corners no scripted conversation there reaches.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence

import pytest
from inline_snapshot import snapshot

from pydantic_ai.messages import (
    BinaryAudio,
    RealtimeInputSpeechEndEvent,
    RealtimeInputSpeechStartEvent,
    RealtimeResponseInterruptedEvent,
)
from pydantic_ai.realtime.codec import (
    CommitAudio,
    ConversationItemCreated,
    InputTranscript,
    OutputTranscript,
    RealtimeCodecEvent,
    ResponseDone,
    ToolCall,
    ToolCallCancelled,
)
from pydantic_ai.realtime.lifecycle import AnswersRevised, InferredLifecycle, LifecycleEvent, RealtimeFrame


def lifecycle(frame: RealtimeFrame | Sequence[LifecycleEvent]) -> list[str]:
    """The lifecycle events in `frame`, by type: what the codec events alone don't say."""
    return [type(event).__name__ for event in frame if isinstance(event, LifecycleEvent)]


def test_a_commit_joins_a_transcribed_turn_that_hasnt_joined_yet() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([InputTranscript('Hello')])) == snapshot(['UserTurnStarted'])
    tracker.input_sent(CommitAudio())
    assert lifecycle(tracker.take_pending()) == snapshot(['UserTurnEnded'])


def test_output_after_a_cut_off_response_is_a_response_of_its_own() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([OutputTranscript('Once upon')])) == snapshot(['ResponseStarted'])
    assert lifecycle(tracker.frame_for([RealtimeResponseInterruptedEvent()])) == snapshot([])
    assert lifecycle(tracker.frame_for([OutputTranscript('Sure.')])) == snapshot(['ResponseEnded', 'ResponseStarted'])


def test_a_call_after_one_whose_usage_follows_leaves_the_response_open() -> None:
    tracker = InferredLifecycle(transcribes=True)
    calls = [
        ToolCall(tool_call_id='c1', tool_name='a', args='{}', response_usage_follows=True),
        ToolCall(tool_call_id='c2', tool_name='b', args='{}'),
    ]
    assert lifecycle(tracker.frame_for(calls)) == snapshot(['ResponseStarted'])


def test_items_a_resumed_conversation_doesnt_replay_say_nothing() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([ConversationItemCreated(item_id='i1')])) == snapshot([])
    assert lifecycle(tracker.frame_for([ConversationItemCreated(replayed=True)])) == snapshot([])


def test_speech_starting_after_a_joined_turn_ends_it() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([InputTranscript('Hi')])) == snapshot(['UserTurnStarted'])
    assert lifecycle(tracker.frame_for([OutputTranscript('Hello.')])) == snapshot(['UserTurnEnded', 'ResponseStarted'])
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechStartEvent()])) == snapshot(['UserTurnClosed'])


def test_a_speech_end_naming_no_turn_joins_the_one_being_transcribed() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([InputTranscript('Hi')])) == snapshot(['UserTurnStarted'])
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechEndEvent()])) == snapshot(['UserTurnEnded'])


def test_a_named_speech_end_for_a_turn_already_joined_says_nothing() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechStartEvent(item_id='u1')])) == snapshot(['UserTurnStarted'])
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechEndEvent(item_id='u1')])) == snapshot(['UserTurnEnded'])
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechEndEvent(item_id='u1')])) == snapshot([])


def test_a_drop_ends_a_joined_turn_still_waiting_for_its_transcript() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechStartEvent(item_id='u1')])) == snapshot(['UserTurnStarted'])
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechEndEvent(item_id='u1')])) == snapshot(['UserTurnEnded'])
    assert lifecycle(tracker.connection_lost()) == snapshot(['UserTurnClosed'])


async def test_frames_take_an_async_iterable_with_nothing_to_close() -> None:
    class _Events:
        """An async iterator that is no generator."""

        def __init__(self) -> None:
            self._events: list[RealtimeCodecEvent] = [OutputTranscript('Hi.'), ResponseDone()]

        def __aiter__(self) -> _Events:
            return self

        async def __anext__(self) -> RealtimeCodecEvent:
            if not self._events:
                raise StopAsyncIteration
            return self._events.pop(0)

    tracker = InferredLifecycle(transcribes=True)
    assert [lifecycle(frame) async for frame in tracker.frames(_Events())] == snapshot(
        [['ResponseStarted'], ['ResponseEnded'], []]
    )


def test_a_commit_that_never_went_out_commits_nothing() -> None:
    tracker = InferredLifecycle(transcribes=False)
    tracker.input_sent(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    tracker.input_failed(tracker.input_sent(CommitAudio()))
    tracker.input_sent('Hello')
    assert lifecycle(tracker.take_pending()) == snapshot(['InputAdded'])


def test_a_failed_commit_of_a_transcribed_turn_leaves_it_to_join_later() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([InputTranscript('Hello')])) == snapshot(['UserTurnStarted'])
    tracker.input_failed(tracker.input_sent(CommitAudio()))
    assert lifecycle(tracker.take_pending()) == snapshot([])
    assert lifecycle(tracker.frame_for([OutputTranscript('Hi.')])) == snapshot(['UserTurnEnded', 'ResponseStarted'])


def test_speech_starting_in_a_turn_its_words_already_started_says_nothing() -> None:
    tracker = InferredLifecycle(transcribes=True)
    assert lifecycle(tracker.frame_for([InputTranscript('Hi')])) == snapshot(['UserTurnStarted'])
    assert lifecycle(tracker.frame_for([RealtimeInputSpeechStartEvent()])) == snapshot([])


class _TranscribesBeforeReplying(InferredLifecycle):
    untranscribed_audio_is_a_turn = False


@pytest.mark.parametrize(
    ('tracker', 'expected'),
    [
        pytest.param(InferredLifecycle(), ['UserTurnStarted', 'UserTurnEnded', 'ResponseStarted'], id='default'),
        pytest.param(_TranscribesBeforeReplying(), ['ResponseStarted'], id='transcribes-before-replying'),
    ],
)
def test_untranscribed_audio_a_reply_answers_is_a_turn_unless_the_provider_transcribes_first(
    tracker: InferredLifecycle, expected: list[str]
) -> None:
    tracker.input_sent(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    assert lifecycle(tracker.frame_for([OutputTranscript('Welcome!')])) == expected


def test_sending_reports_an_input_that_failed_to_go_out() -> None:
    tracker = InferredLifecycle()
    with pytest.raises(OSError), tracker.sending('Hello'):
        raise OSError
    assert tracker.take_pending() == []


def test_words_after_a_second_commit_close_the_turn_the_first_commit_joined() -> None:
    """Audio committed after a turn joined is a turn of its own: the earlier one is over once that one's words come."""
    tracker = InferredLifecycle(transcribes=True)
    audio = BinaryAudio(data=b'\x00\x00', media_type='audio/pcm')
    assert lifecycle(tracker.frame_for([InputTranscript('Hello')])) == snapshot(['UserTurnStarted'])
    tracker.input_sent(audio)
    tracker.input_sent(CommitAudio())
    assert lifecycle(tracker.take_pending()) == snapshot(['UserTurnEnded'])
    tracker.input_sent(audio)
    tracker.input_sent(CommitAudio())
    assert lifecycle(tracker.take_pending()) == snapshot([])
    assert lifecycle(tracker.frame_for([InputTranscript('Again')])) == snapshot(
        ['UserTurnClosed', 'UserTurnStarted', 'UserTurnEnded']
    )


class _CutOffLifecycle(InferredLifecycle):
    inputs_cut_off_replies = True


def revisions(frame: RealtimeFrame | Sequence[LifecycleEvent]) -> list[AnswersRevised]:
    return [event for event in frame if isinstance(event, AnswersRevised)]


def test_an_input_that_cuts_a_reply_off_is_left_to_the_next() -> None:
    """With nothing but typed turns going in, only an input can have cut the reply off: the later one did."""
    tracker = _CutOffLifecycle(transcribes=False)
    tracker.input_sent('Tell me a story.')
    tracker.input_sent('What is 2 plus 2?')
    tracker.take_pending()
    tracker.frame_for([OutputTranscript('Once upon')])
    assert revisions(tracker.frame_for([RealtimeResponseInterruptedEvent(), ResponseDone(interrupted=True)])) == [
        AnswersRevised(response_id='pydantic_ai_response_1', released=(1,))
    ]


def test_a_reply_the_user_speaks_over_keeps_every_input_it_answers() -> None:
    """The provider heard the user start speaking after the inputs went out: that cut the reply off, not an input."""
    tracker = _CutOffLifecycle(transcribes=False)
    tracker.input_sent('Tell me a story.')
    tracker.input_sent('What is 2 plus 2?')
    tracker.take_pending()
    tracker.frame_for([OutputTranscript('Once upon')])
    tracker.speech_heard()
    assert revisions(tracker.frame_for([RealtimeResponseInterruptedEvent(), ResponseDone(interrupted=True)])) == []


def test_a_cancellation_revises_only_the_response_whose_calls_it_names() -> None:
    tracker = _CutOffLifecycle(transcribes=False)
    tracker.input_sent('First.')
    tracker.take_pending()
    tracker.frame_for([ToolCall(tool_call_id='a', tool_name='lookup', args='{}')])
    tracker.input_sent('Second.')
    tracker.input_sent('Third.')
    tracker.take_pending()
    tracker.frame_for([ToolCall(tool_call_id='b', tool_name='lookup', args='{}')])
    # A late cancellation of the first response's call says nothing about the second's inputs...
    assert revisions(tracker.frame_for([ToolCallCancelled(['a'])])) == []
    # ...and leaves the second response's own cancellation to revise it.
    assert revisions(tracker.frame_for([ToolCallCancelled(['b'])])) == [
        AnswersRevised(response_id='pydantic_ai_response_2', released=(2,))
    ]


def test_a_microphone_streaming_silence_is_no_sign_the_user_spoke() -> None:
    """Audio going out says nothing: an always-on microphone streams silence through a typed turn too."""
    tracker = _CutOffLifecycle(transcribes=False)
    tracker.input_sent(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    tracker.input_sent('Tell me a story.')
    tracker.input_sent(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    tracker.input_sent('What is 2 plus 2?')
    tracker.input_sent(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    tracker.take_pending()
    tracker.frame_for([OutputTranscript('Once upon')])
    tracker.input_sent(BinaryAudio(data=b'\x00\x00', media_type='audio/pcm'))
    assert revisions(tracker.frame_for([RealtimeResponseInterruptedEvent(), ResponseDone(interrupted=True)])) == [
        AnswersRevised(response_id='pydantic_ai_response_1', released=(3,))
    ]
