"""The invariant checker's own bookkeeping, on a real simulation with its record adjusted by hand."""

from __future__ import annotations as _annotations

from dataclasses import replace

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.messages import ModelResponse, SpeechPart, TextPart
    from pydantic_ai.realtime.codec import ToolCall

    from . import _findings
    from ._invariants import is_cut_off_copy
    from ._openai_simulation import OpenAISimulation
    from ._simulation import InvariantViolation, Operation

pytestmark = pytest.mark.skipif(not imports_successful(), reason='realtime provider SDKs or hypothesis not installed')


def test_unexpected_error_is_reported_whichever_order_operations_finish() -> None:
    """An operation that finishes after a later one started is still judged, once."""
    with OpenAISimulation(strict=True) as sim:
        first = Operation(name='first', key=None, caller='main', issued=sim.truth.tick())
        second = Operation(name='second', key=None, caller='mic', issued=sim.truth.tick(), completed=sim.truth.tick())
        sim.operations += [first, second]
        sim.check()
        first.completed = sim.truth.tick()
        first.error = RuntimeError('boom')
        with pytest.raises(InvariantViolation, match=r"\[api\.unexpected_error\] first raised RuntimeError\('boom'\)"):
            sim.check()


def test_unexpected_consumer_error_is_reported_once() -> None:
    """An exception other than a `RealtimeError` escaping the session's iterator is a violation, judged once."""
    with OpenAISimulation(strict=True) as sim:
        sim.consumer_errors.append(RuntimeError('boom'))
        with pytest.raises(
            InvariantViolation, match=r"\[api\.unexpected_error\] iterating the session raised RuntimeError\('boom'\)"
        ):
            sim.check()
        sim.check()


def test_ambiguous_send_only_excuses_its_own_duplicate() -> None:
    """The client may resend what an ambiguous send delivered, but nothing else may arrive twice."""
    with OpenAISimulation(strict=False) as sim:
        sim.fail_next_send(fault='ambiguous')
        sim.send_text()
        sim.settle()
        sim.send_text()
        sim.settle()
        truth = sim.truth
        assert truth.ambiguous_inputs == {'t1'}
        t2 = truth.input('t2')
        assert t2 is not None
        truth.inputs.append(replace(t2))
        with pytest.raises(InvariantViolation, match=r"\[wire\.duplicate\] the server received 't2' 2 times"):
            sim.checker.check_at_rest()
        truth.inputs.remove(truth.inputs[-1])
        t1 = truth.input('t1')
        assert t1 is not None
        truth.inputs.append(replace(t1))
        sim.checker.check_at_rest()
        truth.inputs.append(replace(t1))
        with pytest.raises(InvariantViolation, match=r"\[wire\.duplicate\] the server received 't1' 3 times"):
            sim.checker.check_at_rest()


def test_codec_contract_violation_is_reported() -> None:
    """A connection breaking the codec contract fails the next step's check, naming the event that broke it."""
    with OpenAISimulation(strict=True) as sim:
        call = ToolCall(tool_call_id='call_1', tool_name='lookup', args='{}')
        checker = sim.checker
        checker._codec_issues += checker.lifecycle.feed(call) + checker.lifecycle.feed(call)  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(
            InvariantViolation,
            match=r"\[codec\.duplicate_tool_call\] tool call 'call_1' reported twice \(codec event #1\)",
        ):
            sim.check()


def test_a_reply_settled_by_a_response_that_never_saw_the_input_is_reported() -> None:
    """The session must settle an input with its own answer, not a response the server started before it had it."""
    with OpenAISimulation(strict=True) as sim:
        sim.send_audio()
        sim.speech_start()
        sim.speech_stop()
        sim.send_text(respond=False)
        session = sim.session
        assert session is not None
        (text,) = [input_ for input_ in sim.truth.inputs if input_.kind == 'text']
        assert text.client_index is not None
        # What a connection that took the provider's reply for the text's would make the session do.
        session._core._answered_by[text.client_index] = 'resp_1'  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(
            InvariantViolation, match=r"\[obligation\.misattributed\] 't1' is settled by resp_1, which the server"
        ):
            sim.check()


def test_reconnect_without_replay_loses_the_history() -> None:
    """A re-dial the session doesn't replay history into starts a conversation that knows nothing said before."""
    with OpenAISimulation(strict=True) as sim:
        session = sim.session
        assert session is not None
        # What a connection that neither resumes nor replays would do.
        session._connection._message_history = None  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        sim.send_text()
        sim.speak()
        sim.finish()
        sim.settle()
        sim.drop()
        sim.settle()
        sim.send_text()
        with pytest.raises(
            InvariantViolation,
            match=r"\[history\.not_restored\] connection 2 started resp_2 without \['assistant:r1w1', 'user:t1'\]",
        ):
            sim.settle()
        first, second = sim.server.sessions
        assert first.conversation.isdisjoint(second.conversation)


def test_findings_need_what_they_are_about() -> None:
    """A finding's trigger needs the input or response its violation names; a violation naming neither isn't one."""
    with OpenAISimulation(strict=True) as sim:
        sim.send_text()
        sim.call_tool()
        sim.settle()
        nothing = InvariantViolation('history.order', 'detail', sim.trace)
        assert not _findings._sent_before_reply_content(sim, nothing)  # pyright: ignore[reportPrivateUsage]
        assert not _findings._reply_taken_by_earlier_response(sim, nothing)  # pyright: ignore[reportPrivateUsage]
        spoken = InvariantViolation('wait.early', 'detail', sim.trace, {'response': 'resp_2'})
        assert not _findings._reply_taken_by_earlier_response(sim, spoken)  # pyright: ignore[reportPrivateUsage]
        # The text's reply started after the text went out: no earlier response took it.
        [text] = [input_.key for input_ in sim.truth.inputs if input_.kind == 'text']
        answered = InvariantViolation('wait.early', 'detail', sim.trace, {'input': text})
        assert not _findings._reply_taken_by_earlier_response(sim, answered)  # pyright: ignore[reportPrivateUsage]


def test_a_cut_off_copy_only_marks_the_response_interrupted_and_moves_cuts_back() -> None:
    """The one replacement history allows: the same response, interrupted, with no cut moved later."""
    recorded = ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='Hi.', interrupted_at_ms=40)])
    cut = replace(recorded, parts=[replace(recorded.parts[0], interrupted_at_ms=20)], state='interrupted')
    assert is_cut_off_copy(recorded, cut)
    assert not is_cut_off_copy(cut, replace(cut, parts=[replace(recorded.parts[0], interrupted_at_ms=80)]))
    assert not is_cut_off_copy(cut, replace(cut, parts=[replace(recorded.parts[0], interrupted_at_ms=None)]))
    assert not is_cut_off_copy(recorded, replace(cut, parts=[SpeechPart(speaker='assistant', transcript='Bye.')]))
    assert not is_cut_off_copy(recorded, replace(recorded, parts=[TextPart('Hi.')]))
