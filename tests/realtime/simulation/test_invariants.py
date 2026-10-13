"""The invariant checker's own bookkeeping, on a real simulation with its record adjusted by hand."""

from __future__ import annotations as _annotations

from dataclasses import replace

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
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
