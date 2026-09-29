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
