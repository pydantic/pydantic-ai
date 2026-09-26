"""Hypothesis state machines that drive simulations through randomized interleavings.

Each rule is one simulation step: a client operation, a provider behavior, a fault, or letting time
pass. Rules take `ticks` (how long the event loop runs afterwards) and, for provider frames, whether
they are delivered at once, which is how the machine explores which task wins each race. Hypothesis
shrinks a failure to a minimal sequence of steps, and the `InvariantViolation` it reports carries that
sequence as a replayable trace.

Known findings (see `_findings.py`) are tolerated in the default mode: the simulation's checker notes a
violation matching one and carries on checking everything else, and the machine counts it in
Hypothesis' statistics, so exploration stays green on current main while still failing on anything new.
Set `REALTIME_SIMULATION_STRICT=1` to report them too.
"""

from __future__ import annotations as _annotations

from collections import Counter
from collections.abc import Callable
from typing import Any, ClassVar, Literal

from hypothesis import event, note, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, precondition, rule

from ._simulation import InvariantViolation, SessionOptions, Simulation, ToolOutcome

KNOWN_HIT_COUNTS: Counter[str] = Counter()
"""How many examples ran into each known finding, across the whole test process (for reports)."""

TICKS = st.one_of(st.none(), st.integers(min_value=0, max_value=6))
"""How long the loop runs after a step: to rest (`None`), or a few iterations, leaving tasks mid-flight."""

TOOL_OUTCOMES: st.SearchStrategy[ToolOutcome] = st.sampled_from(['ok', 'ok', 'ok', 'retry', 'error'])


class SessionMachine(RuleBasedStateMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Client-side rules shared by every provider; subclasses add the provider's behavior."""

    make_simulation: ClassVar[Callable[[int, SessionOptions, Any], Simulation]]
    """Builds the simulation from a seed, the session options, and a draw of `provider_options`."""
    provider_options: ClassVar[st.SearchStrategy[Any]] = st.none()

    def __init__(self) -> None:
        super().__init__()
        self.sim: Simulation | None = None
        self.failed = False

    # --- plumbing ---------------------------------------------------------------------------------

    @property
    def s(self) -> Simulation:
        assert self.sim is not None
        return self.sim

    def run(self, action: Callable[[], object]) -> None:
        """Run one step. Known findings are tolerated by the simulation's checker; anything else fails."""
        try:
            action()
        except InvariantViolation:  # pragma: no cover (only when exploration finds something new)
            # Hypothesis still runs `teardown()` after a failing step; it must not replace this failure.
            self.failed = True
            raise

    def live(self) -> bool:
        # (Hypothesis can evaluate preconditions before the `@initialize` rule has run.)
        return self.sim is not None and not self.sim.closed and self.sim.close_requested is None

    def teardown(self) -> None:
        sim = self.s
        try:
            if not self.failed:  # pragma: no branch
                self.run(lambda: sim.settle())
                self.run(lambda: sim.close())
                self.run(lambda: sim.settle())
                self.run(lambda: sim.check_handoff())
        finally:
            for finding_id, code in dict.fromkeys(sim.checker.known_hits):
                event(f'known finding {finding_id} ({code})')
                KNOWN_HIT_COUNTS[f'{finding_id} ({code})'] += 1
            sim.close_simulation()

    # --- configuration ----------------------------------------------------------------------------

    @initialize(
        seed=st.integers(min_value=0, max_value=2**16),
        handle_barge_in=st.booleans(),
        latency=st.booleans(),
        request_limit=st.one_of(st.none(), st.integers(min_value=1, max_value=6)),
        data=st.data(),
    )
    def start(
        self, seed: int, handle_barge_in: bool, latency: bool, request_limit: int | None, data: st.DataObject
    ) -> None:
        options = SessionOptions(handle_barge_in=handle_barge_in, latency=latency, request_limit=request_limit)
        provider_options = data.draw(type(self).provider_options, label='provider_options')
        self.sim = type(self).make_simulation(seed, options, provider_options)
        note(f'{type(self.sim).__name__}(seed={seed}, options={options!r}, provider={provider_options!r})')
        self.run(self.s.start)

    # --- client operations ------------------------------------------------------------------------

    @precondition(lambda self: self.live())
    @rule(respond=st.booleans(), ticks=TICKS)
    def send_text(self, respond: bool, ticks: int | None) -> None:
        self.run(lambda: self.s.send_text(respond=respond, ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(chunks=st.integers(min_value=1, max_value=3), ticks=TICKS)
    def send_audio(self, chunks: int, ticks: int | None) -> None:
        self.run(lambda: self.s.send_audio(chunks=chunks, ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(ticks=TICKS)
    def wait_for_reply(self, ticks: int | None) -> None:
        self.run(lambda: self.s.wait_for_reply(ticks=ticks))

    @precondition(lambda self: self.live() and bool(self.s.tools.pending()))
    @rule(call=st.integers(min_value=0, max_value=3), outcome=TOOL_OUTCOMES, ticks=TICKS)
    def finish_tool(self, call: int, outcome: ToolOutcome, ticks: int | None) -> None:
        self.run(lambda: self.s.finish_tool(call=call, outcome=outcome, ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(chunks=st.integers(min_value=1, max_value=4), ticks=TICKS)
    def play(self, chunks: int, ticks: int | None) -> None:
        self.run(lambda: self.s.play(chunks=chunks, ticks=ticks))

    @precondition(lambda self: self.sim is not None)
    @rule(ticks=st.integers(min_value=0, max_value=6))
    def tick(self, ticks: int) -> None:
        self.run(lambda: self.s.tick(ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(seconds=st.sampled_from([0.1, 0.5, 2.5]), ticks=TICKS)
    def advance_time(self, seconds: float, ticks: int | None) -> None:
        self.run(lambda: self.s.advance_time(seconds, ticks=ticks))

    @precondition(lambda self: self.live())
    @rule()
    def settle(self) -> None:
        self.run(lambda: self.s.settle())

    @precondition(lambda self: self.live())
    @rule(ticks=TICKS)
    def close(self, ticks: int | None) -> None:
        self.run(lambda: self.s.close(ticks=ticks))


class ManualTurnMachine(SessionMachine):  # pragma: lax no cover (driven only by randomized exploration)
    """Adds the manual turn-control and barge-in operations, for providers that support them."""

    @precondition(lambda self: self.live())
    @rule(respond=st.booleans(), ticks=TICKS)
    def send_image(self, respond: bool, ticks: int | None) -> None:
        self.run(lambda: self.s.send_image(respond=respond, ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(ticks=TICKS)
    def commit_audio(self, ticks: int | None) -> None:
        self.run(lambda: self.s.commit_audio(ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(ticks=TICKS)
    def clear_audio(self, ticks: int | None) -> None:
        self.run(lambda: self.s.clear_audio(ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(ticks=TICKS)
    def create_response(self, ticks: int | None) -> None:
        self.run(lambda: self.s.create_response(ticks=ticks))

    @precondition(lambda self: self.live())
    @rule(mode=st.sampled_from(['cancel', 'played_ms', 'played_bytes']), ticks=TICKS)
    def interrupt(self, mode: Literal['cancel', 'played_ms', 'played_bytes'], ticks: int | None) -> None:
        self.run(lambda: self.s.interrupt(mode=mode, ticks=ticks))
