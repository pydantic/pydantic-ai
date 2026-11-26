"""The data an engine's conformance declarations are made of: `Support` cells, its `Recovery`, and an `EngineRun`.

They live apart from `pydantic_ai.durable_exec.conformance`, which re-exports them, so that an engine's support table
can be read without pytest. See [`DurableEngineSuite`][pydantic_ai.durable_exec.conformance.DurableEngineSuite].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

__all__ = ('SupportLevel', 'Support', 'RecoveryMode', 'Recovery', 'EngineRun')

SupportLevel = Literal['supported', 'limited', 'refused', 'broken', 'untested']
"""How an engine runs a case or a scenario. See [`Support`][pydantic_ai.durable_exec.conformance.Support]."""

RecoveryMode = Literal['replay', 'restart']
"""How an engine recovers a run. See [`Recovery`][pydantic_ai.durable_exec.conformance.Recovery]."""


@dataclass
class EngineRun:
    """One run of an agent under a durable engine, or its recovery: what it returned and what the engine recorded."""

    id: str
    """The engine's id for the run, which [`DurableEngine.recover`][pydantic_ai.durable_exec.conformance.DurableEngine.recover]
    reads its record by."""
    output: str | BaseException
    """The run's output, or what it raised. A refusal is a [`UserError`][pydantic_ai.exceptions.UserError]:
    an engine that wraps the errors a run raises unwraps it."""
    units: list[str] = field(default_factory=list[str])
    """The durable units the engine recorded for the run, in order: its journal or history, model requests
    included, with the agent's name left out of each unit's name."""
    observed: list[str] = field(default_factory=list[str])
    """What the run observed in workflow code: `event:<text>` per custom or capability event, `enqueued:<text>` per
    enqueued message's event, and `message:<text>` per enqueued message in the history. The suite fills it in for a
    scenario's run; an engine leaves it empty."""


@dataclass(frozen=True)
class Support:
    """How a durable engine runs a case's calls or a scenario: one cell of its support table.

    - `supported`: the suite runs it under the engine and checks what the engine records and recovers.
    - `limited`: the same, and `check` asserts the limit, which `reason` documents.
    - `refused`: the engine raises a [`UserError`][pydantic_ai.exceptions.UserError] matching the `match` regex
      before the refused operation records anything, and the record it leaves recovers unaffected.
    - `broken`: the engine runs it and gets it wrong. `match` is a regex of the symptom and `issue` tracks the bug.
      The suite checks the symptom still shows, so the cell fails, saying to update it, once the bug is fixed.
    - `untested`: no suite runs it under the engine yet. Only the generated docs table uses it: a suite fails a cell
      declared `untested`.
    """

    level: SupportLevel
    reason: str = ''
    """Why the cell isn't `supported`: the caveat the docs show. Required unless `supported`."""
    match: str | None = None
    """A regex of what the engine raises (`refused`) or of the symptom (`broken`), searched for in each way the run
    breaks the suite's checks. Required for those levels, and only for them."""
    check: Callable[[EngineRun], None] | None = None
    """Asserts the limit on the first run. Required for `limited`, and only for it."""
    issue: int | None = None
    """The GitHub issue that tracks the cell: full support for `limited`, the bug for `broken`, a suite for `untested`.

    Required for those three, and recommended for `refused`. The docs badge links it.
    """

    def __post_init__(self) -> None:
        if self.level != 'supported' and not self.reason:
            raise ValueError(f'a `{self.level}` cell needs a `reason`')
        if (self.level in ('refused', 'broken')) != (self.match is not None):
            raise ValueError('a `refused` or `broken` cell, and only one, names what it matches in `match`')
        if (self.level == 'limited') != (self.check is not None):
            raise ValueError('a `limited` cell, and only one, asserts its limit in `check`')
        if self.level in ('limited', 'broken', 'untested') and self.issue is None:
            raise ValueError(f'a `{self.level}` cell needs the `issue` that tracks it')


@dataclass(frozen=True)
class Recovery:
    """How an engine recovers a run, which applies to every cell.

    - `replay`: recovery replays the record, so a recovered run repeats no completed call, and observes the same.
    - `restart`: recovery restarts the run and makes every model and tool call again. The suite treats each cell it
      would otherwise check for recovery equality as `limited`: each counted call is made exactly once more, and the
      output is the same. `reason` and `issue` say why, and the docs show those cells as limited.
    """

    mode: RecoveryMode = 'replay'
    reason: str = ''
    """Why the engine restarts rather than replays. Required for `restart`."""
    issue: int | None = None
    """The GitHub issue that tracks replaying instead. Required for `restart`."""

    def __post_init__(self) -> None:
        if self.mode == 'restart' and (not self.reason or self.issue is None):
            raise ValueError('a `restart` recovery needs a `reason` and the `issue` that tracks it')


INLINE_ANSWER = 'inline result'
"""What a `BackgroundTools` scenario's model answers when the backgrounded call's result came back inline, as its
normal tool return, rather than as a follow-up message."""


def ran_in_foreground(run: EngineRun) -> None:
    """The `limited` check of an engine that runs a backgrounded call in the foreground: its result came back inline.

    The engine runs tool calls one at a time, so `BackgroundTools` runs the call in the foreground.
    """
    assert run.output == INLINE_ANSWER, f'the call ran in the background, or failed: {run.output!r}'
