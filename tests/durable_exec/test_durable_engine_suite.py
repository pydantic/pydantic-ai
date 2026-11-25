"""How `DurableEngineSuite` judges a cell, with a stand-in engine: what each level and recovery mode accepts.

`TestDBOS` (`tests/harness/test_durable_tool_io.py`) and `TestTemporal` (`tests/harness/test_durable_temporal_replay.py`)
run the suite end to end; these check the judgements their engines don't reach, such as a `restart` recovery.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import BaseDurabilityCapability, conformance
from pydantic_ai.durable_exec.conformance import (
    ANSWER,
    PROMPT,
    DurableEngineSuite,
    EngineRun,
    Recovery,
    Support,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets import AbstractToolset

REFUSAL = UserError('not supported inside a durable unit')


@dataclass
class _Script:
    made: tuple[tuple[str, dict[str, object]], ...] = (('work', {}),)
    follow_ups: int = 0

    def build(self, directory: Path, /) -> Sequence[AbstractCapability[None] | AbstractToolset[None]]:
        raise NotImplementedError


@dataclass
class _Engine:
    """Returns the recoveries it's given, in order, each observing what it's given."""

    recoveries: list[tuple[EngineRun, list[str]]] = field(default_factory=list[tuple[EngineRun, list[str]]])
    name: str = 'stand-in'
    afters: list[int | None] = field(default_factory=list[int | None])

    def durability(self) -> BaseDurabilityCapability[None]:
        raise NotImplementedError

    async def run(self, agent: Agent[None, str], prompt: str) -> EngineRun:
        raise NotImplementedError

    async def recover(self, run: EngineRun, *, after: int | None = None) -> EngineRun:
        self.afters.append(after)
        recovered, observed = self.recoveries.pop(0)
        conformance._OBSERVED['observer'] = observed  # pyright: ignore[reportPrivateUsage]
        return recovered


def _counts(*counts: dict[str, int]) -> Callable[[], Awaitable[Counter[str]]]:
    remaining = [Counter(count) for count in counts]

    async def count() -> Counter[str]:
        return remaining.pop(0)

    return count


def _suite(support: dict[str, Support] | None = None, **attributes: object) -> DurableEngineSuite:
    suite_type = type('Suite', (DurableEngineSuite,), {'support': support or {}, **attributes})
    return suite_type()


def _check(run: EngineRun) -> None:
    assert run.output == ANSWER


def test_a_supported_cell_must_not_break() -> None:
    with pytest.raises(AssertionError, match='case under stand-in: lost'):
        _suite()._assert_cell('case', _Engine(), EngineRun('', ANSWER), ['lost'])  # pyright: ignore[reportPrivateUsage]


def test_a_limited_cell_checks_its_limit() -> None:
    suite = _suite({'case': Support('limited', 'runs in the foreground', check=_check)})
    suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), [])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AssertionError):
        suite._assert_cell('case', _Engine(), EngineRun('', 'other'), [])  # pyright: ignore[reportPrivateUsage]


def test_a_refused_cell_must_refuse_before_it_records_anything() -> None:
    suite = _suite({'case': Support('refused', 'unit', match='inside a durable', issue=1)})
    raised = 'the run raised UserError: not supported inside a durable unit'
    suite._assert_cell('case', _Engine(), EngineRun('', REFUSAL), [raised])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AssertionError, match='no longer raises'):
        suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), [])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AssertionError, match='not before it records anything'):
        suite._assert_cell('case', _Engine(), EngineRun('', REFUSAL, ['unit']), [raised, 'recovering it lost a unit'])  # pyright: ignore[reportPrivateUsage]


def test_a_broken_cell_must_still_show_its_symptom() -> None:
    suite = _suite({'case': Support('broken', 'loses an event', match='lost', issue=1)})
    suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), ['recovery lost an event'])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AssertionError, match='it now passes'):
        suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), [])  # pyright: ignore[reportPrivateUsage]


def test_an_untested_cell_fails() -> None:
    suite = _suite({'case': Support('untested', 'no suite', issue=1)})
    with pytest.raises(pytest.fail.Exception, match='run it, or declare it'):
        suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), [])  # pyright: ignore[reportPrivateUsage]


def test_an_unlisted_cell_can_be_pending() -> None:
    suite = _suite(unlisted='pending')
    suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), [])  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(pytest.xfail.Exception, match='declare its support'):
        suite._assert_cell('case', _Engine(), EngineRun('', ANSWER), ['lost'])  # pyright: ignore[reportPrivateUsage]


def test_the_first_run_must_answer_once_and_observe_what_it_should() -> None:
    def expected_units(self: DurableEngineSuite, name: str) -> tuple[str, ...]:
        return ('tool',)

    def model_request_unit(self: DurableEngineSuite, name: str) -> str:
        return 'model'

    suite = _suite(model_request_unit=model_request_unit, expected_units=expected_units)
    run = EngineRun('', 'other', ['model', 'tool', 'model'], ['event:a'])
    assert suite._run_breaks('case', _Script(), ('event:b',), run, Counter({'search': 2})) == [  # pyright: ignore[reportPrivateUsage]
        f"the run returned 'other', not {ANSWER!r}",
        "a request was made more than once: {'search': 2}",
        "the run observed ['event:a'], not ['event:b']",
    ]
    followed = EngineRun('', ANSWER, ['model', 'tool', 'model', 'model'])
    assert suite._run_breaks('case', _Script(follow_ups=1), None, followed, Counter()) == []  # pyright: ignore[reportPrivateUsage]
    drifted = suite._run_breaks('case', _Script(), None, EngineRun('', ANSWER, ['model', 'other']), Counter())  # pyright: ignore[reportPrivateUsage]
    assert drifted == [
        'the run recorded 1 `model` units, not one per call, one per follow-up and one to answer',
        "the calls record ('other',), not ('tool',). A recorded name is what a run from an earlier release recovers "
        'by: change it only with a compatibility note',
    ]


def test_the_defaults_pin_nothing_and_count_nothing() -> None:
    suite = _suite()
    assert suite.expected_units('case') is None
    assert suite.marks('case') == ()
    assert suite.model_request_unit('case') is None
    assert DurableEngineSuite.count_requests._get_wrapped_function()(suite)() == {}  # pyright: ignore[reportPrivateUsage]
    assert suite._run_breaks('case', _Script(), None, EngineRun('', ANSWER, ['model']), Counter()) == []  # pyright: ignore[reportPrivateUsage]


async def test_a_replayed_recovery_must_be_the_same_and_repeat_no_request() -> None:
    run = EngineRun('', ANSWER, ['model', 'tool', 'model'], ['event:a'])
    engine = _Engine(
        [
            (EngineRun('', REFUSAL, ['model']), ['event:b']),
            (EngineRun('', ANSWER, ['model', 'other']), ['event:a']),
        ]
    )
    breaks = await _suite()._recovery_breaks(engine, run, 'observer', True, _counts({'a': 1}, {'a': 2}))  # pyright: ignore[reportPrivateUsage]
    assert breaks == [
        'recovery returned UserError: not supported inside a durable unit, not ' + repr(ANSWER),
        "recovery recorded ['model'], not ['model', 'tool', 'model']",
        "recovery lost ['event:a']",
        "recovery added ['event:b']",
        "recovery made a request again: {'a': 1} became {'a': 2}",
    ]
    assert engine.afters == [None, 1]


async def test_a_restarted_recovery_makes_each_request_once_more() -> None:
    suite = _suite(recovery=Recovery('restart', 'reruns the entry task', issue=1))
    run = EngineRun('', ANSWER, ['tool'])
    engine = _Engine([(EngineRun('', ANSWER, ['tool']), []), (EngineRun('', 'other', ['tool']), [])])
    assert await suite._recovery_breaks(engine, run, None, True, _counts({'a': 1}, {'a': 2})) == []  # pyright: ignore[reportPrivateUsage]
    assert await suite._recovery_breaks(engine, run, None, True, _counts({'a': 1}, {'a': 3})) == [  # pyright: ignore[reportPrivateUsage]
        "a restart made {'a': 2} again, not each of {'a': 1} once",
        f"recovery returned 'other', not {ANSWER!r}",
    ]


async def test_a_refusal_mid_run_must_leave_its_record_unaffected() -> None:
    run = EngineRun('', REFUSAL, ['model'])
    engine = _Engine([(EngineRun('', REFUSAL, ['model']), []), (EngineRun('', ANSWER, ['model', 'tool']), [])])
    assert await _suite()._refusal_breaks(engine, run, None, _counts({}, {})) == []  # pyright: ignore[reportPrivateUsage]
    assert await _suite()._refusal_breaks(engine, run, None, _counts({}, {'a': 1})) == [  # pyright: ignore[reportPrivateUsage]
        "recovering the refused run recorded ['model', 'tool'], not ['model']",
        f'recovering the refused run returned {ANSWER!r}, not the refusal',
        "recovering the refused run made a request again: {} became {'a': 1}",
    ]


@dataclass
class _RefusedAtBuild(_Script):
    def build(self, directory: Path, /) -> Sequence[AbstractCapability[None] | AbstractToolset[None]]:
        raise REFUSAL


async def test_a_refusal_when_the_case_is_built_is_judged_by_its_cell(tmp_path: Path) -> None:
    """A capability that refuses the engine as it's built records nothing, so a `refused` cell holds."""
    cases = {'case': _RefusedAtBuild()}
    refused = _suite({'case': Support('refused', 'unit', match='inside a durable', issue=1)}, cases=cases)
    await refused.test_case('case', _Engine(), True, dict, tmp_path)
    with pytest.raises(AssertionError, match='the run raised UserError'):
        await _suite(cases=cases).test_case('case', _Engine(), True, dict, tmp_path)


async def test_a_refusal_mid_run_may_repeat_requests_under_a_restarting_engine() -> None:
    """A restart makes every call again, so recovering a refused run under one doesn't have to repeat no request."""
    suite = _suite(recovery=Recovery('restart', 'reruns the entry task', issue=1))
    run = EngineRun('', REFUSAL, ['model'])
    engine = _Engine([(EngineRun('', REFUSAL, ['model']), [])])
    assert await suite._refusal_breaks(engine, run, None, _counts({}, {'a': 1})) == []  # pyright: ignore[reportPrivateUsage]


async def test_a_nested_run_on_the_suite_model_is_answered_and_counted() -> None:
    """A capability's own run on the run's model is counted, so a recovery that repeats it breaks the cell."""
    conformance._NESTED.clear()  # pyright: ignore[reportPrivateUsage]
    model = conformance._scripted_model((('ask', {}),))  # pyright: ignore[reportPrivateUsage]

    async def ask() -> str:
        return (await Agent(model).run('nested')).output

    try:
        result = await Agent(model, tools=[ask]).run(PROMPT)
        assert (result.output, conformance._NESTED) == (ANSWER, {'model.unprompted_request': 1})  # pyright: ignore[reportPrivateUsage]
    finally:
        conformance._NESTED.clear()  # pyright: ignore[reportPrivateUsage]
