"""How `DurableEngineSuite` judges a cell, with a stand-in engine: what each level and recovery mode accepts.

`TestDBOS` (`tests/harness/test_durable_tool_io.py`) and `TestTemporal` (`tests/harness/test_durable_temporal_replay.py`)
run the suite end to end; these check the judgements their engines don't reach, such as a `restart` recovery.
They run a stand-in engine and scripted model, with no provider exchange to record, so none is a VCR test.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.conformance import Calls, CapabilityBuilder
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

REFUSAL = UserError('not supported inside a durable unit')


def _unbuilt(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    raise NotImplementedError


def _script(follow_ups: int = 0) -> Calls:
    return Calls((('work', {}),), build=_unbuilt, follow_ups=follow_ups)


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
    suite = _suite({'case': Support('limited', 'runs in the foreground', check=_check, issue=1)})
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
    assert suite._run_breaks('case', _script(), ('event:b',), run, Counter({'search': 2})) == [  # pyright: ignore[reportPrivateUsage]
        f"the run returned 'other', not {ANSWER!r}",
        "a request was made more than once: {'search': 2}",
        "the run observed ['event:a'], not ['event:b']",
    ]
    followed = EngineRun('', ANSWER, ['model', 'tool', 'model', 'model'])
    assert suite._run_breaks('case', _script(follow_ups=1), None, followed, Counter()) == []  # pyright: ignore[reportPrivateUsage]
    drifted = suite._run_breaks('case', _script(), None, EngineRun('', ANSWER, ['model', 'other']), Counter())  # pyright: ignore[reportPrivateUsage]
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
    assert suite._run_breaks('case', _script(), None, EngineRun('', ANSWER, ['model']), Counter()) == []  # pyright: ignore[reportPrivateUsage]


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


def _refused_at_build(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    raise REFUSAL


async def test_a_refusal_when_the_case_is_built_is_judged_by_its_cell(tmp_path: Path) -> None:
    """A capability that refuses the engine as it's built records nothing, so a `refused` cell holds."""
    cases = {'case': Calls((('work', {}),), build=_refused_at_build)}
    refused = _suite({'case': Support('refused', 'unit', match='inside a durable', issue=1)}, cases=cases)
    await refused.test_case('case', _Engine(), True, dict, tmp_path)
    with pytest.raises(AssertionError, match='the run raised UserError'):
        await _suite(cases=cases).test_case('case', _Engine(), True, dict, tmp_path)


def _needs_a_missing_dependency(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    raise ModuleNotFoundError("No module named 'missing_sdk'", name='missing_sdk')


def _hints_at_a_missing_dependency(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    missing = ModuleNotFoundError("No module named 'missing_sdk'", name='missing_sdk')
    raise ImportError('Install it with: pip install "missing-sdk"') from missing


def _imports_a_missing_name(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    raise ImportError("cannot import name 'gone' from 'installed_sdk'")


def _builds_nothing(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    return []


@pytest.mark.parametrize('build', [_needs_a_missing_dependency, _hints_at_a_missing_dependency])
async def test_a_case_whose_dependency_is_missing_is_skipped(
    build: CapabilityBuilder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A case whose build can't find its optional dependency is skipped, not judged: no engine ran it.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    monkeypatch.delenv('PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS', raising=False)
    suite = _suite(cases={'case': Calls((('work', {}),), build=build)})
    with pytest.raises(pytest.skip.Exception, match='case needs `missing_sdk`, from an optional dependency'):
        await suite.test_case('case', _Engine(), True, dict, tmp_path)


async def test_a_missing_dependency_fails_when_extras_are_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS` turns the skip into a failure, naming the missing module.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    monkeypatch.setenv('PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS', 'true')
    suite = _suite(cases={'case': Calls((('work', {}),), build=_needs_a_missing_dependency)})
    with pytest.raises(
        pytest.fail.Exception, match=r'case needs `missing_sdk`.*`PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS`'
    ):
        await suite.test_case('case', _Engine(), True, dict, tmp_path)


async def test_an_import_error_that_is_no_missing_module_fails(tmp_path: Path) -> None:
    """A name a mismatched version lacks is a broken install, not a missing extra: the case fails.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    suite = _suite(cases={'case': Calls((('work', {}),), build=_imports_a_missing_name)})
    with pytest.raises(ImportError, match="cannot import name 'gone'"):
        await suite.test_case('case', _Engine(), True, dict, tmp_path)


def _misses_its_own_module(directory: Path, /) -> Sequence[AbstractCapability[None]]:
    raise ModuleNotFoundError("No module named 'pydantic_ai_harness.gone'", name='pydantic_ai_harness.gone')


async def test_a_module_missing_from_the_suites_own_packages_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A module missing from `pydantic_ai` or `pydantic_ai_harness` is a broken install, not an extra: the case fails.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    monkeypatch.delenv('PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS', raising=False)
    suite = _suite(cases={'case': Calls((('work', {}),), build=_misses_its_own_module)})
    with pytest.raises(ModuleNotFoundError, match=r'pydantic_ai_harness\.gone'):
        await suite.test_case('case', _Engine(), True, dict, tmp_path)


@pytest.mark.parametrize('value', ['0', 'false', ''])
async def test_extras_are_optional_unless_required(value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`0`, `false` or an empty value leave a missing extra a skip.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    monkeypatch.setenv('PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS', value)
    suite = _suite(cases={'case': Calls((('work', {}),), build=_needs_a_missing_dependency)})
    with pytest.raises(pytest.skip.Exception):
        await suite.test_case('case', _Engine(), True, dict, tmp_path)


async def test_an_unknown_value_of_the_extras_variable_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A value the suite doesn't know, such as `yes`, is an error rather than silently leaving extras optional.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    monkeypatch.setenv('PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS', 'yes')
    suite = _suite(cases={'case': Calls((('work', {}),), build=_needs_a_missing_dependency)})
    with pytest.raises(ValueError, match='must be `1`, `true`, `0` or `false`'):
        await suite.test_case('case', _Engine(), True, dict, tmp_path)


@dataclass
class _BrokenEngine(_Engine):
    def durability(self) -> BaseDurabilityCapability[None]:
        raise ModuleNotFoundError("No module named 'engine_sdk'", name='engine_sdk')


async def test_a_missing_module_in_the_engine_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only a case's build can skip it: the engine's own code failing to import fails every cell.

    It runs a stand-in engine and build, with no provider exchange to record, so it isn't a VCR test.
    """
    monkeypatch.delenv('PYDANTIC_AI_REQUIRE_CONFORMANCE_EXTRAS', raising=False)
    suite = _suite(cases={'case': Calls((('work', {}),), build=_builds_nothing)})
    with pytest.raises(ModuleNotFoundError, match='engine_sdk'):
        await suite.test_case('case', _BrokenEngine(), True, dict, tmp_path)


async def test_a_refused_unit_records_no_result() -> None:
    """The unit a refusal raised in is failed, and only that one: an engine that recorded its result breaks the cell."""
    run = EngineRun('', REFUSAL, ['model'], failed=['tool'])
    engine = _Engine([(EngineRun('', REFUSAL, ['model'], failed=['tool']), [])])
    assert await _suite()._refusal_breaks(engine, run, None, _counts({}, {})) == []  # pyright: ignore[reportPrivateUsage]
    completed = EngineRun('', REFUSAL, ['model', 'tool'], failed=['tool', 'other'])
    engine = _Engine([(EngineRun('', REFUSAL, ['model', 'tool']), [])])
    assert await _suite()._refusal_breaks(engine, completed, None, _counts({}, {})) == [  # pyright: ignore[reportPrivateUsage]
        "the refused run recorded more than the refused unit as failed: ['tool', 'other']",
        "the refused run also recorded a result for the unit that raised: ['tool']",
        "recovering the refused run failed [], not ['tool', 'other']",
    ]


async def test_a_refusal_mid_run_may_repeat_requests_under_a_restarting_engine() -> None:
    """A restart makes every call again, so recovering a refused run under one doesn't have to repeat no request."""
    suite = _suite(recovery=Recovery('restart', 'reruns the entry task', issue=1))
    run = EngineRun('', REFUSAL, ['model'])
    engine = _Engine([(EngineRun('', REFUSAL, ['model']), [])])
    assert await suite._refusal_breaks(engine, run, None, _counts({}, {'a': 1})) == []  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('recovery', [Recovery('replay'), Recovery('restart', 'reruns the entry task', issue=1)])
async def test_a_refusal_mid_run_must_keep_what_it_observed(recovery: Recovery) -> None:
    """Recovering a refused run observes what the run did before it refused, each once, in any order.

    It runs a stand-in engine, with no provider exchange to record, so it isn't a VCR test.
    """
    suite = _suite(recovery=recovery)
    run = EngineRun('', REFUSAL, ['model'], ['event:a', 'enqueued:b'])
    reordered = _Engine([(EngineRun('', REFUSAL, ['model']), ['enqueued:b', 'event:a'])])
    assert await suite._refusal_breaks(reordered, run, 'observer', _counts({}, {})) == []  # pyright: ignore[reportPrivateUsage]
    changed = _Engine([(EngineRun('', REFUSAL, ['model']), ['event:a', 'event:a'])])
    assert await suite._refusal_breaks(changed, run, 'observer', _counts({}, {})) == [  # pyright: ignore[reportPrivateUsage]
        "recovering the refused run lost ['enqueued:b']",
        "recovering the refused run added ['event:a']",
    ]


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
