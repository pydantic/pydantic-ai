"""The conformance suite for Pydantic AI durable execution engines.

Subclass `DurableEngineSuite` in a pytest module and provide an `engine` fixture: each test runs one cell, a
capability case's calls or a scenario, under the engine, recovers it, and checks the cell's declared
[`Support`][pydantic_ai.durable_exec.conformance.Support] level. The cases and scenarios are the core capabilities'
[`CALLS`][pydantic_ai.capabilities.conformance.CALLS] and
[`SCENARIO_RUNS`][pydantic_ai.capabilities.conformance.SCENARIO_RUNS]; `pydantic_ai_harness.conformance`'s
`HarnessDurableEngineSuite` adds the Pydantic AI Harness cases. Requires pytest and the anyio pytest plugin.

The suite's class, fixtures, `DurableEngine`, `EngineRun`, `Support`, `Recovery` and the level names are stable in 2.x.
Minor releases can add cases, scenarios and checks, or tighten them: a new cell runs as `supported`, so an engine that
fails it declares its support on upgrade, or sets `unlisted = 'pending'`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

import anyio.to_thread
import pytest

from ..agent import Agent
from ..capabilities import AbstractCapability
from ..capabilities.conformance import CALLS, SCENARIO_RUNS, Calls, ScenarioRun, ToolCall
from ..exceptions import UserError
from ..messages import (
    AgentStreamEvent,
    CapabilityEvent,
    CustomEvent,
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from ..models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from ..run import AgentRunResult
from ..tools import RunContext
from ..toolsets import AbstractToolset
from ._base import BaseDurabilityCapability
from ._support import INLINE_ANSWER, EngineRun, Recovery, RecoveryMode, Support, SupportLevel, ran_in_foreground

__all__ = (
    'DurableEngine',
    'DurableEngineSuite',
    'EngineRun',
    'INLINE_ANSWER',
    'Recovery',
    'RecoveryMode',
    'Support',
    'SupportLevel',
    'ran_in_foreground',
)

PROMPT = 'go'
"""The prompt the suite starts each run with, which tells its model requests apart from a nested run's."""

ANSWER = 'done'
"""What the suite's model answers once it has made its calls: every run's output."""

_UNPROMPTED_REQUEST = 'model.unprompted_request'
"""What the suite counts a model request without `PROMPT` under: a capability's own run on the run's model."""

_NESTED: Counter[str] = Counter()
"""The model requests the suite's model counted itself, since the current cell started."""

_OBSERVED: dict[str, list[str]] = {}
"""What each scenario agent's recorder saw in its latest run, by agent name: an engine may run a copy of it."""


class DurableEngine(Protocol):
    """A durable engine, started in process for the suite: it runs an agent durably, and recovers the run."""

    @property
    def name(self) -> str:
        """The engine's name, for messages."""
        ...

    def durability(self) -> BaseDurabilityCapability[None]:
        """A new durability capability for the engine, which the suite puts last on each agent it builds."""
        ...

    async def run(self, agent: Agent[None, str], prompt: str) -> EngineRun:
        """Run `agent` with `prompt` durably, as the engine runs a workflow, and return what it recorded.

        A run that raises returns the error as its `output` rather than raising it. `EngineRun.units` holds only the
        units whose result the engine recorded; a unit that raised goes in `EngineRun.failed`.
        """
        ...

    async def recover(self, run: EngineRun, *, after: int | None = None) -> EngineRun:
        """Recover `run` from its record, as the engine does after a crash, and return the recovered run.

        Its `units` and `failed` report the recovered run's record the way `run` reports the first run's.

        With `after=k`, keep only the first `k` units of the record, as if the run had stopped there, so the rest
        runs again. Only called when the suite's `can_fork_mid_run` fixture is `True`.
        """
        ...


def _scripted_model(made: Sequence[ToolCall]) -> FunctionModel:
    """A model that makes the calls in `made`, one per response, then answers `ANSWER`; call `i` has the ID `call_{i}`.

    Every request past the calls, such as one a call's follow-up causes, gets the answer. It streams when the run
    does, as it does when a capability listens for events.

    A request whose first message isn't the run's `PROMPT` is a capability's own run on the run's model: it gets the
    answer and is counted, as recovery must not make it again.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del info
        if not any(isinstance(part, UserPromptPart) and part.content == PROMPT for part in messages[0].parts):
            _NESTED[_UNPROMPTED_REQUEST] += 1
            return ModelResponse(parts=[TextPart(ANSWER)])
        step = sum(isinstance(message, ModelResponse) for message in messages)
        if step >= len(made):
            return ModelResponse(parts=[TextPart(ANSWER)])
        tool_name, args = made[step]
        return ModelResponse(parts=[ToolCallPart(tool_name, args, tool_call_id=f'call_{step}')])

    async def stream_response(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for part in respond(messages, info).parts:
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {0: DeltaToolCall(part.tool_name, json.dumps(part.args), tool_call_id=part.tool_call_id)}

    return FunctionModel(respond, stream_function=stream_response)


def _prompts(messages: Sequence[ModelMessage]) -> list[str]:
    """The user prompts in `messages` other than the run's own `PROMPT`: the enqueued ones."""
    return [
        str(part.content)
        for message in messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and part.content != PROMPT
    ]


@dataclass
class _Recorder(AbstractCapability[None]):
    """Records what a scenario's run observes in workflow code: its events, then the messages enqueued into it."""

    agent_name: str

    async def before_run(self, ctx: RunContext[None]) -> None:
        _OBSERVED[self.agent_name] = []

    async def on_event(self, ctx: RunContext[None], *, event: AgentStreamEvent) -> None:
        seen = _OBSERVED.setdefault(self.agent_name, [])
        if isinstance(event, CustomEvent | CapabilityEvent):
            seen.append(f'event:{getattr(event, "text", type(event).__name__)}')
        elif isinstance(event, EnqueuedMessagesEvent):
            seen.extend(f'enqueued:{content}' for content in _prompts(event.messages))

    async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        _OBSERVED.setdefault(self.agent_name, []).extend(
            f'message:{prompt}' for prompt in _prompts(result.all_messages())
        )
        return result


def _describe(output: str | BaseException) -> str:
    return f'{type(output).__name__}: {output}' if isinstance(output, BaseException) else repr(output)


def _repeated(counts: Mapping[str, int]) -> dict[str, int]:
    return {key: count for key, count in counts.items() if count > 1}


class DurableEngineSuite:
    """Subclass in your test suite and provide the `engine` fixture.

    Each test runs one cell under the engine: a case's calls (`test_case`) or a scenario (`test_scenario`). It runs
    the agent, recovers the run, and checks the cell's level in `support`. A cell `support` doesn't list is
    `supported`:

    - `supported`: the run returns the answer and makes each counted request once. Recovery from the whole record
      returns the same, records the same units, observes the same, and makes no counted request again. With
      `can_fork_mid_run`, recovery from the middle of the record returns and observes the same, and keeps the units
      before the cut. When `expected_units` names the units, the run records exactly those.
    - `limited`: the same, and the cell's `check` asserts the limit.
    - `refused`: the engine raises a [`UserError`][pydantic_ai.exceptions.UserError] matching `match`, before the
      refused operation records anything: at build or run start, or mid-run. A refused operation that runs in a unit
      leaves that one unit failed, with no result recorded for it. The units recorded before it are unaffected, so
      recovering the record records the same units, fails the same unit and refuses again, without a counted
      request.
    - `broken`: some check above fails with a message, or the run raises an error, that matches `match`. Once the
      bug is fixed it no longer does, and the cell fails, saying to update it.

    A `restart` [`Recovery`][pydantic_ai.durable_exec.conformance.Recovery] makes the cells that check recovery
    equality limited: a recovery makes each counted request exactly once more, and returns the same output.
    """

    pytestmark = pytest.mark.anyio

    support: ClassVar[Mapping[str, Support]] = {}
    """The cells that aren't `supported`, keyed by case or scenario name."""
    recovery: ClassVar[Recovery] = Recovery()
    """How the engine recovers a run."""
    unlisted: ClassVar[Literal['supported', 'pending']] = 'supported'
    """How to treat a cell `support` doesn't list. `'pending'` reports one that fails as an expected failure rather
    than an error, so a case or scenario a new release adds doesn't fail your CI until you declare its support."""
    cases: ClassVar[Mapping[str, Calls]] = CALLS
    """The cases the suite runs, by capability class name: the core capabilities' `CALLS` by default."""
    scenarios: ClassVar[Mapping[str, ScenarioRun]] = SCENARIO_RUNS
    """The scenarios the suite runs, named `<owner>.<scenario>`: `SCENARIO_RUNS` by default."""

    @pytest.fixture
    def engine(self) -> DurableEngine | AsyncIterator[DurableEngine]:
        raise NotImplementedError('provide an `engine` fixture')

    @pytest.fixture
    def can_fork_mid_run(self) -> bool:
        """Override to return `False` if the engine can't recover a run from part of its record. Document the limit."""
        return True

    @pytest.fixture
    def count_requests(self) -> Callable[[], Mapping[str, int]]:
        """Count the requests the cases' stand-ins make from now on, by kind: override it when they count any.

        The suite counts the model requests of a capability's own runs itself.
        """
        return dict

    def expected_units(self, name: str) -> Sequence[str] | None:
        """The units a run of the case `name` records besides its model requests, in order, to pin their names.

        `None`, the default, leaves them unpinned. Pinning needs `model_request_unit` too.
        """
        return None

    def model_request_unit(self, name: str) -> str | None:
        """The unit the engine records for a model request of the case `name`, which `expected_units` leaves out.

        A run streams its requests when a capability listens for events, which can rename the unit. Pinning needs it.
        """
        return None

    def marks(self, name: str) -> Collection[pytest.MarkDecorator]:
        """The pytest marks to put on the test of the case or scenario `name`."""
        return ()

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        for argname, names in (('case_name', self.cases), ('scenario_name', self.scenarios)):
            if argname in metafunc.fixturenames:
                params = [pytest.param(name, marks=self.marks(name), id=name) for name in sorted(names)]
                metafunc.parametrize(argname, params)

    async def test_case(
        self,
        case_name: str,
        engine: DurableEngine,
        can_fork_mid_run: bool,
        count_requests: Callable[[], Mapping[str, int]],
        tmp_path: Path,
    ) -> None:
        """A case's calls, made one per response, recover the way the cell says."""
        await self._check(case_name, self.cases[case_name], None, engine, can_fork_mid_run, count_requests, tmp_path)

    async def test_scenario(
        self,
        scenario_name: str,
        engine: DurableEngine,
        can_fork_mid_run: bool,
        count_requests: Callable[[], Mapping[str, int]],
        tmp_path: Path,
    ) -> None:
        """What a scenario's run observes, its events and enqueued messages, comes back the same, each once."""
        scenario = self.scenarios[scenario_name]
        await self._check(
            scenario_name, scenario.calls, scenario.observed, engine, can_fork_mid_run, count_requests, tmp_path
        )

    async def _check(
        self,
        name: str,
        calls: Calls,
        observed: tuple[str, ...] | None,
        engine: DurableEngine,
        can_fork_mid_run: bool,
        count_requests: Callable[[], Mapping[str, int]],
        directory: Path,
    ) -> None:
        _NESTED.clear()

        async def counts() -> Counter[str]:
            # The stand-ins may count in files, so the count is read off the event loop.
            counted = await anyio.to_thread.run_sync(count_requests)  # workflow-nondeterminism-ok: test-side only
            return Counter({**counted, **_NESTED})

        agent_name = re.sub(r'\W', '_', name) + '_agent'
        observer = None if observed is None else agent_name
        try:
            # A build may write the files its stand-ins read, so it runs off the event loop. A capability that
            # refuses the engine may already refuse when it's built, which the cell's level judges like any refusal.
            built = await anyio.to_thread.run_sync(calls.build, directory)  # workflow-nondeterminism-ok: test-side
            agent = _agent(agent_name, calls.made, built, engine.durability(), observer=observer)
        except ImportError as error:
            pytest.skip(f'{name} needs an optional dependency that is not installed: {error}')
        except UserError as error:
            run = EngineRun('', error)
        else:
            run = await _observed(engine.run(agent, PROMPT), observer)

        if isinstance(run.output, BaseException):
            breaks = [f'the run raised {_describe(run.output)}']
            breaks += await self._refusal_breaks(engine, run, observer, counts) if run.units or run.failed else []
        else:
            breaks = self._run_breaks(name, calls, observed, run, await counts())
            breaks += await self._recovery_breaks(engine, run, observer, can_fork_mid_run, counts)
        self._assert_cell(name, engine, run, breaks)

    def _run_breaks(
        self, name: str, calls: Calls, observed: tuple[str, ...] | None, run: EngineRun, first: Counter[str]
    ) -> list[str]:
        """How the first run breaks the cell: its answer, its requests, its units and what it observed."""
        breaks = [] if run.output == ANSWER else [f'the run returned {_describe(run.output)}, not {ANSWER!r}']
        if repeated := _repeated(first):
            breaks.append(f'a request was made more than once: {repeated}')
        if observed is not None and tuple(run.observed) != observed:
            breaks.append(f'the run observed {run.observed}, not {list(observed)}')
        expected = self.expected_units(name)
        if expected is not None and (model_request := self.model_request_unit(name)) is not None:
            requests = run.units.count(model_request)
            if requests != len(calls.made) + calls.follow_ups + 1:
                breaks.append(
                    f'the run recorded {requests} `{model_request}` units, not one per call, one per follow-up and '
                    'one to answer'
                )
            recorded = tuple(unit for unit in run.units if unit != model_request)
            if recorded != tuple(expected):
                breaks.append(
                    f'the calls record {recorded!r}, not {tuple(expected)!r}. A recorded name is what a run from an '
                    'earlier release recovers by: change it only with a compatibility note'
                )
        return breaks

    async def _recovery_breaks(
        self,
        engine: DurableEngine,
        run: EngineRun,
        observer: str | None,
        can_fork_mid_run: bool,
        counts: Callable[[], Awaitable[Counter[str]]],
    ) -> list[str]:
        """How recovering `run` breaks the cell: under replay, it must recover the same and repeat no request."""
        first = await counts()
        recovered = await _observed(engine.recover(run), observer)
        after = await counts()
        if self.recovery.mode == 'restart':
            repeated = after - first
            breaks = (
                [] if repeated == first else [f'a restart made {dict(repeated)} again, not each of {dict(first)} once']
            )
            if recovered.output != run.output:
                breaks.append(f'recovery returned {_describe(recovered.output)}, not {run.output!r}')
            return breaks
        breaks = _differences('recovery', run, recovered)
        if after != first:
            breaks.append(f'recovery made a request again: {dict(first)} became {dict(after)}')
        if can_fork_mid_run and (kept := len(run.units) // 2):
            # The units past the cut run again against what the whole first run left behind, such as the files it
            # wrote, so they may differ from the first run's: the ones before it must not.
            recovered = await _observed(engine.recover(run, after=kept), observer)
            breaks += _differences(f'recovery from unit {kept}', run, recovered, kept=kept)
        return breaks

    async def _refusal_breaks(
        self, engine: DurableEngine, run: EngineRun, observer: str | None, counts: Callable[[], Awaitable[Counter[str]]]
    ) -> list[str]:
        """How a run refused mid-run breaks the rule that the units it recorded first are unaffected."""
        before = await counts()
        recovered = await _observed(engine.recover(run), observer)
        after = await counts()
        breaks: list[str] = []
        if len(run.failed) > 1:
            breaks.append(f'the refused run recorded more than the refused unit as failed: {run.failed}')
        if completed := sorted(set(run.failed) & set(run.units)):
            breaks.append(f'the refused run also recorded a result for the unit that raised: {completed}')
        if recovered.units != run.units:
            breaks.append(f'recovering the refused run recorded {recovered.units}, not {run.units}')
        if recovered.failed != run.failed:
            breaks.append(f'recovering the refused run failed {recovered.failed}, not {run.failed}')
        if not isinstance(recovered.output, UserError) or str(recovered.output) != str(run.output):
            breaks.append(f'recovering the refused run returned {_describe(recovered.output)}, not the refusal')
        if self.recovery.mode == 'replay' and after != before:
            breaks.append(f'recovering the refused run made a request again: {dict(before)} became {dict(after)}')
        return breaks

    def _assert_cell(self, name: str, engine: DurableEngine, run: EngineRun, breaks: list[str]) -> None:
        cell = self.support.get(name)
        if cell is None:
            if breaks and self.unlisted == 'pending':
                pytest.xfail(f'{name} is pending under {engine.name}: declare its support ({"; ".join(breaks)})')
            cell = Support('supported')
        where = f'{name} under {engine.name}'
        if cell.level == 'untested':
            pytest.fail(f'{where} is declared `untested`: run it, or declare it `refused` or `broken`')
        elif cell.level in ('supported', 'limited'):
            assert not breaks, f'{where}: ' + '; '.join(breaks)
            if cell.check is not None:
                cell.check(run)
        elif cell.level == 'refused':
            assert cell.match is not None
            refused = isinstance(run.output, UserError) and re.search(cell.match, str(run.output)) is not None
            assert refused, (
                f'{where} no longer raises {cell.match!r}: update its cell ({"; ".join(breaks) or "it passes"})'
            )
            assert breaks[1:] == [], f'{where} is refused, but not before it records anything: ' + '; '.join(breaks[1:])
        else:
            assert cell.match is not None
            assert any(re.search(cell.match, line) for line in breaks), (
                f'{where} no longer shows {cell.match!r}: update its cell ({"; ".join(breaks) or "it now passes"})'
            )


def _agent(
    name: str,
    made: Sequence[ToolCall],
    built: Sequence[AbstractCapability[None] | AbstractToolset[None]],
    durability: BaseDurabilityCapability[None],
    *,
    observer: str | None,
) -> Agent[None, str]:
    """The agent that makes the calls in `made` against `built`, under `durability`.

    With an `observer`, it streams, and records what it observes under that name.
    """
    return Agent(
        _scripted_model(made),
        name=name,
        deps_type=type(None),
        toolsets=[part for part in built if isinstance(part, AbstractToolset)],
        capabilities=[
            *([] if observer is None else [_Recorder(observer)]),
            *(part for part in built if isinstance(part, AbstractCapability)),
            durability,
        ],
    )


async def _observed(running: Awaitable[EngineRun], observer: str | None) -> EngineRun:
    """The run `running` returns, with what the recorder named `observer`, if any, saw it observe."""
    run = await running
    if observer is not None:
        run.observed = _OBSERVED.pop(observer, [])
    return run


def _differences(label: str, run: EngineRun, recovered: EngineRun, *, kept: int | None = None) -> list[str]:
    """How `recovered` differs from `run`: its output, its units, and what it observed, with nothing lost or added.

    With `kept`, only the first `kept` units, which the recovery kept, must be the same.
    """
    breaks: list[str] = []
    if recovered.output != run.output:
        breaks.append(f'{label} returned {_describe(recovered.output)}, not {run.output!r}')
    if recovered.units[:kept] != run.units[:kept]:
        breaks.append(f'{label} recorded {recovered.units}, not {run.units}')
    lost = sorted((Counter(run.observed) - Counter(recovered.observed)).elements())
    added = sorted((Counter(recovered.observed) - Counter(run.observed)).elements())
    breaks += [f'{label} lost {lost}'] if lost else []
    breaks += [f'{label} added {added}'] if added else []
    return breaks
