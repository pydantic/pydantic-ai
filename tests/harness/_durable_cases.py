"""What the built-in engines' `DurableEngineSuite` subclasses share: the cases and scenarios, and how requests count.

`tests/harness/test_durable_tool_io.py` runs the suite under DBOS, and `tests/harness/test_durable_temporal_replay.py`
under Temporal. `BuiltInEngineSuite` feeds them every case D1 applies to and every scenario `SCENARIO_RUNS` scripts,
and pins the units each case records to its `durable_steps`.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal, ParamSpec, TypeVar
from unittest.mock import patch

import pytest

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec.conformance import DurableEngineSuite
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness.capability_creation import CapabilityStore
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.memory import InMemoryMemoryStore
from pydantic_ai_harness.planning import SqlitePlanStore

from ..append_only import ToolCall, skip_unless_installed
from ..capability_cases import SCENARIO_RUNS, Contract, Scenario
from ._contracts import CASES, PUBLIC
from ._counting_clients import AWS_CLI_COUNT_VARIABLE, REQUESTS

P = ParamSpec('P')
T = TypeVar('T')


def build_case(name: str, directory: Path) -> list[AbstractCapability[None]]:
    """What the calls of the case of `name` run against, built in `directory` with the credentials it checks."""
    case = CASES[name]
    assert case.calls is not None
    with patch.dict(os.environ, case.env):
        built = case.calls.build(directory)
    return [built] if isinstance(built, AbstractCapability) else list(built)


@dataclass(frozen=True)
class SuiteCase:
    """A case's calls, as the suite runs them: skipped when its optional dependency isn't installed."""

    name: str
    made: tuple[ToolCall, ...]
    follow_ups: int = 0

    def build(self, directory: Path, /) -> Sequence[AbstractCapability[None] | AbstractToolset[None]]:
        skip_unless_installed(self.name, CASES[self.name].requires)
        return build_case(self.name, directory)


@dataclass(frozen=True)
class SuiteScenario:
    """A scenario `SCENARIO_RUNS` scripts, as the suite runs it: one `work` call, then an answer."""

    name: Scenario
    made: tuple[ToolCall, ...] = (('work', {}),)
    follow_ups: int = 0

    @property
    def observed(self) -> tuple[str, ...]:
        return SCENARIO_RUNS[self.name].observed

    def build(self, directory: Path, /) -> Sequence[AbstractCapability[None] | AbstractToolset[None]]:
        del directory
        return list(SCENARIO_RUNS[self.name].build())


def suite_cases(contract: Contract) -> dict[str, SuiteCase]:
    """The public cases `contract` applies to, for the suite to run."""
    return {
        name: SuiteCase(name, case.calls.made, case.calls.follow_ups)
        for name, case in CASES.items()
        if name in PUBLIC and contract.applies(name, case) and case.calls is not None
    }


SUITE_SCENARIOS = {scenario: SuiteScenario(scenario) for scenario in SCENARIO_RUNS}
"""Every scenario `SCENARIO_RUNS` scripts, for the suite to run."""


def _counted(key: str, method: Callable[P, T]) -> Callable[P, T]:
    """`method`, counting each call in `REQUESTS` under `key`: an async method's call, when its request starts."""

    def counted(*args: P.args, **kwargs: P.kwargs) -> T:
        REQUESTS[key] += 1
        return method(*args, **kwargs)

    return counted


def counting_requests(monkeypatch: pytest.MonkeyPatch, directory: Path) -> Callable[[], Counter[str]]:
    """Count every request the cases' stand-ins, the `aws` stand-in, and the stores make, from now on.

    Both built-in suites count with it, so a request means the same thing in each. `directory` holds the `aws` count.
    """
    REQUESTS.clear()
    aws_count = directory / 'aws_cli_count'
    monkeypatch.setenv(AWS_CLI_COUNT_VARIABLE, str(aws_count))

    monkeypatch.setattr(CapabilityStore, 'write', _counted('store.write', CapabilityStore.write))
    monkeypatch.setattr(CapabilityStore, 'list_all', _counted('store.list_all', CapabilityStore.list_all))
    monkeypatch.setattr(CapabilityStore, 'disable', _counted('store.disable', CapabilityStore.disable))
    # A memory write first looks up its idempotency key; the write is what recovery must not repeat.
    monkeypatch.setattr(InMemoryMemoryStore, 'write', _counted('memory.write', InMemoryMemoryStore.write))
    # The plan tools read the store many times; a write is what recovery must not repeat.
    monkeypatch.setattr(SqlitePlanStore, 'set_items', _counted('plan.set_items', SqlitePlanStore.set_items))
    monkeypatch.setattr(SqlitePlanStore, 'add_item', _counted('plan.add_item', SqlitePlanStore.add_item))
    monkeypatch.setattr(
        SnapshotHistorySource, 'list_runs', _counted('history.list_runs', SnapshotHistorySource.list_runs)
    )

    def snapshot() -> Counter[str]:
        counts = Counter(REQUESTS)
        if aws_count.exists():
            counts['aws_cli'] = len(aws_count.read_text().splitlines())
        return counts

    return snapshot


class BuiltInEngineSuite(DurableEngineSuite):
    """`DurableEngineSuite` with the built-in cases and scenarios, the stand-ins' counts, and the case units pinned.

    The cases' units are pinned by their `durable_steps`, and their model request unit by `durable_steps.streams`; a
    scenario's run streams for its recorder, so its units aren't pinned.
    """

    durable_steps: ClassVar[Literal['dbos', 'temporal']]
    """The `DurableSteps` field that pins the units each case records under the engine."""
    model_request_units: ClassVar[tuple[str, str]]
    """The engine's unit for a model request, and for a streamed one."""
    scenarios = SUITE_SCENARIOS

    @pytest.fixture
    def count_requests(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[], Mapping[str, int]]:
        return counting_requests(monkeypatch, tmp_path)

    def expected_units(self, name: str) -> Sequence[str] | None:
        calls = CASES[name].calls if name in CASES else None
        return None if calls is None else getattr(calls.durable_steps, self.durable_steps)

    def model_request_unit(self, name: str) -> str | None:
        calls = CASES[name].calls if name in CASES else None
        requested, streamed = self.model_request_units
        return None if calls is None else streamed if calls.durable_steps.streams else requested

    def marks(self, name: str) -> Collection[pytest.MarkDecorator]:
        calls = CASES[name].calls if name in CASES else None
        if calls is None:
            return ()
        marks: list[pytest.MarkDecorator] = []
        if reason := calls.moves_cache_prefix:
            marks.append(pytest.mark.moves_cache_prefix(reason=reason))
        if reason := calls.launches_python:
            marks.append(pytest.mark.subprocess(reason=reason, conditional=True))
        return marks
