"""What the built-in engines' `DurableEngineSuite` subclasses share: the cases and scenarios, and how requests count.

`tests/harness/test_durable_tool_io.py` runs the suite under DBOS, and `tests/harness/test_durable_temporal_replay.py`
under Temporal. `BuiltInEngineSuite` feeds them every case D1 applies to and every scenario the public `SCENARIO_RUNS`
scripts, and pins the units each case records to its `durable_steps`.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import ClassVar, Literal, ParamSpec, TypeVar
from unittest.mock import patch

import pytest

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.conformance import Calls
from pydantic_ai_harness.capability_creation import CapabilityStore
from pydantic_ai_harness.conformance import HarnessDurableEngineSuite
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.memory import InMemoryMemoryStore
from pydantic_ai_harness.planning import SqlitePlanStore

from ..append_only import skip_unless_installed
from ..capability_cases import Contract
from ._contracts import CASES, PUBLIC
from ._counting_clients import AWS_CLI_COUNT_VARIABLE, REQUESTS

P = ParamSpec('P')
T = TypeVar('T')


def build_case(name: str, directory: Path) -> list[AbstractCapability[None]]:
    """What the calls of the case of `name` run against, built in `directory` with the credentials it checks.

    Only a scenario's calls run against a toolset of the agent's own, so a case's are all capabilities.
    """
    case = CASES[name]
    assert case.calls is not None
    with patch.dict(os.environ, case.env):
        built = case.calls.build(directory)
    capabilities = [part for part in built if isinstance(part, AbstractCapability)]
    assert len(capabilities) == len(built), f'the calls of {name} run against a toolset, which only a scenario may'
    return capabilities


def _suite_build(name: str, directory: Path) -> list[AbstractCapability[None]]:
    """`build_case`, skipped when the case's optional dependency isn't installed."""
    skip_unless_installed(name, CASES[name].requires)
    return build_case(name, directory)


def suite_cases(contract: Contract) -> dict[str, Calls]:
    """The public cases `contract` applies to, for the suite to run: the public `CALLS` and the test-only ones."""
    return {
        name: Calls(case.calls.made, build=partial(_suite_build, name), follow_ups=case.calls.follow_ups)
        for name, case in CASES.items()
        if name in PUBLIC and contract.applies(name, case) and case.calls is not None
    }


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


class BuiltInEngineSuite(HarnessDurableEngineSuite):
    """`HarnessDurableEngineSuite` with the test-only cases, the stand-ins' counts, and the case units pinned.

    Its `cases` are the public `CALLS`, core and Harness, plus the cases whose calls need a stand-in from the tests,
    each skipped when its optional dependency isn't installed. The cases' units are pinned by their `durable_steps`,
    and their model request unit by `durable_steps.streams`; a scenario's run streams for its recorder, so its units
    aren't pinned.
    """

    durable_steps: ClassVar[Literal['dbos', 'temporal']]
    """The `DurableSteps` field that pins the units each case records under the engine."""
    model_request_units: ClassVar[tuple[str, str]]
    """The engine's unit for a model request, and for a streamed one."""

    @pytest.fixture
    def count_requests(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Callable[[], Mapping[str, int]]:
        return counting_requests(monkeypatch, tmp_path)

    def expected_units(self, name: str) -> Sequence[str] | None:
        case = CASES.get(name)
        if case is None or case.calls is None:
            return None
        return case.durable_steps.dbos if self.durable_steps == 'dbos' else case.durable_steps.temporal

    def model_request_unit(self, name: str) -> str | None:
        case = CASES.get(name)
        requested, streamed = self.model_request_units
        return None if case is None or case.calls is None else streamed if case.durable_steps.streams else requested

    def marks(self, name: str) -> Collection[pytest.MarkDecorator]:
        case = CASES.get(name)
        if case is None or case.calls is None:
            return ()
        marks: list[pytest.MarkDecorator] = []
        if reason := case.calls.moves_cache_prefix:
            marks.append(pytest.mark.moves_cache_prefix(reason=reason))
        if reason := case.launches_python:
            marks.append(pytest.mark.subprocess(reason=reason, conditional=True))
        return marks
