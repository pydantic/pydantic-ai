"""The engines' support tables: every engine runs `DurableEngineSuite` or is pending, and every cell names a real one.

Each engine that runs the suite declares its cells in its own package (`ENGINE_SUPPORT_MODULES`), and its subclass
checks them: `TestDBOS` in `tests/harness/test_durable_tool_io.py`, `TestTemporal` in
`tests/harness/test_durable_temporal_replay.py`. `scripts/durable_exec_support.py` renders them into the docs.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from pydantic_ai.capabilities.conformance import CALLS as CORE_CALLS, SCENARIO_RUNS
from pydantic_ai.durable_exec.conformance import EngineRun, Recovery, Support
from pydantic_ai_harness.conformance import CALLS as HARNESS_CALLS, HarnessDurableEngineSuite

from .. import capability_cases as core
from .._inline_snapshot import snapshot
from ..append_only import discover_capabilities
from ..capability_cases import (
    DURABLE_DBOS,
    DURABLE_TEMPORAL,
    ENGINE_CAPABILITIES,
    ENGINE_SUPPORT_MODULES,
    ENGINES,
    ID_LESS_TOOLSETS,
    PENDING_ENGINES,
    SCENARIOS,
    UNTESTED_ISSUE,
    Contract,
    Engine,
    EngineTable,
    engine_table,
)
from ._contracts import CASES, PUBLIC
from ._durable_cases import suite_cases
from .capability_cases import CASES as HARNESS_CASES, HARNESS_CAPABILITY_PACKAGES

REPO_ROOT = Path(__file__).parents[2]


def _script(name: str) -> ModuleType:
    """The module of `scripts/<name>.py`, imported from its path: `scripts/` is no package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / 'scripts' / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its module's annotations through `sys.modules`.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


durable_exec_contract = _script('durable_exec_contract')
durable_exec_support = _script('durable_exec_support')

DURABILITY_PACKAGES = ('pydantic_ai.durable_exec', *HARNESS_CAPABILITY_PACKAGES)
"""The packages that define durability capabilities, one per engine."""

SUITES: Mapping[Engine, tuple[str, str]] = {
    'dbos': ('tests/harness/test_durable_tool_io.py', 'TestDBOS'),
    'temporal': ('tests/harness/test_durable_temporal_replay.py', 'TestTemporal'),
}
"""The `DurableEngineSuite` subclass of each engine that runs the suite: its module and class."""


def _module_file(module: str) -> Path:
    """The source file of `module`, found without importing it, as its package may need an engine's SDK."""
    top, *rest = module.split('.')
    spec = importlib.util.find_spec(top)
    assert spec is not None and spec.submodule_search_locations is not None
    path = Path(next(iter(spec.submodule_search_locations))).joinpath(*rest)
    return path / '__init__.py' if path.is_dir() else path.with_suffix('.py')


def _is_durability(name: str, module: str) -> bool:
    """Whether the class `name` in `module` subclasses `BaseDurabilityCapability` directly, as every engine's does."""
    tree = ast.parse(_module_file(module).read_text())
    [cls] = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    bases = [base.value if isinstance(base, ast.Subscript) else base for base in cls.bases]
    return any(isinstance(base, ast.Name) and base.id == 'BaseDurabilityCapability' for base in bases)


def _table(engine: Engine) -> EngineTable:
    """The support table of `engine`, skipped when its SDK, which its package imports, isn't installed."""
    try:
        return engine_table(engine)
    except ImportError as error:  # pragma: lax no cover
        pytest.skip(f'the {engine} SDK is not installed: {error}')


def test_every_durability_capability_is_an_engine() -> None:
    """A new engine can't land unnamed: its durability capability must be in `ENGINE_CAPABILITIES`.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    found = {
        ref.name for ref in discover_capabilities(DURABILITY_PACKAGES).values() if _is_durability(ref.name, ref.module)
    }
    assert found == set(ENGINE_CAPABILITIES.values())


def test_every_durability_capability_has_a_suite() -> None:
    """Every engine runs `DurableEngineSuite` against a support table of its own, or is still pending.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    assert set(SUITES).isdisjoint(PENDING_ENGINES)
    assert {*SUITES, *PENDING_ENGINES} == set(ENGINES) == set(ENGINE_CAPABILITIES)
    assert PENDING_ENGINES == snapshot(('prefect', 'aws_lambda', 'absurd', 'pgtask', 'azure_durable_task', 'render'))
    assert set(ENGINE_SUPPORT_MODULES) == set(ENGINES)
    for path, class_name in SUITES.values():
        tree = ast.parse((REPO_ROOT / path).read_text())
        assert any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body), path


def test_measured_cells_only_go_up() -> None:
    """How many cases each engine's suite measures: an engine that joins the suite raises its count, and none lowers it.

    Every cell of an engine with a suite is measured, whatever its level; an engine still pending measures none.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    measured = sum(name in PUBLIC and case.calls is not None for name, case in CASES.items())
    counts = {engine: measured if engine in SUITES else 0 for engine in ENGINES}
    assert counts == snapshot(
        {
            'dbos': 51,
            'temporal': 51,
            'prefect': 0,
            'aws_lambda': 0,
            'absurd': 0,
            'pgtask': 0,
            'azure_durable_task': 0,
            'render': 0,
        }
    )


def test_the_manifest_and_the_docs_name_every_engine() -> None:
    """Adding an engine touches the contract manifest's roots, the docs tables and the suites alike.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    assert set(durable_exec_contract.ENGINES) - {'durable_exec'} == set(ENGINES)
    assert {engine: module for engine, (_, module) in durable_exec_support.TABLES.items()} == ENGINE_SUPPORT_MODULES
    assert durable_exec_support.PENDING == set(PENDING_ENGINES)


@pytest.mark.parametrize('engine', ENGINES)
def test_every_cell_names_a_real_case_or_scenario(engine: Engine) -> None:
    """A cell only means something for a case whose calls the suite runs, or for a scenario.

    An engine with a suite lists only the cells that aren't `supported`; one still pending lists only the scenarios
    a test measures, as nothing runs its cases.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    table = _table(engine)
    cells = table.support
    assert isinstance(table.recovery, Recovery)
    for name, cell in cells.items():
        assert isinstance(cell, Support)
        assert cell.level != 'untested', f'{name}: an unlisted cell is what is untested'
        assert engine in PENDING_ENGINES or cell.level != 'supported', f'{name}: an unlisted cell is `supported`'
        if name in SCENARIOS:
            continue
        assert engine not in PENDING_ENGINES, (
            f'{engine} names the case {name}, but no suite runs cases under it yet, so nothing checks the cell: list '
            f'only scenarios a test measures until it runs `DurableEngineSuite` (#{UNTESTED_ISSUE})'
        )
        assert name in PUBLIC, f'{engine} names {name}, which is neither a scenario nor a public capability'
        assert CASES[name].calls is not None, f'{engine} names {name}, which has no `calls` to run'


def test_temporal_refuses_the_id_less_toolsets() -> None:
    """Temporal refuses exactly the cases whose toolset has no id.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    cells = _table('temporal').support
    refused = {name for name, cell in cells.items() if cell.issue == 9233}
    assert refused == set(ID_LESS_TOOLSETS)


def test_the_d1_exemptions_are_the_cases_a_table_refuses_or_breaks() -> None:
    """D1's exemptions are the cases an engine's table refuses or breaks, and no others.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    contracts: dict[Engine, Contract] = {'dbos': DURABLE_DBOS, 'temporal': DURABLE_TEMPORAL}
    for engine, contract in contracts.items():
        cells = _table(engine).support
        exempt = {name for name, cell in cells.items() if name in CASES and cell.level in ('refused', 'broken')}
        assert (set(contract.exempt), len(contract.exempt)) == (exempt, len(exempt))
    assert DURABLE_TEMPORAL.exempt['Planning'].reason.endswith('https://github.com/pydantic/pydantic-ai/issues/7971')


_check: Callable[[EngineRun], None] = lambda run: None  # noqa: E731


@pytest.mark.parametrize(
    ('make', 'error'),
    [
        (lambda: Support('limited', check=_check, issue=1), 'a `limited` cell needs a `reason`'),
        (lambda: Support('limited', 'foreground', issue=1), 'asserts its limit in `check`'),
        (lambda: Support('limited', 'foreground', check=_check), 'a `limited` cell needs the `issue`'),
        (lambda: Support('supported', check=_check), 'asserts its limit in `check`'),
        (lambda: Support('refused', 'no id'), 'names what it matches'),
        (lambda: Support('supported', match='boom'), 'names what it matches'),
        (lambda: Support('broken', 'wrong order', match='lost'), 'a `broken` cell needs the `issue`'),
        (lambda: Support('untested', 'no suite'), 'an? `untested` cell needs the `issue`'),
        (lambda: Recovery('restart', 'reruns every call'), 'a `restart` recovery needs'),
    ],
)
def test_a_declaration_says_what_it_needs(make: Callable[[], object], error: str) -> None:
    """A support declaration missing what its level or mode needs is rejected when built.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    with pytest.raises(ValueError, match=error):
        make()


SCENARIO_OWNERS_BESIDES_CAPABILITIES = ('RunContext',)
"""What owns a scenario that no capability does: `RunContext`'s `emit` and `enqueue`."""


def test_every_scenario_names_its_owner() -> None:
    """A scenario is `<owner>.<scenario>`, and it is no case: D1's exemptions and `calls` never see it.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    for scenario in SCENARIOS:
        owner, _, name = scenario.partition('.')
        assert (owner in PUBLIC or owner in SCENARIO_OWNERS_BESIDES_CAPABILITIES) and name, scenario
        assert scenario not in CASES, scenario


def test_the_suites_run_every_run_context_scenario() -> None:
    """No capability's tests run a `RunContext` scenario, so the engine suites must, through `SCENARIO_RUNS`.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    run_context = {scenario for scenario in SCENARIOS if scenario.startswith('RunContext.')}
    assert run_context <= set(SCENARIO_RUNS) <= set(SCENARIOS)


def test_the_cases_refer_to_the_public_calls() -> None:
    """A public `Calls` is defined once, in its package's `conformance` module, and its case refers to it.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    for public, cases in ((CORE_CALLS, core.CASES), (HARNESS_CALLS, HARNESS_CASES)):
        for name, calls in public.items():
            case = cases[name].calls
            assert name in PUBLIC and case is not None, f'{name} has public calls but no case that refers to them'
            assert (case.made, case.build, case.follow_ups) == (calls.made, calls.build, calls.follow_ups), name
            assert case.build is calls.build, f'{name} defines its calls again instead of `Calls.of(CALLS[{name!r}])`'


def test_test_only_cases_only_shrink() -> None:
    """The cases whose calls need a stand-in from the tests, which only the built-in engines run.

    A third-party engine's `HarnessDurableEngineSuite` runs every other case the built-in ones do. A case leaves
    this list when its stand-in goes public and its calls move to `CALLS`; none joins it.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    built_in = suite_cases(DURABLE_DBOS)
    assert set(HarnessDurableEngineSuite.cases) <= set(built_in)
    assert sorted(set(built_in) - set(HarnessDurableEngineSuite.cases)) == snapshot(
        [
            'Advisor',
            'BrowserUse',
            'DayAI',
            'ExaAgent',
            'ExaSearch',
            'GitHub',
            'Grain',
            'KeenableSearch',
            'Linear',
            'LocalStack',
            'LogfireMCP',
            'Notion',
            'Ordinal',
            'PostHog',
            'Pylon',
            'Slack',
            'SmartSearch',
            'StackOne',
            'YouResearch',
            'YouSearch',
        ]
    )


def test_the_compatibility_page_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    """`docs/durable_execution/compatibility.md` shows the tables as the engines declare them.

    It reads test tables, docs or source, with no provider exchange to record, so it isn't a VCR test.
    """
    for engine in ENGINES:
        _table(engine)
    assert durable_exec_support.main([]) == 0, capsys.readouterr().err
