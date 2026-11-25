"""The engines' support tables: every engine runs `DurableEngineSuite` or is pending, and every cell names a real one.

Each engine that runs the suite declares its cells in its own package (`ENGINE_SUPPORT_MODULES`), and its subclass
checks them: `TestDBOS` in `tests/harness/test_durable_tool_io.py`, `TestTemporal` in
`tests/harness/test_durable_temporal_replay.py`. `scripts/durable_exec_support.py` renders them into the docs.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from pydantic_ai.durable_exec.conformance import EngineRun, Recovery, Support

from .._inline_snapshot import snapshot
from ..append_only import discover_capabilities
from ..capability_cases import (
    DURABLE_DBOS,
    DURABLE_TEMPORAL,
    ENGINE_CAPABILITIES,
    ENGINE_SUPPORT_MODULES,
    ENGINES,
    ID_LESS_TOOLSETS,
    SCENARIO_RUNS,
    SCENARIOS,
    Contract,
    Engine,
)
from ._contracts import CASES, PUBLIC
from .capability_cases import HARNESS_CAPABILITY_PACKAGES

REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

import durable_exec_contract  # noqa: E402
import durable_exec_support  # noqa: E402

DURABILITY_PACKAGES = ('pydantic_ai.durable_exec', *HARNESS_CAPABILITY_PACKAGES)
"""The packages that define durability capabilities, one per engine."""

SUITES: Mapping[Engine, tuple[str, str]] = {
    'dbos': ('tests/harness/test_durable_tool_io.py', 'TestDBOS'),
    'temporal': ('tests/harness/test_durable_temporal_replay.py', 'TestTemporal'),
}
"""The `DurableEngineSuite` subclass of each engine that runs the suite: its module and class."""

PENDING_ENGINES: tuple[Engine, ...] = ('prefect', 'aws_lambda', 'absurd', 'pgtask', 'azure_durable_task', 'render')
"""The engines that don't run the suite yet. It may only shrink: a new engine joins with a suite of its own."""


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


def _table(engine: Engine) -> ModuleType:
    """The support table of `engine`, skipped when its SDK, which its package imports, isn't installed."""
    try:
        return importlib.import_module(ENGINE_SUPPORT_MODULES[engine])
    except ImportError as error:  # pragma: lax no cover
        pytest.skip(f'the {engine} SDK is not installed: {error}')


def test_every_durability_capability_is_an_engine() -> None:
    """A new engine can't land unnamed: its durability capability must be in `ENGINE_CAPABILITIES`."""
    found = {
        ref.name for ref in discover_capabilities(DURABILITY_PACKAGES).values() if _is_durability(ref.name, ref.module)
    }
    assert found == set(ENGINE_CAPABILITIES.values())


def test_every_durability_capability_has_a_suite() -> None:
    """Every engine runs `DurableEngineSuite` against a support table of its own, or is still pending."""
    assert set(SUITES).isdisjoint(PENDING_ENGINES)
    assert {*SUITES, *PENDING_ENGINES} == set(ENGINES) == set(ENGINE_CAPABILITIES)
    assert PENDING_ENGINES == snapshot(('prefect', 'aws_lambda', 'absurd', 'pgtask', 'azure_durable_task', 'render'))
    assert set(ENGINE_SUPPORT_MODULES) == set(SUITES)
    for path, class_name in SUITES.values():
        tree = ast.parse((REPO_ROOT / path).read_text())
        assert any(isinstance(node, ast.ClassDef) and node.name == class_name for node in tree.body), path


def test_the_manifest_and_the_docs_name_every_engine() -> None:
    """Adding an engine touches the contract manifest's roots, the docs tables and the suites alike."""
    assert set(durable_exec_contract.ENGINES) - {'durable_exec'} == set(ENGINES)
    assert set(durable_exec_support.TABLES) == set(SUITES)
    assert set(durable_exec_support.PENDING) == set(PENDING_ENGINES)
    assert {module for _, module in durable_exec_support.TABLES.values()} == {
        ENGINE_SUPPORT_MODULES[engine] for engine in SUITES
    }


@pytest.mark.parametrize('engine', sorted(SUITES))
def test_every_cell_names_a_real_case_or_scenario(engine: Engine) -> None:
    """A cell only means something for a case whose calls the suite runs, or for a scenario."""
    table = _table(engine)
    cells: Mapping[str, Support] = table.SUPPORT
    assert isinstance(table.RECOVERY, Recovery)
    for name, cell in cells.items():
        assert isinstance(cell, Support)
        assert cell.level != 'untested', f'{name}: a suite runs every cell, so none is `untested`'
        if name in SCENARIOS:
            continue
        assert name in PUBLIC, f'{engine} names {name}, which is neither a scenario nor a public capability'
        assert CASES[name].calls is not None, f'{engine} names {name}, which has no `calls` to run'


def test_temporal_refuses_the_id_less_toolsets() -> None:
    cells: Mapping[str, Support] = _table('temporal').SUPPORT
    refused = {name for name, cell in cells.items() if cell.issue == 9233}
    assert refused == set(ID_LESS_TOOLSETS)


def test_the_d1_exemptions_are_the_cases_a_table_refuses_or_breaks() -> None:
    contracts: dict[Engine, Contract] = {'dbos': DURABLE_DBOS, 'temporal': DURABLE_TEMPORAL}
    for engine, contract in contracts.items():
        cells: Mapping[str, Support] = _table(engine).SUPPORT
        exempt = {name for name, cell in cells.items() if name in CASES and cell.level in ('refused', 'broken')}
        assert (set(contract.exempt), len(contract.exempt)) == (exempt, len(exempt))
    assert DURABLE_TEMPORAL.exempt['Planning'].reason.endswith('https://github.com/pydantic/pydantic-ai/issues/7971')


_check: Callable[[EngineRun], None] = lambda run: None  # noqa: E731


@pytest.mark.parametrize(
    ('make', 'error'),
    [
        (lambda: Support('limited', check=_check), 'a `limited` cell needs a `reason`'),
        (lambda: Support('limited', 'foreground'), 'asserts its limit in `check`'),
        (lambda: Support('supported', check=_check), 'asserts its limit in `check`'),
        (lambda: Support('refused', 'no id'), 'names what it matches'),
        (lambda: Support('supported', match='boom'), 'names what it matches'),
        (lambda: Support('broken', 'wrong order', match='lost'), 'a `broken` cell needs the `issue`'),
        (lambda: Support('untested', 'no suite'), 'an? `untested` cell needs the `issue`'),
        (lambda: Recovery('restart', 'reruns every call'), 'a `restart` recovery needs'),
    ],
)
def test_a_declaration_says_what_it_needs(make: Callable[[], object], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        make()


SCENARIO_OWNERS_BESIDES_CAPABILITIES = ('RunContext',)
"""What owns a scenario that no capability does: `RunContext`'s `emit` and `enqueue`."""


def test_every_scenario_names_its_owner() -> None:
    """A scenario is `<owner>.<scenario>`, and it is no case: D1's exemptions and `calls` never see it."""
    for scenario in SCENARIOS:
        owner, _, name = scenario.partition('.')
        assert (owner in PUBLIC or owner in SCENARIO_OWNERS_BESIDES_CAPABILITIES) and name, scenario
        assert scenario not in CASES, scenario


def test_the_suites_run_every_run_context_scenario() -> None:
    """No capability's tests run a `RunContext` scenario, so the engine suites must, through `SCENARIO_RUNS`."""
    run_context = {scenario for scenario in SCENARIOS if scenario.startswith('RunContext.')}
    assert run_context <= set(SCENARIO_RUNS)


def test_the_compatibility_page_is_current(capsys: pytest.CaptureFixture[str]) -> None:
    """`docs/durable_execution/compatibility.md` shows the tables as the engines declare them."""
    for engine in SUITES:
        _table(engine)
    assert durable_exec_support.main([]) == 0, capsys.readouterr().err
