"""The engine x case table, `ENGINE_SUPPORT`: every engine has a column, and every cell says what its level needs.

The durable suites check the cells themselves: `tests/harness/test_durable_tool_io.py` the DBOS column and
`tests/harness/test_durable_temporal_replay.py` the Temporal one. A `refused` cell must still raise what it names,
and a `broken` one must still break, through the contract exemptions `ENGINE_SUPPORT` derives.
"""

from __future__ import annotations

import ast
import importlib.util
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from .._inline_snapshot import snapshot
from ..append_only import discover_capabilities
from ..capability_cases import ENGINE_CAPABILITIES, ENGINE_SUPPORT, ENGINES, Support, engine_support
from ._contracts import CASES, PUBLIC
from .capability_cases import HARNESS_CAPABILITY_PACKAGES

DURABILITY_PACKAGES = ('pydantic_ai.durable_exec', *HARNESS_CAPABILITY_PACKAGES)
"""The packages that define durability capabilities, one per engine."""


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


def test_every_durability_capability_is_an_engine() -> None:
    """A new engine can't land without a column: its durability capability must be in `ENGINE_CAPABILITIES`."""
    found = {
        ref.name for ref in discover_capabilities(DURABILITY_PACKAGES).values() if _is_durability(ref.name, ref.module)
    }
    assert found == set(ENGINE_CAPABILITIES.values())


def test_every_cell_is_a_case_with_calls() -> None:
    """A cell only means something for a case whose calls a suite runs."""
    for name in ENGINE_SUPPORT:
        assert name in PUBLIC, f'`ENGINE_SUPPORT` names {name}, which is not a public capability'
        assert CASES[name].calls is not None, f'`ENGINE_SUPPORT` names {name}, which has no `calls` to run'


def test_untested_cells_only_go_down() -> None:
    """How many cases each engine runs no suite for: test a cell to lower its engine's count, never raise it."""
    untested = Counter(
        engine
        for name, case in CASES.items()
        if name in PUBLIC and case.calls is not None
        for engine in ENGINES
        if engine_support(name, engine).level == 'untested'
    )
    assert dict(untested) == snapshot(
        {
            'prefect': 43,
            'aws_lambda': 43,
            'absurd': 43,
            'pgtask': 43,
            'azure_durable_task': 43,
            'render': 43,
        }
    )


@pytest.mark.parametrize(
    ('make', 'error'),
    [
        (lambda: Support('limited'), 'a `limited` cell needs a `reason`'),
        (lambda: Support('refused', 'no id'), 'names what the engine raises'),
        (lambda: Support('supported', raises='boom'), 'names what the engine raises'),
        (lambda: Support('broken', 'wrong order'), 'a `broken` cell needs the `issue`'),
        (lambda: Support('untested', 'no suite'), 'an? `untested` cell needs the `issue`'),
    ],
)
def test_a_cell_says_what_its_level_needs(make: Callable[[], Support], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        make()


def test_an_exempt_cell_links_its_issue() -> None:
    assert Support('refused', 'no id', raises='id').exempt.reason == 'no id'
    assert Support('broken', 'wrong order', issue=1).exempt.reason == (
        'wrong order: https://github.com/pydantic/pydantic-ai/issues/1'
    )
