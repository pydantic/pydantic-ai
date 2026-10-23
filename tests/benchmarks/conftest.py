from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        '--benchmark-write-dependencies',
        action='store_true',
        help='Record imported repository files for auditing the CodSpeed dependency filter.',
    )


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Check imports in isolated benchmark runs, without other suites' collection imports."""
    root = Path(__file__).resolve().parents[2]
    if session.config.args != ['tests/benchmarks']:
        return
    loaded: set[str] = set()
    for module in tuple(sys.modules.values()):
        namespace: dict[str, object] = getattr(module, '__dict__', {})
        filename = namespace.get('__file__')
        if isinstance(filename, str):
            path = Path(filename).resolve()
            if path.is_relative_to(root) and path.relative_to(root).parts[0] in (
                'pydantic_ai_slim',
                'pydantic_graph',
                'pydantic_evals',
                'src',
                'clai',
                'tests',
                'examples',
                'docs',
            ):
                loaded.add(path.relative_to(root).as_posix())
    manifest = Path(__file__).with_name('dependencies.json')
    if session.config.getoption('--benchmark-write-dependencies'):
        manifest.write_text(json.dumps({'loaded': sorted(loaded)}, indent=2) + '\n')
        return
    dependencies = TypeAdapter(dict[str, list[str]]).validate_json(manifest.read_text())['loaded']
    missing = loaded.difference(dependencies)
    if missing:
        pytest.exit(
            'Benchmark dependencies changed; audit dependencies.json and the CodSpeed classifier: '
            + ', '.join(sorted(missing)),
            returncode=1,
        )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        'markers', 'codspeed_only: a large workload that only runs when CodSpeed measures the benchmarks'
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Run large workloads only under CodSpeed.

    Ordinary test runs still check every benchmark's correctness on its smallest workload, which reaches the
    same code paths at a fraction of the cost.
    """
    # The same check `pytest-codspeed` makes: `--codspeed` locally, `CODSPEED_ENV` under the CodSpeed action.
    codspeed_enabled = config.getoption('--codspeed') or 'CODSPEED_ENV' in os.environ
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        (selected if codspeed_enabled or not item.get_closest_marker('codspeed_only') else deselected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = selected
