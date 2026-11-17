from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter
from typing_extensions import TypedDict


class Workflow(TypedDict):
    jobs: dict[str, object]


class ClassifierStep(TypedDict):
    run: str


class ClassifierJob(TypedDict):
    steps: list[ClassifierStep]


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Check imports in isolated benchmark runs, without other suites' collection imports."""
    benchmark_dir = Path(__file__).resolve().parent
    if not session.config.args or any(
        not Path(argument.split('::', 1)[0]).resolve().is_relative_to(benchmark_dir) for argument in session.config.args
    ):
        return
    root = benchmark_dir.parent.parent
    loaded: set[str] = set()
    for module in tuple(sys.modules.values()):
        # Inspect recorded metadata without invoking module-level __getattr__ hooks.
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
    try:
        loaded_workflow: object = yaml.safe_load((root / '.github/workflows/benchmark.yml').read_text())
        workflow = TypeAdapter(Workflow).validate_python(loaded_workflow)
        classify_job = TypeAdapter(ClassifierJob).validate_python(workflow['jobs']['classify'])
        lines = classify_job['steps'][0]['run'].splitlines()
        start = lines.index('affects_benchmarks() {')
        end = lines.index('}', start) + 1
        classifier = '\n'.join(lines[start:end])
    except (OSError, yaml.YAMLError, ValueError, LookupError) as error:
        pytest.exit(
            'Cannot extract `affects_benchmarks` from the `classify` step in '
            '`.github/workflows/benchmark.yml`; keep its opening and closing lines unindented or update '
            f'`tests/benchmarks/conftest.py` to match the workflow structure:\n{error}',
            returncode=1,
        )
    # This executes only in the benchmark/test context, never in the token-bearing classifier job.
    result = subprocess.run(
        [
            'bash',
            '-e',
            '-c',
            classifier + '\nuncovered=0\nfor filename; do\n'
            '  if ! affects_benchmarks "$filename"; then\n'
            '    printf \'%s\\n\' "$filename"\n'
            '    uncovered=1\n'
            '  fi\ndone\nexit "$uncovered"\n',
            'benchmark-dependencies',
            *sorted(loaded),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        pytest.exit(
            'Loaded benchmark dependencies would skip CodSpeed; update `affects_benchmarks` in '
            '`.github/workflows/benchmark.yml`:\n' + result.stdout + result.stderr,
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
