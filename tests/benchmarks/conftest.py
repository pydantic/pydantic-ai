from __future__ import annotations

import os

import pytest


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
