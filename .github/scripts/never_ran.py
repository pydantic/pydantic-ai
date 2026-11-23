"""Fail a test that ran in no CI cell.

Every test job writes one JUnit report per pytest session (`PYTEST_JUNIT_DIR`, which `tests/conftest.py` reads) and
uploads them as a `junit-*` artifact. The `never-ran` job downloads them all and runs this script over them, on runs
whose route runs every test job. A test under `tests/` that every report lists as skipped ran nowhere: its skip
condition holds in every cell, so CI never runs it and it can break unseen. It fails unless `NEVER_RAN_EXEMPT` in
`tests/never_ran.py` gives the reason, or `PATH_GATED_JOBS` names the job that runs it and that job was skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT)]
import latest_versions_canary_report as junit
from tests.manifests import assert_classified
from tests.never_ran import NEVER_RAN_EXEMPT, PATH_GATED_JOBS


def never_ran(reports: Iterable[Path], root: Path) -> set[str]:
    """The node ids under `tests/` that `reports` list only as skipped. An expected failure ran."""
    seen: set[str] = set()
    ran: set[str] = set()
    for report in reports:
        for case in junit.testcases(report):
            if (node_id := junit.collected_node_id(case.get('classname', ''), case.get('name', ''), root)) is not None:
                seen.add(node_id)
                if all(result.tag != 'skipped' or result.get('type') == 'pytest.xfail' for result in case):
                    ran.add(node_id)
    return seen - ran


def entry(node_id: str, keys: Iterable[str]) -> str:
    """The key in `keys` that names `node_id`, its file or a directory holding it, or `node_id` itself."""
    return next(
        (
            key
            for key in keys
            if node_id == key
            or node_id.startswith((f'{key}::', f'{key}['))
            or (key[-1] == '/' and node_id.startswith(key))
        ),
        node_id,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fail a test under tests/ that ran in no CI cell.')
    parser.add_argument('reports', type=Path, help='directory holding every JUnit report of the run')
    args = parser.parse_args(argv)
    reports = sorted(args.reports.rglob('*.xml'))
    if not reports:
        print(f'No JUnit report under {args.reports}.')
        return 1
    needs = json.loads(os.environ['NEEDS'])
    # A missing job is a `KeyError`: the `never-ran` job must need every job named here.
    gated = tuple(path for job, paths in PATH_GATED_JOBS.items() if needs[job]['result'] == 'skipped' for path in paths)
    nodes = {node_id for node_id in never_ran(reports, ROOT) if not node_id.startswith(gated)}
    try:
        assert_classified(
            {entry(node_id, NEVER_RAN_EXEMPT) for node_id in nodes},
            NEVER_RAN_EXEMPT,
            complete_walk=True,
            table_name='NEVER_RAN_EXEMPT',
        )
    except AssertionError as error:
        print(
            'Every CI cell skipped these tests, so none ran them. Run them in a cell, or give the reason no cell can '
            'in `NEVER_RAN_EXEMPT` in `tests/never_ran.py`; see "Tests that run in no cell" in `tests/AGENTS.md`.\n'
            f'{error}'
        )
        return 1
    print(f'{len(reports)} reports: every test under tests/ ran in a cell, or `NEVER_RAN_EXEMPT` says why none can.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
