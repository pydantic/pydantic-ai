"""Opt-in evidence collection for the PostgreSQL pilot; production pytest never loads this plugin."""

import json
import os
from pathlib import Path

import pytest
from postgres_pilot import Event

DIRECTORY = Path(os.environ['POSTGRES_PILOT_EVIDENCE'])
CONTROLLER = 'PYTEST_XDIST_WORKER' not in os.environ


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node: object, ids: list[str]) -> None:
    """Record exactly the collected IDs, including parametrization and xdist groups."""
    path = DIRECTORY / 'collection.json'
    if path.exists():
        assert json.loads(path.read_text()) == ids
    else:
        path.write_text(json.dumps(ids, indent=2))


def record_report(report: pytest.TestReport | pytest.CollectReport) -> None:
    """Stream controller reports so failed or interrupted runs retain partial evidence."""
    if not CONTROLLER:
        return
    reason = ''
    if report.skipped:
        if isinstance(report.longrepr, tuple):
            reason = report.longrepr[2]
        else:
            reason = str(report.longrepr)
    xfail: object = getattr(report, 'wasxfail', None)
    if isinstance(xfail, str):
        reason = xfail
    event = Event(
        nodeid=report.nodeid,
        phase=report.when if isinstance(report, pytest.TestReport) else 'collect',
        outcome=report.outcome,
        reason=reason,
    )
    with (DIRECTORY / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(event.model_dump_json() + '\n')


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Record every executed test phase."""
    record_report(report)


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Include skipped modules and collection errors forwarded by xdist."""
    if not report.passed:
        record_report(report)
