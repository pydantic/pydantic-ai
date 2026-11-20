from __future__ import annotations

import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).parent))

import pyright_trend

REPO = 'pydantic/pydantic-ai'
NOW = datetime(2026, 11, 19, 6, 0, tzinfo=UTC)
WORKFLOW = Path(__file__).parents[1] / 'workflows' / pyright_trend.WORKFLOW_FILE


class FakeApi:
    """Serves `main` runs newest first, each finished a minute before the next, with the step's duration or none."""

    def __init__(self, durations: Sequence[float | None], newest_finished: datetime = NOW) -> None:
        self.durations = durations
        self.newest_finished = newest_finished
        self.calls: list[str] = []

    def __call__(self, arguments: Sequence[str]) -> list[str]:
        path = next(argument for argument in arguments if argument.startswith('repos/'))
        self.calls.append(path)
        if '/workflows/' in path:
            return [
                f'{run_id}\t{(self.newest_finished - timedelta(minutes=run_id)).isoformat()}'
                for run_id in range(len(self.durations))
            ]
        seconds = self.durations[int(path.split('/')[-2])]
        if seconds is None:
            return []
        return [
            f'2026-11-19T00:00:00Z\t{(datetime(2026, 11, 19, tzinfo=UTC) + timedelta(seconds=seconds)).isoformat()}'
        ]


def test_the_job_and_step_it_reads_exist_in_ci():
    workflow = TypeAdapter(dict[str | bool, object]).validate_python(
        yaml.safe_load(WORKFLOW.read_text(encoding='utf-8'))
    )
    jobs = TypeAdapter(dict[str, dict[str, object]]).validate_python(workflow['jobs'])
    [job] = [job for job in jobs.values() if job.get('name') == pyright_trend.JOB_NAME]
    steps = TypeAdapter(list[dict[str, object]]).validate_python(job['steps'])
    assert pyright_trend.STEP_NAME in [step.get('name') for step in steps]


@pytest.fixture
def outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('GITHUB_REPOSITORY', REPO)
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(tmp_path / 'summary.md'))
    monkeypatch.setenv('GITHUB_OUTPUT', str(tmp_path / 'output'))
    return tmp_path


@pytest.mark.parametrize(
    ('newer', 'expected'),
    [
        pytest.param(300.0, 'true', id='slower'),
        pytest.param(299.0, 'false', id='just-under'),
    ],
)
def test_a_newer_median_at_one_and_a_half_times_the_older_one_reports_a_slowdown(
    outputs: Path, newer: float, expected: str
):
    # The newest run of each window is an outlier, which the medians ignore.
    api = FakeApi([1000.0, *[newer] * 19, 400.0, *[200.0] * 19])

    assert pyright_trend.main(api, NOW) == 0

    assert (outputs / 'output').read_text() == f'slower={expected}\n'
    assert f'median {newer:.0f}s over the newest 20' in (outputs / 'summary.md').read_text()


def test_runs_that_skipped_the_step_are_left_out(outputs: Path):
    api = FakeApi([300.0, None, *[300.0] * 19, *[None] * 3, *[200.0] * 20])

    pyright_trend.main(api, NOW)

    assert (outputs / 'output').read_text() == 'slower=true\n'
    assert len(api.calls) == 1 + 44


def test_too_few_runs_with_the_step_are_not_judged(outputs: Path):
    api = FakeApi([*[300.0] * 20, *[None] * 10, *[200.0] * 19])

    pyright_trend.main(api, NOW)

    assert (outputs / 'output').read_text() == 'slower=false\n'
    assert 'Only 39 of the newest 49 `main` runs' in (outputs / 'summary.md').read_text()


def test_a_window_whose_newest_run_is_older_than_fresh_for_is_not_judged(outputs: Path):
    api = FakeApi([*[300.0] * 20, *[200.0] * 20], newest_finished=NOW - timedelta(minutes=91))

    pyright_trend.main(api, NOW)

    assert (outputs / 'output').read_text() == 'slower=false\n'
    assert len(api.calls) == 1


def test_no_main_runs_are_not_judged(outputs: Path):
    api = FakeApi([])

    pyright_trend.main(api, NOW)

    assert (outputs / 'output').read_text() == 'slower=false\n'
    assert 'No `main` run finished in the last 90 minutes' in (outputs / 'summary.md').read_text()
    assert len(api.calls) == 1
