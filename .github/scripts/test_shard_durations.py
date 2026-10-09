from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import shard_durations

REPO = 'pydantic/pydantic-ai'
PREFIX = 'test-durations-3.12-all-extras-'


def _zip(durations: dict[str, float]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('test-durations.json', json.dumps(durations))
    return buffer.getvalue()


def _artifact(artifact_id: int, name: str, run_id: int, *, branch: str = 'main', fork: bool = False) -> dict[str, Any]:
    return {
        'id': artifact_id,
        'name': name,
        'expired': False,
        'workflow_run': {
            'id': run_id,
            'head_branch': branch,
            'repository_id': 1,
            'head_repository_id': 2 if fork else 1,
        },
    }


class FakeApi:
    def __init__(self, artifacts: list[dict[str, Any]], zips: dict[int, bytes]) -> None:
        self.artifacts = artifacts
        self.zips = zips

    def __call__(self, path: str) -> bytes:
        if path.startswith(f'repos/{REPO}/actions/artifacts?name='):
            name = path.split('name=')[1].split('&')[0]
            return json.dumps({'artifacts': [a for a in self.artifacts if a['name'] == name]}).encode()
        if path.startswith(f'repos/{REPO}/actions/runs/'):
            run_id = int(path.split('/')[5])
            return json.dumps({'artifacts': [a for a in self.artifacts if a['workflow_run']['id'] == run_id]}).encode()
        artifact_id = int(path.split('/')[5])
        return self.zips[artifact_id]


def test_fetch_merges_shards_of_newest_complete_main_run():
    api = FakeApi(
        artifacts=[
            # Newest first: a fork PR from a branch called `main`, a PR run, a `main` run still
            # missing its second shard, then the run to use.
            _artifact(50, f'{PREFIX}1', 500, fork=True),
            _artifact(51, f'{PREFIX}2', 500, fork=True),
            _artifact(40, f'{PREFIX}1', 400, branch='feature'),
            _artifact(30, f'{PREFIX}1', 300),
            _artifact(20, f'{PREFIX}1', 200),
            _artifact(21, f'{PREFIX}2', 200),
            _artifact(22, 'coverage-3.12-all-extras-1', 200),
        ],
        zips={20: _zip({'a': 1.0}), 21: _zip({'b': 2.0})},
    )
    assert shard_durations.fetch(REPO, PREFIX, 2, api) == (200, {'a': 1.0, 'b': 2.0})


def test_fetch_without_a_complete_run():
    api = FakeApi(artifacts=[_artifact(30, f'{PREFIX}1', 300)], zips={})
    assert shard_durations.fetch(REPO, PREFIX, 2, api) is None


def test_compare_flags_new_slow_and_slower_tests():
    baseline = {'steady': 1.0, 'doubled': 1.0, 'tiny': 0.1, 'gone': 5.0}
    current = {'steady': 1.5, 'doubled': 2.5, 'tiny': 0.9, 'new_fast': 0.5, 'new_slow': 3.0}
    report = shard_durations.compare(baseline, current)
    assert report == shard_durations.Report(new=[('new_slow', 3.0)], slower=[('doubled', 1.0, 2.5)])
    rendered = shard_durations.render(report, '3.12 shard 1')
    assert '| `new_slow` | 3.0 |' in rendered
    assert '| `doubled` | 1.0 | 2.5 |' in rendered
    assert shard_durations.render(shard_durations.Report(new=[], slower=[]), 'x') == ''


def test_report_writes_the_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    baseline = tmp_path / 'baseline.json'
    current = tmp_path / 'current.json'
    summary = tmp_path / 'summary.md'
    baseline.write_text(json.dumps({'a': 1.0}))
    current.write_text(json.dumps({'a': 1.0, 'b': 4.0}))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(summary))
    args = ['report', '--baseline', str(baseline), '--current', str(current), '--title', 't']
    assert shard_durations.main(args) == 0
    assert '| `b` | 4.0 |' in summary.read_text()


def test_fetch_failure_does_not_fail_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.delenv('GITHUB_REPOSITORY', raising=False)
    output = tmp_path / 'durations.json'
    assert shard_durations.main(['fetch', '--prefix', PREFIX, '--shards', '2', '--output', str(output)]) == 0
    assert not output.exists()
    assert 'splitting shards by test count' in capsys.readouterr().out
