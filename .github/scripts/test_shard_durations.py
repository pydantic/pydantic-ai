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


def _artifact(artifact_id: int, name: str) -> dict[str, Any]:
    return {'id': artifact_id, 'name': name, 'expired': False}


class FakeApi:
    def __init__(self, runs: dict[int, list[dict[str, Any]]], zips: dict[int, bytes]) -> None:
        self.runs = runs
        self.zips = zips
        self.paths: list[str] = []

    def __call__(self, path: str) -> bytes:
        self.paths.append(path)
        parts = path.split('?')[0].split('/')
        if parts[3:5] == ['actions', 'workflows']:
            return json.dumps({'workflow_runs': [{'id': run_id} for run_id in self.runs]}).encode()
        if parts[3:5] == ['actions', 'runs']:
            return json.dumps({'artifacts': self.runs[int(parts[5])]}).encode()
        return self.zips[int(parts[5])]


def test_fetch_merges_shards_of_newest_complete_main_run():
    api = FakeApi(
        runs={
            # Newest first: a run still missing its second shard, then the run to use.
            300: [_artifact(30, f'{PREFIX}1')],
            200: [
                _artifact(20, f'{PREFIX}1'),
                _artifact(21, f'{PREFIX}2'),
                _artifact(22, 'coverage-3.12-all-extras-1'),
            ],
            100: [_artifact(10, f'{PREFIX}1'), _artifact(11, f'{PREFIX}2')],
        },
        zips={20: _zip({'a': 1.0}), 21: _zip({'b': 2.0})},
    )
    assert shard_durations.fetch(REPO, PREFIX, 2, '2026-10-09T12:00:00Z', api) == (200, {'a': 1.0, 'b': 2.0})
    assert api.paths[0] == (
        f'repos/{REPO}/actions/workflows/ci.yml/runs?branch=main&event=push&created=%3C2026-10-09T12:00:00Z&per_page=20'
    )


def test_fetch_without_a_complete_run():
    api = FakeApi(runs={300: [_artifact(30, f'{PREFIX}1')]}, zips={})
    assert shard_durations.fetch(REPO, PREFIX, 2, '2026-10-09T12:00:00Z', api) is None


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
