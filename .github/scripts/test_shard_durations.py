from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from typing_extensions import TypedDict

sys.path.insert(0, str(Path(__file__).parent))

import shard_durations

REPO = 'pydantic/pydantic-ai'
PREFIX = 'test-durations-3.12-all-extras-'
CREATED_AT = '2026-10-09T12:00:00Z'
CURRENT_RUN_PATH = f'repos/{REPO}/actions/runs/999'
HISTORY_PATH = f'repos/{REPO}/actions/workflows/ci.yml/runs?branch=main&event=push&created=%3C{CREATED_AT}&per_page=20'
BASELINES: list[tuple[str, int]] = [
    ('3.11-all-extras', 3),
    ('3.11-lowest-versions', 2),
    ('3.12-all-extras', 2),
    ('3.12-lowest-versions', 2),
    ('3.13-all-extras', 2),
    ('3.13-lowest-versions', 2),
    ('3.14-all-extras', 2),
    ('3.14-lowest-versions', 2),
    ('3.15-all-extras', 2),
]


def _zip(durations: dict[str, float]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('test-durations.json', json.dumps(durations))
    return buffer.getvalue()


class Artifact(TypedDict):
    id: int
    name: str
    expired: bool


def _artifact(artifact_id: int, name: str) -> Artifact:
    return {'id': artifact_id, 'name': name, 'expired': False}


class FakeApi:
    def __init__(self, runs: dict[int, list[Artifact]], zips: dict[int, bytes]) -> None:
        self.runs = runs
        self.zips = zips
        self.paths: list[str] = []
        self.rate_limit_paths: list[str] = []
        self.failures: dict[str, Exception] = {}

    def __call__(self, path: str, *, show_rate_limit: bool = False) -> bytes:
        self.paths.append(path)
        if show_rate_limit:
            self.rate_limit_paths.append(path)
        if path in self.failures:
            raise self.failures[path]
        if path == CURRENT_RUN_PATH:
            # Reruns keep created_at; neither updated_at nor the current clock is the cutoff.
            return json.dumps({'created_at': CREATED_AT, 'updated_at': '2026-10-20T12:00:00Z'}).encode()
        parts = path.split('?')[0].split('/')
        if parts[3:5] == ['actions', 'workflows']:
            return json.dumps({'workflow_runs': [{'id': run_id} for run_id in self.runs]}).encode()
        if parts[3:5] == ['actions', 'runs']:
            return json.dumps({'artifacts': self.runs[int(parts[5])]}).encode()
        return self.zips[int(parts[5])]


def _fetch_args(directory: Path, baselines: Sequence[tuple[str, int]]) -> list[str]:
    args = ['fetch']
    for name, shards in baselines:
        args += [
            '--prefix',
            f'test-durations-{name}-',
            '--shards',
            str(shards),
            '--output',
            str(directory / f'{name}.json'),
        ]
    return args


@pytest.fixture
def step_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'github-output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(path))
    return path


@pytest.fixture
def fetch_environment(monkeypatch: pytest.MonkeyPatch, step_output: Path):
    monkeypatch.setenv('GITHUB_REPOSITORY', REPO)
    monkeypatch.setenv('GITHUB_RUN_ID', '999')


@pytest.mark.usefixtures('fetch_environment')
def test_batch_fetch_shares_reads_and_preserves_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step_output: Path
):
    artifacts: list[Artifact] = []
    zips: dict[int, bytes] = {}
    for name, shards in BASELINES:
        for shard in range(1, shards + 1):
            artifact_id = len(artifacts) + 1
            artifacts.append(_artifact(artifact_id, f'test-durations-{name}-{shard}'))
            zips[artifact_id] = _zip({f'z::{shard}': float(shard), f'a::{shard}': 0.5})
    api = FakeApi(runs={200: artifacts}, zips=zips)
    monkeypatch.setattr(shard_durations, 'gh_api', api)
    single = tmp_path / 'single'
    single.mkdir()
    for baseline in BASELINES:
        assert shard_durations.main(_fetch_args(single, [baseline])) == 0
    single_paths = Counter(api.paths)
    assert single_paths[CURRENT_RUN_PATH] == single_paths[HISTORY_PATH] == 9
    assert single_paths[f'repos/{REPO}/actions/runs/200/artifacts?per_page=100'] == 9
    assert sum(path.endswith('/zip') for path in api.paths) == 19
    assert len(api.paths) == 46

    api.paths.clear()
    api.rate_limit_paths.clear()
    step_output.unlink()
    batch = tmp_path / 'batch'
    batch.mkdir()
    assert shard_durations.main(_fetch_args(batch, BASELINES)) == 0
    assert api.rate_limit_paths == [CURRENT_RUN_PATH]
    # A `main` run caches this baseline only when no prefix failed.
    assert step_output.read_text(encoding='utf-8') == 'errors=0\n'
    batch_paths = Counter(api.paths)
    assert batch_paths[CURRENT_RUN_PATH] == batch_paths[HISTORY_PATH] == 1
    assert batch_paths[f'repos/{REPO}/actions/runs/200/artifacts?per_page=100'] == 1
    assert sum(path.endswith('/zip') for path in api.paths) == 19
    assert len(api.paths) == 22
    assert {path.name for path in batch.iterdir()} == {f'{name}.json' for name, _ in BASELINES}
    for name, shards in BASELINES:
        expected: dict[str, float] = {
            f'{key}::{shard}': value
            for shard in range(1, shards + 1)
            for key, value in [('z', float(shard)), ('a', 0.5)]
        }
        assert (
            (batch / f'{name}.json').read_text()
            == (single / f'{name}.json').read_text()
            == json.dumps(expected, sort_keys=True)
        )


@pytest.mark.usefixtures('fetch_environment')
def test_batch_fetch_selects_each_prefix_independently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    api = FakeApi(
        runs={
            300: [
                _artifact(30, 'test-durations-a-1'),
                {**_artifact(31, 'test-durations-a-2'), 'expired': True},
                _artifact(32, 'test-durations-b-1'),
            ],
            200: [_artifact(20, 'test-durations-a-2'), _artifact(21, 'test-durations-a-1')],
            100: [_artifact(10, 'test-durations-a-1'), _artifact(11, 'test-durations-a-2')],
        },
        zips={20: _zip({'a': 2.0, 'shared': 2.0}), 21: _zip({'z': 1.0, 'shared': 1.0}), 32: _zip({'b': 3.0})},
    )
    monkeypatch.setattr(shard_durations, 'gh_api', api)
    assert shard_durations.main(_fetch_args(tmp_path, [('a', 2), ('b', 1), ('missing', 1)])) == 0
    assert (tmp_path / 'a.json').read_text() == '{"a": 2.0, "shared": 2.0, "z": 1.0}'
    assert (tmp_path / 'b.json').read_text() == '{"b": 3.0}'
    assert (tmp_path / 'missing.json').read_text() == '{}'
    out = capsys.readouterr().out
    assert 'from run 200' in out and 'from run 300' in out
    assert 'No `main` run has `test-durations-missing-*`' in out
    assert api.paths.count(HISTORY_PATH) == 1
    assert all(count == 1 for count in Counter(api.paths).values())
    assert not any(path.endswith('/31/zip') or path.endswith('/10/zip') for path in api.paths)


@pytest.mark.usefixtures('fetch_environment')
@pytest.mark.parametrize('failure', [OSError('offline'), subprocess.CalledProcessError(1, ['gh'])])
@pytest.mark.parametrize(
    'failed_path',
    [
        CURRENT_RUN_PATH,
        HISTORY_PATH,
        f'repos/{REPO}/actions/runs/200/artifacts?per_page=100',
        f'repos/{REPO}/actions/artifacts/20/zip',
    ],
)
def test_batch_fetch_failures_are_cached_and_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    step_output: Path,
    failed_path: str,
    failure: Exception,
):
    api = FakeApi(
        runs={
            200: [
                _artifact(20, 'test-durations-a-1'),
                _artifact(21, 'test-durations-b-1'),
                _artifact(22, 'test-durations-c-1'),
            ]
        },
        zips={20: _zip({'a': 1.0}), 21: _zip({'b': 2.0}), 22: _zip({'c': 3.0})},
    )
    api.failures[failed_path] = failure
    monkeypatch.setattr(shard_durations, 'gh_api', api)
    assert shard_durations.main(_fetch_args(tmp_path, [('b', 1), ('a', 1), ('c', 1)])) == 0
    isolated = failed_path.endswith('/zip')
    assert json.loads((tmp_path / 'a.json').read_text()) == {}
    assert json.loads((tmp_path / 'b.json').read_text()) == ({'b': 2.0} if isolated else {})
    assert json.loads((tmp_path / 'c.json').read_text()) == ({'c': 3.0} if isolated else {})
    assert api.paths.count(failed_path) == 1
    assert capsys.readouterr().out.count('::warning::') == (1 if isolated else 3)
    assert step_output.read_text(encoding='utf-8') == f'errors={1 if isolated else 3}\n'


@pytest.mark.usefixtures('fetch_environment')
def test_batch_fetch_manifest_failure_preserves_other_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    api = FakeApi(
        runs={
            300: [_artifact(30, 'test-durations-b-1'), _artifact(31, 'test-durations-c-1')],
            200: [],
        },
        zips={30: _zip({'b': 2.0}), 31: _zip({'c': 3.0})},
    )
    failed_path = f'repos/{REPO}/actions/runs/200/artifacts?per_page=100'
    api.failures[failed_path] = OSError('offline')
    monkeypatch.setattr(shard_durations, 'gh_api', api)
    assert shard_durations.main(_fetch_args(tmp_path, [('b', 1), ('a', 1), ('missing', 1), ('c', 1)])) == 0
    assert (tmp_path / 'a.json').read_text() == (tmp_path / 'missing.json').read_text() == '{}'
    assert json.loads((tmp_path / 'b.json').read_text()) == {'b': 2.0}
    assert json.loads((tmp_path / 'c.json').read_text()) == {'c': 3.0}
    assert api.paths.count(failed_path) == 1
    assert capsys.readouterr().out.count('::warning::') == 2


@pytest.mark.usefixtures('fetch_environment')
@pytest.mark.parametrize(
    'failed_path',
    [
        CURRENT_RUN_PATH,
        HISTORY_PATH,
        f'repos/{REPO}/actions/runs/200/artifacts?per_page=100',
        f'repos/{REPO}/actions/artifacts/20/zip',
    ],
)
def test_batch_fetch_invalid_responses_remain_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failed_path: str
):
    api = FakeApi(
        runs={200: [_artifact(20, 'test-durations-a-1'), _artifact(21, 'test-durations-b-1')]},
        zips={20: b'invalid ZIP', 21: _zip({'b': 2.0})},
    )

    def invalid_response(path: str, *, show_rate_limit: bool = False) -> bytes:
        content = api(path)
        return b'invalid JSON' if path == failed_path and not path.endswith('/zip') else content

    monkeypatch.setattr(shard_durations, 'gh_api', invalid_response)
    assert shard_durations.main(_fetch_args(tmp_path, [('a', 1), ('b', 1)])) == 0
    assert (tmp_path / 'a.json').read_text() == '{}'
    assert json.loads((tmp_path / 'b.json').read_text()) == ({'b': 2.0} if failed_path.endswith('/zip') else {})
    assert api.paths.count(failed_path) == 1
    assert '::warning::' in capsys.readouterr().out


def test_batch_fetch_rejects_unmatched_options(tmp_path: Path):
    args = _fetch_args(tmp_path, [('a', 1)]) + ['--prefix', 'test-durations-b-']
    with pytest.raises(SystemExit, match='2'):
        shard_durations.main(args)
    assert list(tmp_path.iterdir()) == []


def test_baseline_workflow_invocation_and_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step_output: Path
):
    workflow = yaml.safe_load((Path(__file__).parents[1] / 'workflows' / 'ci.yml').read_text())
    jobs = workflow['jobs']
    job = jobs['test-durations-baseline']
    assert not job.get('needs')
    assert 'test-durations-baseline' not in jobs['classify'].get('needs', [])
    for consumer in ['test-all-extras', 'test-lowest-versions']:
        assert jobs[consumer]['needs'] == ['classify', 'test-durations-baseline']
    steps = {step.get('id'): step for step in job['steps']}
    # Capture the shell's actual argv without starting a Python interpreter.
    capture = tmp_path / 'args'
    subprocess.run(
        ['bash', '-e', '-c', 'python3() { printf "%s\\0" "$@" >> "$FETCH_ARGS"; }\n' + steps['fetch']['run']],
        cwd=tmp_path,
        env={**os.environ, 'FETCH_ARGS': str(capture)},
        check=True,
    )
    args = capture.read_text().split('\0')[:-1]
    assert args == ['.github/scripts/shard_durations.py', *_fetch_args(Path('test-durations'), BASELINES)]
    upload = job['steps'][-1]['with']
    assert upload['name'] == 'test-durations-baseline'
    assert upload['path'] == 'test-durations'
    assert upload['if-no-files-found'] == 'error'

    # A restored baseline skips the REST fetch; `main` saves what it fetched under the same key and path.
    key_step = steps['key']
    assert (
        key_step['env']['PREFIX'] == "test-durations-baseline-${{ hashFiles('.github/scripts/shard_durations.py') }}-"
    )
    key_output = tmp_path / 'key-output'
    subprocess.run(
        ['bash', '-e', '-c', key_step['run']],
        env={**os.environ, 'PREFIX': 'test-durations-baseline-abc-', 'GITHUB_OUTPUT': str(key_output)},
        check=True,
    )
    assert re.fullmatch(
        r'prefix=test-durations-baseline-abc-\nkey=test-durations-baseline-abc-\d{10}\n',
        key_output.read_text(encoding='utf-8'),
    )
    restore = steps['restore']
    save = next(step for step in job['steps'] if step.get('uses', '').startswith('actions/cache/save@'))
    assert restore['uses'].replace('/restore@', '/save@') == save['uses']
    assert restore['with']['key'] == save['with']['key'] == '${{ steps.key.outputs.key }}'
    assert restore['with']['path'] == save['with']['path'] == upload['path']
    assert '&& steps.key.outputs.prefix ||' in restore['with']['restore-keys']
    assert steps['fetch']['if'] == "steps.restore.outputs.cache-matched-key == ''"
    assert save['if'] == (
        "github.event_name == 'push' && github.ref == 'refs/heads/main' && github.run_attempt == '1' "
        "&& steps.fetch.outputs.errors == '0'"
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('GITHUB_REPOSITORY', raising=False)
    assert shard_durations.main(args[1:]) == 0
    assert {path.name: path.read_text() for path in Path('test-durations').iterdir()} == {
        f'{name}.json': '{}' for name, _ in BASELINES
    }
    assert step_output.read_text(encoding='utf-8') == f'errors={len(BASELINES)}\n'


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], step_output: Path
):
    monkeypatch.delenv('GITHUB_REPOSITORY', raising=False)
    output = tmp_path / 'durations.json'
    assert shard_durations.main(['fetch', '--prefix', PREFIX, '--shards', '2', '--output', str(output)]) == 0
    # Every shard reads this file, so an empty one still makes them all fall back the same way.
    assert json.loads(output.read_text()) == {}
    assert 'splitting by module size' in capsys.readouterr().out
    assert step_output.read_text(encoding='utf-8') == 'errors=1\n'


def test_gh_api_shows_the_rate_limit_headers(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    body = b'{"created_at": "2026-10-21T03:00:00Z"}'
    # `gh api --include` ends the status line with LF and each header with CRLF.
    included = (
        b'HTTP/2.0 200 OK\nContent-Type: application/json\r\nX-Ratelimit-Limit: 15000\r\n'
        b'X-Ratelimit-Remaining: 4055\r\nX-Ratelimit-Reset: 1792552704\r\nX-Ratelimit-Resource: core\r\n'
        b'X-Ratelimit-Used: 10945\r\n\r\n' + body
    )
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=included if '--include' in command else body)

    monkeypatch.setattr(subprocess, 'run', run)
    assert shard_durations.gh_api('repos/x', show_rate_limit=True) == body
    assert shard_durations.gh_api('repos/x') == body
    assert commands == [['gh', 'api', '--include', 'repos/x'], ['gh', 'api', 'repos/x']]
    assert capsys.readouterr().out == (
        'REST rate limit: resource=core limit=15000 remaining=4055 used=10945 reset=2026-10-21T03:18:24+00:00\n'
    )


def test_module_weights_sum_tests_and_estimate_new_modules(tmp_path: Path):
    durations = {
        'tests/test_a.py::test_one': 1.0,
        'tests/test_a.py::test_two[x::y]': 2.0,
        'tests/test_b.py::TestB::test_one': 5.0,
        'tests/test_c.py::test_one': 7.0,
        'tests/test_removed.py::test_one': 100.0,
    }
    modules = ['tests/test_a.py', 'tests/test_b.py', 'tests/test_c.py', 'tests/test_new.py']
    assert shard_durations.module_weights(modules, durations, tmp_path) == {
        'tests/test_a.py': 3.0,
        'tests/test_b.py': 5.0,
        'tests/test_c.py': 7.0,
        'tests/test_new.py': 5.0,
    }

    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / 'test_a.py').write_text('x' * 10)
    assert shard_durations.module_weights(['tests/test_a.py'], {}, tmp_path) == {'tests/test_a.py': 10.0}


def test_assign_puts_every_module_in_exactly_one_balanced_shard():
    weights = {'tests/test_a.py': 9.0, 'tests/test_b.py': 5.0, 'tests/test_c.py': 4.0, 'tests/test_d.py': 4.0}
    assert shard_durations.assign(weights, 2) == [
        ['tests/test_a.py', 'tests/test_d.py'],
        ['tests/test_b.py', 'tests/test_c.py'],
    ]


def test_select_ignores_the_other_shards_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    durations = tmp_path / 'durations.json'
    durations.write_text(json.dumps({'tests/test_slow.py::test': 10.0, 'tests/test_fast.py::test': 1.0}))

    def find_test_modules(root: Path, exclude: Sequence[str]) -> list[str]:
        return ['tests/test_fast.py', 'tests/test_slow.py']

    monkeypatch.setattr(shard_durations, 'find_test_modules', find_test_modules)
    args = ['select', '--durations', str(durations), '--shards', '2', '--shard', '1']
    assert shard_durations.main(args) == 0
    out, err = capsys.readouterr()
    assert out.split() == ['--ignore=tests/test_fast.py']
    assert '* shard 1: 1 modules, estimated 10s' in err


def test_find_test_modules_lists_this_repo():
    root = Path(__file__).parents[2]
    modules = shard_durations.find_test_modules(root, ['tests/durable_exec'])
    assert 'tests/test_agent.py' in modules
    assert 'tests/conftest.py' not in modules
    assert not any(path.startswith('tests/durable_exec/') for path in modules)
