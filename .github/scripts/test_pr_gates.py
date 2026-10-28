"""Tests for the PR gates in `.github/workflows/pr-gates.yml`."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pr_gates
import pytest
import yaml
from pr_gates import ChangedFile, GateError, GhCli, compatibility_waiver, run
from pydantic import BaseModel, Field

WORKFLOW = Path(__file__).parent.parent / 'workflows' / 'pr-gates.yml'
ALLOWLIST = '.github/api-compatibility-allowlist.json'
WAIVER = {
    'against': 'v2.59.0',
    'fingerprint': 'a' * 64,
    'reason': 'Deprecated in v2.40, removed on schedule.',
    'pull_request': 'https://github.com/pydantic/pydantic-ai/pull/1',
}


def _allowlist(*waivers: dict[str, str]) -> str:
    return json.dumps({'allowed_breakages': list(waivers)})


@dataclass
class FakeGitHub:
    files: list[str]
    current_labels: set[str] = field(default_factory=set[str])
    texts: dict[str, str | None] = field(default_factory=dict[str, str | None])

    def changed_files(self) -> list[ChangedFile]:
        return [ChangedFile(filename=path) for path in self.files]

    def labels(self) -> set[str]:
        return set(self.current_labels)

    def merge_base(self, base: str, head: str) -> str:
        return 'merge-base'

    def file_text(self, path: str, ref: str) -> str | None:
        assert path == ALLOWLIST
        return self.texts.get(ref)


@pytest.mark.parametrize(
    ('head', 'base', 'labels', 'fails'),
    [
        pytest.param(_allowlist(WAIVER), _allowlist(), set[str](), True, id='new-waiver-unlabeled'),
        pytest.param(_allowlist(WAIVER), _allowlist(), {'compatibility impact'}, False, id='new-waiver-labeled'),
        # A file added by the PR has no version at the merge base.
        pytest.param(_allowlist(WAIVER), None, set[str](), True, id='new-file'),
        pytest.param(_allowlist(), _allowlist(WAIVER), set[str](), False, id='waiver-removed'),
        pytest.param(_allowlist(WAIVER), _allowlist(WAIVER), set[str](), False, id='reformatted'),
        pytest.param('{"allowed_breakages": [{}]}', _allowlist(), {'compatibility impact'}, True, id='invalid'),
    ],
)
def test_compatibility_waiver(head: str, base: str | None, labels: set[str], fails: bool):
    github = FakeGitHub(files=[ALLOWLIST], current_labels=labels, texts={'head': head, 'merge-base': base})

    assert bool(compatibility_waiver(github, base_sha='base', head_sha='head').failures) is fails


def test_an_untouched_allowlist_passes_without_reading_it():
    github = FakeGitHub(files=['pydantic_ai_slim/pydantic_ai/agent.py'])

    assert compatibility_waiver(github, base_sha='base', head_sha='head').failures == []


def test_run_reports_failures_as_annotations_and_a_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    summary = tmp_path / 'summary.md'
    github = FakeGitHub(files=[ALLOWLIST], texts={'head': _allowlist(WAIVER), 'merge-base': _allowlist()})

    status = run(
        'compatibility-waiver', github, {'BASE_SHA': 'base', 'HEAD_SHA': 'head', 'GITHUB_STEP_SUMMARY': str(summary)}
    )

    assert status == 1
    assert capsys.readouterr().out.startswith('::error title=Compatibility waiver labeled::Adds 1 API compatibility')
    assert summary.read_text().startswith('## Compatibility waiver labeled: failed')


def test_run_fails_closed_when_github_cannot_answer(capsys: pytest.CaptureFixture[str]):
    class Unreachable(FakeGitHub):
        def changed_files(self) -> list[ChangedFile]:
            raise GateError('Retrieved only 3000 of 3001 changed files')

    assert run('compatibility-waiver', Unreachable(files=[]), {'BASE_SHA': 'base', 'HEAD_SHA': 'head'}) == 1
    assert 'Retrieved only 3000 of 3001' in capsys.readouterr().out


def test_gh_cli_parses_github_responses(monkeypatch: pytest.MonkeyPatch):
    responses: dict[str, str | None] = {
        'repos/o/r/pulls/7': '{"changed_files": 2}',
        'repos/o/r/pulls/7/files?per_page=100': '[[{"filename": "a"}], [{"filename": "b", "previous_filename": "c"}]]',
        'repos/o/r/compare/x...y': '{"merge_base_commit": {"sha": "m"}}',
        'repos/o/r/issues/7/labels?per_page=100': '[[{"name": "compatibility impact"}]]',
        f'repos/o/r/contents/{ALLOWLIST}?ref=y': '{}',
        f'repos/o/r/contents/{ALLOWLIST}?ref=m': None,
    }

    def fake_run(args: list[str], **_: bool) -> subprocess.CompletedProcess[str]:
        output = responses[args[-1]]
        if output is None:
            return subprocess.CompletedProcess(args, 1, '', 'gh: Not Found (HTTP 404)')
        return subprocess.CompletedProcess(args, 0, output, '')

    monkeypatch.setattr(pr_gates.subprocess, 'run', fake_run)
    github = GhCli('o/r', 7)

    assert github.changed_files() == [ChangedFile(filename='a'), ChangedFile(filename='b', previous_filename='c')]
    assert github.merge_base('x', 'y') == 'm'
    assert github.labels() == {'compatibility impact'}
    assert github.file_text(ALLOWLIST, 'y') == '{}'
    assert github.file_text(ALLOWLIST, 'm') is None

    responses['repos/o/r/pulls/7'] = '{"changed_files": 3}'
    with pytest.raises(GateError, match='Retrieved only 2 of 3'):
        github.changed_files()


class _Trigger(BaseModel):
    types: list[str]
    paths: list[str] | None = None


class _Step(BaseModel):
    run: str = ''


class _Job(BaseModel):
    condition: str = Field(alias='if')
    steps: list[_Step]


class _Workflow(BaseModel):
    on: dict[str, _Trigger]
    jobs: dict[str, _Job]


def test_the_workflow_runs_every_gate_on_every_pr_event_and_skips_only_the_merge_queue():
    # `BaseLoader` keeps the `on` key a string; the default loader reads it as `True`.
    workflow = _Workflow.model_validate(yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader))

    assert set(workflow.on) == {'pull_request', 'merge_group'}
    assert workflow.on['pull_request'].paths is None
    assert {'opened', 'synchronize', 'reopened', 'labeled', 'unlabeled', 'edited'} <= set(
        workflow.on['pull_request'].types
    )
    for job in workflow.jobs.values():
        assert job.condition == "github.event_name == 'pull_request'"
        for step in job.steps:
            assert '${{' not in step.run
