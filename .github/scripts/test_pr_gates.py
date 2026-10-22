"""Tests for the PR gates in `.github/workflows/pr-gates.yml`.

The workflow triggers on `pull_request_target`, so GitHub runs it from the default branch and a
PR that changes it never exercises its own change. These tests are the only check of its logic
before `main`.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pr_gates
import pytest
import yaml
from pr_gates import ChangedFile, GateError, GhCli, LabelEvent, compatibility_waiver, protected_paths, run
from pydantic import BaseModel, ConfigDict, Field

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


def _labeled(label: str, login: str = 'maintainer', actor_type: str = 'User') -> LabelEvent:
    return LabelEvent.model_validate(
        {'event': 'labeled', 'actor': {'login': login, 'type': actor_type}, 'label': {'name': label}}
    )


@dataclass
class FakeGitHub:
    files: list[str]
    current_labels: set[str] = field(default_factory=set[str])
    events: list[LabelEvent] = field(default_factory=list[LabelEvent])
    permissions: dict[str, str] = field(default_factory=lambda: {'maintainer': 'write'})
    # `None` models a push whose changes can't be listed.
    pushed_files: list[str] | None = field(default_factory=list[str])
    texts: dict[str, str | None] = field(default_factory=dict[str, str | None])
    renamed_from: dict[str, str] = field(default_factory=dict[str, str])
    removed: list[str] = field(default_factory=list[str])

    def changed_files(self) -> list[ChangedFile]:
        return [ChangedFile(filename=path, previous_filename=self.renamed_from.get(path)) for path in self.files]

    def changed_files_between(self, base: str, head: str) -> list[ChangedFile] | None:
        assert (base, head) == ('before', 'after')
        return None if self.pushed_files is None else [ChangedFile(filename=path) for path in self.pushed_files]

    def labels(self) -> set[str]:
        return set(self.current_labels)

    def label_events(self) -> list[LabelEvent]:
        return self.events

    def permission(self, login: str) -> str:
        return self.permissions.get(login, 'none')

    def remove_label(self, name: str) -> None:
        self.removed.append(name)
        self.current_labels.discard(name)

    def merge_base(self, base: str, head: str) -> str:
        return 'merge-base'

    def file_text(self, path: str, ref: str) -> str | None:
        assert path == ALLOWLIST
        return self.texts.get(ref)


@pytest.mark.parametrize(
    ('path', 'label'),
    [
        ('uv.lock', 'approved:dependencies'),
        ('pyproject.toml', 'approved:dependencies'),
        # Every package's manifest declares dependencies, not just the root one.
        ('pydantic_ai_slim/pyproject.toml', 'approved:dependencies'),
        ('.github/workflows/ci.yml', 'approved:ci-config'),
        ('.github/actions/fetch-dynamic-prompt/action.yml', 'approved:ci-config'),
        ('.github/scripts/pr_gates.py', 'approved:ci-config'),
        ('.pre-commit-config.yaml', 'approved:ci-config'),
        ('Makefile', 'approved:ci-config'),
    ],
)
def test_a_protected_path_needs_its_scope_label(path: str, label: str):
    github = FakeGitHub(files=['docs/index.md', path])

    verdict = protected_paths(github, pushed=None)

    assert len(verdict.failures) == 1
    assert f'`{label}`' in verdict.failures[0]
    assert f'`{path}`' in verdict.failures[0]


@pytest.mark.parametrize('path', ['docs/Makefile', '.github/dependabot.yml', 'tests/pyproject_test.py', 'README.md'])
def test_unprotected_paths_pass(path: str):
    assert protected_paths(FakeGitHub(files=[path]), pushed=None).failures == []


def test_a_rename_out_of_a_protected_path_counts():
    github = FakeGitHub(files=['tools/ci.yml'], renamed_from={'tools/ci.yml': '.github/workflows/ci.yml'})

    assert 'approved:ci-config' in protected_paths(github, pushed=None).failures[0]


def test_a_maintainer_label_approves_its_scope():
    github = FakeGitHub(
        files=['uv.lock', '.github/workflows/ci.yml'],
        current_labels={'approved:dependencies', 'approved:ci-config'},
        events=[_labeled('approved:dependencies'), _labeled('approved:ci-config', login='admin-user')],
        permissions={'maintainer': 'write', 'admin-user': 'admin'},
    )

    assert protected_paths(github, pushed=None).failures == []


def test_one_scope_label_does_not_approve_the_other():
    github = FakeGitHub(
        files=['uv.lock', 'Makefile'],
        current_labels={'approved:dependencies'},
        events=[_labeled('approved:dependencies')],
    )

    failures = protected_paths(github, pushed=None).failures

    assert len(failures) == 1
    assert '`approved:ci-config`' in failures[0]


@pytest.mark.parametrize(
    ('event', 'permissions'),
    [
        pytest.param(_labeled('approved:dependencies', login='contributor'), {'contributor': 'read'}, id='read'),
        pytest.param(_labeled('approved:dependencies', login='triager'), {'triager': 'triage'}, id='triage'),
        # The collaborator lookup failing reads as `none`, which blocks.
        pytest.param(_labeled('approved:dependencies', login='unknown'), {}, id='unreadable'),
        # GitHub Apps push branches into this repo, so a bot never vouches, whatever its access.
        pytest.param(
            _labeled('approved:dependencies', login='pydanty[bot]', actor_type='Bot'),
            {'pydanty[bot]': 'write'},
            id='bot',
        ),
    ],
)
def test_a_label_from_anyone_but_a_maintainer_does_not_approve(event: LabelEvent, permissions: dict[str, str]):
    github = FakeGitHub(
        files=['uv.lock'], current_labels={'approved:dependencies'}, events=[event], permissions=permissions
    )

    assert len(protected_paths(github, pushed=None).failures) == 1


def test_the_latest_application_of_a_label_is_the_one_that_counts():
    github = FakeGitHub(
        files=['uv.lock'],
        current_labels={'approved:dependencies'},
        events=[_labeled('approved:dependencies'), _labeled('approved:dependencies', login='contributor')],
        permissions={'maintainer': 'write', 'contributor': 'read'},
    )

    assert len(protected_paths(github, pushed=None).failures) == 1


def test_a_label_with_no_labeled_event_does_not_approve():
    github = FakeGitHub(files=['uv.lock'], current_labels={'approved:dependencies'})

    assert 'Cannot tell who applied' in protected_paths(github, pushed=None).failures[0]


def test_a_push_to_an_approved_scope_voids_its_label():
    github = FakeGitHub(
        files=['uv.lock', 'Makefile'],
        current_labels={'approved:dependencies', 'approved:ci-config'},
        events=[_labeled('approved:dependencies'), _labeled('approved:ci-config')],
        pushed_files=['uv.lock', 'docs/index.md'],
    )

    failures = protected_paths(github, pushed=('before', 'after')).failures

    assert github.removed == ['approved:dependencies']
    assert len(failures) == 1
    assert '`approved:dependencies`' in failures[0]


def test_a_push_outside_the_approved_scopes_keeps_the_labels():
    github = FakeGitHub(
        files=['uv.lock'],
        current_labels={'approved:dependencies'},
        events=[_labeled('approved:dependencies')],
        pushed_files=['pydantic_ai_slim/pydantic_ai/agent.py'],
    )

    assert protected_paths(github, pushed=('before', 'after')).failures == []
    assert github.removed == []


def test_a_push_that_cannot_be_listed_voids_every_approval_the_pr_needs():
    github = FakeGitHub(
        files=['uv.lock', 'Makefile'],
        current_labels={'approved:dependencies', 'approved:ci-config', 'pkg:core'},
        events=[_labeled('approved:dependencies'), _labeled('approved:ci-config')],
        pushed_files=None,
    )

    assert len(protected_paths(github, pushed=('before', 'after')).failures) == 2
    assert github.removed == ['approved:ci-config', 'approved:dependencies']


def test_a_pr_that_touches_no_protected_path_reads_nothing_else():
    github = FakeGitHub(files=['docs/index.md'], pushed_files=None)

    assert protected_paths(github, pushed=('before', 'after')).failures == []
    assert github.removed == []


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
    github = FakeGitHub(files=['uv.lock'])

    status = run('protected-paths', github, {'EVENT_ACTION': 'opened', 'GITHUB_STEP_SUMMARY': str(summary)})

    assert status == 1
    assert capsys.readouterr().out.startswith('::error title=Protected paths approved::Changes to dependencies need')
    assert summary.read_text().startswith('## Protected paths approved: failed')


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
        'repos/o/r/compare/x...y': '{"merge_base_commit": {"sha": "m"}, "files": [{"filename": "uv.lock"}]}',
        'repos/o/r/issues/7/labels?per_page=100': '[[{"name": "approved:ci-config"}]]',
        'repos/o/r/collaborators/someone/permission': '{"permission": "admin"}',
        'repos/o/r/collaborators/stranger/permission': None,
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
    assert github.changed_files_between('x', 'y') == [ChangedFile(filename='uv.lock')]
    assert github.merge_base('x', 'y') == 'm'
    assert github.labels() == {'approved:ci-config'}
    assert github.permission('someone') == 'admin'
    assert github.permission('stranger') == 'none'
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
    model_config = ConfigDict(extra='allow')

    condition: str = Field(alias='if')
    steps: list[_Step]


class _Workflow(BaseModel):
    model_config = ConfigDict(extra='allow')

    on: dict[str, _Trigger]
    jobs: dict[str, _Job]


def test_the_workflow_runs_every_gate_on_every_pr_event_and_skips_only_the_merge_queue():
    # `BaseLoader` keeps the `on` key a string; the default loader reads it as `True`.
    workflow = _Workflow.model_validate(yaml.load(WORKFLOW.read_text(encoding='utf-8'), Loader=yaml.BaseLoader))

    assert set(workflow.on) == {'pull_request_target', 'merge_group'}
    assert workflow.on['pull_request_target'].paths is None
    assert {'synchronize', 'labeled', 'unlabeled', 'edited'} <= set(workflow.on['pull_request_target'].types)
    assert 'concurrency' not in (workflow.model_extra or {})
    for job in workflow.jobs.values():
        assert job.condition == "github.event_name == 'pull_request_target'"
        assert 'concurrency' not in (job.model_extra or {})
        for step in job.steps:
            assert '${{' not in step.run
