"""Tests for the CI provenance attributes that `Dataset.evaluate` puts on its experiment span.

These attributes exist for consumers that compare experiment runs (such as a "this run vs. the base branch"
pull request comment), so the tests pin the exact span attributes a consumer would read.
"""

from __future__ import annotations as _annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed'),
]

_PROVENANCE_PREFIXES = ('vcs.', 'cicd.')

_GITHUB_ENV = {
    'GITHUB_ACTIONS': 'true',
    'GITHUB_SERVER_URL': 'https://github.com',
    'GITHUB_REPOSITORY': 'acme/widgets',
    'GITHUB_RUN_ID': '1234567890',
}


@pytest.fixture(autouse=True)
def _clean_provenance_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Turn detection back on, from a clean slate: no CI variables, and no `git` checkout around the current directory.

    The suite itself runs in GitHub Actions and in this repository's checkout.
    """
    for name in list(os.environ):
        if name.startswith(('GITHUB_', 'GITLAB_', 'CI_', 'PYDANTIC_EVALS_', 'GIT_')):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    # `git` looks in the current directory but not in its parents.
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))


async def _task(inputs: Any) -> str:
    return 'ok'


def _set_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    for name, value in env.items():
        monkeypatch.setenv(name, value)


async def _provenance(capfire: CaptureLogfire) -> dict[str, Any]:
    await Dataset[Any, Any, Any](name='provenance', cases=[Case(inputs='x')]).evaluate(_task, progress=False)
    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    [experiment_span] = [s for s in spans if s['name'] == 'evaluate {name}']
    return {k: v for k, v in experiment_span['attributes'].items() if k.startswith(_PROVENANCE_PREFIXES)}


async def test_not_in_ci(capfire: CaptureLogfire):
    """Outside CI and outside a `git` checkout, nothing is detected."""
    assert await _provenance(capfire) == {}


_GITHUB_REPO_ATTRS = {
    'vcs.provider.name': 'github',
    'vcs.repository.url.full': 'https://github.com/acme/widgets',
    'cicd.pipeline.run.url.full': 'https://github.com/acme/widgets/actions/runs/1234567890',
}


async def test_github_push(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': 'push',
            'GITHUB_SHA': 'abc123',
            'GITHUB_REF': 'refs/heads/main',
            'GITHUB_REF_NAME': 'main',
            'GITHUB_REF_TYPE': 'branch',
        },
    )
    assert await _provenance(capfire) == snapshot(
        {
            **_GITHUB_REPO_ATTRS,
            'vcs.ref.head.revision': 'abc123',
            'vcs.ref.head.name': 'main',
            'vcs.ref.head.type': 'branch',
        }
    )


async def test_github_tag_push(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': 'push',
            'GITHUB_SHA': 'abc123',
            'GITHUB_REF': 'refs/tags/v1.0.0',
            'GITHUB_REF_NAME': 'v1.0.0',
            'GITHUB_REF_TYPE': 'tag',
        },
    )
    assert await _provenance(capfire) == snapshot(
        {
            **_GITHUB_REPO_ATTRS,
            'vcs.ref.head.revision': 'abc123',
            'vcs.ref.head.name': 'v1.0.0',
            'vcs.ref.head.type': 'tag',
        }
    )


@pytest.fixture
def pull_request_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A `pull_request` event, where `GITHUB_SHA`/`GITHUB_REF_NAME` point at the synthetic merge commit."""
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': 'pull_request',
            'GITHUB_SHA': 'merge-commit-sha',
            'GITHUB_REF': 'refs/pull/42/merge',
            'GITHUB_REF_NAME': '42/merge',
            'GITHUB_REF_TYPE': 'branch',
            'GITHUB_HEAD_REF': 'feature-x',
            'GITHUB_BASE_REF': 'main',
        },
    )


@pytest.mark.usefixtures('pull_request_env')
async def test_github_pull_request(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    event_path = tmp_path / 'event.json'
    event_path.write_text(
        json.dumps(
            {
                'action': 'synchronize',
                'number': 42,
                'pull_request': {
                    'number': 42,
                    'head': {'ref': 'feature-x', 'sha': 'head-sha'},
                    'base': {'ref': 'main', 'sha': 'base-sha'},
                },
            }
        )
    )
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))

    assert await _provenance(capfire) == snapshot(
        {
            **_GITHUB_REPO_ATTRS,
            'vcs.ref.head.revision': 'head-sha',
            'vcs.ref.head.name': 'feature-x',
            'vcs.ref.head.type': 'branch',
            'vcs.ref.base.name': 'main',
            'vcs.change.id': '42',
        }
    )


@pytest.mark.usefixtures('pull_request_env')
@pytest.mark.parametrize('readable_payload', [True, False])
async def test_github_pull_request_target(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, readable_payload: bool
):
    """`pull_request_target` checks out the base branch by default, so the PR head isn't reported as the code under test."""
    if readable_payload:
        event_path = tmp_path / 'event.json'
        event_path.write_text(
            json.dumps(
                {
                    'pull_request': {
                        'number': 42,
                        'head': {'ref': 'feature-x', 'sha': 'head-sha'},
                        'base': {'ref': 'main'},
                    }
                }
            )
        )
        monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))
    # Unlike `pull_request`, these point at the base branch rather than a merge commit.
    monkeypatch.setenv('GITHUB_EVENT_NAME', 'pull_request_target')
    monkeypatch.setenv('GITHUB_SHA', 'base-tip-sha')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/main')
    monkeypatch.setenv('GITHUB_REF_NAME', 'main')

    expected: dict[str, Any] = {**_GITHUB_REPO_ATTRS, 'vcs.ref.base.name': 'main'}
    if readable_payload:
        expected['vcs.change.id'] = '42'
    assert await _provenance(capfire) == expected


@pytest.mark.usefixtures('pull_request_env')
@pytest.mark.parametrize('event_file', ['missing', 'not json', '{"pull_request": {"number": "not a number"}}'])
async def test_github_pull_request_unreadable_event(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, event_file: str
):
    """Without a readable payload the head commit is unknown, so no SHA is reported rather than the merge commit's."""
    event_path = tmp_path / 'event.json'
    if event_file != 'missing':
        event_path.write_text(event_file)
    monkeypatch.setenv('GITHUB_EVENT_PATH', str(event_path))

    assert await _provenance(capfire) == snapshot(
        {
            **_GITHUB_REPO_ATTRS,
            'vcs.ref.head.name': 'feature-x',
            'vcs.ref.head.type': 'branch',
            'vcs.ref.base.name': 'main',
            'vcs.change.id': '42',
        }
    )


async def test_github_minimal_env(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    """Missing variables omit their attributes instead of producing partial values."""
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_HEAD_REF', 'feature-x')
    monkeypatch.setenv('GITHUB_REF', 'refs/heads/feature-x')
    assert await _provenance(capfire) == snapshot(
        {'vcs.provider.name': 'github', 'vcs.ref.head.name': 'feature-x', 'vcs.ref.head.type': 'branch'}
    )


async def test_github_without_run_id(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    monkeypatch.setenv('GITHUB_SERVER_URL', 'https://github.com')
    monkeypatch.setenv('GITHUB_REPOSITORY', 'acme/widgets')
    assert await _provenance(capfire) == snapshot(
        {'vcs.provider.name': 'github', 'vcs.repository.url.full': 'https://github.com/acme/widgets'}
    )


_GITLAB_ENV = {
    'GITLAB_CI': 'true',
    'CI_PROJECT_URL': 'https://gitlab.example.com/acme/widgets',
    'CI_PIPELINE_URL': 'https://gitlab.example.com/acme/widgets/-/pipelines/987',
    'CI_COMMIT_SHA': 'commit-sha',
}

_GITLAB_REPO_ATTRS = {
    'vcs.provider.name': 'gitlab',
    'vcs.repository.url.full': 'https://gitlab.example.com/acme/widgets',
    'cicd.pipeline.run.url.full': 'https://gitlab.example.com/acme/widgets/-/pipelines/987',
}


async def test_gitlab_branch_pipeline(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, {**_GITLAB_ENV, 'CI_COMMIT_BRANCH': 'main', 'CI_COMMIT_REF_NAME': 'main'})
    assert await _provenance(capfire) == snapshot(
        {
            **_GITLAB_REPO_ATTRS,
            'vcs.ref.head.revision': 'commit-sha',
            'vcs.ref.head.name': 'main',
            'vcs.ref.head.type': 'branch',
        }
    )


async def test_gitlab_tag_pipeline(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(monkeypatch, {**_GITLAB_ENV, 'CI_COMMIT_TAG': 'v1.0.0', 'CI_COMMIT_REF_NAME': 'v1.0.0'})
    assert await _provenance(capfire) == snapshot(
        {
            **_GITLAB_REPO_ATTRS,
            'vcs.ref.head.revision': 'commit-sha',
            'vcs.ref.head.name': 'v1.0.0',
            'vcs.ref.head.type': 'tag',
        }
    )


@pytest.mark.parametrize(
    ('event_type', 'revision'),
    [
        ('detached', 'commit-sha'),
        # `CI_COMMIT_SHA` is the merge commit GitLab builds, so the source branch head is reported instead.
        ('merged_result', 'source-head-sha'),
        ('merge_train', 'source-head-sha'),
    ],
)
async def test_gitlab_merge_request_pipeline(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, event_type: str, revision: str
):
    _set_env(
        monkeypatch,
        {
            **_GITLAB_ENV,
            'CI_MERGE_REQUEST_IID': '7',
            'CI_MERGE_REQUEST_EVENT_TYPE': event_type,
            'CI_MERGE_REQUEST_SOURCE_BRANCH_NAME': 'feature-x',
            'CI_MERGE_REQUEST_TARGET_BRANCH_NAME': 'main',
            **({'CI_MERGE_REQUEST_SOURCE_BRANCH_SHA': 'source-head-sha'} if event_type != 'detached' else {}),
        },
    )
    assert await _provenance(capfire) == {
        **_GITLAB_REPO_ATTRS,
        'vcs.ref.head.revision': revision,
        'vcs.ref.head.name': 'feature-x',
        'vcs.ref.head.type': 'branch',
        'vcs.ref.base.name': 'main',
        'vcs.change.id': '7',
    }


async def test_gitlab_minimal_env(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    """Missing variables omit their attributes instead of producing partial values."""
    monkeypatch.setenv('GITLAB_CI', 'true')
    monkeypatch.setenv('CI_MERGE_REQUEST_IID', '7')
    monkeypatch.setenv('CI_MERGE_REQUEST_EVENT_TYPE', 'merged_result')
    assert await _provenance(capfire) == snapshot({'vcs.provider.name': 'gitlab', 'vcs.change.id': '7'})


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.com', *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def git_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A `git` checkout as the current directory, outside any CI."""
    _git(tmp_path, 'init', '--quiet', '--initial-branch=feature-x')
    _git(tmp_path, 'commit', '--quiet', '--allow-empty', '-m', 'init')
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def test_git_checkout(capfire: CaptureLogfire, git_checkout: Path):
    commit_sha = _git(git_checkout, 'rev-parse', 'HEAD')
    assert await _provenance(capfire) == {
        'vcs.ref.head.revision': commit_sha,
        'vcs.ref.head.name': 'feature-x',
        'vcs.ref.head.type': 'branch',
    }


async def test_git_checkout_detached_head(capfire: CaptureLogfire, git_checkout: Path):
    _git(git_checkout, 'checkout', '--quiet', '--detach')
    assert await _provenance(capfire) == {'vcs.ref.head.revision': _git(git_checkout, 'rev-parse', 'HEAD')}


@pytest.mark.usefixtures('git_checkout')
async def test_git_not_installed(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PATH', '')
    assert await _provenance(capfire) == {}


@pytest.mark.parametrize('value', ['false', 'FALSE', '0'])
async def test_opt_out(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, value: str):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_SHA': 'abc123',
            'PYDANTIC_EVALS_GIT_BRANCH': 'main',
            'PYDANTIC_EVALS_CI_METADATA': value,
        },
    )
    assert await _provenance(capfire) == {}


async def test_overrides_outside_ci(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYDANTIC_EVALS_GIT_COMMIT_SHA', 'def456')
    monkeypatch.setenv('PYDANTIC_EVALS_GIT_BRANCH', 'release')
    assert await _provenance(capfire) == snapshot(
        {'vcs.ref.head.revision': 'def456', 'vcs.ref.head.name': 'release', 'vcs.ref.head.type': 'branch'}
    )


async def test_overrides_win_over_github(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_SHA': 'abc123',
            'GITHUB_REF_NAME': 'v1.0.0',
            'GITHUB_REF_TYPE': 'tag',
            'PYDANTIC_EVALS_GIT_COMMIT_SHA': 'def456',
            'PYDANTIC_EVALS_GIT_BRANCH': 'release',
        },
    )
    assert await _provenance(capfire) == snapshot(
        {
            **_GITHUB_REPO_ATTRS,
            'vcs.ref.head.revision': 'def456',
            'vcs.ref.head.name': 'release',
            'vcs.ref.head.type': 'branch',
        }
    )
