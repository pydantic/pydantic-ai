"""Tests for the CI provenance attributes that `Dataset.evaluate` puts on its experiment span.

These attributes exist for consumers that compare experiment runs (such as a "this run vs. the base branch"
pull request comment), so the tests pin the exact span attributes a consumer would read.
"""

from __future__ import annotations as _annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset, dataset as dataset_module

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
def _clean_provenance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn detection back on, from a clean slate, since the suite itself runs in GitHub Actions."""
    for name in list(os.environ):
        if name.startswith(('GITHUB_', 'GITLAB_', 'CI_', 'PYDANTIC_EVALS_')):
            monkeypatch.delenv(name)


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


@pytest.mark.parametrize('event_name', ['push', 'schedule', 'workflow_dispatch'])
async def test_github_branch_run(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, event_name: str):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_EVENT_NAME': event_name,
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
            'vcs.ref.base.revision': 'base-sha',
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


@pytest.mark.parametrize('event_name', ['issue_comment', 'workflow_run', 'merge_group', None])
async def test_github_event_whose_ref_is_not_under_test(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, event_name: str | None
):
    """On these events `GITHUB_SHA`/`GITHUB_REF` point at the default branch, not at what the workflow tests.

    Reporting them would make, say, a `/eval` comment on a pull request look like a run on the default branch.
    """
    env = {**_GITHUB_ENV, 'GITHUB_SHA': 'main-tip-sha', 'GITHUB_REF_NAME': 'main', 'GITHUB_REF_TYPE': 'branch'}
    if event_name is not None:
        env['GITHUB_EVENT_NAME'] = event_name
    _set_env(monkeypatch, env)
    assert await _provenance(capfire) == _GITHUB_REPO_ATTRS


async def test_github_run_attempt(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    """Each re-run of a workflow run is a separate attempt, with its own URL."""
    _set_env(monkeypatch, {**_GITHUB_ENV, 'GITHUB_RUN_ATTEMPT': '2'})
    assert await _provenance(capfire) == snapshot(
        {
            'vcs.provider.name': 'github',
            'vcs.repository.url.full': 'https://github.com/acme/widgets',
            'cicd.pipeline.run.url.full': 'https://github.com/acme/widgets/actions/runs/1234567890/attempts/2',
        }
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
    ('event_type', 'revision', 'base_revision'),
    [
        # GitLab sets no target branch commit in these pipelines.
        ('detached', 'commit-sha', None),
        # `CI_COMMIT_SHA` is the merge commit GitLab builds, so the source branch head is reported instead.
        ('merged_result', 'source-head-sha', 'target-tip-sha'),
        ('merge_train', 'source-head-sha', 'target-tip-sha'),
    ],
)
async def test_gitlab_merge_request_pipeline(
    capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, event_type: str, revision: str, base_revision: str | None
):
    _set_env(
        monkeypatch,
        {
            **_GITLAB_ENV,
            'CI_MERGE_REQUEST_IID': '7',
            'CI_MERGE_REQUEST_EVENT_TYPE': event_type,
            'CI_MERGE_REQUEST_SOURCE_BRANCH_NAME': 'feature-x',
            'CI_MERGE_REQUEST_TARGET_BRANCH_NAME': 'main',
            **(
                {
                    'CI_MERGE_REQUEST_SOURCE_BRANCH_SHA': 'source-head-sha',
                    'CI_MERGE_REQUEST_TARGET_BRANCH_SHA': 'target-tip-sha',
                }
                if event_type != 'detached'
                else {}
            ),
        },
    )
    expected = {
        **_GITLAB_REPO_ATTRS,
        'vcs.ref.head.revision': revision,
        'vcs.ref.head.name': 'feature-x',
        'vcs.ref.head.type': 'branch',
        'vcs.ref.base.name': 'main',
        'vcs.change.id': '7',
    }
    if base_revision is not None:
        expected['vcs.ref.base.revision'] = base_revision
    assert await _provenance(capfire) == expected


async def test_gitlab_minimal_env(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch):
    """Missing variables omit their attributes instead of producing partial values."""
    monkeypatch.setenv('GITLAB_CI', 'true')
    monkeypatch.setenv('CI_MERGE_REQUEST_IID', '7')
    monkeypatch.setenv('CI_MERGE_REQUEST_EVENT_TYPE', 'merged_result')
    assert await _provenance(capfire) == snapshot({'vcs.provider.name': 'gitlab', 'vcs.change.id': '7'})


@pytest.mark.parametrize('value', ['false', 'FALSE', '0', 'no', 'off'])
async def test_opt_out(capfire: CaptureLogfire, monkeypatch: pytest.MonkeyPatch, value: str):
    _set_env(
        monkeypatch,
        {
            **_GITHUB_ENV,
            'GITHUB_SHA': 'abc123',
            'PYDANTIC_EVALS_GIT_BRANCH': 'main',
            'PYDANTIC_EVALS_CI_PROVENANCE': value,
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


async def test_not_detected_when_span_is_not_recording(monkeypatch: pytest.MonkeyPatch):
    """Without a recording span, detection doesn't run, so nothing reads files or runs `git`."""
    real_logfire_span = dataset_module.logfire_span

    @contextmanager
    def non_recording_span(*args: Any, **kwargs: Any) -> Generator[Any, None, None]:
        with real_logfire_span(*args, **kwargs) as span:
            monkeypatch.setattr(span, 'is_recording', lambda: False)
            yield span

    monkeypatch.setattr(dataset_module, 'logfire_span', non_recording_span)
    monkeypatch.setattr(
        dataset_module, 'ci_provenance_attributes', lambda: pytest.fail('detection ran on a non-recording span')
    )
    await Dataset[Any, Any, Any](name='provenance', cases=[Case(inputs='x')]).evaluate(_task, progress=False)
