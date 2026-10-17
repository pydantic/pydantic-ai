"""Detect the CI context an experiment runs in, so its span can be matched to a commit, branch and pull request.

Detection only reads environment variables and, for GitHub Actions pull request events, the event payload file.
It never runs `git` and never raises: provenance that can't be determined is simply omitted.
"""

from __future__ import annotations as _annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

__all__ = ('ci_provenance_attributes',)

_DISABLE_ENV_VAR = 'PYDANTIC_EVALS_CI_METADATA'
_COMMIT_SHA_ENV_VAR = 'PYDANTIC_EVALS_GIT_COMMIT_SHA'
_BRANCH_ENV_VAR = 'PYDANTIC_EVALS_GIT_BRANCH'

_COMMIT_SHA_ATTR = 'logfire.experiment.git.commit_sha'
_BRANCH_ATTR = 'logfire.experiment.git.branch'
_BASE_BRANCH_ATTR = 'logfire.experiment.git.base_branch'
_PULL_REQUEST_ATTR = 'logfire.experiment.ci.pull_request'
_RUN_URL_ATTR = 'logfire.experiment.ci.run_url'

_PULL_REQUEST_REF_RE = re.compile(r'refs/pull/(\d+)/')


class _GitHubRef(BaseModel):
    ref: str | None = None
    sha: str | None = None


class _GitHubPullRequest(BaseModel):
    number: int | None = None
    head: _GitHubRef = Field(default_factory=_GitHubRef)
    base: _GitHubRef = Field(default_factory=_GitHubRef)


class _GitHubEvent(BaseModel):
    pull_request: _GitHubPullRequest | None = None


def ci_provenance_attributes() -> dict[str, str | int]:
    """Return the `logfire.experiment.git.*` and `logfire.experiment.ci.*` span attributes for the current run.

    GitHub Actions is detected automatically. Other CI systems can supply the commit SHA and branch through
    `PYDANTIC_EVALS_GIT_COMMIT_SHA` and `PYDANTIC_EVALS_GIT_BRANCH`, which also override what GitHub Actions
    reports. Setting `PYDANTIC_EVALS_CI_METADATA=false` disables all of it.
    """
    env = os.environ
    if env.get(_DISABLE_ENV_VAR, '').strip().lower() in ('0', 'false'):
        return {}

    attributes: dict[str, str | int] = {}
    if env.get('GITHUB_ACTIONS') == 'true':
        attributes.update(_github_actions_attributes(env))
    if commit_sha := env.get(_COMMIT_SHA_ENV_VAR):
        attributes[_COMMIT_SHA_ATTR] = commit_sha
    if branch := env.get(_BRANCH_ENV_VAR):
        attributes[_BRANCH_ATTR] = branch
    return attributes


def _github_actions_attributes(env: Mapping[str, str]) -> dict[str, str | int]:
    attributes: dict[str, str | int] = {}
    if (
        (server := env.get('GITHUB_SERVER_URL'))
        and (repo := env.get('GITHUB_REPOSITORY'))
        and (run_id := env.get('GITHUB_RUN_ID'))
    ):
        attributes[_RUN_URL_ATTR] = f'{server}/{repo}/actions/runs/{run_id}'

    pull_request = _read_pull_request(env.get('GITHUB_EVENT_PATH'))
    if pull_request is not None:
        # On pull request events, `GITHUB_SHA` and `GITHUB_REF_NAME` point at the synthetic merge commit
        # (`refs/pull/<number>/merge`), so the PR head commit and branch come from the event payload.
        commit_sha = pull_request.head.sha
        branch = pull_request.head.ref or env.get('GITHUB_HEAD_REF')
        base_branch = pull_request.base.ref or env.get('GITHUB_BASE_REF')
        number = pull_request.number
    elif head_ref := env.get('GITHUB_HEAD_REF'):
        # A pull request event whose payload couldn't be read: the head commit is unknown, and
        # `GITHUB_SHA` would be the merge commit, so the SHA is omitted rather than misreported.
        commit_sha = None
        branch = head_ref
        base_branch = env.get('GITHUB_BASE_REF')
        match = _PULL_REQUEST_REF_RE.match(env.get('GITHUB_REF', ''))
        number = int(match.group(1)) if match else None
    else:
        commit_sha = env.get('GITHUB_SHA')
        # `GITHUB_REF_NAME` is a tag name when the run was triggered by a tag.
        branch = env.get('GITHUB_REF_NAME') if env.get('GITHUB_REF_TYPE') == 'branch' else None
        base_branch = None
        number = None

    if env.get('GITHUB_EVENT_NAME') == 'pull_request_target':
        # These runs check out the base branch by default, so whether the PR head is the code under test
        # is unknown, and its commit and branch are omitted rather than possibly misreported.
        commit_sha = branch = None

    if commit_sha:
        attributes[_COMMIT_SHA_ATTR] = commit_sha
    if branch:
        attributes[_BRANCH_ATTR] = branch
    if base_branch:
        attributes[_BASE_BRANCH_ATTR] = base_branch
    if number is not None:
        attributes[_PULL_REQUEST_ATTR] = number
    return attributes


def _read_pull_request(event_path: str | None) -> _GitHubPullRequest | None:
    if not event_path:
        return None
    try:
        return _GitHubEvent.model_validate_json(Path(event_path).read_bytes()).pull_request
    except (OSError, ValidationError):
        return None
