"""Detect where an experiment runs, so its span can be matched to a commit, branch and pull request.

GitHub Actions and GitLab CI are detected from their environment variables (and, for GitHub Actions pull request
events, the event payload file). Anywhere else, the commit and branch are read from the `git` checkout in the current
directory. Detection never raises: provenance that can't be determined is simply omitted.

The attribute names are the OpenTelemetry `vcs.*` and `cicd.*` semantic conventions.
"""

from __future__ import annotations as _annotations

import os
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

__all__ = ('ci_provenance_attributes',)

_DISABLE_ENV_VAR = 'PYDANTIC_EVALS_CI_METADATA'
_COMMIT_SHA_ENV_VAR = 'PYDANTIC_EVALS_GIT_COMMIT_SHA'
_BRANCH_ENV_VAR = 'PYDANTIC_EVALS_GIT_BRANCH'

_PROVIDER_ATTR = 'vcs.provider.name'
_REPOSITORY_URL_ATTR = 'vcs.repository.url.full'
_REVISION_ATTR = 'vcs.ref.head.revision'
_HEAD_NAME_ATTR = 'vcs.ref.head.name'
_HEAD_TYPE_ATTR = 'vcs.ref.head.type'
_BASE_NAME_ATTR = 'vcs.ref.base.name'
_CHANGE_ID_ATTR = 'vcs.change.id'
_RUN_URL_ATTR = 'cicd.pipeline.run.url.full'

_GIT_TIMEOUT_SECONDS = 5

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


def ci_provenance_attributes() -> dict[str, str]:
    """Return the `vcs.*` and `cicd.*` span attributes for the current run.

    `PYDANTIC_EVALS_GIT_COMMIT_SHA` and `PYDANTIC_EVALS_GIT_BRANCH` override the detected commit SHA and branch.
    Setting `PYDANTIC_EVALS_CI_METADATA=false` disables all of it.
    """
    env = os.environ
    if env.get(_DISABLE_ENV_VAR, '').strip().lower() in ('0', 'false'):
        return {}

    if env.get('GITHUB_ACTIONS') == 'true':
        attributes = _github_actions_attributes(env)
    elif env.get('GITLAB_CI') == 'true':
        attributes = _gitlab_ci_attributes(env)
    else:
        attributes = _git_checkout_attributes()

    if commit_sha := env.get(_COMMIT_SHA_ENV_VAR):
        attributes[_REVISION_ATTR] = commit_sha
    if branch := env.get(_BRANCH_ENV_VAR):
        _set_head_ref(attributes, branch, 'branch')
    return attributes


def _set_head_ref(attributes: dict[str, str], name: str | None, ref_type: Literal['branch', 'tag']) -> None:
    if name:
        attributes[_HEAD_NAME_ATTR] = name
        attributes[_HEAD_TYPE_ATTR] = ref_type


def _set_if_known(attributes: dict[str, str], key: str, value: str | None) -> None:
    if value:
        attributes[key] = value


def _github_actions_attributes(env: Mapping[str, str]) -> dict[str, str]:
    attributes: dict[str, str] = {_PROVIDER_ATTR: 'github'}
    server, repo = env.get('GITHUB_SERVER_URL'), env.get('GITHUB_REPOSITORY')
    if server and repo:
        attributes[_REPOSITORY_URL_ATTR] = f'{server}/{repo}'
        if run_id := env.get('GITHUB_RUN_ID'):
            attributes[_RUN_URL_ATTR] = f'{server}/{repo}/actions/runs/{run_id}'

    head_type: Literal['branch', 'tag'] = 'branch'
    pull_request = _read_pull_request(env.get('GITHUB_EVENT_PATH'))
    if pull_request is not None:
        # On pull request events, `GITHUB_SHA` and `GITHUB_REF_NAME` point at the synthetic merge commit
        # (`refs/pull/<number>/merge`), so the PR head commit and branch come from the event payload.
        commit_sha = pull_request.head.sha
        head_name = pull_request.head.ref or env.get('GITHUB_HEAD_REF')
        base_name = pull_request.base.ref or env.get('GITHUB_BASE_REF')
        number = pull_request.number
    elif head_ref := env.get('GITHUB_HEAD_REF'):
        # A pull request event whose payload couldn't be read: the head commit is unknown, and
        # `GITHUB_SHA` would be the merge commit, so the SHA is omitted rather than misreported.
        commit_sha = None
        head_name = head_ref
        base_name = env.get('GITHUB_BASE_REF')
        match = _PULL_REQUEST_REF_RE.match(env.get('GITHUB_REF', ''))
        number = int(match.group(1)) if match else None
    else:
        commit_sha = env.get('GITHUB_SHA')
        ref_type = env.get('GITHUB_REF_TYPE')
        head_name = env.get('GITHUB_REF_NAME') if ref_type in ('branch', 'tag') else None
        head_type = 'tag' if ref_type == 'tag' else 'branch'
        base_name = None
        number = None

    if env.get('GITHUB_EVENT_NAME') == 'pull_request_target':
        # These runs check out the base branch by default, so whether the PR head is the code under test
        # is unknown, and its commit and branch are omitted rather than possibly misreported.
        commit_sha = head_name = None

    _set_if_known(attributes, _REVISION_ATTR, commit_sha)
    _set_head_ref(attributes, head_name, head_type)
    _set_if_known(attributes, _BASE_NAME_ATTR, base_name)
    if number is not None:
        attributes[_CHANGE_ID_ATTR] = str(number)
    return attributes


def _read_pull_request(event_path: str | None) -> _GitHubPullRequest | None:
    if not event_path:
        return None
    try:
        return _GitHubEvent.model_validate_json(Path(event_path).read_bytes()).pull_request
    except (OSError, ValidationError):
        return None


def _gitlab_ci_attributes(env: Mapping[str, str]) -> dict[str, str]:
    attributes: dict[str, str] = {_PROVIDER_ATTR: 'gitlab'}
    _set_if_known(attributes, _REPOSITORY_URL_ATTR, env.get('CI_PROJECT_URL'))
    _set_if_known(attributes, _RUN_URL_ATTR, env.get('CI_PIPELINE_URL'))

    if merge_request_iid := env.get('CI_MERGE_REQUEST_IID'):
        if env.get('CI_MERGE_REQUEST_EVENT_TYPE') in ('merged_result', 'merge_train'):
            # These pipelines build a merge of the source and target branches, so `CI_COMMIT_SHA` is that
            # merge commit. GitLab reports the source branch head separately.
            commit_sha = env.get('CI_MERGE_REQUEST_SOURCE_BRANCH_SHA')
        else:
            commit_sha = env.get('CI_COMMIT_SHA')
        _set_if_known(attributes, _REVISION_ATTR, commit_sha)
        _set_head_ref(attributes, env.get('CI_MERGE_REQUEST_SOURCE_BRANCH_NAME'), 'branch')
        _set_if_known(attributes, _BASE_NAME_ATTR, env.get('CI_MERGE_REQUEST_TARGET_BRANCH_NAME'))
        attributes[_CHANGE_ID_ATTR] = merge_request_iid
    else:
        _set_if_known(attributes, _REVISION_ATTR, env.get('CI_COMMIT_SHA'))
        if tag := env.get('CI_COMMIT_TAG'):
            _set_head_ref(attributes, tag, 'tag')
        else:
            _set_head_ref(attributes, env.get('CI_COMMIT_BRANCH'), 'branch')
    return attributes


def _git_checkout_attributes() -> dict[str, str]:
    """The commit and branch of the `git` checkout in the current directory, if there is one.

    The remote URL is not recorded, because it can contain credentials.
    """
    attributes: dict[str, str] = {}
    if commit_sha := _git('rev-parse', 'HEAD'):
        attributes[_REVISION_ATTR] = commit_sha
        # `symbolic-ref` fails on a detached HEAD, which has no branch.
        _set_head_ref(attributes, _git('symbolic-ref', '--quiet', '--short', 'HEAD'), 'branch')
    return attributes


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ['git', *args],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output if result.returncode == 0 and output else None
