"""Merge gates that a script answers from a pull request's metadata alone.

`.github/workflows/pr-gates.yml` runs each subcommand as its own check. It triggers on
`pull_request_target`, so this script runs from the default branch and never sees the PR's
own code: everything a gate judges comes from the GitHub API. A gate fails closed, so
anything it cannot read counts against the PR.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Literal, Protocol

from check_api_compatibility import Allowlist
from pydantic import BaseModel, TypeAdapter, ValidationError

ALLOWLIST = '.github/api-compatibility-allowlist.json'
COMPATIBILITY_LABEL = 'compatibility impact'
APPROVER_PERMISSIONS = frozenset({'write', 'admin'})
DEPENDENCY_FILES = frozenset({'pyproject.toml', 'uv.lock'})
CI_CONFIG_DIRS = ('.github/workflows/', '.github/actions/', '.github/scripts/')
CI_CONFIG_FILES = frozenset({'.pre-commit-config.yaml', 'Makefile'})
TITLES = {'protected-paths': 'Protected paths approved', 'compatibility-waiver': 'Compatibility waiver labeled'}


class GateError(Exception):
    """GitHub returned something a gate cannot decide on."""


class ChangedFile(BaseModel):
    """One file a pull request or push changes."""

    filename: str
    # Set on a rename, which changes the old path as surely as an edit in place.
    previous_filename: str | None = None


class _Actor(BaseModel):
    login: str
    type: str


class _Label(BaseModel):
    name: str


class LabelEvent(BaseModel):
    """One entry of a pull request's issue events."""

    event: str
    actor: _Actor | None = None
    label: _Label | None = None


class _PullRequest(BaseModel):
    changed_files: int


class _Commit(BaseModel):
    sha: str


class _Comparison(BaseModel):
    status: Literal['ahead', 'behind', 'diverged', 'identical']
    merge_base_commit: _Commit
    files: list[ChangedFile] = []


class _Permission(BaseModel):
    permission: str


class GitHub(Protocol):
    """What the gates read from, and change on, one pull request."""

    def changed_files(self) -> list[ChangedFile]: ...

    def changed_files_between(self, base: str, head: str) -> list[ChangedFile] | None:
        """Files changed from `base` to `head`, or `None` when the full list is unknowable."""
        ...

    def labels(self) -> set[str]: ...

    def label_events(self) -> list[LabelEvent]: ...

    def permission(self, login: str) -> str:
        """The user's `.permission` on this repository, or `none` when it cannot be read."""
        ...

    def remove_label(self, name: str) -> None: ...

    def merge_base(self, base: str, head: str) -> str: ...

    def file_text(self, path: str, ref: str) -> str | None:
        """The file's contents at `ref`, or `None` when it does not exist there."""
        ...


@dataclass
class Verdict:
    """A gate's report: the step summary lines, and the failures that make it fail."""

    summary: list[str] = field(default_factory=list[str])
    failures: list[str] = field(default_factory=list[str])


def _paths(files: Iterable[ChangedFile]) -> set[str]:
    return {path for file in files for path in (file.filename, file.previous_filename) if path is not None}


def _scoped(paths: Iterable[str]) -> dict[str, list[str]]:
    """Group changed paths by the approval scope they need."""
    by_scope: dict[str, list[str]] = {}
    for path in sorted(paths):
        if PurePosixPath(path).name in DEPENDENCY_FILES:
            by_scope.setdefault('dependencies', []).append(path)
        if path.startswith(CI_CONFIG_DIRS) or path in CI_CONFIG_FILES:
            by_scope.setdefault('ci-config', []).append(path)
    return by_scope


def protected_paths(github: GitHub, *, pushed: tuple[str, str] | None = None, unlisted_change: bool = False) -> Verdict:
    """Require an `approved:<scope>` label, applied by a maintainer, for every protected scope touched.

    `pushed` is the `(before, after)` pair of a `synchronize` event. A push that changes the
    PR's files in a scope voids that scope's approval, so its label is removed before the verdict.
    `unlisted_change` marks an event that changes the PR without a push to list, which voids
    every approval the PR depends on.
    """
    verdict = Verdict()
    pr_paths = _paths(github.changed_files())
    touched = _scoped(pr_paths)
    if not touched:
        verdict.summary.append('No protected path changed.')
        return verdict

    labels = github.labels()
    if pushed is not None or unlisted_change:
        delta = None if pushed is None else github.changed_files_between(*pushed)
        # Only the PR's own files count: a merge of `main` brings in protected changes the PR doesn't make.
        voided = set(touched) if delta is None else set(_scoped(_paths(delta) & pr_paths))
        for scope in sorted(voided):
            label = f'approved:{scope}'
            if label in labels:
                github.remove_label(label)
                labels.discard(label)
                reason = (
                    'the changes since it was applied cannot be listed'
                    if delta is None
                    else 'a push changed this scope after it was applied'
                )
                verdict.summary.append(f'Removed `{label}`: {reason}.')

    # Events come oldest first, so the last `labeled` event for a label names who applied it now.
    appliers: dict[str, _Actor | None] = {}
    for event in github.label_events():
        if event.event == 'labeled' and event.label is not None:
            appliers[event.label.name] = event.actor

    for scope, paths in sorted(touched.items()):
        label = f'approved:{scope}'
        listed = ', '.join(f'`{path}`' for path in paths)
        verdict.summary.append(f'Scope `{scope}`: {listed}')
        if label not in labels:
            verdict.failures.append(f'Changes to {scope} need the `{label}` label from a maintainer: {listed}')
            continue
        applier = appliers.get(label)
        if applier is None:
            verdict.failures.append(f'Cannot tell who applied `{label}`; a maintainer has to re-apply it.')
        elif applier.type != 'User':
            verdict.failures.append(
                f'`{label}` was applied by {applier.login}, not a person; a maintainer has to re-apply it.'
            )
        elif (permission := github.permission(applier.login)) not in APPROVER_PERMISSIONS:
            verdict.failures.append(
                f'`{label}` was applied by {applier.login}, whose permission is `{permission}`; it needs write or admin.'
            )
        else:
            verdict.summary.append(f'`{label}` applied by {applier.login} ({permission}).')
    return verdict


def compatibility_waiver(github: GitHub, *, base_sha: str, head_sha: str) -> Verdict:
    """Require `compatibility impact` on a PR that adds an API compatibility waiver."""
    verdict = Verdict()
    if ALLOWLIST not in _paths(github.changed_files()):
        verdict.summary.append(f'`{ALLOWLIST}` is unchanged.')
        return verdict

    merge_base = github.merge_base(base_sha, head_sha)
    try:
        added = _waivers(github.file_text(ALLOWLIST, head_sha)) - _waivers(github.file_text(ALLOWLIST, merge_base))
    except ValidationError as error:
        verdict.failures.append(f'Cannot read `{ALLOWLIST}`: {error.error_count()} validation error(s).')
        return verdict

    if not added:
        verdict.summary.append('No waiver added.')
    elif COMPATIBILITY_LABEL in github.labels():
        verdict.summary.append(f'{len(added)} waiver(s) added, labeled `{COMPATIBILITY_LABEL}`.')
    else:
        verdict.failures.append(
            f'Adds {len(added)} API compatibility waiver(s) without the `{COMPATIBILITY_LABEL}` label, '
            'so the break would not appear in the release notes.'
        )
    return verdict


def _waivers(text: str | None) -> set[tuple[str, str]]:
    if text is None:
        return set()
    return {(waiver.against, waiver.fingerprint) for waiver in Allowlist.model_validate_json(text).allowed_breakages}


_FILE_PAGES = TypeAdapter(list[list[ChangedFile]])
_LABEL_PAGES = TypeAdapter(list[list[_Label]])
_EVENT_PAGES = TypeAdapter(list[list[LabelEvent]])


class GhCli:
    """`GitHub`, through `gh api` and the token in `GH_TOKEN`."""

    def __init__(self, repo: str, number: int) -> None:
        self._repo = f'repos/{repo}'
        self._issue = f'repos/{repo}/issues/{number}'
        self._pull = f'repos/{repo}/pulls/{number}'

    def _api(self, *args: str) -> str:
        result = subprocess.run(['gh', 'api', *args], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise GateError(f'`gh api {args[-1]}` failed: {result.stderr.strip()}')
        return result.stdout

    def changed_files(self) -> list[ChangedFile]:
        total = _PullRequest.model_validate_json(self._api(self._pull)).changed_files
        pages = _FILE_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._pull}/files?per_page=100'))
        files = [file for page in pages for file in page]
        # The endpoint caps at 3000 files and truncates silently; deciding on a partial list
        # could wave a protected change through.
        if len(files) < total:
            raise GateError(f'Retrieved only {len(files)} of {total} changed files (the PR files API caps at 3000).')
        return files

    def changed_files_between(self, base: str, head: str) -> list[ChangedFile] | None:
        try:
            comparison = _Comparison.model_validate_json(self._api(f'{self._repo}/compare/{base}...{head}'))
        except GateError:
            # A force-push can leave `base` unreachable from this repository.
            return None
        # The comparison runs from the merge base, so it is the push itself only when `head`
        # descends from `base`. A rollback (`behind`) or rewritten history (`diverged`) can hide a
        # changed file. The compare API also lists at most 300 files.
        if comparison.status not in ('ahead', 'identical') or len(comparison.files) >= 300:
            return None
        return comparison.files

    def labels(self) -> set[str]:
        pages = _LABEL_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._issue}/labels?per_page=100'))
        return {label.name for page in pages for label in page}

    def label_events(self) -> list[LabelEvent]:
        pages = _EVENT_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._issue}/events?per_page=100'))
        return [event for page in pages for event in page]

    def permission(self, login: str) -> str:
        try:
            output = self._api(f'{self._repo}/collaborators/{urllib.parse.quote(login, safe="")}/permission')
        except GateError:
            return 'none'
        return _Permission.model_validate_json(output).permission

    def remove_label(self, name: str) -> None:
        self._api('--method', 'DELETE', f'{self._issue}/labels/{urllib.parse.quote(name, safe="")}')

    def merge_base(self, base: str, head: str) -> str:
        return _Comparison.model_validate_json(self._api(f'{self._repo}/compare/{base}...{head}')).merge_base_commit.sha

    def file_text(self, path: str, ref: str) -> str | None:
        try:
            return self._api('-H', 'Accept: application/vnd.github.raw+json', f'{self._repo}/contents/{path}?ref={ref}')
        except GateError as error:
            if 'HTTP 404' in str(error):
                return None
            raise


def run(gate: str, github: GitHub, env: Mapping[str, str]) -> int:
    """Run one gate, report it as annotations and a step summary, and return the exit status."""
    title = TITLES[gate]
    try:
        if gate == 'protected-paths':
            action = env.get('EVENT_ACTION')
            pushed = (env['BEFORE_SHA'], env['AFTER_SHA']) if action == 'synchronize' else None
            # A closed PR's branch takes pushes that fire no `synchronize`, and a new base branch
            # changes the diff without a push.
            unlisted_change = action == 'reopened' or (action == 'edited' and bool(env.get('BASE_FROM')))
            verdict = protected_paths(github, pushed=pushed, unlisted_change=unlisted_change)
        else:
            verdict = compatibility_waiver(github, base_sha=env['BASE_SHA'], head_sha=env['HEAD_SHA'])
    except (GateError, ValidationError) as error:
        verdict = Verdict(failures=[f'{type(error).__name__}: {error}'])

    for failure in verdict.failures:
        escaped = failure.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')
        print(f'::error title={title}::{escaped}')
    report = '\n\n'.join(
        [f'## {title}: {"failed" if verdict.failures else "passed"}', *verdict.summary, *verdict.failures]
    )
    if summary_path := env.get('GITHUB_STEP_SUMMARY'):
        with open(summary_path, 'a', encoding='utf-8') as summary:
            summary.write(report + '\n')
    else:
        print(report)
    return 1 if verdict.failures else 0


def main() -> int:
    """Run the gate named on the command line against the PR in the environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gate', choices=list(TITLES))
    args = parser.parse_args()
    env = os.environ
    if env.get('PR_STATE', 'open') != 'open':
        # `edited` also fires on closed PRs, where there is no merge left to gate.
        print(f'PR is {env["PR_STATE"]}; nothing to gate.')
        return 0
    return run(args.gate, GhCli(env['GITHUB_REPOSITORY'], int(env['PR_NUMBER'])), env)


if __name__ == '__main__':
    sys.exit(main())
