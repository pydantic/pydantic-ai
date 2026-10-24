"""Merge gates that a script answers from a pull request's metadata alone.

`.github/workflows/pr-gates.yml` runs each subcommand as its own check. Everything a gate judges
comes from the GitHub API. A gate fails closed, so anything it cannot read counts against the PR.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, assert_never

from check_api_compatibility import Allowlist
from pydantic import BaseModel, TypeAdapter, ValidationError

ALLOWLIST = '.github/api-compatibility-allowlist.json'
COMPATIBILITY_LABEL = 'compatibility impact'
Gate = Literal['compatibility-waiver']
TITLES: dict[Gate, str] = {'compatibility-waiver': 'Compatibility waiver labeled'}


class GateError(Exception):
    """GitHub returned something a gate cannot decide on."""


class ChangedFile(BaseModel):
    """One file a pull request changes."""

    filename: str
    # Set on a rename, which changes the old path as surely as an edit in place.
    previous_filename: str | None = None


class _Label(BaseModel):
    name: str


class _PullRequest(BaseModel):
    changed_files: int


class _Commit(BaseModel):
    sha: str


class _Comparison(BaseModel):
    merge_base_commit: _Commit


class GitHub(Protocol):
    """What the gates read from one pull request."""

    def changed_files(self) -> list[ChangedFile]: ...

    def labels(self) -> set[str]: ...

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
        # could miss a changed allowlist.
        if len(files) < total:
            raise GateError(f'Retrieved only {len(files)} of {total} changed files (the PR files API caps at 3000).')
        return files

    def labels(self) -> set[str]:
        pages = _LABEL_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._issue}/labels?per_page=100'))
        return {label.name for page in pages for label in page}

    def merge_base(self, base: str, head: str) -> str:
        return _Comparison.model_validate_json(self._api(f'{self._repo}/compare/{base}...{head}')).merge_base_commit.sha

    def file_text(self, path: str, ref: str) -> str | None:
        try:
            return self._api('-H', 'Accept: application/vnd.github.raw+json', f'{self._repo}/contents/{path}?ref={ref}')
        except GateError as error:
            if 'HTTP 404' in str(error):
                return None
            raise


def run(gate: Gate, github: GitHub, env: Mapping[str, str]) -> int:
    """Run one gate, report it as annotations and a step summary, and return the exit status."""
    title = TITLES[gate]
    try:
        match gate:
            case 'compatibility-waiver':
                verdict = compatibility_waiver(github, base_sha=env['BASE_SHA'], head_sha=env['HEAD_SHA'])
            case _:
                assert_never(gate)
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
