"""Merge gates that a script answers from a pull request's metadata alone.

`.github/workflows/pr-gates.yml` runs each subcommand as its own check. Everything a gate judges
comes from the GitHub API. A gate fails closed, so anything it cannot read counts against the PR.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol, assert_never

from check_api_compatibility import Allowlist
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

ALLOWLIST = '.github/api-compatibility-allowlist.json'
COMPATIBILITY_LABEL = 'compatibility impact'
Gate = Literal['compatibility-waiver']
TITLES: dict[Gate, str] = {'compatibility-waiver': 'Compatibility waiver labeled'}
ReviewGate = Literal['douwebot', 'human']
Size = Literal['size: S', 'size: M', 'size: L', 'size: XL']
REVIEW_CONTEXTS: dict[ReviewGate, str] = {
    'douwebot': 'Gate: clean douwebot approval',
    'human': 'Gate: human approval',
}
DOUWEBOT_MARKER = re.compile(r'\n\n<!-- douwebot:review run=([1-9][0-9]*) head=([a-f0-9]{40}) -->\Z')


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


class ReviewUser(BaseModel):
    """An API identity; repository permission is checked separately."""

    id: int
    login: str
    type: str


class ReviewPullRequest(BaseModel):
    """The live head, base, author, and labels that determine the review bar."""

    state: Literal['open', 'closed']
    head: _Commit
    base: _Commit
    user: ReviewUser
    labels: list[_Label]
    changed_files: int


class Review(BaseModel):
    """A submitted review, including its immutable reviewed commit."""

    id: int
    user: ReviewUser
    state: Literal['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED', 'DISMISSED', 'PENDING']
    commit_id: str
    body: str
    submitted_at: datetime | None


class ReviewComment(BaseModel):
    """The parent review identifies the head; `commit_id` can move on later pushes."""

    pull_request_review_id: int
    user: ReviewUser
    original_commit_id: str
    body: str


class ReviewRun(BaseModel):
    """Trusted workflow metadata for a marked douwebot publication."""

    name: str
    path: str
    event: str
    display_title: str
    conclusion: str | None


class ReviewJob(BaseModel):
    """A job's terminal outcome in the originating run."""

    name: str
    conclusion: str | None


class _Jobs(BaseModel):
    jobs: list[ReviewJob]


class _Permission(BaseModel):
    permission: Literal['admin', 'write', 'read', 'none']


class _SizedFile(ChangedFile):
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)


class _SizeResult(BaseModel):
    size: Size
    score: int


class ReviewGitHub(Protocol):
    """The metadata needed by the review gates."""

    def pull_request(self) -> ReviewPullRequest: ...
    def size(self, pull: ReviewPullRequest) -> Size: ...
    def reviews(self) -> list[Review]: ...
    def review_comments(self) -> list[ReviewComment]: ...
    def permission(self, login: str) -> str: ...
    def verified_review_run(self, run_id: int, head: str) -> bool: ...
    def status(self, head: str, gate: ReviewGate, state: str, description: str, url: str) -> None: ...


def review_verdicts(
    github: ReviewGitHub,
    pull: ReviewPullRequest,
    reviews: list[Review],
    comments: list[ReviewComment],
) -> dict[ReviewGate, Verdict]:
    """Apply the size-dependent review bar to a live metadata snapshot."""
    verdicts: dict[ReviewGate, Verdict] = {'douwebot': Verdict(), 'human': Verdict()}
    size = github.size(pull)
    size_labels = {label.name for label in pull.labels if label.name.startswith('size:')}
    if size_labels != {size}:
        for verdict in verdicts.values():
            verdict.failures.append(f'Expected exactly the recomputed `{size}` label; found {sorted(size_labels)!r}.')
        return verdicts
    if size == 'size: S':
        for verdict in verdicts.values():
            verdict.summary.append(
                'Size S: automated review enforcement is skipped; the manual merge bar still applies.'
            )
        return verdicts

    verdicts['douwebot'] = douwebot_approval(github, pull, reviews, comments)
    if size in ('size: L', 'size: XL'):
        verdicts['human'] = human_approval(github, pull, reviews)
    else:
        verdicts['human'].summary.append('Size M does not require a human approval.')
    return verdicts


def douwebot_approval(
    github: ReviewGitHub,
    pull: ReviewPullRequest,
    reviews: list[Review],
    comments: list[ReviewComment],
) -> Verdict:
    """Verify review provenance, then associate findings with their immutable review head."""
    verdict = Verdict()
    own_reviews: list[Review] = []
    marked_ids: set[int] = set()
    for review in reviews:
        if review.user.login != 'github-actions[bot]' or review.user.type != 'Bot' or review.commit_id != pull.head.sha:
            continue
        marker = DOUWEBOT_MARKER.search(review.body)
        if marker and marker[2] == pull.head.sha:
            marked_ids.add(review.id)
            if github.verified_review_run(int(marker[1]), pull.head.sha):
                own_reviews.append(review)

    decisions = [review for review in own_reviews if review.state in ('APPROVED', 'CHANGES_REQUESTED')]
    if (
        not decisions
        or max(
            decisions, key=lambda review: (review.submitted_at.timestamp() if review.submitted_at else 0, review.id)
        ).state
        != 'APPROVED'
    ):
        verdict.failures.append('Requires a marked, successful douwebot approval on the current head.')
    if any(
        review.state == 'APPROVED' and review.body.split('\n\n<!-- douwebot:review')[0] != 'No blocking findings.'
        for review in own_reviews
    ):
        verdict.failures.append('Douwebot has discussion findings on this head.')
    by_id = {review.id: review for review in reviews}
    for comment in comments:
        if comment.user.login != 'github-actions[bot]' or comment.user.type != 'Bot':
            continue
        parent = by_id.get(comment.pull_request_review_id)
        reviewed_head = parent.commit_id if parent else comment.original_commit_id
        if reviewed_head != pull.head.sha:
            continue
        if comment.pull_request_review_id in marked_ids:
            verdict.failures.append('Douwebot has inline findings on this head, including resolved/optional ones.')
        else:
            # gh-aw gives its reviews a separate receipt despite sharing the account.
            receipt = f'{parent.body if parent else ""}\n{comment.body}'
            if any(
                f'<!-- gh-aw-workflow-call-id: {workflow} -->' in receipt
                or f'[gh-aw-workflow-call-id]: # "{workflow.replace("/", "%2F")}"' in receipt
                for workflow in (
                    'pydantic/pydantic-ai/pydantic-ai-pr-review',
                    'pydantic/pydantic-ai/pydantic-ai-ui-security-review',
                )
            ):
                continue
            verdict.failures.append(
                'Cannot attribute an unmarked bot comment on this head; obtain a fresh review/head.'
            )
        break
    if any(review.state == 'COMMENTED' for review in own_reviews):
        verdict.failures.append('Douwebot published an incomplete review on this head.')
    return verdict


def human_approval(github: ReviewGitHub, pull: ReviewPullRequest, reviews: list[Review]) -> Verdict:
    """Accept the latest non-dismissed decision from a different human with current write access."""
    verdict = Verdict()
    latest: dict[int, Review] = {}
    for review in sorted(
        reviews, key=lambda review: (review.submitted_at.timestamp() if review.submitted_at else 0, review.id)
    ):
        if (
            review.user.type == 'User'
            and review.user.id != pull.user.id
            and review.state in ('APPROVED', 'CHANGES_REQUESTED')
        ):
            latest[review.user.id] = review
    if not any(
        review.state == 'APPROVED' and github.permission(review.user.login) in ('write', 'admin')
        for review in latest.values()
    ):
        verdict.failures.append('Requires a non-dismissed approval from another human with write/admin permission.')
    return verdict


def reconcile_reviews(github: ReviewGitHub, env: Mapping[str, str]) -> int:
    """Serialize externally, read live state, and replace the statuses on that exact head."""
    url = f'{env["GITHUB_SERVER_URL"]}/{env["GITHUB_REPOSITORY"]}/actions/runs/{env["GITHUB_RUN_ID"]}'
    if group := env.get('MERGE_GROUP_SHA'):
        for gate in REVIEW_CONTEXTS:
            github.status(group, gate, 'success', 'Admission-only review gate; merge-group evaluation skipped.', url)
        return 0
    head = env['HEAD_SHA']
    verdicts: dict[ReviewGate, Verdict]
    try:
        for _ in range(3):
            pull = github.pull_request()
            if pull.state != 'open':
                return 0
            head = pull.head.sha
            for gate in REVIEW_CONTEXTS:
                github.status(head, gate, 'pending', 'Reading current size and reviews.', url)
            reviews = github.reviews()
            comments = github.review_comments()
            verdicts = review_verdicts(github, pull, reviews, comments)
            # A push or relabel during the API reads must not publish an old snapshot.
            if github.pull_request() == pull and github.reviews() == reviews and github.review_comments() == comments:
                break
        else:
            raise GateError('PR changed repeatedly while evaluating reviews; retry the gate.')
    except (GateError, ValidationError, OSError) as error:
        verdicts = {gate: Verdict(failures=[f'{type(error).__name__}: {error}']) for gate in REVIEW_CONTEXTS}
    failed = False
    for gate, verdict in verdicts.items():
        failed |= bool(verdict.failures)
        state = 'failure' if verdict.failures else 'success'
        description = verdict.failures[0] if verdict.failures else 'Review requirement satisfied.'
        github.status(head, gate, state, description[:140], url)
        report = '\n\n'.join([f'## {REVIEW_CONTEXTS[gate]}: {state}', *verdict.summary, *verdict.failures])
        print(report)
        if summary_path := env.get('GITHUB_STEP_SUMMARY'):
            with open(summary_path, 'a', encoding='utf-8') as summary:
                summary.write(report + '\n')
    return int(failed)


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

    def _api(self, *args: str, input: str | None = None) -> str:
        result = subprocess.run(['gh', 'api', *args], input=input, check=False, capture_output=True, text=True)
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

    def pull_request(self) -> ReviewPullRequest:
        """Read current PR metadata, never an event's stale labels/head."""
        return ReviewPullRequest.model_validate_json(self._api(self._pull))

    def size(self, pull: ReviewPullRequest) -> Size:
        """Use the same Git-attribute-aware calculation as PR Bots."""
        adapter = TypeAdapter(list[list[_SizedFile]])
        pages = adapter.validate_json(self._api('--paginate', '--slurp', f'{self._pull}/files?per_page=100'))
        files = [file for page in pages for file in page]
        if len(files) != pull.changed_files:
            raise GateError('Incomplete PR files response; cannot determine size.')
        # Fetch base objects as data only; never check out or execute PR code.
        present = subprocess.run(['git', 'cat-file', '-e', f'{pull.base.sha}^{{commit}}'], capture_output=True)
        if present.returncode:
            fetched = subprocess.run(
                ['git', 'fetch', '--no-tags', '--depth=1', 'origin', pull.base.sha], capture_output=True
            )
            if fetched.returncode:
                raise GateError('Cannot fetch base attributes for size calculation.')
        result = subprocess.run(
            ['bash', str(Path(__file__).with_name('pr_size.sh')), pull.base.sha],
            input=TypeAdapter(list[_SizedFile]).dump_json(files).decode(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise GateError(f'Size calculation failed: {result.stderr.strip()}')
        return _SizeResult.model_validate_json(result.stdout).size

    def reviews(self) -> list[Review]:
        """Read every review, including superseded and dismissed decisions."""
        pages = TypeAdapter(list[list[Review]]).validate_json(
            self._api('--paginate', '--slurp', f'{self._pull}/reviews?per_page=100')
        )
        return [review for page in pages for review in page]

    def review_comments(self) -> list[ReviewComment]:
        """Read every inline comment; resolution does not make a finding clean."""
        pages = TypeAdapter(list[list[ReviewComment]]).validate_json(
            self._api('--paginate', '--slurp', f'{self._pull}/comments?per_page=100')
        )
        return [comment for page in pages for comment in page]

    def permission(self, login: str) -> str:
        """Resolve the reviewer's current base permission."""
        return _Permission.model_validate_json(self._api(f'{self._repo}/collaborators/{login}/permission')).permission

    def verified_review_run(self, run_id: int, head: str) -> bool:
        """Only the trusted PR Bots publisher can supply a qualifying marker."""
        run = ReviewRun.model_validate_json(self._api(f'{self._repo}/actions/runs/{run_id}'))
        number = self._pull.rsplit('/', 1)[1]
        if not (
            run.path == '.github/workflows/bots.yml'
            and run.event == 'pull_request_target'
            and run.name == 'PR Bots'
            and run.display_title == f'PR Bots #{number} @ {head}'
            and run.conclusion == 'success'
        ):
            return False
        pages = TypeAdapter(list[_Jobs]).validate_json(
            self._api('--paginate', '--slurp', f'{self._repo}/actions/runs/{run_id}/jobs?per_page=100')
        )
        jobs = {job.name: job.conclusion for page in pages for job in page.jobs}
        return jobs.get('douwebot') == 'success' and jobs.get('Publish douwebot review') == 'success'

    def status(self, head: str, gate: ReviewGate, state: str, description: str, url: str) -> None:
        """Latest-per-context commit statuses replace a previous failing verdict."""
        payload: dict[str, str] = {
            'context': REVIEW_CONTEXTS[gate],
            'state': state,
            'description': description,
            'target_url': url,
        }
        self._api('--method', 'POST', '--input', '-', f'{self._repo}/statuses/{head}', input=json.dumps(payload))

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
    parser.add_argument('gate', choices=[*TITLES, 'review-gates'])
    args = parser.parse_args()
    env = os.environ
    if args.gate == 'review-gates':
        return reconcile_reviews(GhCli(env['GITHUB_REPOSITORY'], int(env.get('PR_NUMBER') or '0')), env)
    if env.get('PR_STATE', 'open') != 'open':
        # `edited` also fires on closed PRs, where there is no merge left to gate.
        print(f'PR is {env["PR_STATE"]}; nothing to gate.')
        return 0
    return run(args.gate, GhCli(env['GITHUB_REPOSITORY'], int(env['PR_NUMBER'])), env)


if __name__ == '__main__':
    sys.exit(main())
