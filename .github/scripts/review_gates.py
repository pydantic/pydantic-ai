"""Publish size-dependent review statuses from trusted default-branch code."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pr_gates import ChangedFile, Commit, GateError, GhCli, Label, Verdict
from publish_douwebot_review import APPROVAL_BODY, DOUWEBOT_MARKER, REVIEW_BODIES
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

ReviewGate = Literal['douwebot', 'human']
Size = Literal['size: S', 'size: M', 'size: L', 'size: XL']
Permission = Literal['admin', 'write', 'read', 'none']
CommitStatus = Literal['pending', 'success', 'failure']
REVIEW_CONTEXTS: dict[ReviewGate, str] = {
    'douwebot': 'Gate: clean douwebot approval',
    'human': 'Gate: human approval',
}


class ReviewUser(BaseModel):
    """An API identity; repository permission is checked separately."""

    id: int
    login: str
    type: str


class _Repository(BaseModel):
    id: int = Field(gt=0)


class _Head(Commit):
    repo: _Repository


class ReviewPullRequest(BaseModel):
    """The live head, base, author, and labels that determine the review bar."""

    state: Literal['open', 'closed']
    head: _Head
    base: Commit
    user: ReviewUser
    labels: list[Label]
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
    permission: Permission


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
    def permission(self, login: str) -> Permission: ...
    def verified_review_run(self, run_id: int, head: str) -> bool: ...
    def status(self, head: str, gate: ReviewGate, state: CommitStatus, description: str, url: str) -> None: ...


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
            body = review.body[: marker.start()]
            if body not in REVIEW_BODIES.values() or (review.state == 'APPROVED' and body != APPROVAL_BODY):
                verdict.failures.append('Douwebot has discussion findings on this head.')
            if github.verified_review_run(int(marker[1]), pull.head.sha):
                own_reviews.append(review)

    decisions: list[Review] = [
        review for review in own_reviews if review.state in ('APPROVED', 'CHANGES_REQUESTED', 'COMMENTED')
    ]
    if not decisions or max(decisions, key=review_order).state != 'APPROVED':
        verdict.failures.append('Requires a marked, successful douwebot approval on the current head.')
    by_id: dict[int, Review] = {review.id: review for review in reviews}
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
    return verdict


def human_approval(github: ReviewGitHub, pull: ReviewPullRequest, reviews: list[Review]) -> Verdict:
    """Accept the latest non-dismissed decision from a different human with current write access."""
    verdict = Verdict()
    latest: dict[int, Review] = {}
    for review in sorted(reviews, key=review_order):
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


def review_order(review: Review) -> tuple[float, int]:
    """Order submitted decisions chronologically, including reviews created as drafts."""
    return (review.submitted_at.timestamp() if review.submitted_at else 0, review.id)


def reconcile_reviews(github: ReviewGitHub, env: Mapping[str, str]) -> int:
    """Serialize externally, read live state, and replace the statuses on that exact head."""
    url = f'{env["GITHUB_SERVER_URL"]}/{env["GITHUB_REPOSITORY"]}/actions/runs/{env["GITHUB_RUN_ID"]}'
    if group := env.get('MERGE_GROUP_SHA'):
        for gate in REVIEW_CONTEXTS:
            github.status(group, gate, 'success', 'Admission-only review gate; merge-group evaluation skipped.', url)
        return 0
    expected_head = env['EXPECTED_HEAD_SHA']
    expected_repository = int(env['EXPECTED_HEAD_REPOSITORY_ID'])
    head = ''
    verdicts: dict[ReviewGate, Verdict]
    try:
        for _ in range(3):
            pull = github.pull_request()
            if pull.state != 'open' or pull.head.sha != expected_head or pull.head.repo.id != expected_repository:
                print('Observed PR head is no longer open/current; no new statuses written.')
                return 0
            head = pull.head.sha
            reviews = github.reviews()
            comments = github.review_comments()
            verdicts = review_verdicts(github, pull, reviews, comments)
            # A push or relabel during the API reads must not publish an old snapshot.
            if github.pull_request() == pull and github.reviews() == reviews and github.review_comments() == comments:
                break
        else:
            raise GateError('PR changed repeatedly while evaluating reviews; retry the gate.')
    except (GateError, ValidationError, OSError) as error:
        if not head:
            # Setup already left the independently verified target pending.
            # Never write to a caller-supplied SHA when no PR snapshot was read.
            print(f'Cannot read live PR: {type(error).__name__}: {error}')
            return 1
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


_SIZED_FILE_PAGES = TypeAdapter(list[list[_SizedFile]])
_SIZED_FILES = TypeAdapter(list[_SizedFile])
_REVIEW_PAGES = TypeAdapter(list[list[Review]])
_COMMENT_PAGES = TypeAdapter(list[list[ReviewComment]])
_JOB_PAGES = TypeAdapter(list[_Jobs])


class ReviewGhCli(GhCli):
    """Review metadata and commit statuses through the shared GitHub client."""

    def pull_request(self) -> ReviewPullRequest:
        """Read current PR metadata, never an event's stale labels/head."""
        return ReviewPullRequest.model_validate_json(self._api(self._pull))

    def size(self, pull: ReviewPullRequest) -> Size:
        """Use the same Git-attribute-aware calculation as PR Bots."""
        files = self.file_pages(_SIZED_FILE_PAGES, pull.changed_files, exact=True)
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
            input=_SIZED_FILES.dump_json(files).decode(),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise GateError(f'Size calculation failed: {result.stderr.strip()}')
        return _SizeResult.model_validate_json(result.stdout).size

    def reviews(self) -> list[Review]:
        """Read every review, including superseded and dismissed decisions."""
        pages = _REVIEW_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._pull}/reviews?per_page=100'))
        return [review for page in pages for review in page]

    def review_comments(self) -> list[ReviewComment]:
        """Read every inline comment; resolution does not make a finding clean."""
        pages = _COMMENT_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._pull}/comments?per_page=100'))
        return [comment for page in pages for comment in page]

    def permission(self, login: str) -> Permission:
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
        pages = _JOB_PAGES.validate_json(
            self._api('--paginate', '--slurp', f'{self._repo}/actions/runs/{run_id}/jobs?per_page=100')
        )
        jobs: dict[str, str | None] = {job.name: job.conclusion for page in pages for job in page.jobs}
        return jobs.get('douwebot') == 'success' and jobs.get('Publish douwebot review') == 'success'

    def status(self, head: str, gate: ReviewGate, state: CommitStatus, description: str, url: str) -> None:
        """Latest-per-context commit statuses replace a previous failing verdict."""
        payload: dict[str, str] = {
            'context': REVIEW_CONTEXTS[gate],
            'state': state,
            'description': description,
            'target_url': url,
        }
        self._api('--method', 'POST', '--input', '-', f'{self._repo}/statuses/{head}', input=json.dumps(payload))


def main() -> int:
    """Evaluate the verified PR or merge group pinned by the workflow."""
    env = os.environ
    return reconcile_reviews(ReviewGhCli(env['GITHUB_REPOSITORY'], int(env.get('PR_NUMBER') or '0')), env)


if __name__ == '__main__':
    sys.exit(main())
