"""Publish the review statuses and the `.github/` change gate from trusted default-branch code."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pr_gates import ChangedFile, Commit, GateError, GhCli, Label, Verdict
from publish_douwebot_review import (
    APPROVAL_BODY,
    DOUWEBOT_MARKER,
    EARLIER_REVIEW_ATTRIBUTIONS,
    REVIEW_BODIES,
    REVIEW_ENGINES,
)
from publish_pr_review_checks import MARKER as PR_REVIEW_CHECKS_MARKER
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

ReviewGate = Literal['douwebot', 'github-dir', 'dispositions']
Size = Literal['size: S', 'size: M', 'size: L', 'size: XL']
CommitStatus = Literal['pending', 'success', 'failure']
REVIEW_CONTEXTS: dict[ReviewGate, str] = {
    'douwebot': 'Gate: clean douwebot approval',
    'github-dir': 'Gate: .github changes by a maintainer',
    'dispositions': 'Gate: review findings dispositioned',
}
# Dependabot owns the action-pin bumps in `.github/workflows/` and holds no repository permission of its own.
# Exact logins, not a `*[bot]` glob: `pydanty[bot]` builds branches out of outside text.
GITHUB_DIR_BOTS = frozenset({'dependabot[bot]'})
# The douwebot prompt in `bots.yml` asks for this prefix on every optional finding.
OPTIONAL_FINDING_PREFIX = 'Optional:'
# Review bots by REST login. `finding_reviewer` tells the `github-actions[bot]` reviewers apart by receipt.
REVIEWER_LOGINS: dict[str, str] = {
    'chatgpt-codex-connector[bot]': 'Codex',
    'claude[bot]': 'Claude',
    'coderabbitai[bot]': 'CodeRabbit',
    'Copilot': 'Copilot',
    'cursor[bot]': 'Cursor',
    'devin-ai-integration[bot]': 'Devin',
    'github-advanced-security[bot]': 'CodeQL',
    'macroscopeapp[bot]': 'Macroscope',
    'pydanty[bot]': 'pydanty',
    'veria-ai[bot]': 'Veria',
}
# gh-aw reviews share `github-actions[bot]`; each run's receipt names its workflow.
GH_AW_REVIEWERS: dict[str, str] = {
    'pydantic/pydantic-ai/pydantic-ai-pr-review': 'CI Review',
    # Retired, but an open PR's head can still carry its comments.
    'pydantic/pydantic-ai/pydantic-ai-ui-security-review': 'UI security review',
}
# Resolving a thread settled a finding started before this time. GitHub records no resolution time.
# Delete this baseline once no open PR has a bot thread started before it.
RESOLVED_THREADS_STARTED_BEFORE = datetime(2026, 11, 27, 0, tzinfo=UTC)
# A reply's first line in this form answers its finding; the reply can open with an attribution line.
DISPOSITION = re.compile(
    r'^(?:Fixed in (?P<sha>[0-9a-f]{7,40})\b|Tracked in #(?P<number>[1-9][0-9]{0,8})\b|(?P<declined>Declined: \S))',
    re.MULTILINE,
)


class ReviewUser(BaseModel):
    """The GitHub identity attached to a review or review comment."""

    id: int
    login: str
    type: str


class _Repository(BaseModel):
    id: int = Field(gt=0)


class _Branch(Commit):
    """A pull request's head or base: its commit and the repository that holds it."""

    repo: _Repository


class ReviewPullRequest(BaseModel):
    """The live head, base, and labels that determine the review bar."""

    state: Literal['open', 'closed']
    number: int
    html_url: str
    user: ReviewUser
    head: _Branch
    base: _Branch
    labels: list[Label]
    changed_files: int


class Review(BaseModel):
    """A submitted review; GitHub can move its `commit_id` after a push, but not its body's marker."""

    id: int
    user: ReviewUser
    state: Literal['APPROVED', 'CHANGES_REQUESTED', 'COMMENTED', 'DISMISSED', 'PENDING']
    commit_id: str
    body: str
    submitted_at: datetime | None


class ReviewComment(BaseModel):
    """The parent review's marker identifies the head, else `original_commit_id`; `commit_id` can move on later pushes."""

    id: int
    pull_request_review_id: int
    # REST omits it on a thread's first comment and points every reply at that comment.
    in_reply_to_id: int | None = None
    user: ReviewUser
    original_commit_id: str
    created_at: datetime
    body: str


class ReviewRun(BaseModel):
    """Trusted workflow metadata for a marked douwebot publication."""

    name: str
    path: str
    event: str
    head_sha: str
    conclusion: str | None


class ReviewJob(BaseModel):
    """A job's terminal outcome in the originating run."""

    name: str
    conclusion: str | None


class _Jobs(BaseModel):
    jobs: list[ReviewJob]


class _CommentId(BaseModel):
    databaseId: int


class _ThreadComments(BaseModel):
    nodes: list[_CommentId]


class _Thread(BaseModel):
    isResolved: bool
    comments: _ThreadComments


class _Threads(BaseModel):
    nodes: list[_Thread]


class _ThreadsPull(BaseModel):
    reviewThreads: _Threads


class _ThreadsRepository(BaseModel):
    pullRequest: _ThreadsPull


class _ThreadsData(BaseModel):
    repository: _ThreadsRepository


class _ThreadsPage(BaseModel):
    data: _ThreadsData


class _SizedFile(ChangedFile):
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)


class _SizeResult(BaseModel):
    size: Size
    score: int


class ReviewGitHub(Protocol):
    """The metadata needed by the review gates."""

    def pull_request(self) -> ReviewPullRequest: ...
    def changed_files(self) -> list[ChangedFile]: ...
    def permission(self, login: str) -> str: ...
    def size(self, pull: ReviewPullRequest) -> Size: ...
    def reviews(self) -> list[Review]: ...
    def review_comments(self) -> list[ReviewComment]: ...
    def resolved_comment_ids(self) -> set[int]: ...
    def verified_review_run(self, run_id: int, head: str) -> bool: ...
    def own_commits(self, base: str, head: str, shas: set[str]) -> set[str]: ...
    def issue_created_at(self, number: int) -> datetime | None: ...
    def status(self, head: str, gate: ReviewGate, state: CommitStatus, description: str, url: str) -> None: ...


def review_verdicts(
    github: ReviewGitHub,
    pull: ReviewPullRequest,
    reviews: list[Review],
    comments: list[ReviewComment],
    resolved: set[int],
) -> dict[ReviewGate, Verdict]:
    """Apply the `.github/` change gate and the size-dependent review bar to a live metadata snapshot."""
    verdicts: dict[ReviewGate, Verdict] = {'douwebot': Verdict(), 'github-dir': github_dir_change(github, pull)}
    size = github.size(pull)
    size_labels = {label.name for label in pull.labels if label.name.startswith('size:')}
    if not size_labels:
        # The status description keeps only 140 characters, so the next step must fit in them.
        next_step = (
            'PR Bots labels a PR whose author has write access, or with `trusted:ci-run`.'
            if pull.head.repo.id == pull.base.repo.id
            else 'PR Bots skips fork PRs. A maintainer recreates the change as a same-repository PR.'
        )
        verdicts['douwebot'].failures.append(f'No size label (computed `{size}`): {next_step}')
        return verdicts
    if size_labels != {size}:
        verdicts['douwebot'].failures.append(
            f'Expected exactly the recomputed `{size}` label; found {sorted(size_labels)!r}.'
        )
        return verdicts
    if size == 'size: S':
        verdicts['douwebot'].summary.append(
            'Size S: automated review enforcement is skipped; the manual merge bar still applies.'
        )
        return verdicts

    verdicts['douwebot'] = douwebot_approval(github, pull, reviews, comments, resolved)
    return verdicts


def github_dir_change(github: ReviewGitHub, pull: ReviewPullRequest) -> Verdict:
    """Files under `.github/` run with the repository's credentials, so only a maintainer may change them.

    A pull request runs its own copy of `protect-github-dir.yml`, which a fork can edit; this verdict comes from
    default-branch code. The author's resolved permission is the only trust signal: an app pushes branches into this
    repository without holding write access, and `author_association` misreports private org members.
    """
    protected = sorted(
        {
            path
            for file in github.changed_files()
            for path in (file.filename, file.previous_filename)
            if path is not None and path.startswith('.github/')
        }
    )
    if not protected:
        return Verdict(summary=['No file under `.github/` changed.'])
    author = pull.user
    if author.login in GITHUB_DIR_BOTS or (
        author.type == 'User' and github.permission(author.login) in {'write', 'admin'}
    ):
        return Verdict(summary=[f'@{author.login} may change `.github/`.'])
    return Verdict(failures=[f'Only a maintainer may change `.github/`; this PR changes {", ".join(protected)}.'])


def douwebot_approval(
    github: ReviewGitHub,
    pull: ReviewPullRequest,
    reviews: list[Review],
    comments: list[ReviewComment],
    resolved: set[int],
) -> Verdict:
    """Verify provenance and associate findings with their immutable review head.

    GitHub can move a review's and a comment's `commit_id` after a push. Use the parent's
    marker head, or `original_commit_id` when the parent is unavailable or unmarked.

    An optional inline finding on a verified approval stops blocking once its review thread
    is resolved, given by the thread root ids in `resolved`. Every other finding blocks.
    """
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
            if (engine := marker[3]) in ('claude', 'opencode', 'opencode-zai', 'opencode-deepinfra'):
                attributions = (REVIEW_ENGINES[engine], *EARLIER_REVIEW_ATTRIBUTIONS.get(engine, ()))
                suffix = next((f'\n\n{text}' for text in attributions if body.endswith(f'\n\n{text}')), None)
                if suffix:
                    body = body[: -len(suffix)]
                else:
                    verdict.failures.append('Douwebot review engine attribution is missing.')
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
        marker = DOUWEBOT_MARKER.search(parent.body) if parent else None
        reviewed_head = marker[2] if marker else comment.original_commit_id
        if reviewed_head != pull.head.sha:
            continue
        if comment.pull_request_review_id in marked_ids:
            if parent in own_reviews and settled_optional_finding(comment, parent, resolved):
                continue
            verdict.failures.append('Douwebot has blocking or unresolved optional inline findings on this head.')
        else:
            if other_review_receipt(comment, parent):
                continue
            verdict.failures.append(
                'Cannot attribute an unmarked bot comment on this head; obtain a fresh review/head.'
            )
        break
    return verdict


def other_review_receipt(comment: ReviewComment, parent: Review | None) -> bool:
    """gh-aw and PR Review Checks give their reviews separate receipts despite sharing the account."""
    if parent and PR_REVIEW_CHECKS_MARKER.search(parent.body):
        return True
    return gh_aw_reviewer(comment, parent) is not None


def gh_aw_reviewer(comment: ReviewComment, parent: Review | None) -> str | None:
    """Name the gh-aw workflow whose receipt the comment or its parent review carries."""
    receipt = f'{parent.body if parent else ""}\n{comment.body}'
    return next(
        (
            reviewer
            for workflow, reviewer in GH_AW_REVIEWERS.items()
            if f'<!-- gh-aw-workflow-call-id: {workflow} -->' in receipt
            or f'[gh-aw-workflow-call-id]: # "{workflow.replace("/", "%2F")}"' in receipt
        ),
        None,
    )


def finding_reviewer(comment: ReviewComment, parent: Review | None) -> str | None:
    """Name the registered reviewer that started a thread, or `None` for a bot nobody registered."""
    if comment.user.login != 'github-actions[bot]':
        return REVIEWER_LOGINS.get(comment.user.login)
    if parent and DOUWEBOT_MARKER.search(parent.body):
        return 'douwebot'
    if parent and PR_REVIEW_CHECKS_MARKER.search(parent.body):
        return 'PR Review Checks'
    return gh_aw_reviewer(comment, parent)


def _rejection(
    login: str,
    line: re.Match[str],
    finding: ReviewComment,
    reviewer: str | None,
    pull: ReviewPullRequest,
    writers: set[str],
    own: set[str],
    opened: dict[str, datetime],
) -> str | None:
    """Why a disposition reply does not answer its finding, or `None` when it does."""
    writer = login in writers
    if reviewer is None and not writer:
        return f'only write access answers a thread no registered reviewer started, and `{login}` has none'
    if not writer and login != pull.user.login:
        return f'`{login}` is neither the PR author nor a user with write access'
    if sha := line['sha']:
        return None if sha in own else f'`{sha}` is not a commit of this PR'
    if number := line['number']:
        if (at := opened.get(number)) is None:
            return f'#{number} is not another issue or PR'
        return None if writer or at >= finding.created_at else f'#{number} was opened before the finding'
    optional = reviewer == 'douwebot' and finding.body.startswith(OPTIONAL_FINDING_PREFIX)
    return None if writer or optional else '`Declined:` needs write access'


def finding_dispositions(
    github: ReviewGitHub,
    pull: ReviewPullRequest,
    reviews: list[Review],
    comments: list[ReviewComment],
    resolved: set[int],
) -> Verdict:
    """Every inline thread a review bot started, on any head, needs a person's recorded answer.

    A reply counts from the PR author or from write access. A reply line `Fixed in <sha>` names a commit of this PR,
    and `Tracked in #N` another issue or PR, opened at or after the finding unless the replier has write access.
    `Declined: <reason>` counts from write access, and from the author on a douwebot `Optional:` finding.
    On a thread no registered reviewer started, only a reply from write access counts.
    A resolved thread started before `RESOLVED_THREADS_STARTED_BEFORE` is settled, whatever bot started it.
    """
    parents = {review.id: review for review in reviews}
    findings = [comment for comment in comments if comment.in_reply_to_id is None and comment.user.type == 'Bot']
    answers: dict[int, list[tuple[str, re.Match[str]]]] = {}
    finding_ids = {finding.id for finding in findings}
    for comment in comments:
        if (
            comment.in_reply_to_id in finding_ids
            and comment.user.type == 'User'
            and (line := DISPOSITION.search(comment.body))
        ):
            answers.setdefault(comment.in_reply_to_id, []).append((comment.user.login, line))
    lines = [line for replies in answers.values() for _, line in replies]
    own = github.own_commits(pull.base.sha, pull.head.sha, {line['sha'] for line in lines if line['sha']})
    tracked = {number for line in lines if (number := line['number']) and int(number) != pull.number}
    opened = {number: at for number in tracked if (at := github.issue_created_at(int(number)))}
    repliers = {login for replies in answers.values() for login, _ in replies}
    writers = {login for login in repliers if github.permission(login) in {'write', 'admin'}}
    unanswered: list[str] = []
    unregistered: list[str] = []
    for finding in findings:
        if finding.created_at < RESOLVED_THREADS_STARTED_BEFORE and finding.id in resolved:
            continue
        url = f'{pull.html_url}#discussion_r{finding.id}'
        reviewer = finding_reviewer(finding, parents.get(finding.pull_request_review_id))
        name = f'the {reviewer} finding' if reviewer else f'the `{finding.user.login}` thread'
        if reviewer is None:
            unregistered.append(f'{url}: no registered reviewer started this `{finding.user.login}` thread.')
        rejections = [
            _rejection(login, line, finding, reviewer, pull, writers, own, opened)
            for login, line in answers.get(finding.id, [])
        ]
        if None in rejections:
            continue
        reasons = '; '.join(reason for reason in rejections if reason)
        unanswered.append(
            f'{url}: {name} has no accepted disposition reply: {reasons}.'
            if reasons
            else f'{url}: {name} has no disposition reply.'
        )
    summary = [f'{len(findings)} review bot findings.']
    if unregistered:
        summary += [
            f'{len(unregistered)} review bot threads have no registered reviewer, so only a reply from write access '
            'answers them. Register the bot in `REVIEWER_LOGINS`, or see #14014 for a `github-actions[bot]` thread '
            'without a receipt.',
            *unregistered,
        ]
    failures: list[str] = []
    if unanswered:
        failures += [
            f'{len(unanswered)} of {len(findings)} review bot findings need a reply: '
            '`Fixed in <sha>`, `Tracked in #N` or `Declined: <reason>`.',
            *unanswered,
        ]
    return Verdict(summary=summary, failures=failures)


def settled_optional_finding(comment: ReviewComment, parent: Review | None, resolved: set[int]) -> bool:
    """An approval's optional finding stops blocking once its thread is resolved."""
    return (
        parent is not None
        and parent.state == 'APPROVED'
        and comment.body.startswith(OPTIONAL_FINDING_PREFIX)
        and comment.id in resolved
    )


def review_order(review: Review) -> tuple[float, int]:
    """Order submitted decisions chronologically, including reviews created as drafts."""
    return (review.submitted_at.timestamp() if review.submitted_at else 0, review.id)


def _disposition_verdict(
    github: ReviewGitHub,
    pull: ReviewPullRequest,
    reviews: list[Review],
    comments: list[ReviewComment],
    resolved: set[int],
) -> tuple[Verdict, bool]:
    """A failed disposition lookup fails only its own status and the job; the douwebot verdict stands."""
    try:
        return finding_dispositions(github, pull, reviews, comments, resolved), False
    except (GateError, ValidationError, OSError) as error:
        return Verdict(failures=[f'{type(error).__name__}: {error}']), True


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
    failed = False
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
            resolved = github.resolved_comment_ids()
            verdicts = review_verdicts(github, pull, reviews, comments, resolved)
            verdicts['dispositions'], failed = _disposition_verdict(github, pull, reviews, comments, resolved)
            # A push, relabel or thread resolution during the API reads must not publish an old snapshot.
            if (
                github.pull_request() == pull
                and github.reviews() == reviews
                and github.review_comments() == comments
                and github.resolved_comment_ids() == resolved
            ):
                break
        else:
            raise GateError('PR changed repeatedly while evaluating reviews; retry the gate.')
    except (GateError, ValidationError, OSError) as error:
        failed = True
        if not head:
            # Setup already left the independently verified target pending.
            # Never write to a caller-supplied SHA when no PR snapshot was read.
            print(f'Cannot read live PR: {type(error).__name__}: {error}')
            return 1
        verdicts = {gate: Verdict(failures=[f'{type(error).__name__}: {error}']) for gate in REVIEW_CONTEXTS}
    for gate, verdict in verdicts.items():
        state = 'failure' if verdict.failures else 'success'
        description = verdict.failures[0] if verdict.failures else 'Review requirement satisfied.'
        github.status(head, gate, state, description[:140], url)
        report = '\n\n'.join([f'## {REVIEW_CONTEXTS[gate]}: {state}', *verdict.summary, *verdict.failures])
        print(report)
        if summary_path := env.get('GITHUB_STEP_SUMMARY'):
            with open(summary_path, 'a', encoding='utf-8') as summary:
                summary.write(report + '\n')
    return int(failed)


class _Permission(BaseModel):
    permission: str


class _CommitComparison(BaseModel):
    status: str


class _Issue(BaseModel):
    created_at: datetime


_PERMISSION = TypeAdapter(_Permission)
_ISSUE = TypeAdapter(_Issue)
_COMMIT_PAGES = TypeAdapter(list[list[Commit]])
_SIZED_FILE_PAGES = TypeAdapter(list[list[_SizedFile]])
_SIZED_FILES = TypeAdapter(list[_SizedFile])
_REVIEW_PAGES = TypeAdapter(list[list[Review]])
_COMMENT_PAGES = TypeAdapter(list[list[ReviewComment]])
_JOB_PAGES = TypeAdapter(list[_Jobs])
_THREAD_PAGES = TypeAdapter(list[_ThreadsPage])
# A douwebot finding always starts its own thread, so the first comment identifies it.
_THREADS_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $endCursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100, after: $endCursor) {
        nodes { isResolved comments(first: 1) { nodes { databaseId } } }
        pageInfo { hasNextPage endCursor }
      }
    }
  }
}
"""


class ReviewGhCli(GhCli):
    """Review metadata and commit statuses through the shared GitHub client."""

    def __init__(self, repo: str, number: int) -> None:
        super().__init__(repo, number)
        self._owner, self._name = repo.split('/')
        self._number = number

    def pull_request(self) -> ReviewPullRequest:
        """Read current PR metadata, never an event's stale labels/head."""
        return ReviewPullRequest.model_validate_json(self._api(self._pull))

    def permission(self, login: str) -> str:
        """Fail closed: an account the endpoint cannot resolve has no write access."""
        try:
            permission = _PERMISSION.validate_json(self._api(f'{self._repo}/collaborators/{login}/permission'))
        except GateError as error:
            print(f'No repository permission for @{login}: {error}')
            return 'none'
        return permission.permission

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
        """Read every inline comment, resolved or not."""
        pages = _COMMENT_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._pull}/comments?per_page=100'))
        return [comment for page in pages for comment in page]

    def resolved_comment_ids(self) -> set[int]:
        """Read the root comment id of every resolved review thread; REST does not expose resolution."""
        pages = _THREAD_PAGES.validate_json(
            self._api(
                '--paginate',
                '--slurp',
                '-f',
                f'owner={self._owner}',
                '-f',
                f'name={self._name}',
                '-F',
                f'number={self._number}',
                '-f',
                f'query={_THREADS_QUERY}',
                'graphql',
            )
        )
        return {
            comment.databaseId
            for page in pages
            for thread in page.data.repository.pullRequest.reviewThreads.nodes
            if thread.isResolved
            for comment in thread.comments.nodes
        }

    def verified_review_run(self, run_id: int, head: str) -> bool:
        """Only the trusted PR Bots publisher can supply a qualifying marker."""
        run = ReviewRun.model_validate_json(self._api(f'{self._repo}/actions/runs/{run_id}'))
        if not (
            run.path == '.github/workflows/bots.yml'
            # Receipts on heads reviewed before `bots.yml` moved to `pull_request` come from `main`'s copy
            # on `pull_request_target`. `main` no longer has that trigger, so no new run can carry it.
            and run.event in {'pull_request', 'pull_request_target'}
            and run.name == 'PR Bots'
            and run.head_sha == head
            and run.conclusion == 'success'
        ):
            return False
        pages = _JOB_PAGES.validate_json(
            self._api('--paginate', '--slurp', f'{self._repo}/actions/runs/{run_id}/jobs?per_page=100')
        )
        jobs: dict[str, str | None] = {job.name: job.conclusion for page in pages for job in page.jobs}
        return jobs.get('douwebot') == 'success' and jobs.get('Publish douwebot review') == 'success'

    def own_commits(self, base: str, head: str, shas: set[str]) -> set[str]:
        """Return the given SHAs, full or abbreviated, that name a commit this PR adds to its base."""
        if not shas:
            return set()
        pages = _COMMIT_PAGES.validate_json(self._api('--paginate', '--slurp', f'{self._pull}/commits?per_page=100'))
        listed = [commit.sha for page in pages for commit in page]
        own = {sha for sha in shas if any(commit.startswith(sha) for commit in listed)}
        # The endpoint lists at most 250 commits; past that, compare each unmatched SHA with both ends.
        if len(listed) < 250:
            return own
        for sha in shas - own:
            try:
                # The compare endpoint finds a fork's commit only by its full SHA.
                full = Commit.model_validate_json(self._api(f'{self._repo}/commits/{sha}')).sha
                after_base = self._api(f'{self._repo}/compare/{base}...{full}?per_page=1')
                in_head = self._api(f'{self._repo}/compare/{full}...{head}?per_page=1')
            except GateError as error:
                if 'HTTP 404' in str(error) or 'HTTP 422' in str(error):
                    continue
                raise
            if _CommitComparison.model_validate_json(after_base).status in {'ahead', 'diverged'} and (
                _CommitComparison.model_validate_json(in_head).status in {'ahead', 'identical'}
            ):
                own.add(sha)
        return own

    def issue_created_at(self, number: int) -> datetime | None:
        """The issues endpoint also serves pull requests; `None` when the number names neither."""
        try:
            issue = _ISSUE.validate_json(self._api(f'{self._repo}/issues/{number}'))
        except GateError as error:
            if 'HTTP 404' in str(error) or 'HTTP 410' in str(error):
                return None
            raise
        return issue.created_at

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
