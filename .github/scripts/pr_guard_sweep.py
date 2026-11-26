"""Run PR Guard on every open pull request that an opened, reopened, ready-for-review or edited event changed.

A fork pull request's own runs hold a read-only token, so PR Guard runs on a schedule from the
default branch instead. Each sweep reads the events since the last successful sweep started, and runs
`pr_guard.sh` for each pull request they touched: as the author when the author acted, otherwise as the
user who acted last. A failed sweep leaves the next one to read the same events again.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter

# GitHub's search index and a sweep's own run time both lag; reading a little further back costs only a repeat
# check, which closes nothing it would not have closed the first time.
OVERLAP = timedelta(minutes=5)
FIRST_SWEEP = timedelta(hours=1)
# A sweep anchors on the last successful sweep, so one pull request that keeps failing widens every later
# sweep's window. The cap bounds that repeated work. When no sweep succeeds within the cap after an event,
# the guard never judges that event, and its pull request is left to a maintainer.
LONGEST_LOOKBACK = timedelta(hours=3)
GUARD = Path(__file__).with_name('pr_guard.sh')

_QUERY = """
query($q: String!, $since: DateTime!, $endCursor: String) {
  search(query: $q, type: ISSUE, first: 50, after: $endCursor) {
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number
        state
        isDraft
        createdAt
        authorAssociation
        baseRefName
        author { __typename login }
        userContentEdits(last: 20) { nodes { editedAt editor { __typename login } } }
        timelineItems(
          since: $since
          last: 20
          itemTypes: [REOPENED_EVENT, READY_FOR_REVIEW_EVENT, BASE_REF_CHANGED_EVENT, RENAMED_TITLE_EVENT]
        ) {
          nodes {
            ... on ReopenedEvent { createdAt actor { __typename login } }
            ... on ReadyForReviewEvent { createdAt actor { __typename login } }
            ... on BaseRefChangedEvent { createdAt actor { __typename login } }
            ... on RenamedTitleEvent { createdAt actor { __typename login } }
          }
        }
      }
    }
  }
}
"""


class Actor(BaseModel):
    """GraphQL drops the `[bot]` suffix that REST logins, and so `pr_guard.sh`, carry."""

    typename: str = Field(alias='__typename')
    login: str

    @property
    def rest_login(self) -> str:
        return f'{self.login}[bot]' if self.typename == 'Bot' else self.login


class TimelineEvent(BaseModel):
    """When something happened to a pull request, and who did it."""

    createdAt: datetime
    actor: Actor | None


class _Timeline(BaseModel):
    nodes: list[TimelineEvent]


class BodyEdit(BaseModel):
    """One edit of the description; the oldest is the original description."""

    editedAt: datetime
    editor: Actor | None


class _BodyEdits(BaseModel):
    nodes: list[BodyEdit]


class PullRequest(BaseModel):
    """What `pr_guard.sh` reads about one pull request, and the events it reacts to."""

    number: int
    state: Literal['OPEN', 'CLOSED', 'MERGED']
    isDraft: bool
    createdAt: datetime
    authorAssociation: str
    baseRefName: str
    author: Actor | None
    userContentEdits: _BodyEdits
    timelineItems: _Timeline


class _PageInfo(BaseModel):
    hasNextPage: bool
    endCursor: str | None


class _Search(BaseModel):
    pageInfo: _PageInfo
    nodes: list[PullRequest]


class _SearchData(BaseModel):
    search: _Search


class _SearchPage(BaseModel):
    data: _SearchData


class _Repository(BaseModel):
    default_branch: str


class _Run(BaseModel):
    run_started_at: datetime


class _Runs(BaseModel):
    workflow_runs: list[_Run]


_SEARCH_PAGES = TypeAdapter(list[_SearchPage])


def _gh(*args: str) -> str:
    return subprocess.run(['gh', 'api', *args], check=True, capture_output=True, text=True).stdout


def sweep_start(last_success: datetime | None, now: datetime) -> datetime:
    """Read from just before the last successful sweep started, but never further back than `LONGEST_LOOKBACK`."""
    if last_success is None:
        return now - FIRST_SWEEP
    return max(last_success - OVERLAP, now - LONGEST_LOOKBACK)


def deciding_event(pull: PullRequest, since: datetime) -> TimelineEvent | None:
    """The event since `since` that `pr_guard.sh` should judge, or None when nothing it reacts to happened.

    The author's latest event decides when there is one: a later event by someone trusted, such as a bot
    rewriting the description, does not exempt the author's own. Otherwise the latest event decides, and
    the guard's trusted-actor bypass judges its actor.
    """
    events = [
        *pull.timelineItems.nodes,
        TimelineEvent(createdAt=pull.createdAt, actor=pull.author),
        *(TimelineEvent(createdAt=edit.editedAt, actor=edit.editor) for edit in pull.userContentEdits.nodes),
    ]
    recent = [event for event in events if event.createdAt >= since]
    by_author = [event for event in recent if pull.author is None or event.actor == pull.author]
    return max(by_author or recent, key=lambda event: event.createdAt, default=None)


def guard_env(pull: PullRequest, event: TimelineEvent) -> dict[str, str]:
    """The variables `pr_guard.sh` reads, as that event's `pull_request` payload would set them."""
    # GitHub reports a deleted account as no actor; treat its pull request and events as the author's own.
    author = pull.author.rest_login if pull.author is not None else 'ghost'
    return {
        'GITHUB_EVENT_SENDER_LOGIN': event.actor.rest_login if event.actor is not None else author,
        'GITHUB_EVENT_PULL_REQUEST_NUMBER': str(pull.number),
        'GITHUB_EVENT_PULL_REQUEST_USER_LOGIN': author,
        'GITHUB_EVENT_PULL_REQUEST_AUTHOR_ASSOCIATION': pull.authorAssociation,
        'GITHUB_EVENT_PULL_REQUEST_CREATED_AT': pull.createdAt.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'GITHUB_EVENT_PULL_REQUEST_DRAFT': str(pull.isDraft).lower(),
        'GITHUB_EVENT_PULL_REQUEST_STATE': pull.state.lower(),
        'GITHUB_EVENT_PULL_REQUEST_BASE_REF': pull.baseRefName,
    }


def main() -> int:
    """Check every pull request touched since the last sweep; fail when any check could not decide."""
    repo = os.environ['REPO']
    now = datetime.now(UTC)
    # A `schedule` event's payload names no repository, so ask for its default branch.
    default_branch = _Repository.model_validate_json(_gh(f'repos/{repo}')).default_branch
    # A dispatch from another branch runs that branch's scripts, so only a default-branch sweep anchors the window.
    runs = _Runs.model_validate_json(
        _gh(f'repos/{repo}/actions/workflows/pr-guard.yml/runs?status=success&branch={default_branch}&per_page=1')
    )
    since = sweep_start(runs.workflow_runs[0].run_started_at if runs.workflow_runs else None, now)
    stamp = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f'Reading pull request events since {stamp}.')
    pages = _SEARCH_PAGES.validate_json(
        _gh(
            'graphql',
            '--paginate',
            '--slurp',
            '-f',
            f'q=repo:{repo} is:pr is:open updated:>={stamp}',
            '-f',
            f'since={stamp}',
            '-f',
            f'query={_QUERY}',
        )
    )
    failed: list[int] = []
    for pull in (pull for page in pages for pull in page.data.search.nodes):
        if (event := deciding_event(pull, since)) is None:
            continue
        print(f'::group::PR #{pull.number}', flush=True)
        result = subprocess.run(
            ['bash', str(GUARD)],
            env={**os.environ, 'DEFAULT_BRANCH': default_branch, **guard_env(pull, event)},
            check=False,
        )
        print('::endgroup::', flush=True)
        if result.returncode:
            failed.append(pull.number)
    if failed:
        print(f'::error::PR Guard could not check {", ".join(f"#{number}" for number in failed)}.')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
