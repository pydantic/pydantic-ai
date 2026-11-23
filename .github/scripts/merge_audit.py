"""Audit `main` after the fact, and report what the audit finds to one rolling issue.

The audit compares the live `main` ruleset with `.github/rulesets/main.json`, checks each pull request merged in the
window for a green `check`, the one context the ruleset requires, and for green review contexts, which it does not
require, and lists CodeQL alerts dismissed in the window whose comment links no issue.

`python3 .github/scripts/merge_audit.py [--since ISO] [--until ISO] [--publish]` audits the 24 hours before `--until`,
now by default, and prints the report. `--publish` also adds it to the rolling issue when it finds anything.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import _commands
from pydantic import BaseModel, TypeAdapter

T = TypeVar('T')

REPO = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
RULESET = Path(__file__).parents[1] / 'rulesets' / 'main.json'
# The fields of a ruleset that say what it enforces. The rest identify it or change on every edit, and
# `bypass_actors` is visible only to repository admins.
RULESET_FIELDS = ('name', 'target', 'enforcement', 'conditions', 'rules')
REQUIRED_CHECK = 'check'
MARKER = '<!-- merge-audit -->'
ISSUE_TITLE = 'Merge audit findings'
ISSUE_LINK = re.compile(r'github\.com/[\w.-]+/[\w.-]+/issues/\d+|(?<![\w/])#\d+\b')
_MERGED = """
query($q: String!, $endCursor: String) {
  search(type: ISSUE, query: $q, first: 50, after: $endCursor) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest { number title mergedAt headRefOid mergeCommit { oid } } }
  }
}
"""


class _Commit(BaseModel):
    oid: str


class MergedPull(BaseModel):
    """A pull request merged into `main`, as the search query returns it."""

    number: int
    title: str
    mergedAt: datetime
    headRefOid: str
    mergeCommit: _Commit | None


class _CheckRun(BaseModel):
    conclusion: str | None
    completed_at: datetime | None


class _Status(BaseModel):
    context: str
    state: str


class _Alert(BaseModel):
    html_url: str
    dismissed_at: datetime | None
    dismissed_reason: str | None
    dismissed_comment: str | None


class _Issue(BaseModel):
    number: int
    body: str


def _gh(*args: str) -> str:
    return _commands.run(['gh', *args], check=True).stdout


def _rows(adapter: TypeAdapter[T], output: str) -> list[T]:
    """Validate one JSON document per line, as `gh --jq` prints the items of a paginated listing."""
    return [adapter.validate_json(line) for line in output.splitlines() if line]


def ruleset_drift(committed: Mapping[str, object], live: Mapping[str, object]) -> list[str]:
    """The ruleset fields whose live value differs from the committed one."""
    return [field for field in RULESET_FIELDS if committed.get(field) != live.get(field)]


def live_ruleset() -> dict[str, object]:
    """The enforcing fields of the live repository ruleset named `main`."""
    ids = _gh('api', f'repos/{REPO}/rulesets?includes_parents=false', '--jq', '.[] | select(.name == "main") | .id')
    [ruleset_id] = ids.split()
    live = TypeAdapter(dict[str, object]).validate_json(_gh('api', f'repos/{REPO}/rulesets/{ruleset_id}'))
    return {field: live[field] for field in RULESET_FIELDS if field in live}


def merged_pulls(since: datetime, until: datetime) -> list[MergedPull]:
    """The pull requests merged into `main` in the window."""
    window = f'{since:%Y-%m-%dT%H:%M:%SZ}..{until:%Y-%m-%dT%H:%M:%SZ}'
    query = f'repo:{REPO} is:pr is:merged base:main merged:{window}'
    output = _gh(
        'api', 'graphql', '--paginate', '-f', f'query={_MERGED}', '-f', f'q={query}', '--jq', '.data.search.nodes[]'
    )
    return _rows(TypeAdapter(MergedPull), output)


def _check_passed_on(sha: str, by: datetime) -> bool:
    output = _gh(
        'api', f'repos/{REPO}/commits/{sha}/check-runs?check_name={REQUIRED_CHECK}&filter=all&per_page=100',
        '--paginate', '--jq', '.check_runs[] | {conclusion, completed_at}',
    )  # fmt: skip
    runs = _rows(TypeAdapter(_CheckRun), output)
    return any(run.conclusion == 'success' and run.completed_at is not None and run.completed_at <= by for run in runs)


def check_passed(pull: MergedPull) -> bool:
    """Whether `check` succeeded by the merge, on the head or, for a merge-queue merge, on the commit the queue tested.

    The queue's commit becomes the merge commit. Looked up by name: a head can carry more check suites than a listing
    of its suites returns, and a run after the merge, such as the push run on `main`, does not count.
    """
    commits = [pull.headRefOid, *([pull.mergeCommit.oid] if pull.mergeCommit else [])]
    return any(_check_passed_on(sha, pull.mergedAt) for sha in commits)


def failed_review_contexts(pull: MergedPull) -> list[str]:
    """The head's `PR Review Checks` and `Gate:` statuses that were not green, as `context: state`."""
    output = _gh(
        'api', f'repos/{REPO}/commits/{pull.headRefOid}/status?per_page=100', '--jq', '.statuses[] | {context, state}'
    )
    return sorted(
        f'{status.context}: {status.state}'
        for status in _rows(TypeAdapter(_Status), output)
        if (status.context == 'PR Review Checks' or status.context.startswith('Gate: ')) and status.state != 'success'
    )


def unlinked_dismissals(since: datetime, until: datetime) -> list[_Alert]:
    """The CodeQL alerts dismissed in the window whose dismissal comment links no issue."""
    output = _gh(
        'api', f'repos/{REPO}/code-scanning/alerts?state=dismissed&tool_name=CodeQL&per_page=100', '--paginate',
        '--jq', '.[] | {html_url, dismissed_at, dismissed_reason, dismissed_comment}',
    )  # fmt: skip
    return [
        alert
        for alert in _rows(TypeAdapter(_Alert), output)
        if alert.dismissed_at is not None
        and since <= alert.dismissed_at < until
        and not ISSUE_LINK.search(alert.dismissed_comment or '')
    ]


def report(since: datetime, until: datetime) -> str | None:
    """The audit's report for the window, or `None` when it finds nothing."""
    drift = ruleset_drift(TypeAdapter(dict[str, object]).validate_json(RULESET.read_bytes()), live_ruleset())
    pulls = merged_pulls(since, until)
    red = [pull for pull in pulls if not check_passed(pull)]
    reviews = [(pull, contexts) for pull in pulls if (contexts := failed_review_contexts(pull))]
    alerts = unlinked_dismissals(since, until)
    if not (drift or red or reviews or alerts):
        return None
    lines = [MARKER, f'## Merge audit, {since:%Y-%m-%d %H:%M} to {until:%Y-%m-%d %H:%M} UTC', '']
    if drift:
        fields = ', '.join(f'`{field}`' for field in drift)
        lines += [
            '### `main` ruleset',
            '',
            f'The live ruleset differs from `.github/rulesets/main.json` in {fields}.',
            '',
        ]
    if red:
        lines += [f'### Merged before `{REQUIRED_CHECK}` passed', '']
        lines += [f'- https://github.com/{REPO}/pull/{pull.number} {pull.title}' for pull in red] + ['']
    if reviews:
        lines += ['### Merged with a review context not green', '', 'The ruleset does not require these contexts.', '']
        lines += [
            f'- https://github.com/{REPO}/pull/{pull.number}: {", ".join(contexts)}' for pull, contexts in reviews
        ]
        lines += ['']
    if alerts:
        lines += ['### CodeQL alerts dismissed without a linked issue', '']
        lines += [f'- {alert.html_url} ({alert.dismissed_reason})' for alert in alerts] + ['']
    return '\n'.join(lines)


def publish(body: str) -> None:
    """Add `body` to the open rolling issue, or open it with `body`."""
    output = _gh(
        'issue', 'list', '--repo', REPO, '--state', 'open', '--author', 'app/github-actions', '--limit', '100',
        '--json', 'number,body', '--jq', '.[]',
    )  # fmt: skip
    issues = [issue for issue in _rows(TypeAdapter(_Issue), output) if issue.body.startswith(MARKER)]
    if issues:
        _gh('issue', 'comment', str(issues[0].number), '--repo', REPO, '--body', body)
    else:
        _gh('issue', 'create', '--repo', REPO, '--title', ISSUE_TITLE, '--body', body)


def _utc(value: str) -> datetime:
    """Parse an ISO 8601 time with its UTC offset, such as `2026-11-23T06:00:00Z`, as UTC."""
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise argparse.ArgumentTypeError(f'{value} has no UTC offset')
    return moment.astimezone(UTC)


def main(argv: list[str] | None = None) -> None:
    """Print the audit's report, and with `--publish` add it to the rolling issue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--until', type=_utc, default=datetime.now(UTC))
    parser.add_argument('--since', type=_utc)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args(argv)
    until: datetime = args.until
    since: datetime = args.since or until - timedelta(hours=24)
    body = report(since, until)
    print(body or 'The merge audit found nothing.')
    if body and args.publish:
        publish(body)


if __name__ == '__main__':
    main()
