#!/usr/bin/env python3
"""Notify maintainers when open issues and PRs have no GitHub activity for three days.

The state machine is deterministic: GitHub activity, current assignees,
collaborator permissions, and prior hidden comment markers are the only inputs.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import functools
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import NotRequired, TypedDict

_API = 'https://api.github.com'
_SLA = dt.timedelta(days=3)
_MAX_CANDIDATES = 20
_MAX_WORKERS = 12
_DEFAULT_RECIPIENT = 'adtyavrdhn'
_DOUWE_RECIPIENT = 'DouweM'
_MAINTAINER_ASSOCIATIONS = frozenset({'OWNER', 'MEMBER', 'COLLABORATOR'})
_MAINTAINER_PERMISSIONS = frozenset({'admin', 'maintain', 'write'})
_MARKER = re.compile(
    r'<!-- pydantic-ai-attention-monitor stage=(?P<stage>[123]) '
    r'recipient=(?P<recipient>[A-Za-z0-9-]+) -->'
)


class Assignee(TypedDict):
    """Assignee identity plus repository permission classification."""

    login: str
    permission: str | None
    is_maintainer: bool


class Activity(TypedDict):
    """Relevant issue or pull-request timeline activity."""

    kind: str
    author: str
    author_association: str
    created_at: str
    body: str
    state: NotRequired[str]


class MonitorState(TypedDict):
    """Latest active reminder marker emitted by this workflow."""

    stage: Literal[1, 2, 3]
    recipient: str
    created_at: str


def _parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _is_bot(login: str) -> bool:
    return login.endswith('[bot]') or login in {'github-actions', 'github-actions[bot]'}


def _activity_time(activity: Mapping[str, object]) -> dt.datetime:
    return _parse_time(str(activity['created_at']))


def _is_maintainer_activity(activity: Mapping[str, object]) -> bool:
    return (
        not _is_bot(str(activity.get('author', ''))) and activity.get('author_association') in _MAINTAINER_ASSOCIATIONS
    )


def choose_primary_recipient(assignees: Iterable[Assignee]) -> str:
    """Select the first assigned maintainer, falling back to Aditya."""
    return next((a['login'] for a in assignees if a['is_maintainer']), _DEFAULT_RECIPIENT)


def latest_monitor_state(activities: Iterable[Activity], primary_recipient: str) -> MonitorState | None:
    """Return an active reminder, reset by reassignment or maintainer response."""
    markers: list[tuple[dt.datetime, MonitorState]] = []
    timeline = list(activities)
    for activity in timeline:
        match = _MARKER.search(activity.get('body', ''))
        if match:
            stage_text = match.group('stage')
            stage: Literal[1, 2, 3]
            if stage_text == '1':
                stage = 1
            elif stage_text == '2':
                stage = 2
            else:
                stage = 3
            markers.append(
                (
                    _activity_time(activity),
                    MonitorState(
                        stage=stage,
                        recipient=match.group('recipient'),
                        created_at=activity['created_at'],
                    ),
                )
            )
    if not markers:
        return None

    marker_time, state = max(markers, key=lambda entry: entry[0])
    if state['stage'] in {1, 2} and state['recipient'].casefold() != primary_recipient.casefold():
        return None
    if any(
        _activity_time(a) > marker_time
        and (_is_maintainer_activity(a) or str(a.get('author', '')).casefold() == state['recipient'].casefold())
        for a in timeline
    ):
        return None
    return state


def next_reminder(
    *,
    activities: Iterable[Activity],
    assignees: Iterable[Assignee],
    last_activity_at: dt.datetime,
    now: dt.datetime,
) -> tuple[Literal[1, 2, 3], str] | None:
    """Return the due reminder stage and deterministic recipient."""
    primary = choose_primary_recipient(assignees)
    state = latest_monitor_state(activities, primary)
    if state is None:
        if now - last_activity_at < _SLA:
            return None
        return 1, primary
    if state['stage'] == 3:
        return None
    if now - _parse_time(state['created_at']) < _SLA:
        return None
    if state['stage'] == 1:
        return 2, primary
    return 3, _DOUWE_RECIPIENT


class GitHubClient:
    """Small authenticated REST client with Link-header pagination."""

    def __init__(self, token: str) -> None:
        self._token = token

    def _request(self, path: str) -> tuple[Any, str | None]:
        request = urllib.request.Request(
            f'{_API}{path}',
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self._token}',
                'User-Agent': 'pydantic-ai-attention-monitor',
                'X-GitHub-Api-Version': '2022-11-28',
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response), response.headers.get('Link')

    def get(self, path: str) -> Any:
        return self._request(path)[0]

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        """POST one JSON payload and return the decoded response."""
        request = urllib.request.Request(
            f'{_API}{path}',
            data=json.dumps(payload).encode(),
            method='POST',
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self._token}',
                'Content-Type': 'application/json',
                'User-Agent': 'pydantic-ai-attention-monitor',
                'X-GitHub-Api-Version': '2022-11-28',
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)

    def paginate(self, path: str) -> list[dict[str, Any]]:
        separator = '&' if '?' in path else '?'
        url = f'{path}{separator}per_page=100'
        items: list[dict[str, Any]] = []
        while url:
            page, links = self._request(url)
            items.extend(page)
            url = _next_path(links)
        return items

    def search_issues(self, query: str) -> list[dict[str, Any]]:
        """Return every issue-search result page, up to GitHub's search limit."""
        encoded_query = urllib.parse.quote_plus(query)
        url = f'/search/issues?q={encoded_query}&sort=updated&order=asc&per_page=100'
        items: list[dict[str, Any]] = []
        while url:
            page, links = self._request(url)
            items.extend(page['items'])
            url = _next_path(links)
        return items


def _next_path(links: str | None) -> str:
    if not links:
        return ''
    for entry in links.split(','):
        if 'rel="next"' in entry:
            url = entry[entry.index('<') + 1 : entry.index('>')]
            parsed = urllib.parse.urlparse(url)
            return f'{parsed.path}?{parsed.query}'
    return ''


def _author(entry: Mapping[str, Any]) -> str:
    user = entry.get('user') or entry.get('author')
    if isinstance(user, Mapping):
        return str(cast(Mapping[str, object], user).get('login') or '')
    return ''


def _activity(kind: str, entry: Mapping[str, Any], *, created_key: str = 'created_at') -> Activity:
    return Activity(
        kind=kind,
        author=_author(entry),
        author_association=str(entry.get('author_association') or ''),
        created_at=str(entry[created_key]),
        body=str(entry.get('body') or ''),
        state=str(entry.get('state') or ''),
    )


def _hydrate(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> dict[str, Any]:
    number = int(item['number'])
    comments = [_activity('comment', c) for c in client.paginate(f'/repos/{repo}/issues/{number}/comments')]
    is_pr = 'pull_request' in item
    reviews: list[Activity] = []
    review_comments: list[Activity] = []
    if is_pr:
        reviews = [
            _activity('review', r, created_key='submitted_at')
            for r in client.paginate(f'/repos/{repo}/pulls/{number}/reviews')
            if r.get('submitted_at')
        ]
        review_comments = [
            _activity('review_comment', c) for c in client.paginate(f'/repos/{repo}/pulls/{number}/comments')
        ]
    activities = [*comments, *reviews, *review_comments]
    return {
        'kind': 'pull_request' if is_pr else 'issue',
        'number': number,
        'title': item['title'],
        'url': item['html_url'],
        'author': item['user']['login'],
        'body': item.get('body') or '',
        'created_at': item['created_at'],
        'updated_at': item['updated_at'],
        'labels': [label['name'] for label in item.get('labels', [])],
        'assignee_logins': [a['login'] for a in item.get('assignees', [])],
        'activities': activities,
        'last_activity_at': item['updated_at'],
    }


def _safe_hydrate(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> dict[str, Any] | None:
    """Ignore an item that closes between discovery and hydration."""
    try:
        return _hydrate(client, repo, item)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _permission(client: GitHubClient, repo: str, login: str) -> str | None:
    try:
        return str(client.get(f'/repos/{repo}/collaborators/{login}/permission')['permission'])
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404}:
            return None
        raise


def render_comment(item: Mapping[str, Any], stage: Literal[1, 2, 3], recipient: str) -> str:
    """Render one reminder plus its machine-readable state marker."""
    noun = 'PR' if item['kind'] == 'pull_request' else 'issue'
    if stage == 1:
        message = f'@{recipient} this {noun} has had no GitHub activity for three days.'
    elif stage == 2:
        message = f'@{recipient} second ping: this {noun} is still waiting three days after the first reminder.'
    else:
        message = f'@{recipient} pinging you because this {noun} is still waiting three days after the second reminder.'
    return (
        f'{message}\n\n'
        f'**Last GitHub activity:** {item["last_activity_at"]}\n\n'
        f'<!-- pydantic-ai-attention-monitor stage={stage} recipient={recipient} -->'
    )


def _write_summary(path: str | None, *, staged: bool, total: int, due: list[dict[str, Any]]) -> None:
    if not path:
        return
    lines = [
        '# Issue and PR attention monitor',
        '',
        f'- Mode: {"staged" if staged else "live"}',
        f'- Open items scanned: {total}',
        f'- Notifications selected: {len(due)}',
    ]
    for item in due:
        reminder = item['next_reminder']
        lines.append(
            f'- {item["url"]}: reminder {reminder["stage"]} to '
            f'`@{reminder["recipient"]}` (last activity {item["last_activity_at"]})'
        )
    with Path(path).open('a', encoding='utf-8') as summary:
        summary.write('\n'.join(lines) + '\n')


def main() -> int:
    """Scan all open items and post or preview bounded notifications."""
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    staged = os.environ.get('ATTENTION_MONITOR_STAGED', 'true').casefold() != 'false'
    if not token:
        print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
        return 1

    client = GitHubClient(token)
    items = client.paginate(f'/repos/{repo}/issues?state=open&sort=updated&direction=asc')
    now = dt.datetime.now(dt.timezone.utc)
    stale = [item for item in items if now - _parse_time(item['updated_at']) >= _SLA]
    query = f'repo:{repo} is:open "pydantic-ai-attention-monitor" in:comments'
    pending_reminders = client.search_issues(query)
    candidates_by_number = {int(item['number']): item for item in [*stale, *pending_reminders]}
    candidate_items: list[dict[str, Any]] = list(candidates_by_number.values())
    hydrate = functools.partial(_safe_hydrate, client, repo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        hydration_results = executor.map(hydrate, candidate_items)
        hydrated = [item for item in hydration_results if item is not None]

    assignee_logins = sorted({login for item in hydrated for login in item['assignee_logins']})
    permissions = {login: _permission(client, repo, login) for login in assignee_logins}
    due: list[dict[str, Any]] = []
    for item in hydrated:
        assignees: list[Assignee] = [
            Assignee(
                login=login,
                permission=permissions[login],
                is_maintainer=permissions[login] in _MAINTAINER_PERMISSIONS,
            )
            for login in item.pop('assignee_logins')
        ]
        last_activity = _parse_time(item['last_activity_at'])
        reminder = next_reminder(
            activities=item['activities'], assignees=assignees, last_activity_at=last_activity, now=now
        )
        if reminder is None:
            continue
        stage, recipient = reminder
        item['assignees'] = assignees
        item['next_reminder'] = {'stage': stage, 'recipient': recipient}
        due.append(item)

    due.sort(key=lambda item: (-item['next_reminder']['stage'], item['last_activity_at']))
    selected = due[:_MAX_CANDIDATES]
    for item in selected:
        reminder = item['next_reminder']
        body = render_comment(item, reminder['stage'], reminder['recipient'])
        if staged:
            print(f'WOULD COMMENT {item["url"]}:\n{body}\n')
        else:
            client.post(f'/repos/{repo}/issues/{item["number"]}/comments', {'body': body})
            print(f'commented on {item["url"]}')
    _write_summary(
        os.environ.get('GITHUB_STEP_SUMMARY'),
        staged=staged,
        total=len(items),
        due=selected,
    )
    print(f'{"previewed" if staged else "posted"} {len(selected)} of {len(due)} due notifications')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
