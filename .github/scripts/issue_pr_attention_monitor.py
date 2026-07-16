#!/usr/bin/env python3
"""Apply the deterministic policy around agentic issue and PR triage.

The model decides only whether a maintainer is the next actor. This module owns
label changes, ownership, reminder timing, and every write to GitHub.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import TypedDict

_API = 'https://api.github.com'
_SLA = dt.timedelta(days=3)
_MAX_REMINDERS = 20
_DEFAULT_OWNER = 'adtyavrdhn'
_FINAL_RECIPIENT = 'DouweM'
_ACTION_LABEL = 'needs-maintainer-action'
_MAINTAINER_PERMISSIONS = frozenset({'admin', 'maintain', 'write'})
_MARKER = re.compile(
    r'<!-- pydantic-ai-attention-monitor stage=(?P<stage>[123]) '
    r'recipients=(?P<recipients>[A-Za-z0-9,-]+) -->'
)


class Activity(TypedDict):
    """One timeline event relevant to ownership or reminder state."""

    kind: str
    author: str
    created_at: str
    body: str
    label: str
    assignee: str
    is_maintainer: bool


class Reminder(TypedDict):
    """A reminder that is due now."""

    stage: Literal[1, 2, 3]
    recipients: list[str]


class Decision(TypedDict):
    """The complete model-controlled surface."""

    item_number: int
    next_actor: Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain']
    confidence: Literal['high', 'medium', 'low']


def _parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _is_bot(login: str) -> bool:
    return login.endswith('[bot]') or login in {'github-actions', 'github-actions[bot]'}


def _labels(item: Mapping[str, Any]) -> set[str]:
    return {str(label['name']) for label in item.get('labels', [])}


def _marker_body(stage: int, recipients: Sequence[str]) -> str:
    return f'<!-- pydantic-ai-attention-monitor stage={stage} recipients={",".join(recipients)} -->'


def _current_recipients(assignees: Iterable[str], permissions: Mapping[str, str | None]) -> list[str]:
    maintainers = sorted(login for login in assignees if permissions.get(login) in _MAINTAINER_PERMISSIONS)
    return maintainers or [_DEFAULT_OWNER]


def _active_label_since(activities: Sequence[Activity]) -> dt.datetime | None:
    active_since: dt.datetime | None = None
    for activity in sorted(activities, key=lambda value: _parse_time(value['created_at'])):
        if activity['label'] != _ACTION_LABEL:
            continue
        if activity['kind'] == 'labeled':
            active_since = _parse_time(activity['created_at'])
        elif activity['kind'] == 'unlabeled':
            active_since = None
    return active_since


def _resets_clock(activity: Activity, permissions: Mapping[str, str | None]) -> bool:
    if activity['is_maintainer'] and activity['kind'] in {'comment', 'review', 'review_comment'}:
        return True
    return (
        activity['kind'] in {'assigned', 'unassigned'}
        and permissions.get(activity['assignee']) in _MAINTAINER_PERMISSIONS
    )


def _trusted_marker(activity: Activity) -> tuple[int, list[str]] | None:
    if activity['author'] != 'github-actions[bot]':
        return None
    matches = list(_MARKER.finditer(activity['body']))
    if not matches:
        return None
    match = matches[-1]
    return int(match.group('stage')), match.group('recipients').split(',')


def next_reminder(
    *,
    activities: Sequence[Activity],
    assignees: Iterable[str],
    permissions: Mapping[str, str | None],
    now: dt.datetime,
) -> Reminder | None:
    """Return the next reminder after three quiet days, or None."""
    label_since = _active_label_since(activities)
    if label_since is None:
        return None

    reset_at = label_since
    latest_marker: tuple[dt.datetime, int, list[str]] | None = None
    for activity in sorted(activities, key=lambda value: _parse_time(value['created_at'])):
        created_at = _parse_time(activity['created_at'])
        if created_at < label_since:
            continue
        if _resets_clock(activity, permissions):
            reset_at = created_at
            latest_marker = None
        marker = _trusted_marker(activity)
        if marker:
            latest_marker = created_at, marker[0], marker[1]

    recipients = _current_recipients(assignees, permissions)
    if latest_marker is None:
        return Reminder(stage=1, recipients=recipients) if now >= reset_at + _SLA else None

    marker_at, stage, previous_recipients = latest_marker
    if stage == 3:
        return None
    if {login.casefold() for login in previous_recipients} != {login.casefold() for login in recipients}:
        return Reminder(stage=1, recipients=recipients) if now >= marker_at + _SLA else None
    if now < marker_at + _SLA:
        return None
    if stage == 1:
        return Reminder(stage=2, recipients=recipients)
    if any(login.casefold() == _FINAL_RECIPIENT.casefold() for login in recipients):
        return None
    return Reminder(stage=3, recipients=[_FINAL_RECIPIENT])


def render_comment(reminder: Reminder, primary_recipients: Sequence[str]) -> str:
    """Render a concise fixed reminder; no model text reaches GitHub."""
    mentions = ' '.join(f'@{login}' for login in reminder['recipients'])
    if reminder['stage'] == 1:
        message = f'{mentions} This needs an eye — could you please take a call on what should be done here?'
    elif reminder['stage'] == 2:
        message = f'{mentions} A second nudge on this — it still needs a maintainer call.'
    else:
        owners = ' '.join(f'@{login}' for login in primary_recipients)
        message = (
            f'@{_FINAL_RECIPIENT} This still needs an eye after two pings to {owners} — '
            'could you please take a call on the next step?'
        )
    return f'{message}\n\n{_marker_body(reminder["stage"], reminder["recipients"])}'


class GitHubClient:
    """Small authenticated GitHub REST client."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._permission_cache: dict[tuple[str, str], str | None] = {}

    def request(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f'{_API}{path}',
            data=data,
            method=method,
            headers={
                'Accept': 'application/vnd.github+json',
                'Authorization': f'Bearer {self._token}',
                'Content-Type': 'application/json',
                'User-Agent': 'pydantic-ai-attention-monitor',
                'X-GitHub-Api-Version': '2022-11-28',
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return None if response.status == 204 else json.load(response)

    def get(self, path: str) -> Any:
        return self.request('GET', path)

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        return self.request('POST', path, payload)

    def delete(self, path: str) -> Any:
        return self.request('DELETE', path)

    def permission(self, repo: str, login: str) -> str | None:
        if not login or _is_bot(login):
            return None
        key = repo, login.casefold()
        if key not in self._permission_cache:
            try:
                value = self.get(f'/repos/{repo}/collaborators/{login}/permission')
                self._permission_cache[key] = str(value['permission'])
            except urllib.error.HTTPError as exc:
                if exc.code not in {403, 404}:
                    raise
                self._permission_cache[key] = None
        return self._permission_cache[key]

    def paginate(self, path: str) -> list[dict[str, Any]]:
        separator = '&' if '?' in path else '?'
        page = 1
        items: list[dict[str, Any]] = []
        while True:
            batch = cast(list[dict[str, Any]], self.get(f'{path}{separator}per_page=100&page={page}'))
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1


def _actor(entry: Mapping[str, Any]) -> str:
    user = entry.get('user') or entry.get('actor')
    return str(cast(Mapping[str, object], user).get('login') or '') if isinstance(user, Mapping) else ''


def _activity(kind: str, entry: Mapping[str, Any], *, created_key: str = 'created_at') -> Activity:
    label = entry.get('label')
    assignee = entry.get('assignee')
    return Activity(
        kind=kind,
        author=_actor(entry),
        created_at=str(entry[created_key]),
        body=str(entry.get('body') or ''),
        label=str(cast(Mapping[str, object], label).get('name') or '') if isinstance(label, Mapping) else '',
        assignee=(
            str(cast(Mapping[str, object], assignee).get('login') or '') if isinstance(assignee, Mapping) else ''
        ),
        is_maintainer=False,
    )


def hydrate(
    client: GitHubClient, repo: str, number: int
) -> tuple[dict[str, Any], list[Activity], dict[str, str | None]]:
    """Load authoritative item, activity, and maintainer permissions."""
    item = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    timeline = [
        _activity(str(event.get('event') or ''), event)
        for event in client.paginate(f'/repos/{repo}/issues/{number}/timeline')
        if event.get('created_at')
    ]
    comments = [_activity('comment', entry) for entry in client.paginate(f'/repos/{repo}/issues/{number}/comments')]
    reviews: list[Activity] = []
    review_comments: list[Activity] = []
    if 'pull_request' in item:
        reviews = [
            _activity('review', entry, created_key='submitted_at')
            for entry in client.paginate(f'/repos/{repo}/pulls/{number}/reviews')
            if entry.get('submitted_at')
        ]
        review_comments = [
            _activity('review_comment', entry) for entry in client.paginate(f'/repos/{repo}/pulls/{number}/comments')
        ]
    activities = [*timeline, *comments, *reviews, *review_comments]
    logins = {
        *[str(assignee['login']) for assignee in item.get('assignees', [])],
        *[activity['author'] for activity in activities if activity['author']],
        *[activity['assignee'] for activity in activities if activity['assignee']],
    }
    permissions = {login: client.permission(repo, login) for login in logins}
    for activity in activities:
        activity['is_maintainer'] = permissions.get(activity['author']) in _MAINTAINER_PERMISSIONS
    return item, activities, permissions


def _add_label(client: GitHubClient, repo: str, number: int) -> None:
    client.post(f'/repos/{repo}/issues/{number}/labels', {'labels': [_ACTION_LABEL]})


def _remove_label(client: GitHubClient, repo: str, number: int) -> None:
    try:
        client.delete(f'/repos/{repo}/issues/{number}/labels/{urllib.parse.quote(_ACTION_LABEL, safe="")}')
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise


def _assign(client: GitHubClient, repo: str, number: int, login: str) -> None:
    client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [login]})


def ensure_label(client: GitHubClient, repo: str) -> None:
    """Create the single policy label if it does not exist."""
    encoded = urllib.parse.quote(_ACTION_LABEL, safe='')
    try:
        client.get(f'/repos/{repo}/labels/{encoded}')
        return
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
    try:
        client.post(
            f'/repos/{repo}/labels',
            {
                'name': _ACTION_LABEL,
                'color': 'd4c5f9',
                'description': 'The next meaningful action must come from a maintainer',
            },
        )
    except urllib.error.HTTPError as exc:
        if exc.code != 422:  # A concurrent run may have created it.
            raise


def handle_event(client: GitHubClient, repo: str, event: Mapping[str, Any], event_name: str) -> list[str]:
    """Enforce label authority and claim unowned items on maintainer response."""
    item = event.get('issue') or event.get('pull_request')
    if not isinstance(item, Mapping):
        return ['event has no issue or pull request']
    item_mapping = cast(Mapping[str, object], item)
    raw_number = item_mapping['number']
    if not isinstance(raw_number, int):
        raise ValueError('Event item number must be an integer')
    number = raw_number
    actor = str(cast(Mapping[str, object], event.get('sender') or {}).get('login') or '')
    is_maintainer = client.permission(repo, actor) in _MAINTAINER_PERMISSIONS and not _is_bot(actor)
    action = str(event.get('action') or '')
    label = event.get('label')
    label_name = str(cast(Mapping[str, object], label).get('name') or '') if isinstance(label, Mapping) else ''

    if label_name == _ACTION_LABEL and not is_maintainer and actor != 'github-actions[bot]':
        if action == 'labeled':
            _remove_label(client, repo, number)
            return [f'ignored non-maintainer label addition on #{number}']
        if action == 'unlabeled':
            _add_label(client, repo, number)
            return [f'restored non-maintainer label removal on #{number}']

    response_events = {'issue_comment', 'pull_request_review', 'pull_request_review_comment'}
    if not (is_maintainer and (event_name in response_events or label_name == _ACTION_LABEL)):
        return [f'no deterministic change for #{number}']
    current = cast(Mapping[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    assignees = [str(value['login']) for value in current.get('assignees', [])]
    has_owner = any(client.permission(repo, login) in _MAINTAINER_PERMISSIONS for login in assignees)
    if is_maintainer and event_name in response_events and action in {'created', 'submitted'} and not has_owner:
        _assign(client, repo, number, actor)
        return [f'assigned @{actor} to #{number}']
    if is_maintainer and action == 'labeled' and label_name == _ACTION_LABEL and not has_owner:
        _assign(client, repo, number, _DEFAULT_OWNER)
        return [f'assigned @{_DEFAULT_OWNER} to #{number}']
    return [f'no deterministic change for #{number}']


def _latest_maintainer_label_event(activities: Sequence[Activity]) -> Activity | None:
    events = [activity for activity in activities if activity['label'] == _ACTION_LABEL and activity['is_maintainer']]
    return max(events, key=lambda value: _parse_time(value['created_at']), default=None)


def _action_label_is_authorized(activities: Sequence[Activity]) -> bool:
    """Require the current label addition to come from a maintainer or this workflow."""
    events = [activity for activity in activities if activity['label'] == _ACTION_LABEL]
    latest = max(events, key=lambda value: _parse_time(value['created_at']), default=None)
    return bool(
        latest
        and latest['kind'] == 'labeled'
        and (latest['is_maintainer'] or latest['author'] == 'github-actions[bot]')
    )


def _parse_decisions(path: str) -> list[Decision]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Agent output must contain an `items` list')
    data = cast(Mapping[str, object], loaded)
    raw_items = data.get('items')
    if not isinstance(raw_items, list):
        raise ValueError('Agent output must contain an `items` list')
    decisions: list[Decision] = []
    for value in cast(list[object], raw_items):
        if not isinstance(value, Mapping):
            continue
        raw = cast(Mapping[str, object], value)
        if raw.get('type') != 'record_attention_decision':
            continue
        number = raw.get('item_number')
        actor = raw.get('next_actor')
        confidence = raw.get('confidence')
        if not isinstance(number, str) or re.fullmatch(r'[1-9][0-9]*', number) is None:
            raise ValueError('Decision `item_number` must be a positive decimal string')
        if actor not in {'maintainer', 'contributor', 'automation', 'none', 'uncertain'}:
            raise ValueError(f'Invalid `next_actor`: {actor!r}')
        if confidence not in {'high', 'medium', 'low'}:
            raise ValueError(f'Invalid `confidence`: {confidence!r}')
        decisions.append(
            Decision(
                item_number=int(number),
                next_actor=cast(Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain'], actor),
                confidence=cast(Literal['high', 'medium', 'low'], confidence),
            )
        )
    return decisions


def eligible_numbers(event: Mapping[str, Any], event_name: str) -> set[int]:
    """Return the only item numbers an agent output may mutate."""
    item = event.get('issue') or event.get('pull_request')
    if isinstance(item, Mapping):
        number = cast(Mapping[str, object], item).get('number')
        return {number} if isinstance(number, int) and number > 0 else set()
    if event_name != 'check_suite' or not isinstance(event.get('check_suite'), Mapping):
        return set()
    pull_requests = cast(Mapping[str, object], event['check_suite']).get('pull_requests')
    if not isinstance(pull_requests, Sequence):
        return set()
    return {
        number
        for value in cast(Sequence[object], pull_requests)[:5]
        if isinstance(value, Mapping)
        and isinstance(number := cast(Mapping[str, object], value).get('number'), int)
        and number > 0
    }


def apply_decision(client: GitHubClient, repo: str, decision: Decision, *, staged: bool) -> str:
    """Apply one decision after reloading authoritative state."""
    number = decision['item_number']
    item, activities, permissions = hydrate(client, repo, number)
    if item.get('state', 'open') != 'open':
        return f'#{number}: ignored because it is closed'
    labels = _labels(item)
    latest_label = _latest_maintainer_label_event(activities)
    wants_attention = decision['next_actor'] == 'maintainer' and decision['confidence'] == 'high'

    # A maintainer's latest manual label action is authoritative until another
    # maintainer changes it. Model classifications never override it.
    if latest_label:
        return f'#{number}: preserved maintainer label decision'
    if decision['confidence'] != 'high':
        return f'#{number}: abstained at {decision["confidence"]} confidence'
    if wants_attention and _ACTION_LABEL not in labels:
        if staged:
            return f'#{number}: would add {_ACTION_LABEL}'
        ensure_label(client, repo)
        assignees = [str(value['login']) for value in item.get('assignees', [])]
        has_owner = any(permissions.get(login) in _MAINTAINER_PERMISSIONS for login in assignees)
        _add_label(client, repo, number)
        try:
            if not has_owner:
                _assign(client, repo, number, _DEFAULT_OWNER)
        except Exception:
            _remove_label(client, repo, number)
            raise
        return f'#{number}: added {_ACTION_LABEL}'
    if not wants_attention and _ACTION_LABEL in labels:
        if not staged:
            _remove_label(client, repo, number)
        return f'#{number}: {"would remove" if staged else "removed"} {_ACTION_LABEL}'
    return f'#{number}: label already matches decision'


def run_reminders(client: GitHubClient, repo: str, *, staged: bool, now: dt.datetime) -> list[str]:
    """Post at most a bounded number of due reminders on labeled open items."""
    if not staged:
        ensure_label(client, repo)
    encoded = urllib.parse.quote(_ACTION_LABEL, safe='')
    items = client.paginate(f'/repos/{repo}/issues?state=open&labels={encoded}&sort=updated&direction=asc')
    lines: list[str] = []
    for item in items:
        number = int(item['number'])
        current, activities, permissions = hydrate(client, repo, number)
        if _ACTION_LABEL not in _labels(current) or not _action_label_is_authorized(activities):
            continue
        assignees = [str(value['login']) for value in current.get('assignees', [])]
        primary = _current_recipients(assignees, permissions)
        reminder = next_reminder(activities=activities, assignees=assignees, permissions=permissions, now=now)
        if reminder is None:
            continue
        body = render_comment(reminder, primary)
        if not staged:
            client.post(f'/repos/{repo}/issues/{number}/comments', {'body': body})
        lines.append(f'#{number}: {"would post" if staged else "posted"} reminder {reminder["stage"]}')
        if len(lines) >= _MAX_REMINDERS:
            break
    return lines


def _write_summary(lines: Sequence[str]) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with Path(path).open('a', encoding='utf-8') as summary:
            summary.write('## Issue and PR attention monitor\n\n')
            summary.write('\n'.join(f'- {line}' for line in lines) or '- No changes')
            summary.write('\n')


def main() -> int:
    """Run event handling, decision application, or scheduled reminders."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['event', 'apply-decisions', 'remind'])
    parser.add_argument('--event-path', default=os.environ.get('GITHUB_EVENT_PATH'))
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token:
        print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
        return 1

    client = GitHubClient(token)
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    staged = os.environ.get('ATTENTION_TRIAGE_STAGED', 'true').casefold() != 'false'
    event_name = os.environ.get('GITHUB_EVENT_NAME', '')
    if args.mode == 'remind':
        lines = run_reminders(client, repo, staged=staged, now=dt.datetime.now(dt.timezone.utc))
    else:
        if not args.event_path:
            parser.error('--event-path is required')
        event = cast(dict[str, Any], json.loads(Path(args.event_path).read_text(encoding='utf-8')))
        if args.mode == 'event':
            ensure_label(client, repo)
            lines = handle_event(client, repo, event, event_name)
        else:
            if not args.agent_output:
                parser.error('--agent-output is required')
            decisions = _parse_decisions(args.agent_output)
            allowed = eligible_numbers(event, event_name)
            numbers = [decision['item_number'] for decision in decisions]
            if len(numbers) != len(set(numbers)) or set(numbers) - allowed:
                raise ValueError('Agent output contains duplicate or ineligible item numbers')
            lines = [apply_decision(client, repo, decision, staged=staged) for decision in decisions]
    _write_summary(lines)
    for line in lines:
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
