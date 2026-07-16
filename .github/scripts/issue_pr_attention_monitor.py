#!/usr/bin/env python3
"""Apply deterministic ownership and reminder policy around agentic triage.

The model may only propose the next actor and a short rationale. This module
owns every GitHub mutation, validates human overrides, and computes all clocks.
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
_DEFAULT_RECIPIENT = 'adtyavrdhn'
_DOUWE_RECIPIENT = 'DouweM'
_ACTION_LABEL = 'needs-maintainer-action'
_FORCE_LABEL = 'attention:force'
_SKIP_LABEL = 'attention:skip'
_PRIORITY_LABEL = 'p:1-highest'
_MAINTAINER_PERMISSIONS = frozenset({'admin', 'maintain', 'write'})
_LABEL_DEFINITIONS: dict[str, tuple[str, str]] = {
    _ACTION_LABEL: ('d4c5f9', 'The next meaningful action must come from a maintainer'),
    _FORCE_LABEL: ('b60205', 'Maintainer override: force attention triage'),
    _SKIP_LABEL: ('cfd3d7', 'Maintainer override: do not request attention'),
}
_MARKER = re.compile(
    r'<!-- pydantic-ai-attention-monitor stage=(?P<stage>[123]) '
    r'recipients=(?P<recipients>[A-Za-z0-9,-]+) -->'
)


class Activity(TypedDict):
    """One timeline event relevant to reminder state."""

    kind: str
    author: str
    created_at: str
    body: str
    label: str
    assignee: str
    is_maintainer: bool


class Reminder(TypedDict):
    """A deterministic reminder that is currently due."""

    stage: Literal[1, 2, 3]
    recipients: list[str]
    due_at: str
    terminal_without_comment: bool


class Decision(TypedDict):
    """The narrow structured judgment accepted from the agent."""

    item_number: int
    next_actor: Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain']
    confidence: Literal['high', 'medium', 'low']
    recommended_action: str
    context: str
    urgent: bool


def _parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _is_bot(login: str) -> bool:
    return login.endswith('[bot]') or login in {'github-actions', 'github-actions[bot]'}


def _labels(item: Mapping[str, Any]) -> set[str]:
    return {str(label['name']) for label in item.get('labels', [])}


def _marker_body(stage: int, recipients: Sequence[str]) -> str:
    return f'<!-- pydantic-ai-attention-monitor stage={stage} recipients={",".join(recipients)} -->'


def _trusted_marker(activity: Activity) -> re.Match[str] | None:
    """Return the host-appended marker, never model or contributor text."""
    if activity['author'] != 'github-actions[bot]':
        return None
    matches = list(_MARKER.finditer(activity['body']))
    return matches[-1] if matches else None


def _current_recipients(assignees: Iterable[str], permissions: Mapping[str, str | None]) -> list[str]:
    maintainers = sorted(login for login in assignees if permissions.get(login) in _MAINTAINER_PERMISSIONS)
    return maintainers or [_DEFAULT_RECIPIENT]


def _active_label_since(activities: Sequence[Activity]) -> dt.datetime | None:
    active_since: dt.datetime | None = None
    for activity in sorted(activities, key=lambda a: _parse_time(a['created_at'])):
        if activity['label'] != _ACTION_LABEL:
            continue
        if activity['kind'] == 'labeled':
            active_since = _parse_time(activity['created_at'])
        elif activity['kind'] == 'unlabeled':
            active_since = None
    return active_since


def _latest_label_event(activities: Sequence[Activity], label: str) -> Activity | None:
    events = [activity for activity in activities if activity['label'] == label]
    return max(events, key=lambda a: _parse_time(a['created_at']), default=None)


def _valid_override(labels: set[str], activities: Sequence[Activity], label: str) -> bool:
    """Trust maintainer overrides and their deterministic workflow projection."""
    latest = _latest_label_event(activities, label)
    return (
        label in labels
        and latest is not None
        and latest['kind'] == 'labeled'
        and (latest['is_maintainer'] or latest['author'] == 'github-actions[bot]')
    )


def _resets_reminder(activity: Activity, permissions: Mapping[str, str | None]) -> bool:
    """Return whether an activity starts a fresh owner-response window."""
    if activity['is_maintainer'] and activity['kind'] in {'comment', 'review', 'review_comment'}:
        return True
    return (
        activity['kind'] in {'assigned', 'unassigned'}
        and permissions.get(activity['assignee']) in _MAINTAINER_PERMISSIONS
    )


def _latest_reminder_marker(
    activities: Sequence[Activity],
    permissions: Mapping[str, str | None],
    label_since: dt.datetime,
) -> tuple[dt.datetime, int, list[str]] | None:
    """Return the latest marker in the current label and ownership-response cycle."""
    latest_marker: tuple[dt.datetime, int, list[str]] | None = None
    for activity in sorted(activities, key=lambda a: _parse_time(a['created_at'])):
        created_at = _parse_time(activity['created_at'])
        if created_at < label_since:
            continue
        if _resets_reminder(activity, permissions):
            latest_marker = None
        if match := _trusted_marker(activity):
            latest_marker = (created_at, int(match.group('stage')), match.group('recipients').split(','))
    return latest_marker


def next_reminder(
    *,
    activities: Sequence[Activity],
    assignees: Iterable[str],
    permissions: Mapping[str, str | None],
    now: dt.datetime,
) -> Reminder | None:
    """Return the next due reminder from label, maintainer, assignment, and marker events."""
    label_since = _active_label_since(activities)
    if label_since is None:
        return None
    recipients = _current_recipients(assignees, permissions)
    reset_at = label_since
    for activity in sorted(activities, key=lambda a: _parse_time(a['created_at'])):
        created_at = _parse_time(activity['created_at'])
        if created_at < label_since:
            continue
        if _resets_reminder(activity, permissions):
            reset_at = max(reset_at, created_at)
    latest_marker = _latest_reminder_marker(activities, permissions, label_since)

    if latest_marker is None:
        due_at = reset_at + _SLA
        if now < due_at:
            return None
        return Reminder(stage=1, recipients=recipients, due_at=due_at.isoformat(), terminal_without_comment=False)

    marker_at, stage, marker_recipients = latest_marker
    if stage == 3:
        return None
    if stage in {1, 2} and {r.casefold() for r in marker_recipients} != {r.casefold() for r in recipients}:
        due_at = max(reset_at, marker_at) + _SLA
        if now < due_at:
            return None
        return Reminder(stage=1, recipients=recipients, due_at=due_at.isoformat(), terminal_without_comment=False)
    due_at = marker_at + _SLA
    if now < due_at:
        return None
    if stage == 1:
        return Reminder(stage=2, recipients=recipients, due_at=due_at.isoformat(), terminal_without_comment=False)
    terminal = any(r.casefold() == _DOUWE_RECIPIENT.casefold() for r in recipients)
    return Reminder(
        stage=3,
        recipients=[_DOUWE_RECIPIENT],
        due_at=due_at.isoformat(),
        terminal_without_comment=terminal,
    )


def _clean_context(value: str) -> str:
    fallback = 'A maintainer decision is still the next step.'
    text = ' '.join(value.split()).strip()
    if re.search(r'https?://|www\.|<|>|!\[|\]\s*\(', text, flags=re.IGNORECASE):
        return fallback
    text = re.sub(r'[`*_~#|\[\]()]', '', text).replace('@', '@\u200b')
    if len(text) > 240:
        text = f'{text[:237].rstrip()}...'
    return text.rstrip('.') + '.' if text else fallback


def render_comment(reminder: Reminder, context: str, primary_recipients: Sequence[str]) -> str:
    """Render a concise reminder with hidden deterministic state."""
    mentions = ' '.join(f'@{login}' for login in reminder['recipients'])
    if reminder['stage'] == 1:
        message = f'{mentions} This needs an eye — could you please take a call on what should be done here?'
    elif reminder['stage'] == 2:
        message = f'{mentions} A second nudge on this — it still needs a maintainer call.'
    else:
        owners = ' '.join(f'@{login}' for login in primary_recipients)
        message = (
            f'@{_DOUWE_RECIPIENT} This still needs an eye after two pings to {owners} — '
            'could you please take a call on the next step?'
        )
    return f'{message}\n\n**Context:** {_clean_context(context)}\n\n{_marker_body(reminder["stage"], reminder["recipients"])}'


class GitHubClient:
    """Minimal authenticated GitHub REST client."""

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
            if response.status == 204:
                return None
            return json.load(response)

    def get(self, path: str) -> Any:
        return self.request('GET', path)

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        return self.request('POST', path, payload)

    def delete(self, path: str) -> Any:
        return self.request('DELETE', path)

    def permission(self, repo: str, login: str) -> str | None:
        """Return a collaborator permission, cached for this process."""
        if not login or _is_bot(login):
            return None
        key = (repo, login.casefold())
        if key in self._permission_cache:
            return self._permission_cache[key]
        try:
            permission = str(self.get(f'/repos/{repo}/collaborators/{login}/permission')['permission'])
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                permission = None
            else:
                raise
        self._permission_cache[key] = permission
        return permission

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


def _permission(client: GitHubClient, repo: str, login: str) -> str | None:
    return client.permission(repo, login)


def _actor(entry: Mapping[str, Any]) -> str:
    user = entry.get('user') or entry.get('actor')
    user_mapping = cast(Mapping[str, object], user) if isinstance(user, Mapping) else None
    return str(user_mapping.get('login') or '') if user_mapping else ''


def _activity(kind: str, entry: Mapping[str, Any], *, created_key: str = 'created_at') -> Activity:
    label = entry.get('label')
    assignee = entry.get('assignee')
    label_mapping = cast(Mapping[str, object], label) if isinstance(label, Mapping) else None
    assignee_mapping = cast(Mapping[str, object], assignee) if isinstance(assignee, Mapping) else None
    return Activity(
        kind=kind,
        author=_actor(entry),
        created_at=str(entry[created_key]),
        body=str(entry.get('body') or ''),
        label=str(label_mapping.get('name') or '') if label_mapping else '',
        assignee=str(assignee_mapping.get('login') or '') if assignee_mapping else '',
        is_maintainer=False,
    )


def hydrate(
    client: GitHubClient, repo: str, number: int
) -> tuple[dict[str, Any], list[Activity], dict[str, str | None]]:
    """Load one item, its relevant timeline, and actor permissions."""
    item = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    timeline = [
        _activity(str(event.get('event') or ''), event)
        for event in client.paginate(f'/repos/{repo}/issues/{number}/timeline')
        if event.get('created_at')
    ]
    comments = [_activity('comment', c) for c in client.paginate(f'/repos/{repo}/issues/{number}/comments')]
    reviews: list[Activity] = []
    review_comments: list[Activity] = []
    if 'pull_request' in item:
        reviews = [
            _activity('review', r, created_key='submitted_at')
            for r in client.paginate(f'/repos/{repo}/pulls/{number}/reviews')
            if r.get('submitted_at')
        ]
        review_comments = [
            _activity('review_comment', c) for c in client.paginate(f'/repos/{repo}/pulls/{number}/comments')
        ]
    activities = [*timeline, *comments, *reviews, *review_comments]
    logins = {
        *[str(a['login']) for a in item.get('assignees', [])],
        *[activity['author'] for activity in activities if activity['author']],
        *[activity['assignee'] for activity in activities if activity['assignee']],
    }
    permissions = {login: _permission(client, repo, login) for login in logins}
    for activity in activities:
        activity['is_maintainer'] = permissions.get(activity['author']) in _MAINTAINER_PERMISSIONS
    return item, activities, permissions


def _add_labels(client: GitHubClient, repo: str, number: int, labels: Sequence[str]) -> None:
    client.post(f'/repos/{repo}/issues/{number}/labels', {'labels': list(labels)})


def _remove_label(client: GitHubClient, repo: str, number: int, label: str) -> None:
    try:
        client.delete(f'/repos/{repo}/issues/{number}/labels/{urllib.parse.quote(label, safe="")}')
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise


def _assign(client: GitHubClient, repo: str, number: int, login: str) -> None:
    client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [login]})


def ensure_labels(client: GitHubClient, repo: str) -> list[str]:
    """Ensure the three policy labels exist without changing existing definitions."""
    created: list[str] = []
    for name, (color, description) in _LABEL_DEFINITIONS.items():
        try:
            client.get(f'/repos/{repo}/labels/{urllib.parse.quote(name, safe="")}')
            continue
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        try:
            client.post(f'/repos/{repo}/labels', {'name': name, 'color': color, 'description': description})
            created.append(name)
        except urllib.error.HTTPError as exc:
            if exc.code != 422:  # another run may have created it concurrently
                raise
    return created


def handle_event(client: GitHubClient, repo: str, event: Mapping[str, Any]) -> list[str]:
    """Apply deterministic ownership and maintainer override behavior."""
    item = event.get('issue') or event.get('pull_request')
    if not isinstance(item, Mapping):
        return ['event has no issue or pull request']
    item_mapping = cast(Mapping[str, object], item)
    number = int(cast(int | str, item_mapping['number']))
    actor = str(cast(Mapping[str, Any], event.get('sender') or {}).get('login') or '')
    actor_permission = _permission(client, repo, actor)
    is_maintainer = actor_permission in _MAINTAINER_PERMISSIONS and not _is_bot(actor)
    action = str(event.get('action') or '')
    label = event.get('label')
    label_mapping = cast(Mapping[str, object], label) if isinstance(label, Mapping) else None
    label_name = str(label_mapping.get('name') or '') if label_mapping else ''
    messages: list[str] = []

    if (
        label_name in {_ACTION_LABEL, _FORCE_LABEL, _SKIP_LABEL}
        and not is_maintainer
        and actor != 'github-actions[bot]'
    ):
        if action == 'labeled':
            _remove_label(client, repo, number, label_name)
            return [f'ignored non-maintainer {label_name} override on #{number}']
        if action == 'unlabeled':
            _add_labels(client, repo, number, [label_name])
            return [f'restored non-maintainer removal of {label_name} on #{number}']

    if action == 'labeled' and label_name == _SKIP_LABEL:
        if _ACTION_LABEL in _labels(item_mapping):
            _remove_label(client, repo, number, _ACTION_LABEL)
        return [f'applied {_SKIP_LABEL} override on #{number}']

    assignees = [
        str(assignee.get('login') or '')
        for assignee in cast(Sequence[Mapping[str, object]], item_mapping.get('assignees') or [])
    ]
    permissions = {login: _permission(client, repo, login) for login in assignees}
    maintainer_assignees = [login for login in assignees if permissions.get(login) in _MAINTAINER_PERMISSIONS]
    claim_events = {'issue_comment', 'pull_request_review', 'pull_request_review_comment'}
    event_name = os.environ.get('GITHUB_EVENT_NAME', '')
    if is_maintainer and action in {'created', 'submitted'} and event_name in claim_events and not maintainer_assignees:
        _assign(client, repo, number, actor)
        messages.append(f'assigned @{actor} to #{number}')
        maintainer_assignees = [actor]

    if is_maintainer and action == 'unlabeled' and label_name == _ACTION_LABEL:
        _add_labels(client, repo, number, [_SKIP_LABEL])
        messages.append(f'added {_SKIP_LABEL} to #{number}')

    if (
        action == 'labeled'
        and label_name in {_ACTION_LABEL, _FORCE_LABEL}
        and not maintainer_assignees
        and _SKIP_LABEL not in _labels(item_mapping)
    ):
        if label_name == _FORCE_LABEL:
            _add_labels(client, repo, number, [_ACTION_LABEL])
        _assign(client, repo, number, _DEFAULT_RECIPIENT)
        messages.append(f'assigned @{_DEFAULT_RECIPIENT} to #{number}')
        maintainer_assignees = [_DEFAULT_RECIPIENT]
    if action == 'labeled' and label_name == _PRIORITY_LABEL and not maintainer_assignees:
        messages.append(f'ANOMALY #{number}: urgent item has no maintainer owner')
    return messages


def _parse_decisions(path: str) -> list[Decision]:
    raw_data: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw_data, Mapping):
        raise ValueError('Agent output must be an object')
    data = cast(Mapping[str, object], raw_data)
    raw_items = data.get('items')
    if not isinstance(raw_items, list):
        raise ValueError('Agent output must contain an `items` list')
    decisions: list[Decision] = []
    for raw_item in cast(list[object], raw_items):
        if not isinstance(raw_item, Mapping):
            raise ValueError('Every agent output item must be an object')
        raw = cast(Mapping[str, object], raw_item)
        if raw.get('type') != 'record_attention_decision':
            continue
        item_number = raw.get('item_number')
        if not isinstance(item_number, str) or re.fullmatch(r'[1-9][0-9]*', item_number) is None:
            raise ValueError('Decision `item_number` must be a positive decimal string')
        next_actor = raw.get('next_actor')
        if not isinstance(next_actor, str) or next_actor not in {
            'maintainer',
            'contributor',
            'automation',
            'none',
            'uncertain',
        }:
            raise ValueError(f'Invalid `next_actor`: {next_actor!r}')
        confidence = raw.get('confidence')
        if not isinstance(confidence, str) or confidence not in {'high', 'medium', 'low'}:
            raise ValueError(f'Invalid `confidence`: {confidence!r}')
        recommended_action = raw.get('recommended_action')
        context = raw.get('context')
        if not isinstance(recommended_action, str) or not isinstance(context, str):
            raise ValueError('Decision text fields must be strings')
        urgent = raw.get('urgent', False)
        if not isinstance(urgent, bool) and urgent not in {'true', 'false'}:
            raise ValueError('Decision `urgent` must be a boolean')
        decisions.append(
            Decision(
                item_number=int(item_number),
                next_actor=cast(Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain'], next_actor),
                confidence=cast(Literal['high', 'medium', 'low'], confidence),
                recommended_action=recommended_action,
                context=context,
                urgent=urgent is True or str(urgent).casefold() == 'true',
            )
        )
    return decisions


def _event_item_number(event: Mapping[str, Any]) -> int | None:
    item = event.get('issue') or event.get('pull_request')
    if not isinstance(item, Mapping):
        return None
    number = cast(Mapping[str, object], item).get('number')
    return number if isinstance(number, int) and number > 0 else None


def _recent_sample_numbers(client: GitHubClient, repo: str) -> set[int]:
    """Return the bounded manual shadow sample described in the agent prompt."""
    numbers: set[int] = set()
    for item_type in ('issue', 'pr'):
        query = urllib.parse.quote(f'repo:{repo} is:open is:{item_type}')
        result = cast(
            Mapping[str, Any],
            client.get(f'/search/issues?q={query}&sort=updated&order=desc&per_page=5'),
        )
        for item in cast(Sequence[Mapping[str, object]], result.get('items') or []):
            number = item.get('number')
            if isinstance(number, int) and number > 0:
                numbers.add(number)
    return numbers


def _reconciliation_numbers(client: GitHubClient, repo: str) -> set[int]:
    encoded = urllib.parse.quote(_ACTION_LABEL, safe='')
    items = cast(
        Sequence[Mapping[str, object]],
        client.get(f'/repos/{repo}/issues?state=open&labels={encoded}&sort=updated&direction=asc&per_page=10'),
    )
    numbers: set[int] = set()
    for item in items:
        number = item.get('number')
        if isinstance(number, int) and number > 0:
            numbers.add(number)
    return numbers


def eligible_decision_numbers(
    client: GitHubClient,
    repo: str,
    *,
    event_name: str,
    event: Mapping[str, Any],
    staged: bool,
) -> set[int]:
    """Derive the only repository items an agent output may target."""
    event_number = _event_item_number(event)
    if event_number is not None:
        return {event_number}
    if event_name == 'check_suite':
        check_suite = event.get('check_suite')
        if not isinstance(check_suite, Mapping):
            return set()
        pull_requests = cast(Mapping[str, object], check_suite).get('pull_requests')
        if not isinstance(pull_requests, Sequence):
            return set()
        numbers: set[int] = set()
        for pull_request in cast(Sequence[object], pull_requests)[:5]:
            if isinstance(pull_request, Mapping):
                number = cast(Mapping[str, object], pull_request).get('number')
                if isinstance(number, int) and number > 0:
                    numbers.add(number)
        return numbers
    if event_name == 'schedule':
        return set() if staged else _reconciliation_numbers(client, repo)
    if event_name == 'workflow_dispatch':
        return _recent_sample_numbers(client, repo) if staged else _reconciliation_numbers(client, repo)
    return set()


def _validate_decision_targets(decisions: Sequence[Decision], eligible_numbers: set[int]) -> None:
    numbers = [decision['item_number'] for decision in decisions]
    if len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains duplicate decisions for one item')
    unexpected = set(numbers) - eligible_numbers
    if unexpected:
        raise ValueError(f'Agent output targets ineligible item numbers: {sorted(unexpected)}')


def _apply_new_attention_label(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    has_maintainer_owner: bool,
    staged: bool,
) -> str:
    """Atomically add attention and a fallback owner when one is needed."""
    if not staged:
        _add_labels(client, repo, number, [_ACTION_LABEL])
        try:
            if not has_maintainer_owner:
                _assign(client, repo, number, _DEFAULT_RECIPIENT)
        except Exception:
            _remove_label(client, repo, number, _ACTION_LABEL)
            raise
    owner_note = 'existing maintainer owner' if has_maintainer_owner else 'fallback owner'
    return f'#{number}: {"would add" if staged else "added"} {_ACTION_LABEL} with {owner_note}'


def _wants_maintainer_action(labels: set[str], decision: Decision) -> bool:
    """Resolve the force override and high-confidence classifier threshold."""
    return _FORCE_LABEL in labels or (decision['next_actor'] == 'maintainer' and decision['confidence'] == 'high')


def _latest_removal_is_maintainer_override(labels: set[str], activities: Sequence[Activity]) -> bool:
    """Whether the current absence of attention came from a maintainer removal."""
    latest = _latest_label_event(activities, _ACTION_LABEL)
    return (
        _ACTION_LABEL not in labels and latest is not None and latest['kind'] == 'unlabeled' and latest['is_maintainer']
    )


def _apply_skip_override(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    labels: set[str],
    activities: Sequence[Activity],
    staged: bool,
) -> list[str] | None:
    """Apply or preserve a human-maintainer suppression, if one exists."""
    if _SKIP_LABEL in labels:
        if _ACTION_LABEL in labels and not staged:
            _remove_label(client, repo, number, _ACTION_LABEL)
        return [f'#{number}: maintainer skip override preserved']
    if _latest_removal_is_maintainer_override(labels, activities):
        if not staged:
            _add_labels(client, repo, number, [_SKIP_LABEL])
        return [f'#{number}: preserved latest maintainer label-removal override']
    return None


def _remove_attention_if_current(client: GitHubClient, repo: str, number: int) -> str:
    """Remove attention only if no racing authoritative state supersedes it."""
    item, activities, _ = hydrate(client, repo, number)
    labels = _labels(item)
    if item.get('state', 'open') != 'open' or _ACTION_LABEL not in labels:
        return f'#{number}: state changed before label removal; left unchanged'
    if _valid_override(labels, activities, _FORCE_LABEL):
        return f'#{number}: force override appeared before label removal; left unchanged'
    if _valid_override(labels, activities, _SKIP_LABEL):
        return f'#{number}: maintainer skip override preserved'
    _remove_label(client, repo, number, _ACTION_LABEL)
    return f'#{number}: removed {_ACTION_LABEL}'


def _add_attention_if_current(client: GitHubClient, repo: str, number: int) -> str:
    """Add attention using current ownership and suppression state."""
    item, activities, permissions = hydrate(client, repo, number)
    labels = _labels(item)
    if item.get('state', 'open') != 'open':
        return f'#{number}: state changed before attention was added; left unchanged'
    if _valid_override(labels, activities, _SKIP_LABEL) or _latest_removal_is_maintainer_override(labels, activities):
        return f'#{number}: maintainer suppression appeared; left unchanged'
    if _ACTION_LABEL in labels:
        return f'#{number}: attention was already added by another actor'
    assignees = [str(a['login']) for a in item.get('assignees', [])]
    has_maintainer_owner = any(permissions.get(login) in _MAINTAINER_PERMISSIONS for login in assignees)
    return _apply_new_attention_label(
        client,
        repo,
        number,
        has_maintainer_owner=has_maintainer_owner,
        staged=False,
    )


def _post_reminder_if_current(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    context: str,
    now: dt.datetime,
) -> str:
    """Recompute authoritative state immediately before posting a reminder."""
    item, activities, permissions = hydrate(client, repo, number)
    labels = _labels(item)
    if item.get('state', 'open') != 'open' or _ACTION_LABEL not in labels:
        return f'#{number}: state changed before reminder; no comment posted'
    if _valid_override(labels, activities, _SKIP_LABEL):
        return f'#{number}: maintainer skip override appeared; no comment posted'
    assignees = [str(a['login']) for a in item.get('assignees', [])]
    primary = _current_recipients(assignees, permissions)
    reminder = next_reminder(activities=activities, assignees=assignees, permissions=permissions, now=now)
    if reminder is None:
        return f'#{number}: label retained; no reminder due'
    if reminder['terminal_without_comment']:
        return f'ANOMALY #{number}: Douwe already owns this; two unanswered pings exhausted'
    body = render_comment(reminder, context, primary)
    client.post(f'/repos/{repo}/issues/{number}/comments', {'body': body})
    return f'#{number}: posted reminder {reminder["stage"]}'


def apply_decision(
    client: GitHubClient,
    repo: str,
    decision: Decision,
    *,
    staged: bool,
    now: dt.datetime,
) -> list[str]:
    """Validate and apply one model decision through the deterministic policy."""
    number = decision['item_number']
    item, activities, permissions = hydrate(client, repo, number)
    if item.get('state', 'open') != 'open':
        return [f'#{number}: ignored because the item is not open']
    labels = _labels(item)
    messages: list[str] = []
    for override in (_FORCE_LABEL, _SKIP_LABEL):
        if override in labels and not _valid_override(labels, activities, override):
            if not staged:
                _remove_label(client, repo, number, override)
            labels.remove(override)
            messages.append(f'ANOMALY #{number}: ignored non-maintainer {override} override')
    if _FORCE_LABEL in labels and _SKIP_LABEL in labels:
        return [*messages, f'ANOMALY #{number}: {_FORCE_LABEL} and {_SKIP_LABEL} coexist; suppressed']
    skip_result = _apply_skip_override(
        client,
        repo,
        number,
        labels=labels,
        activities=activities,
        staged=staged,
    )
    if skip_result is not None:
        return [*messages, *skip_result]

    wants_action = _wants_maintainer_action(labels, decision)
    if decision['urgent'] and not wants_action:
        return [*messages, f'ANOMALY #{number}: urgent item was not confidently routed to a maintainer']
    if not wants_action:
        if _ACTION_LABEL in labels and decision['confidence'] == 'high' and not staged:
            messages.append(_remove_attention_if_current(client, repo, number))
        else:
            messages.append(f'#{number}: abstained or next actor is not a maintainer')
        return messages

    if _ACTION_LABEL not in labels:
        if not staged:
            return [*messages, _add_attention_if_current(client, repo, number)]
        assignees = [str(a['login']) for a in item.get('assignees', [])]
        has_maintainer_owner = any(permissions.get(login) in _MAINTAINER_PERMISSIONS for login in assignees)
        return [
            *messages,
            _apply_new_attention_label(
                client,
                repo,
                number,
                has_maintainer_owner=has_maintainer_owner,
                staged=staged,
            ),
        ]

    if not staged:
        return [*messages, _post_reminder_if_current(client, repo, number, context=decision['context'], now=now)]

    assignees = [str(a['login']) for a in item.get('assignees', [])]
    reminder = next_reminder(activities=activities, assignees=assignees, permissions=permissions, now=now)
    if reminder is None:
        return [*messages, f'#{number}: label retained; no reminder due']
    if reminder['terminal_without_comment']:
        return [*messages, f'ANOMALY #{number}: Douwe already owns this; two unanswered pings exhausted']
    messages.append(f'#{number}: {"would post" if staged else "posted"} reminder {reminder["stage"]}')
    return messages


def anomaly_report(client: GitHubClient, repo: str, *, now: dt.datetime) -> list[str]:
    """Collect exception-only items for the weekly Slack report."""
    items_by_number: dict[int, dict[str, Any]] = {}
    for label in (_ACTION_LABEL, _FORCE_LABEL, _SKIP_LABEL, _PRIORITY_LABEL):
        encoded = urllib.parse.quote(label, safe='')
        for item in client.paginate(f'/repos/{repo}/issues?state=open&labels={encoded}&sort=updated&direction=asc'):
            items_by_number[int(item['number'])] = item
    anomalies: list[str] = []
    for item in items_by_number.values():
        number = int(item['number'])
        labels = _labels(item)
        _, activities, permissions = hydrate(client, repo, number)
        for override in (_FORCE_LABEL, _SKIP_LABEL):
            if override in labels and not _valid_override(labels, activities, override):
                anomalies.append(f'<{item["html_url"]}|#{number}> has a non-maintainer {override} override')
                labels.remove(override)
        if _FORCE_LABEL in labels and _SKIP_LABEL in labels:
            anomalies.append(f'<{item["html_url"]}|#{number}> has conflicting force and skip overrides')
            continue
        if _SKIP_LABEL in labels:
            continue
        assignees = [str(a['login']) for a in item.get('assignees', [])]
        maintainer_assignees = [login for login in assignees if permissions.get(login) in _MAINTAINER_PERMISSIONS]
        reminder = next_reminder(activities=activities, assignees=assignees, permissions=permissions, now=now)
        label_since = _active_label_since(activities)
        latest_marker = (
            _latest_reminder_marker(activities, permissions, label_since) if label_since is not None else None
        )
        terminal_marker = latest_marker is not None and latest_marker[1] == 3
        if terminal_marker:
            anomalies.append(f'<{item["html_url"]}|#{number}> is still waiting after the Douwe ping')
        elif reminder and reminder['terminal_without_comment']:
            anomalies.append(f'<{item["html_url"]}|#{number}> exhausted two pings with Douwe already assigned')
        elif _PRIORITY_LABEL in labels and not any(
            permissions.get(login) in _MAINTAINER_PERMISSIONS for login in assignees
        ):
            anomalies.append(f'<{item["html_url"]}|#{number}> is urgent but has no maintainer owner')
        elif not maintainer_assignees:
            anomalies.append(f'<{item["html_url"]}|#{number}> has no maintainer owner')
        elif reminder:
            anomalies.append(f'<{item["html_url"]}|#{number}> reminder stage {reminder["stage"]} remains due')
    workflow_runs = cast(
        Mapping[str, Any],
        client.get(
            f'/repos/{repo}/actions/workflows/pydantic-ai-attention-triage.lock.yml/runs'
            '?event=schedule&status=success&per_page=1'
        ),
    )
    runs = cast(list[dict[str, Any]], workflow_runs.get('workflow_runs') or [])
    if not runs or now - _parse_time(str(runs[0]['created_at'])) > dt.timedelta(hours=8):
        anomalies.append('The six-hour attention reconciliation has no successful run in the last eight hours')
    return anomalies


def _write_summary(lines: Sequence[str]) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with Path(path).open('a', encoding='utf-8') as summary:
            summary.write('## Issue and PR attention monitor\n\n')
            summary.write('\n'.join(f'- {line}' for line in lines) or '- No changes')
            summary.write('\n')


def main() -> int:
    """Run deterministic event, decision-application, or reporting mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['setup', 'event', 'apply-decisions', 'report'])
    parser.add_argument('--event-path', default=os.environ.get('GITHUB_EVENT_PATH'))
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    parser.add_argument('--report-path')
    parser.add_argument('--github-output')
    args = parser.parse_args()
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token:
        print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
        return 1
    client = GitHubClient(token)
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    staged = os.environ.get('ATTENTION_TRIAGE_STAGED', 'true').casefold() != 'false'
    now = dt.datetime.now(dt.timezone.utc)
    exit_code = 0

    if args.mode == 'setup':
        lines = [f'created label {label}' for label in ensure_labels(client, repo)]
    elif args.mode == 'event':
        if not args.event_path:
            parser.error('--event-path is required')
        created = ensure_labels(client, repo)
        event = cast(dict[str, Any], json.loads(Path(args.event_path).read_text(encoding='utf-8')))
        lines = [*[f'created label {label}' for label in created], *handle_event(client, repo, event)]
        if any(line.startswith('ANOMALY ') for line in lines):
            exit_code = 2
    elif args.mode == 'apply-decisions':
        if not args.agent_output:
            parser.error('--agent-output is required')
        if not args.event_path:
            parser.error('--event-path is required')
        lines: list[str] = []
        decisions = _parse_decisions(args.agent_output)
        event = cast(dict[str, Any], json.loads(Path(args.event_path).read_text(encoding='utf-8')))
        eligible_numbers = eligible_decision_numbers(
            client,
            repo,
            event_name=os.environ.get('GITHUB_EVENT_NAME', ''),
            event=event,
            staged=staged,
        )
        _validate_decision_targets(decisions, eligible_numbers)
        urgent_number = next((decision['item_number'] for decision in decisions if decision['urgent']), None)
        if args.github_output and urgent_number is not None:
            with Path(args.github_output).open('a', encoding='utf-8') as github_output:
                github_output.write(f'urgent_number={urgent_number}\n')
        for decision in decisions:
            decision_lines = apply_decision(client, repo, decision, staged=staged, now=now)
            lines.extend(decision_lines)
            if decision['urgent'] and any(line.startswith('ANOMALY ') for line in decision_lines):
                exit_code = 2
    else:
        lines = anomaly_report(client, repo, now=now)
        if args.report_path:
            Path(args.report_path).write_text(json.dumps({'anomalies': lines}), encoding='utf-8')
    _write_summary(lines)
    for line in lines:
        print(line)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
