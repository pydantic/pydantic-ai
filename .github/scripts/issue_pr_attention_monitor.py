#!/usr/bin/env python3
"""Prepare attention advice and enforce maintainer-approved reminder policy.

The model can only recommend items in a fixed Slack report. A maintainer opts an
item into deterministic assignment and reminders by applying the action label.
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
_CANDIDATE_WINDOW = dt.timedelta(hours=12)
_CANDIDATE_LIMIT = 20
_CANDIDATE_HYDRATION_LIMIT = 40
_MAX_REMINDERS = 10
_MAX_RECONCILE_ITEMS = 25
_MAX_LABELED_ITEMS = 100
_MAX_ACTIVITY_PAGES = 2
_MAX_PARTICIPANTS = 50
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
        and activity['author'] != 'github-actions[bot]'
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
    reset_after_label = False
    latest_marker: tuple[dt.datetime, int, list[str]] | None = None
    for activity in sorted(activities, key=lambda value: _parse_time(value['created_at'])):
        created_at = _parse_time(activity['created_at'])
        if created_at < label_since:
            continue
        if _resets_clock(activity, permissions):
            reset_at = created_at
            reset_after_label = True
            latest_marker = None
        marker = _trusted_marker(activity)
        if marker:
            latest_marker = created_at, marker[0], marker[1]

    recipients = _current_recipients(assignees, permissions)
    if latest_marker is None:
        due_at = reset_at + _SLA if reset_after_label else label_since
        return Reminder(stage=1, recipients=recipients) if now >= due_at else None

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

    def paginate(self, path: str, *, max_pages: int = _MAX_ACTIVITY_PAGES) -> list[dict[str, Any]]:
        separator = '&' if '?' in path else '?'
        page = 1
        items: list[dict[str, Any]] = []
        while page <= max_pages:
            batch = cast(list[dict[str, Any]], self.get(f'{path}{separator}per_page=100&page={page}'))
            items.extend(batch)
            if len(batch) < 100:
                return items
            if page == max_pages:
                raise RuntimeError(f'GitHub activity exceeds the {max_pages * 100}-event safety limit')
            page += 1
        return items


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
    if len(logins) > _MAX_PARTICIPANTS:
        raise RuntimeError(f'GitHub activity exceeds the {_MAX_PARTICIPANTS}-participant safety limit')
    permissions = {login: client.permission(repo, login) for login in sorted(logins)}
    for activity in activities:
        activity['is_maintainer'] = permissions.get(activity['author']) in _MAINTAINER_PERMISSIONS
    return item, activities, permissions


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


def _latest_maintainer_label_event(activities: Sequence[Activity]) -> Activity | None:
    events = [activity for activity in activities if activity['label'] == _ACTION_LABEL and activity['is_maintainer']]
    return max(events, key=lambda value: _parse_time(value['created_at']), default=None)


def _action_label_is_authorized(activities: Sequence[Activity]) -> bool:
    """Require the current label addition to come from a maintainer."""
    events = [activity for activity in activities if activity['label'] == _ACTION_LABEL]
    latest = max(events, key=lambda value: _parse_time(value['created_at']), default=None)
    return bool(latest and latest['kind'] == 'labeled' and latest['is_maintainer'])


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
    if len(decisions) > _CANDIDATE_LIMIT:
        raise ValueError(f'Agent output exceeds the {_CANDIDATE_LIMIT}-item limit')
    return decisions


def _rotated(values: Sequence[dict[str, Any]], *, now: dt.datetime, stride: int) -> list[dict[str, Any]]:
    if not values:
        return []
    six_hour_slot = int(now.timestamp()) // (6 * 60 * 60)
    offset = (six_hour_slot * stride) % len(values)
    return [*values[offset:], *values[:offset]]


def build_candidates(client: GitHubClient, repo: str, *, now: dt.datetime) -> tuple[list[dict[str, Any]], int, int]:
    """Build a bounded snapshot of items crossing the three-day threshold."""
    earliest = now - _SLA - _CANDIDATE_WINDOW
    latest = now - _SLA
    since = urllib.parse.quote(earliest.isoformat(), safe='')
    items = client.paginate(
        f'/repos/{repo}/issues?state=open&since={since}&sort=updated&direction=asc',
        max_pages=2,
    )
    eligible_items = [
        item
        for item in items
        if earliest <= _parse_time(str(item['updated_at'])) <= latest and _ACTION_LABEL not in _labels(item)
    ]
    candidates: list[dict[str, Any]] = []
    attempts = 0
    skipped_count = 0
    for item in _rotated(eligible_items, now=now, stride=_CANDIDATE_LIMIT):
        if attempts >= _CANDIDATE_HYDRATION_LIMIT or len(candidates) >= _CANDIDATE_LIMIT:
            break
        attempts += 1
        number = int(item['number'])
        try:
            current, activities, _ = hydrate(client, repo, number)
        except (RuntimeError, urllib.error.HTTPError):
            skipped_count += 1
            continue
        current_updated_at = _parse_time(str(current['updated_at']))
        if (
            current.get('state') != 'open'
            or not earliest <= current_updated_at <= latest
            or _ACTION_LABEL in _labels(current)
        ):
            continue
        latest_label = _latest_maintainer_label_event(activities)
        if latest_label and latest_label['kind'] == 'unlabeled':
            continue
        recent = sorted(activities, key=lambda value: _parse_time(value['created_at']))[-30:]
        candidates.append(
            {
                'number': number,
                'kind': 'pull_request' if 'pull_request' in current else 'issue',
                'url': str(current['html_url']),
                'title': str(current.get('title') or '')[:500],
                'body': str(current.get('body') or '')[:4_000],
                'updated_at': str(current['updated_at']),
                'assignees': [str(value['login']) for value in current.get('assignees', [])],
                'labels': sorted(_labels(current)),
                'recent_activity': [
                    {
                        'kind': activity['kind'],
                        'author': activity['author'],
                        'created_at': activity['created_at'],
                        'body': activity['body'][:1_000],
                        'is_maintainer': activity['is_maintainer'],
                    }
                    for activity in recent
                ],
            }
        )
    deferred_count = max(0, len(eligible_items) - attempts)
    return candidates, deferred_count, skipped_count


def write_snapshot(client: GitHubClient, repo: str, path: str, *, now: dt.datetime) -> list[str]:
    """Write public candidate data for the sandboxed advisory agent."""
    candidates, deferred_count, skipped_count = build_candidates(client, repo, now=now)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                'generated_at': now.isoformat(),
                'candidates': candidates,
                'deferred_count': deferred_count,
                'skipped_count': skipped_count,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    lines = [f'wrote {len(candidates)} advisory candidate(s)']
    if deferred_count:
        lines.append(f'deferred {deferred_count} candidate(s) beyond the safety limit')
    if skipped_count:
        lines.append(f'skipped {skipped_count} candidate(s) that could not be safely hydrated')
    if (deferred_count or skipped_count) and not candidates:
        raise RuntimeError('No advisory candidates could be safely hydrated')
    return lines


def _snapshot_allowlist(path: str) -> tuple[set[int], int, int]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Snapshot must be an object')
    data = cast(Mapping[str, object], loaded)
    raw_candidates = data.get('candidates')
    deferred_count = data.get('deferred_count', 0)
    skipped_count = data.get('skipped_count', 0)
    if (
        not isinstance(raw_candidates, list)
        or not isinstance(deferred_count, int)
        or deferred_count < 0
        or not isinstance(skipped_count, int)
        or skipped_count < 0
    ):
        raise ValueError('Snapshot has invalid candidates or warning counts')
    numbers: set[int] = set()
    for candidate in cast(list[object], raw_candidates):
        if not isinstance(candidate, Mapping):
            raise ValueError('Snapshot candidate must be an object')
        number = cast(Mapping[str, object], candidate).get('number')
        if not isinstance(number, int) or number < 1 or number in numbers:
            raise ValueError('Snapshot candidate number must be unique and positive')
        numbers.add(number)
    if len(numbers) > _CANDIDATE_LIMIT:
        raise ValueError('Snapshot exceeds the candidate limit')
    return numbers, deferred_count, skipped_count


def render_advisory(repo: str, path: str, snapshot_path: str) -> tuple[bool, str]:
    """Validate model recommendations and render fixed Slack text."""
    decisions = _parse_decisions(path)
    numbers = [decision['item_number'] for decision in decisions]
    if len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains duplicate item numbers')
    eligible, deferred_count, skipped_count = _snapshot_allowlist(snapshot_path)
    if set(numbers) != eligible:
        raise ValueError('Agent output must contain exactly one decision for every candidate')
    recommended = sorted(
        decision['item_number']
        for decision in decisions
        if decision['next_actor'] == 'maintainer' and decision['confidence'] == 'high'
    )
    if not recommended and not deferred_count and not skipped_count:
        return False, ''
    lines: list[str] = []
    if recommended:
        links = '\n'.join(f'• <https://github.com/{repo}/issues/{number}|#{number}>' for number in recommended)
        lines.append(
            ':eyes: Attention triage found items that may need a maintainer call. '
            f'Apply `{_ACTION_LABEL}` to approve:\n{links}'
        )
    if deferred_count:
        lines.append(
            f':warning: Attention triage deferred {deferred_count} additional candidate(s); '
            'the review limit was reached.'
        )
    if skipped_count:
        lines.append(f':warning: Attention triage skipped {skipped_count} candidate(s) that could not be read safely.')
    return True, '\n'.join(lines)


def _first_maintainer_responder(activities: Sequence[Activity]) -> str | None:
    responses = [
        activity
        for activity in activities
        if activity['kind'] in {'comment', 'review', 'review_comment'}
        and activity['is_maintainer']
        and not _is_bot(activity['author'])
    ]
    first = min(responses, key=lambda value: _parse_time(value['created_at']), default=None)
    return first['author'] if first else None


def run_reminders(client: GitHubClient, repo: str, *, staged: bool, now: dt.datetime) -> list[str]:
    """Post at most a bounded number of due reminders on labeled open items."""
    if not staged:
        ensure_label(client, repo)
    encoded = urllib.parse.quote(_ACTION_LABEL, safe='')
    items = cast(
        list[dict[str, Any]],
        client.get(
            f'/repos/{repo}/issues?state=open&labels={encoded}&sort=updated&direction=asc&per_page={_MAX_LABELED_ITEMS}'
        ),
    )
    if len(items) == _MAX_LABELED_ITEMS:
        raise RuntimeError(f'Attention label backlog reached the {_MAX_LABELED_ITEMS}-item safety limit')
    lines: list[str] = []
    failures = 0
    deferred_reminders = 0
    posted_reminders = 0
    stable_items = sorted(items, key=lambda value: int(value['number']))
    for item in _rotated(stable_items, now=now, stride=_MAX_RECONCILE_ITEMS)[:_MAX_RECONCILE_ITEMS]:
        number = int(item['number'])
        try:
            current, activities, permissions = hydrate(client, repo, number)
        except (RuntimeError, urllib.error.HTTPError):
            failures += 1
            continue
        if _ACTION_LABEL not in _labels(current) or not _action_label_is_authorized(activities):
            continue
        assignees = [str(value['login']) for value in current.get('assignees', [])]
        has_owner = any(permissions.get(login) in _MAINTAINER_PERMISSIONS for login in assignees)
        if not has_owner:
            owner = _first_maintainer_responder(activities) or _DEFAULT_OWNER
            if not staged:
                _assign(client, repo, number, owner)
            assignees.append(owner)
            permissions = {**permissions, owner: client.permission(repo, owner)}
            lines.append(f'#{number}: {"would assign" if staged else "assigned"} @{owner}')
        primary = _current_recipients(assignees, permissions)
        reminder = next_reminder(activities=activities, assignees=assignees, permissions=permissions, now=now)
        if reminder is None:
            continue
        if posted_reminders >= _MAX_REMINDERS:
            deferred_reminders += 1
            continue
        body = render_comment(reminder, primary)
        if not staged:
            client.post(f'/repos/{repo}/issues/{number}/comments', {'body': body})
        lines.append(f'#{number}: {"would post" if staged else "posted"} reminder {reminder["stage"]}')
        posted_reminders += 1
    if failures or deferred_reminders:
        raise RuntimeError(
            f'Could not reconcile {failures} unsafe item(s); deferred {deferred_reminders} due reminder(s)'
        )
    return lines


def _write_summary(lines: Sequence[str]) -> None:
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if path:
        with Path(path).open('a', encoding='utf-8') as summary:
            summary.write('## Issue and PR attention monitor\n\n')
            summary.write('\n'.join(f'- {line}' for line in lines) or '- No changes')
            summary.write('\n')


def _write_action_output(name: str, value: str) -> None:
    path = os.environ.get('GITHUB_OUTPUT')
    if not path:
        return
    with Path(path).open('a', encoding='utf-8') as output:
        output.write(f'{name}<<PYDANTIC_AI_ATTENTION_EOF\n{value}\nPYDANTIC_AI_ATTENTION_EOF\n')


def main() -> int:
    """Build advisory input, render its report, or reconcile reminders."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'report', 'reconcile'])
    parser.add_argument('--snapshot-path', default='/tmp/gh-aw/agent/attention-candidates.json')
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    staged = os.environ.get('ATTENTION_TRIAGE_STAGED', 'true').casefold() != 'false'
    now = dt.datetime.now(dt.timezone.utc)
    if args.mode == 'report':
        if not args.agent_output:
            parser.error('--agent-output is required')
        has_report, report_text = render_advisory(repo, args.agent_output, args.snapshot_path)
        _write_action_output('has_report', str(has_report).lower())
        _write_action_output('report_text', report_text)
        lines = ['prepared advisory Slack report' if has_report else 'no advisory Slack report needed']
    else:
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if not token:
            print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
            return 1
        client = GitHubClient(token)
        if args.mode == 'snapshot':
            lines = write_snapshot(client, repo, args.snapshot_path, now=now)
        else:
            lines = run_reminders(client, repo, staged=staged, now=now)
    _write_summary(lines)
    for line in lines:
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
