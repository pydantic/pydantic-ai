#!/usr/bin/env python3
"""Classify stale issues and PRs, then apply a bounded reminder policy."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

# Stdlib-only imports: production invokes this script with the runner's bare
# `python`, which has no third-party packages installed. The repo-wide ban on
# `typing.TypedDict` exists for pydantic validation on Python 3.10/3.11, and
# this script uses no pydantic.
from typing import Any, Literal, TypedDict, cast  # noqa: TID251

_API = 'https://api.github.com'
_SLA = dt.timedelta(days=3)
_RECENT_ACTIVITY_WINDOW = dt.timedelta(days=45)
_CANDIDATE_LIMIT = 10
_RECENT_CANDIDATE_LIMIT = 5
_BACKLOG_CANDIDATE_LIMIT = _CANDIDATE_LIMIT - _RECENT_CANDIDATE_LIMIT
_RECONCILE_LIMIT = 25
_EVENT_PAGE_LIMIT = 10
_COMMENT_PAGE_LIMIT = 10
_COLLABORATOR_PAGE_LIMIT = 2
_RESPONSE_LIMIT = 5_000_000
_SNAPSHOT_LIMIT = 80_000
_FALLBACK_OWNER = 'adtyavrdhn'
_MAINTAINER_PERMISSIONS = frozenset({'admin', 'maintain', 'write'})
_ACK_ASSOCIATIONS = frozenset({'MEMBER', 'OWNER', 'COLLABORATOR'})
_ACTION_LABEL = 'needs-maintainer-action'
_NOTIFIED_LABEL = 'attention-notified'
_PINGED_LABEL = 'attention-pinged'
_ESCALATED_LABEL = 'attention-escalated'
_STAGE_LABELS = (_PINGED_LABEL, _ESCALATED_LABEL)
_LIFECYCLE_LABELS = (_NOTIFIED_LABEL, *_STAGE_LABELS)
_LABELS = {
    _ACTION_LABEL: ('d4c5f9', 'The next meaningful action must come from a maintainer'),
    _NOTIFIED_LABEL: ('c5def5', 'The triage channel received the initial maintainer attention notice'),
    _PINGED_LABEL: ('fbca04', 'The triage channel received one maintainer attention reminder'),
    _ESCALATED_LABEL: ('d93f0b', 'The triage channel received the terminal maintainer escalation'),
}


class Decision(TypedDict):
    """The complete model-controlled surface."""

    item_number: int
    next_actor: Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain']
    confidence: Literal['high', 'medium', 'low']


class Notice(TypedDict):
    """A fixed host-built notification for the private triage channel."""

    number: int
    kind: Literal['initial', 'reminder', 'escalation']
    expected_stage: Literal[0, 1, 2]
    transition_id: int | str
    title: str
    recipients: list[str]


class NoticeRef(TypedDict):
    """The bounded state reference carried from Slack delivery to finalization."""

    number: int
    kind: Literal['initial', 'reminder', 'escalation']
    expected_stage: Literal[0, 1, 2]
    transition_id: int | str
    recipients: list[str]


_Transition = tuple[dt.datetime, dict[str, Any]]


class GitHubClient:
    """Small GitHub REST client with bounded response parsing."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._maintainers: dict[str, frozenset[str]] = {}

    def _request(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> tuple[Any, str | None]:
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
                return None, response.headers.get('Link')
            body = response.read(_RESPONSE_LIMIT + 1)
            if len(body) > _RESPONSE_LIMIT:
                raise RuntimeError(f'GitHub response exceeds {_RESPONSE_LIMIT} bytes')
            return json.loads(body), response.headers.get('Link')

    def request(self, method: str, path: str, payload: Mapping[str, object] | None = None) -> Any:
        return self._request(method, path, payload)[0]

    def get(self, path: str) -> Any:
        return self.request('GET', path)

    def post(self, path: str, payload: Mapping[str, object]) -> Any:
        return self.request('POST', path, payload)

    def delete(self, path: str) -> Any:
        return self.request('DELETE', path)

    def last_pages(self, path: str, *, count: int = 1) -> list[dict[str, Any]]:
        """Return up to `count` newest pages for an ascending GitHub collection."""
        separator = '&' if '?' in path else '?'
        first_path = f'{path}{separator}per_page=100&page=1'
        first, links = self._request('GET', first_path)
        last_path = _link_path(links, 'last')
        if not last_path:
            return cast(list[dict[str, Any]], first)
        parsed = urllib.parse.urlparse(last_path)
        query = urllib.parse.parse_qs(parsed.query)
        last = int(query['page'][0])
        pages: list[dict[str, Any]] = []
        for page in range(max(1, last - count + 1), last + 1):
            query['page'] = [str(page)]
            page_path = f'{parsed.path}?{urllib.parse.urlencode(query, doseq=True)}'
            pages.extend(cast(list[dict[str, Any]], self.get(page_path)))
        return pages

    def pages(self, path: str, *, count: int = 1) -> Iterator[list[dict[str, Any]]]:
        """Yield up to `count` pages from the start of a GitHub collection."""
        separator = '&' if '?' in path else '?'
        page_path = f'{path}{separator}per_page=100&page=1'
        for _ in range(count):
            values, links = self._request('GET', page_path)
            yield cast(list[dict[str, Any]], values)
            if not (page_path := _link_path(links, 'next')):
                return
        raise RuntimeError(f'GitHub collection exceeds the {count}-page safety limit')

    def maintainer_logins(self, repo: str) -> frozenset[str]:
        if repo not in self._maintainers:
            maintainers: set[str] = set()
            for page in self.pages(
                f'/repos/{repo}/collaborators?permission=push',
                count=_COLLABORATOR_PAGE_LIMIT,
            ):
                maintainers.update(login.casefold() for value in page if (login := str(value.get('login') or '')))
            self._maintainers[repo] = frozenset(maintainers)
        return self._maintainers[repo]


def _parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _link_path(links: str | None, relation: str) -> str:
    if not links:
        return ''
    for entry in links.split(','):
        if f'rel="{relation}"' in entry:
            url = entry[entry.index('<') + 1 : entry.index('>')]
            parsed = urllib.parse.urlparse(url)
            return f'{parsed.path}?{parsed.query}'
    return ''


def _labels(item: Mapping[str, Any]) -> set[str]:
    return {str(label['name']) for label in item.get('labels', [])}


def _login(entry: Mapping[str, Any]) -> str:
    user = entry.get('user')
    return str(cast(Mapping[str, object], user).get('login') or '') if isinstance(user, Mapping) else ''


def _last_page(total: int, page_size: int) -> int:
    return max(1, math.ceil(total / page_size))


def _candidate_context(
    client: GitHubClient, repo: str, item: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, object] | None]:
    """Return bounded conversation and PR state without walking full history."""
    number = int(item['number'])
    page_size = 8
    comments = cast(
        list[dict[str, Any]],
        client.get(
            f'/repos/{repo}/issues/{number}/comments?per_page={page_size}'
            f'&page={_last_page(int(item.get("comments") or 0), page_size)}'
        ),
    )
    entries: list[tuple[str, dict[str, Any]]] = [('comment', comment) for comment in comments]
    pr_context: dict[str, object] | None = None
    if 'pull_request' in item:
        pull = cast(dict[str, Any], client.get(f'/repos/{repo}/pulls/{number}'))
        review_count = int(pull.get('review_comments') or 0)
        review_comments = cast(
            list[dict[str, Any]],
            client.get(
                f'/repos/{repo}/pulls/{number}/comments?per_page={page_size}&page={_last_page(review_count, page_size)}'
            ),
        )
        entries.extend(('review_comment', comment) for comment in review_comments)
        reviews = client.last_pages(f'/repos/{repo}/pulls/{number}/reviews')
        entries.extend(('review', review) for review in reviews if review.get('submitted_at'))
        head = cast(Mapping[str, object], pull['head'])
        sha = str(head['sha'])
        checks = cast(dict[str, Any], client.get(f'/repos/{repo}/commits/{sha}/check-runs?per_page=100')).get(
            'check_runs', []
        )
        check_runs = cast(list[dict[str, Any]], checks)
        pr_context = {
            'draft': bool(pull.get('draft')),
            'mergeable_state': str(pull.get('mergeable_state') or 'unknown'),
            'requested_reviewers': [str(value['login']) for value in pull.get('requested_reviewers', [])],
            'checks': [
                {
                    'name': str(check.get('name') or '')[:100],
                    'status': str(check.get('status') or ''),
                    'conclusion': str(check.get('conclusion') or ''),
                }
                for check in check_runs[:10]
            ],
        }
    recent = sorted(entries, key=lambda entry: str(entry[1].get('created_at') or entry[1].get('submitted_at') or ''))[
        -page_size:
    ]
    return [
        {
            'kind': kind,
            'author': _login(entry),
            'author_association': str(entry.get('author_association') or ''),
            'created_at': str(entry.get('created_at') or entry.get('submitted_at') or ''),
            'body': str(entry.get('body') or '')[:500],
            'state': str(entry.get('state') or '') if kind == 'review' else '',
        }
        for kind, entry in recent
    ], pr_context


def _candidate_page(client: GitHubClient, repo: str, *, now: dt.datetime) -> list[dict[str, Any]]:
    before = (now - _SLA).date().isoformat()
    # An escalated item stays dormant until the reconcile sweep sees real
    # activity and removes the marker; only then may a fresh lifecycle start.
    excluded = f'-label:"{_ACTION_LABEL}" -label:"{_ESCALATED_LABEL}"'
    raw_query = f'repo:{repo} is:open updated:<{before} {excluded}'
    query = urllib.parse.quote_plus(raw_query)
    first = cast(dict[str, Any], client.get(f'/search/issues?q={query}&sort=updated&order=asc&per_page=1'))
    total = min(int(first.get('total_count') or 0), 1_000)
    if not total:
        return []
    slot = int(now.timestamp()) // int(_SLA.total_seconds() / 12)
    recent_after = (now - _RECENT_ACTIVITY_WINDOW).date().isoformat()
    recent_query = urllib.parse.quote_plus(f'{raw_query} updated:>={recent_after}')
    recent_first = cast(
        dict[str, Any],
        client.get(f'/search/issues?q={recent_query}&sort=updated&order=desc&per_page=1'),
    )
    recent_total = min(int(recent_first.get('total_count') or 0), 1_000)
    recent_items: list[dict[str, Any]] = []
    if recent_total:
        recent_pages = math.ceil(recent_total / _RECENT_CANDIDATE_LIMIT)
        recent_page = slot % recent_pages + 1
        recent = cast(
            dict[str, Any],
            client.get(
                f'/search/issues?q={recent_query}&sort=updated&order=desc'
                f'&per_page={_RECENT_CANDIDATE_LIMIT}&page={recent_page}'
            ),
        )
        recent_items = cast(list[dict[str, Any]], recent.get('items') or [])
    pages = math.ceil(total / _BACKLOG_CANDIDATE_LIMIT)
    page = slot % pages + 1
    backlog = cast(
        dict[str, Any],
        client.get(f'/search/issues?q={query}&sort=updated&order=asc&per_page={_BACKLOG_CANDIDATE_LIMIT}&page={page}'),
    )
    candidates: dict[int, dict[str, Any]] = {}
    for item in [
        *recent_items,
        *cast(list[dict[str, Any]], backlog.get('items') or []),
    ]:
        candidates.setdefault(int(item['number']), item)
    return list(candidates.values())[:_CANDIDATE_LIMIT]


def build_snapshot(client: GitHubClient, repo: str, *, now: dt.datetime) -> dict[str, object]:
    """Build the bounded public input consumed by the sandboxed agent."""
    cutoff = now - _SLA
    candidates: list[dict[str, object]] = []
    for result in _candidate_page(client, repo, now=now):
        number = int(result['number'])
        current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        labels = _labels(current)
        updated_at = str(current['updated_at'])
        if (
            current.get('state') != 'open'
            or _parse_time(updated_at) > cutoff
            or _ACTION_LABEL in labels
            or _ESCALATED_LABEL in labels
        ):
            continue
        recent_activity, pr_context = _candidate_context(client, repo, current)
        candidates.append(
            {
                'number': number,
                'kind': 'pull_request' if 'pull_request' in current else 'issue',
                'title': str(current.get('title') or '')[:300],
                'body': str(current.get('body') or '')[:2_000],
                'updated_at': updated_at,
                'assignees': [str(value['login']) for value in current.get('assignees', [])],
                'labels': sorted(labels),
                'recent_activity': recent_activity,
                'pr': pr_context,
            }
        )
    snapshot: dict[str, object] = {'generated_at': now.isoformat(), 'candidates': candidates}
    if len(json.dumps(snapshot, indent=2, ensure_ascii=False).encode()) > _SNAPSHOT_LIMIT:
        raise RuntimeError(f'Attention snapshot exceeds {_SNAPSHOT_LIMIT} bytes')
    return snapshot


def write_snapshot(client: GitHubClient, repo: str, path: str, *, now: dt.datetime) -> list[str]:
    """Write one immutable, size-bounded candidate snapshot."""
    snapshot = build_snapshot(client, repo, now=now)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    candidates = cast(list[object], snapshot['candidates'])
    return [f'wrote {len(candidates)} attention candidate(s)']


def _snapshot_candidates(path: str) -> dict[int, str]:
    """Return the trusted candidate map (number -> snapshot updated_at)."""
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Snapshot must contain a candidates list')
    data = cast(Mapping[str, object], loaded)
    raw_candidates = data.get('candidates')
    if not isinstance(raw_candidates, list):
        raise ValueError('Snapshot must contain a candidates list')
    candidates: dict[int, str] = {}
    for value in cast(list[object], raw_candidates):
        if not isinstance(value, Mapping):
            raise ValueError('Snapshot candidate must be an object')
        candidate = cast(Mapping[str, object], value)
        number = candidate.get('number')
        updated_at = candidate.get('updated_at')
        if not isinstance(number, int) or number < 1 or number in candidates or not isinstance(updated_at, str):
            raise ValueError('Snapshot candidates must have unique positive numbers and timestamps')
        candidates[number] = updated_at
    if len(candidates) > _CANDIDATE_LIMIT:
        raise ValueError('Snapshot exceeds the candidate limit')
    return candidates


def _parse_decisions(path: str) -> list[Decision]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Agent output must contain an items list')
    data = cast(Mapping[str, object], loaded)
    raw_items = data.get('items')
    if not isinstance(raw_items, list):
        raise ValueError('Agent output must contain an items list')
    decisions: list[Decision] = []
    for value in cast(list[object], raw_items):
        if not isinstance(value, Mapping):
            continue
        decision = cast(Mapping[str, object], value)
        if decision.get('type') != 'record_attention_decision':
            continue
        number = decision.get('item_number')
        actor = decision.get('next_actor')
        confidence = decision.get('confidence')
        if not isinstance(number, str) or re.fullmatch(r'[1-9][0-9]*', number) is None:
            raise ValueError('Decision item_number must be a positive decimal string')
        if actor not in {'maintainer', 'contributor', 'automation', 'none', 'uncertain'}:
            raise ValueError(f'Invalid next_actor: {actor!r}')
        if confidence not in {'high', 'medium', 'low'}:
            raise ValueError(f'Invalid confidence: {confidence!r}')
        decisions.append(
            Decision(
                item_number=int(number),
                next_actor=cast(Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain'], actor),
                confidence=cast(Literal['high', 'medium', 'low'], confidence),
            )
        )
    numbers = [decision['item_number'] for decision in decisions]
    if len(numbers) > _CANDIDATE_LIMIT or len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains too many or duplicate decisions')
    return decisions


def ensure_labels(client: GitHubClient, repo: str) -> None:
    """Create the fixed workflow labels if they are absent."""
    for name, (color, description) in _LABELS.items():
        encoded = urllib.parse.quote(name, safe='')
        try:
            client.get(f'/repos/{repo}/labels/{encoded}')
            continue
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code != 404:
                raise
        try:
            client.post(f'/repos/{repo}/labels', {'name': name, 'color': color, 'description': description})
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code != 422:
                raise


def _add_labels(client: GitHubClient, repo: str, number: int, labels: Sequence[str]) -> None:
    client.post(f'/repos/{repo}/issues/{number}/labels', {'labels': list(labels)})


def _collaborator_permission(client: GitHubClient, repo: str, login: str) -> object:
    encoded = urllib.parse.quote(login, safe='')
    return cast(Mapping[str, object], client.get(f'/repos/{repo}/collaborators/{encoded}/permission')).get('permission')


def _maintainer_assignees(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> list[str]:
    maintainers: list[str] = []
    for assignee in item.get('assignees', []):
        login = str(assignee['login'])
        if _collaborator_permission(client, repo, login) in _MAINTAINER_PERMISSIONS:
            maintainers.append(login)
    return sorted(maintainers, key=str.casefold)


def _first_maintainer_in_discussion(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> str | None:
    if 'pull_request' in item:
        return None

    maintainers = client.maintainer_logins(repo)

    author = _login(item)
    if author and author.casefold() in maintainers:
        return author

    number = int(item['number'])
    comment_pages = min(_last_page(int(item.get('comments') or 0), 100), _COMMENT_PAGE_LIMIT)
    for page in client.pages(f'/repos/{repo}/issues/{number}/comments', count=comment_pages):
        for comment in page:
            login = _login(comment)
            if login and login.casefold() in maintainers:
                return login
    return None


def _validate_attention_state(
    previous: Mapping[str, Any], current: Mapping[str, Any], *, check_updated_at: bool = True
) -> None:
    previous_labels = _labels(previous)
    if _ACTION_LABEL in previous_labels and (
        current.get('state') != 'open'
        or (check_updated_at and current.get('updated_at') != previous.get('updated_at'))
        or _ACTION_LABEL not in (current_labels := _labels(current))
        or _stage(current_labels) != _stage(previous_labels)
    ):
        raise RuntimeError('Attention state changed during owner selection')


def _ensure_recipients(
    client: GitHubClient,
    repo: str,
    item: Mapping[str, Any],
    maintainers: Sequence[str],
    transition: _Transition,
) -> list[str]:
    if maintainers:
        number = int(item['number'])
        current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        _validate_attention_state(item, current)
        if current_maintainers := _maintainer_assignees(client, repo, current):
            _validate_attention_transition(client, repo, item, transition)
            return current_maintainers

    owner = _first_maintainer_in_discussion(client, repo, item) or _FALLBACK_OWNER
    number = int(item['number'])
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    _validate_attention_state(item, current)
    current_maintainers = _maintainer_assignees(client, repo, current)
    if current_maintainers:
        _validate_attention_transition(client, repo, item, transition)
        return current_maintainers

    assigned = cast(
        dict[str, Any],
        client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [owner]}),
    )
    _validate_attention_state(item, assigned, check_updated_at=False)
    _validate_attention_transition(client, repo, assigned, transition)
    assigned_maintainers = _maintainer_assignees(client, repo, assigned)
    owner_login = owner.casefold()
    if owner_login not in {login.casefold() for login in assigned_maintainers}:
        raise RuntimeError(f'GitHub did not assign @{owner}')
    return assigned_maintainers


def _remove_label(client: GitHubClient, repo: str, number: int, label: str) -> None:
    encoded = urllib.parse.quote(label, safe='')
    try:
        client.delete(f'/repos/{repo}/issues/{number}/labels/{encoded}')
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code != 404:
            raise


def apply_decisions(client: GitHubClient, repo: str, output_path: str, snapshot_path: str) -> list[str]:
    """Revalidate allowlisted model decisions, then assign and label them."""
    candidates = _snapshot_candidates(snapshot_path)
    decisions = _parse_decisions(output_path)
    unknown = {decision['item_number'] for decision in decisions} - candidates.keys()
    if unknown:
        raise ValueError(f'Agent output contains numbers outside the snapshot: {sorted(unknown)}')
    if {decision['item_number'] for decision in decisions} != candidates.keys():
        raise ValueError('Agent output must classify every snapshot candidate exactly once')
    ensure_labels(client, repo)
    lines: list[str] = []
    failures: list[str] = []
    for decision in decisions:
        number = decision['item_number']
        try:
            current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            labels = _labels(current)
            if (
                current.get('state') != 'open'
                or str(current.get('updated_at')) != candidates[number]
                or _ACTION_LABEL in labels
            ):
                lines.append(f'#{number}: skipped because the item changed after classification')
                continue
            if decision['confidence'] != 'high' or decision['next_actor'] == 'uncertain':
                lines.append(f'#{number}: left unclassified for a future run')
                continue
            if decision['next_actor'] != 'maintainer':
                lines.append(f'#{number}: did not request maintainer attention')
                continue
            for label in labels.intersection(_LIFECYCLE_LABELS):
                _remove_label(client, repo, number, label)
            _add_labels(client, repo, number, [_ACTION_LABEL])
            expected = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if (
                expected.get('state') != 'open'
                or _ACTION_LABEL not in _labels(expected)
                or _stage(_labels(expected)) != 0
            ):
                raise RuntimeError('Attention state changed while applying the request')
            events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
            transition = _transition(events, 0)
            if transition is None or _actor(transition[1]) != 'github-actions[bot]':
                raise RuntimeError('Could not verify the new attention transition')
            recipients = _ensure_recipients(
                client,
                repo,
                expected,
                _maintainer_assignees(client, repo, expected),
                transition,
            )
            mentions = ' '.join(f'@{login}' for login in recipients)
            lines.append(f'#{number}: requested maintainer attention from {mentions}')
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('Failed to apply attention: ' + '; '.join(failures))
    return lines


def _stage(labels: set[str]) -> Literal[0, 1, 2]:
    if _ESCALATED_LABEL in labels:
        return 2
    if _PINGED_LABEL in labels:
        return 1
    return 0


def _advance_stage(client: GitHubClient, repo: str, number: int, labels: set[str], stage: Literal[1, 2]) -> None:
    next_label = _STAGE_LABELS[stage - 1]
    _add_labels(client, repo, number, [next_label])
    for label in labels.intersection(_STAGE_LABELS):
        if label != next_label:
            _remove_label(client, repo, number, label)


def _event_time(event: Mapping[str, Any]) -> dt.datetime | None:
    value = event.get('created_at') or event.get('submitted_at')
    return _parse_time(str(value)) if value else None


def _label_transition(timeline: Sequence[dict[str, Any]], label: str) -> _Transition | None:
    transitions = [
        (time, index, event)
        for index, event in enumerate(timeline)
        if event.get('event') == 'labeled'
        and isinstance(event.get('label'), Mapping)
        and cast(Mapping[str, object], event['label']).get('name') == label
        and (time := _event_time(event)) is not None
    ]
    latest = max(transitions, key=lambda value: (value[0], value[1]), default=None)
    return (latest[0], latest[2]) if latest is not None else None


def _transition(timeline: Sequence[dict[str, Any]], stage: Literal[0, 1, 2]) -> _Transition | None:
    label = _ACTION_LABEL if stage == 0 else _STAGE_LABELS[stage - 1]
    return _label_transition(timeline, label)


def _validate_attention_transition(
    client: GitHubClient,
    repo: str,
    item: Mapping[str, Any],
    expected: _Transition,
    *,
    check_updated_at: bool = True,
) -> None:
    number = int(item['number'])
    stage = _stage(_labels(item))
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    current = _transition(events, stage)
    expected_id = expected[1].get('id')
    if current is None or (
        current[1].get('id') != expected_id if expected_id is not None else current[0] != expected[0]
    ):
        raise RuntimeError('Attention transition changed during owner selection')
    latest = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    _validate_attention_state(item, latest, check_updated_at=check_updated_at)


def _actor(event: Mapping[str, Any]) -> str:
    value = event.get('actor') or event.get('user')
    return str(cast(Mapping[str, object], value).get('login') or '') if isinstance(value, Mapping) else ''


# `mentioned` and `subscribed` can fire as side effects of GitHub activity and
# must never count as acknowledgement.
_NON_ACK_EVENTS = frozenset({'mentioned', 'subscribed'})


def _acknowledged(timeline: Sequence[dict[str, Any]], since: dt.datetime, recipients: Sequence[str]) -> bool:
    recipient_logins = {login.casefold() for login in recipients}
    return any(
        (event_time := _event_time(event)) is not None
        and event_time >= since
        and event.get('event') not in _NON_ACK_EVENTS
        and (
            _actor(event).casefold() in recipient_logins
            or (
                event.get('event') in {'commented', 'reviewed'} and event.get('author_association') in _ACK_ASSOCIATIONS
            )
        )
        for event in timeline
    )


def _closed_since(timeline: Sequence[dict[str, Any]], since: dt.datetime) -> bool:
    return any(
        event.get('event') == 'closed' and (event_time := _event_time(event)) is not None and event_time >= since
        for event in timeline
    )


def _complete(client: GitHubClient, repo: str, number: int, labels: set[str]) -> None:
    for label in labels.intersection(_LIFECYCLE_LABELS):
        _remove_label(client, repo, number, label)
    _remove_label(client, repo, number, _ACTION_LABEL)


def _notice(
    item: Mapping[str, Any],
    kind: Literal['initial', 'reminder', 'escalation'],
    stage: Literal[0, 1, 2],
    transition: _Transition,
    recipients: Sequence[str],
) -> Notice:
    transition_id = transition[1].get('id')
    if not isinstance(transition_id, (int, str)) or isinstance(transition_id, bool) or not recipients:
        raise RuntimeError('Could not build a durable attention notice')
    return Notice(
        number=int(item['number']),
        kind=kind,
        expected_stage=stage,
        transition_id=transition_id,
        title=str(item.get('title') or '')[:300],
        recipients=list(recipients),
    )


def _reconcile_item(
    client: GitHubClient, repo: str, number: int, *, now: dt.datetime
) -> tuple[str, Notice | None] | None:
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    if current.get('state') != 'open':
        # Closing an item is the ultimate resolution: tear down the lifecycle
        # labels so a later reopen can't wake an ancient SLA clock.
        if _ACTION_LABEL in labels:
            _complete(client, repo, number, labels)
            return f'#{number}: completed after the item was closed', None
        return None
    if _ACTION_LABEL not in labels:
        return None
    current_stage = _stage(labels)
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    transition = _transition(events, current_stage)
    if transition is None:
        raise RuntimeError('Could not find the current attention transition')
    transition_at, transition_event = transition
    if _actor(transition_event) != 'github-actions[bot]':
        _complete(client, repo, number, labels)
        return f'#{number}: removed a foreign attention transition', None
    if _closed_since(timeline, transition_at):
        _complete(client, repo, number, labels)
        return f'#{number}: completed after the item was closed', None
    current_stage_label = _STAGE_LABELS[current_stage - 1] if current_stage else None
    for label in labels.intersection(_STAGE_LABELS):
        if label != current_stage_label:
            _remove_label(client, repo, number, label)
    maintainers = _maintainer_assignees(client, repo, current)
    reminder_transition = _transition(events, 1) if current_stage == 2 else None
    acknowledged_since = reminder_transition[0] if reminder_transition is not None else transition_at
    if _acknowledged(timeline, acknowledged_since, maintainers or [_FALLBACK_OWNER]):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer acknowledged the request', None
    recipients = _ensure_recipients(client, repo, current, maintainers, transition)
    if current_stage == 2:
        return (
            f'#{number}: queued terminal triage channel escalation',
            _notice(current, 'escalation', 2, transition, recipients),
        )
    notified_transition = _label_transition(events, _NOTIFIED_LABEL)
    if (
        _NOTIFIED_LABEL not in labels
        or notified_transition is None
        or _actor(notified_transition[1]) != 'github-actions[bot]'
    ):
        if _NOTIFIED_LABEL in labels:
            _remove_label(client, repo, number, _NOTIFIED_LABEL)
        return (
            f'#{number}: queued initial triage channel notice',
            _notice(current, 'initial', current_stage, transition, recipients),
        )
    sla_started_at = max(
        transition_at,
        notified_transition[0],
    )
    if now - sla_started_at < _SLA:
        return None
    if current_stage == 0:
        return (
            f'#{number}: queued triage channel reminder',
            _notice(current, 'reminder', 0, transition, recipients),
        )
    return (
        f'#{number}: queued terminal triage channel escalation',
        _notice(current, 'escalation', 1, transition, recipients),
    )


def _sweep_escalated_item(client: GitHubClient, repo: str, number: int) -> str | None:
    """Wake or retire one dormant escalated item."""
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    if _ACTION_LABEL in labels or _ESCALATED_LABEL not in labels:
        return None
    if current.get('state') != 'open':
        _complete(client, repo, number, labels)
        return f'#{number}: cleared escalation marker after the item was closed'
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    transition = _transition(events, 2)
    if transition is None or _actor(transition[1]) != 'github-actions[bot]':
        _complete(client, repo, number, labels)
        return f'#{number}: removed a foreign attention transition'
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if any(
        _actor(event) != 'github-actions[bot]'
        and (event_time := _event_time(event)) is not None
        and event_time >= transition[0]
        for event in timeline
    ):
        _remove_label(client, repo, number, _ESCALATED_LABEL)
        return f'#{number}: restored attention eligibility after new activity'
    return None


def _active_items(client: GitHubClient, repo: str, *, now: dt.datetime) -> list[dict[str, Any]]:
    """Prioritize channel-undelivered requests, then fill from the oldest active items."""
    query = urllib.parse.quote_plus(f'repo:{repo} is:open label:"{_ACTION_LABEL}" -label:"{_NOTIFIED_LABEL}"')
    first = cast(dict[str, Any], client.get(f'/search/issues?q={query}&sort=updated&order=asc&per_page=1'))
    total = min(int(first.get('total_count') or 0), 1_000)
    priority: list[dict[str, Any]] = []
    if total:
        pages = math.ceil(total / _RECONCILE_LIMIT)
        slot = int(now.timestamp()) // int(_SLA.total_seconds() / 12)
        page = slot % pages + 1
        result = cast(
            dict[str, Any],
            client.get(f'/search/issues?q={query}&sort=updated&order=asc&per_page={_RECONCILE_LIMIT}&page={page}'),
        )
        priority = cast(list[dict[str, Any]], result.get('items') or [])

    encoded = urllib.parse.quote(_ACTION_LABEL, safe='')
    fallback = cast(
        list[dict[str, Any]],
        client.get(
            # state=all so a closed item still completes its lifecycle instead
            # of leaving a dormant clock that a reopen wakes.
            f'/repos/{repo}/issues?state=all&labels={encoded}&sort=updated&direction=asc&per_page={_RECONCILE_LIMIT}'
        ),
    )
    items: dict[int, dict[str, Any]] = {}
    for item in [*priority, *fallback]:
        items.setdefault(int(item['number']), item)
    return list(items.values())[:_RECONCILE_LIMIT]


def reconcile(
    client: GitHubClient, repo: str, *, now: dt.datetime, notices: list[Notice] | None = None
) -> tuple[list[str], list[str]]:
    """Advance a bounded batch of active attention requests.

    Per-item failures are returned rather than raised so that notices queued
    by healthy items always reach the triage channel delivery job.
    """
    ensure_labels(client, repo)
    items = _active_items(client, repo, now=now)
    lines: list[str] = []
    failures: list[str] = []
    for item in items:
        number = int(item['number'])
        try:
            if result := _reconcile_item(client, repo, number, now=now):
                line, notice = result
                lines.append(line)
                if notice is not None and notices is not None:
                    notices.append(notice)
        except (urllib.error.HTTPError, RuntimeError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if len(items) == _RECONCILE_LIMIT:
        lines.append('additional attention items remain for a later rotated batch')
    encoded_escalated = urllib.parse.quote(_ESCALATED_LABEL, safe='')
    dormant = cast(
        list[dict[str, Any]],
        client.get(
            # Recent-first makes renewed activity on an old escalated item
            # visible immediately. Processed items lose the escalation label,
            # so bursts larger than the bound drain over subsequent runs.
            f'/repos/{repo}/issues?state=all&labels={encoded_escalated}'
            f'&sort=updated&direction=desc&per_page={_RECONCILE_LIMIT}'
        ),
    )
    for item in dormant:
        number = int(item['number'])
        if _ACTION_LABEL in _labels(item):
            continue
        try:
            if line := _sweep_escalated_item(client, repo, number):
                lines.append(line)
        except (urllib.error.HTTPError, RuntimeError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    return lines, failures


def _slack_escape(value: str) -> str:
    normalized = ' '.join(value.split())
    for character in '*_~`|\\':
        normalized = normalized.replace(character, '')
    normalized = ' '.join(normalized.split())
    return normalized.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _notice_ref(notice: Notice) -> NoticeRef:
    return NoticeRef(
        number=notice['number'],
        kind=notice['kind'],
        expected_stage=notice['expected_stage'],
        transition_id=notice['transition_id'],
        recipients=notice['recipients'],
    )


def _write_notices(repo: str, notices: Sequence[Notice]) -> None:
    if output_path := os.environ.get('GITHUB_OUTPUT'):
        reasons = {
            'initial': 'triage identified a maintainer as the next actor',
            'reminder': 'there has been no maintainer activity for three days',
            'escalation': 'the reminder has had no maintainer activity for three more days',
        }
        details: list[str] = []
        for notice in notices:
            owners = ', '.join(f'@{_slack_escape(login)}' for login in notice['recipients'])
            title = _slack_escape(notice['title']) or '(untitled)'
            details.append(
                f'• *{notice["kind"].title()}*: '
                f'<https://github.com/{repo}/issues/{notice["number"]}|#{notice["number"]} {title}> — '
                f'owner {owners}; why: {reasons[notice["kind"]]}'
            )
        payload = {
            'text': '\n'.join(
                [
                    '<!channel> *Maintainer attention requested*',
                    *details,
                    '',
                    '*Expected action:* Open each item and make its next maintainer decision there. A reply, review, '
                    'merge, close, or request for changes counts. If no work is needed, say so briefly. Do not remove '
                    'the attention labels; the monitor clears them after maintainer activity.',
                ]
            )
        }
        refs = [_notice_ref(notice) for notice in notices]
        with Path(output_path).open('a', encoding='utf-8') as output:
            output.write(f'has_notices={str(bool(notices)).lower()}\n')
            output.write(f'notice_items={json.dumps(refs, separators=(",", ":"))}\n')
            output.write(f'slack_payload={json.dumps(payload, separators=(",", ":"))}\n')


_LOGIN_PATTERN = re.compile(r'(?=.{1,39}\Z)[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?')
_NOTICE_STAGES: dict[str, frozenset[int]] = {
    'initial': frozenset({0, 1}),
    'reminder': frozenset({0, 1}),
    'escalation': frozenset({1, 2}),
}


def _notice_refs(loaded: object) -> list[NoticeRef]:
    if not isinstance(loaded, Mapping):
        raise ValueError('Notices must contain only an items list')
    data = cast(Mapping[str, object], loaded)
    if set(data) != {'items'} or not isinstance(data['items'], list):
        raise ValueError('Notices must contain only an items list')
    values = cast(list[object], data['items'])
    if len(values) > _RECONCILE_LIMIT:
        raise ValueError('Notices exceed the batch limit')
    notices: list[NoticeRef] = []
    for value in values:
        if not isinstance(value, Mapping) or set(cast(Mapping[object, object], value)) != {
            'number',
            'kind',
            'expected_stage',
            'transition_id',
            'recipients',
        }:
            raise ValueError('Each notice must contain the complete bounded state reference')
        notice = cast(Mapping[str, object], value)
        number = notice['number']
        kind = notice['kind']
        stage = notice['expected_stage']
        transition_id = notice['transition_id']
        recipients = notice['recipients']
        if not isinstance(recipients, list):
            raise ValueError('Notice contains an invalid state reference')
        recipient_values = cast(list[object], recipients)
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number < 1
            or not isinstance(kind, str)
            or kind not in _NOTICE_STAGES
            or not isinstance(stage, int)
            or isinstance(stage, bool)
            or stage not in _NOTICE_STAGES[kind]
            or not isinstance(transition_id, (int, str))
            or isinstance(transition_id, bool)
            or transition_id == ''
            or (isinstance(transition_id, str) and len(transition_id) > 100)
            or not 1 <= len(recipient_values) <= 10
            or any(not isinstance(login, str) or _LOGIN_PATTERN.fullmatch(login) is None for login in recipient_values)
        ):
            raise ValueError('Notice contains an invalid state reference')
        recipient_logins = cast(list[str], recipient_values)
        if len({login.casefold() for login in recipient_logins}) != len(recipient_logins):
            raise ValueError('Notice recipients must be unique')
        notices.append(
            NoticeRef(
                number=number,
                kind=cast(Literal['initial', 'reminder', 'escalation'], kind),
                expected_stage=cast(Literal[0, 1, 2], stage),
                transition_id=transition_id,
                recipients=recipient_logins,
            )
        )
    numbers = [notice['number'] for notice in notices]
    if len(numbers) != len(set(numbers)):
        raise ValueError('Notices must contain unique item numbers')
    return notices


def _notice_state(
    client: GitHubClient, repo: str, notice: NoticeRef
) -> tuple[dict[str, Any], set[str], _Transition, list[str]] | None:
    """Return a notice's exact live state, or `None` if it has changed."""
    number = notice['number']
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    current_stage = _stage(labels)
    resumed_stage = (notice['kind'] == 'reminder' and notice['expected_stage'] == 0 and current_stage == 1) or (
        notice['kind'] == 'escalation' and notice['expected_stage'] == 1 and current_stage == 2
    )
    if (
        current.get('state') != 'open'
        or _ACTION_LABEL not in labels
        or (current_stage != notice['expected_stage'] and not resumed_stage)
    ):
        return None
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    transition = _transition(events, notice['expected_stage'])
    if transition is None or transition[1].get('id') != notice['transition_id']:
        return None
    if resumed_stage:
        resumed_transition = _transition(events, current_stage)
        if (
            resumed_transition is None
            or _actor(resumed_transition[1]) != 'github-actions[bot]'
            or resumed_transition[0] < transition[0]
        ):
            return None
    maintainers = _maintainer_assignees(client, repo, current)
    if not maintainers or {login.casefold() for login in maintainers} != {
        login.casefold() for login in notice['recipients']
    }:
        return None
    latest = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    if (
        latest.get('updated_at') != current.get('updated_at')
        or latest.get('state') != 'open'
        or _labels(latest) != labels
        or {str(assignee['login']).casefold() for assignee in latest.get('assignees', [])}
        != {str(assignee['login']).casefold() for assignee in current.get('assignees', [])}
    ):
        return None
    return latest, labels, transition, maintainers


def _finalize_notice(client: GitHubClient, repo: str, notice: NoticeRef) -> str | None:
    number = notice['number']
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    if current.get('state') != 'open':
        if _ACTION_LABEL in labels:
            _complete(client, repo, number, labels)
        return None
    if _notice_state(client, repo, notice) is None:
        return None
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if (state := _notice_state(client, repo, notice)) is None:
        return None
    _, labels, transition, maintainers = state
    reminder_transition = _transition(timeline, 1) if notice['expected_stage'] == 2 else None
    acknowledged_since = reminder_transition[0] if reminder_transition is not None else transition[0]
    if _closed_since(timeline, transition[0]):
        _complete(client, repo, number, labels)
        return f'#{number}: completed after the item was closed'
    if _acknowledged(timeline, acknowledged_since, maintainers):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer acknowledged the delivered notice'
    if notice['kind'] == 'initial':
        _add_labels(client, repo, number, [_NOTIFIED_LABEL])
        return f'#{number}: recorded initial triage channel notice'
    if notice['kind'] == 'reminder':
        _add_labels(client, repo, number, [_NOTIFIED_LABEL])
        if _stage(labels) == 0:
            _advance_stage(client, repo, number, labels | {_NOTIFIED_LABEL}, 1)
        timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
        if _acknowledged(timeline, acknowledged_since, maintainers):
            _complete(client, repo, number, labels | {_NOTIFIED_LABEL, _PINGED_LABEL})
            return f'#{number}: maintainer acknowledged the delivered notice'
        return f'#{number}: recorded triage channel reminder'
    if _stage(labels) == 1:
        _advance_stage(client, repo, number, labels, 2)
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if _acknowledged(timeline, acknowledged_since, maintainers):
        _complete(client, repo, number, labels | {_ESCALATED_LABEL})
        return f'#{number}: maintainer acknowledged the delivered notice'
    for label in labels.intersection(_STAGE_LABELS):
        if label != _ESCALATED_LABEL:
            _remove_label(client, repo, number, label)
    _remove_label(client, repo, number, _NOTIFIED_LABEL)
    _remove_label(client, repo, number, _ACTION_LABEL)
    return f'#{number}: recorded terminal triage channel escalation'


def finalize_notices(client: GitHubClient, repo: str, notices: Sequence[NoticeRef]) -> list[str]:
    """Advance attention state only after the triage channel delivery succeeds."""
    lines: list[str] = []
    failures: list[str] = []
    for notice in notices:
        number = notice['number']
        try:
            if line := _finalize_notice(client, repo, notice):
                lines.append(line)
        except (urllib.error.HTTPError, RuntimeError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('Failed to finalize attention: ' + '; '.join(failures))
    return lines


def _write_summary(lines: Sequence[str]) -> None:
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(path).open('a', encoding='utf-8') as summary:
            summary.write('## Issue and PR attention monitor\n\n')
            summary.write('\n'.join(f'- {line}' for line in lines) or '- No changes')
            summary.write('\n')


def main() -> int:
    """Build a snapshot, apply decisions, or reconcile reminders."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'apply', 'reconcile', 'finalize'])
    parser.add_argument('--snapshot-path', default='attention-candidates.json')
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token:
        print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
        return 1
    client = GitHubClient(token)
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    now = dt.datetime.now(dt.timezone.utc)
    failures: list[str] = []
    if args.mode == 'snapshot':
        lines = write_snapshot(client, repo, args.snapshot_path, now=now)
    elif args.mode == 'apply':
        if not args.agent_output:
            parser.error('--agent-output is required')
        lines = apply_decisions(client, repo, args.agent_output, args.snapshot_path)
    elif args.mode == 'reconcile':
        notices: list[Notice] = []
        lines, failures = reconcile(client, repo, now=now, notices=notices)
        _write_notices(repo, notices)
    else:
        source = os.environ.get('ATTENTION_NOTICES')
        if source is None:
            parser.error('ATTENTION_NOTICES is required')
        lines = finalize_notices(client, repo, _notice_refs(json.loads(source)))
    _write_summary(lines + [f'failed: {failure}' for failure in failures])
    for line in lines:
        print(line)
    for failure in failures:
        print(f'failed: {failure}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
