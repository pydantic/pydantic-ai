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
_RECENT_CANDIDATE_LIMIT = _CANDIDATE_LIMIT // 2
_BACKLOG_CANDIDATE_LIMIT = _CANDIDATE_LIMIT - _RECENT_CANDIDATE_LIMIT
_RECONCILE_LIMIT = 25
_ACTIVE_OPEN_LIMIT = 20
_CLOSED_CLEANUP_LIMIT = _RECONCILE_LIMIT - _ACTIVE_OPEN_LIMIT
_EVENT_PAGE_LIMIT = 10
_COMMENT_PAGE_LIMIT = 10
_COLLABORATOR_PAGE_LIMIT = 10
_RESPONSE_LIMIT = 5_000_000
_SNAPSHOT_LIMIT = 80_000
_FALLBACK_OWNER = 'adtyavrdhn'
_ACTION_LABEL = 'needs-maintainer-action'
_PINGED_LABEL = 'attention-pinged'
_ESCALATED_LABEL = 'attention-escalated'
_DELIVERED_LABEL = 'attention-delivered'
_STAGE_LABELS = (_PINGED_LABEL, _ESCALATED_LABEL)
_LIFECYCLE_LABELS = (*_STAGE_LABELS, _DELIVERED_LABEL)
_LABELS = {
    _ACTION_LABEL: ('d4c5f9', 'The next meaningful action must come from a maintainer'),
    _PINGED_LABEL: ('fbca04', 'The assigned maintainer has received one reminder'),
    _ESCALATED_LABEL: ('d93f0b', 'The maintainer attention request has been escalated after one reminder'),
    _DELIVERED_LABEL: ('ededed', 'A delivered channel escalation is waiting for GitHub state cleanup'),
}


class Decision(TypedDict):
    """The complete model-controlled surface."""

    item_number: int
    next_actor: Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain']
    confidence: Literal['high', 'medium', 'low']


class Notice(TypedDict):
    """One fixed channel notification."""

    number: int
    kind: Literal['reminder', 'escalation']
    expected_stage: Literal[0, 1, 2]
    transition_id: int | str
    title: str
    recipients: list[str]


class NoticeRef(TypedDict):
    """The item and state that a delivered channel notice described."""

    number: int
    expected_stage: Literal[0, 1, 2]
    transition_id: int | str
    recipients: list[str]


class GitHubClient:
    """Small GitHub REST client with bounded response parsing."""

    def __init__(self, token: str) -> None:
        self._token = token
        self._maintainers: dict[str, dict[str, str]] = {}

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

    def delete(self, path: str, payload: Mapping[str, object] | None = None) -> Any:
        return self.request('DELETE', path, payload)

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

    def pages(self, path: str, *, count: int) -> Iterator[list[dict[str, Any]]]:
        """Yield an ascending collection without silently truncating it."""
        separator = '&' if '?' in path else '?'
        page_path = f'{path}{separator}per_page=100&page=1'
        for _ in range(count):
            values, links = self._request('GET', page_path)
            yield cast(list[dict[str, Any]], values)
            if not (page_path := _link_path(links, 'next')):
                return
        raise RuntimeError(f'GitHub collection exceeds the {count}-page safety limit')

    def maintainer_logins(self, repo: str) -> Mapping[str, str]:
        if repo not in self._maintainers:
            maintainers: dict[str, str] = {}
            for page in self.pages(
                f'/repos/{repo}/collaborators?affiliation=all',
                count=_COLLABORATOR_PAGE_LIMIT,
            ):
                for collaborator in page:
                    permissions = collaborator.get('permissions')
                    if (
                        isinstance(permissions, Mapping)
                        and cast(Mapping[str, object], permissions).get('push') is True
                        and (login := str(collaborator.get('login') or ''))
                    ):
                        maintainers[login.casefold()] = login
            self._maintainers[repo] = maintainers
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


def _rotated_search(
    client: GitHubClient,
    query: str,
    *,
    order: Literal['asc', 'desc'],
    limit: int,
    slot: int,
) -> list[dict[str, Any]]:
    encoded = urllib.parse.quote_plus(query)
    first = cast(
        dict[str, Any],
        client.get(f'/search/issues?q={encoded}&sort=updated&order={order}&per_page=1'),
    )
    total = min(int(first.get('total_count') or 0), 1_000)
    if not total:
        return []
    page = slot % math.ceil(total / limit) + 1
    result = cast(
        dict[str, Any],
        client.get(f'/search/issues?q={encoded}&sort=updated&order={order}&per_page={limit}&page={page}'),
    )
    return cast(list[dict[str, Any]], result.get('items') or [])


def _candidate_page(client: GitHubClient, repo: str, *, now: dt.datetime) -> list[dict[str, Any]]:
    before = (now - _SLA).date().isoformat()
    # An escalated item stays dormant until the reconcile sweep sees real
    # activity and removes the marker; only then may a fresh lifecycle start.
    excluded = f'-label:"{_ACTION_LABEL}" -label:"{_ESCALATED_LABEL}"'
    base_query = f'repo:{repo} is:open updated:<{before} {excluded}'
    slot = int(now.timestamp()) // int(_SLA.total_seconds() / 12)
    recent_after = (now - _RECENT_ACTIVITY_WINDOW).date().isoformat()
    recent = _rotated_search(
        client,
        f'{base_query} updated:>={recent_after}',
        order='desc',
        limit=_RECENT_CANDIDATE_LIMIT,
        slot=slot,
    )
    backlog = _rotated_search(
        client,
        base_query,
        order='asc',
        limit=_BACKLOG_CANDIDATE_LIMIT,
        slot=slot,
    )
    candidates: dict[int, dict[str, Any]] = {}
    for item in [*recent, *backlog]:
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


def _maintainer_assignees(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> list[str]:
    maintainers = client.maintainer_logins(repo)
    return sorted(
        (
            maintainers[login.casefold()]
            for assignee in item.get('assignees', [])
            if (login := str(assignee['login'])) and login.casefold() in maintainers
        ),
        key=str.casefold,
    )


def _first_maintainer_in_discussion(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> str | None:
    """Return the first current maintainer who authored or discussed an issue."""
    if 'pull_request' in item:
        return None
    maintainers = client.maintainer_logins(repo)

    def maintainer(entry: Mapping[str, Any]) -> str | None:
        login = _login(entry)
        return maintainers.get(login.casefold()) if login else None

    if author := maintainer(item):
        return author
    comments = int(item.get('comments') or 0)
    if comments:
        pages = min(_last_page(comments, 100), _COMMENT_PAGE_LIMIT)
        for page in client.pages(f'/repos/{repo}/issues/{int(item["number"])}/comments', count=pages):
            for comment in page:
                if login := maintainer(comment):
                    return login
    return None


def _ensure_recipients(
    client: GitHubClient,
    repo: str,
    item: Mapping[str, Any],
) -> list[str]:
    number = int(item['number'])
    owner = _first_maintainer_in_discussion(client, repo, item)

    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    if current.get('state') != 'open' or _ACTION_LABEL not in _labels(current):
        raise RuntimeError('Attention state changed during owner selection')
    if current.get('updated_at') != item.get('updated_at'):
        owner = _first_maintainer_in_discussion(client, repo, current)
    current_maintainers = _maintainer_assignees(client, repo, current)
    if current_maintainers:
        if owner is None:
            return current_maintainers

    owner = owner or _FALLBACK_OWNER
    if owner.casefold() not in {login.casefold() for login in current_maintainers}:
        client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [owner]})
    if other_owners := [login for login in current_maintainers if login.casefold() != owner.casefold()]:
        client.delete(f'/repos/{repo}/issues/{number}/assignees', {'assignees': other_owners})

    assigned = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    if assigned.get('state') != 'open' or _ACTION_LABEL not in _labels(assigned):
        raise RuntimeError('Attention state changed during owner assignment')
    assigned_maintainers = _maintainer_assignees(client, repo, assigned)
    if [login.casefold() for login in assigned_maintainers] != [owner.casefold()]:
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
            attention_item = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if attention_item.get('state') != 'open' or _ACTION_LABEL not in _labels(attention_item):
                raise RuntimeError('Attention state changed while applying the request')
            recipients = _ensure_recipients(client, repo, attention_item)
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


def _transition(
    timeline: Sequence[dict[str, Any]], stage: Literal[0, 1, 2]
) -> tuple[dt.datetime, dict[str, Any]] | None:
    label = _ACTION_LABEL if stage == 0 else _STAGE_LABELS[stage - 1]
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


def _actor(event: Mapping[str, Any]) -> str:
    value = event.get('actor') or event.get('user')
    return str(cast(Mapping[str, object], value).get('login') or '') if isinstance(value, Mapping) else ''


# `mentioned` and `subscribed` can be generated as activity side effects, so
# they must never count as acknowledgement.
_NON_ACK_EVENTS = frozenset({'mentioned', 'subscribed'})
_ACK_ASSOCIATIONS = frozenset({'MEMBER', 'OWNER', 'COLLABORATOR'})


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


def _complete(client: GitHubClient, repo: str, number: int, labels: set[str]) -> None:
    for label in labels.intersection(_LIFECYCLE_LABELS):
        _remove_label(client, repo, number, label)
    _remove_label(client, repo, number, _ACTION_LABEL)


def _transition_id(transition: tuple[dt.datetime, dict[str, Any]]) -> int | str:
    transition_id = transition[1].get('id')
    if not isinstance(transition_id, (int, str)) or isinstance(transition_id, bool):
        raise RuntimeError('Could not build a durable attention notice')
    return transition_id


def _notice(
    item: Mapping[str, Any],
    kind: Literal['reminder', 'escalation'],
    stage: Literal[0, 1, 2],
    transition: tuple[dt.datetime, dict[str, Any]],
    recipients: Sequence[str],
) -> Notice:
    return Notice(
        number=int(item['number']),
        kind=kind,
        expected_stage=stage,
        transition_id=_transition_id(transition),
        title=str(item.get('title') or '')[:300],
        recipients=list(recipients),
    )


def _notice_if_current(
    client: GitHubClient,
    repo: str,
    number: int,
    kind: Literal['reminder', 'escalation'],
    stage: Literal[0, 1, 2],
    transition_id: int | str,
    recipients: Sequence[str],
) -> Notice | None:
    """Build a notice only if its transition and owners are still live."""
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    current_transition = _transition(events, stage)
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    maintainers = _maintainer_assignees(client, repo, current)
    if (
        current.get('state') != 'open'
        or _ACTION_LABEL not in labels
        or _stage(labels) != stage
        or not {login.casefold() for login in recipients} <= {login.casefold() for login in maintainers}
    ):
        return None
    if (
        current_transition is None
        or current_transition[1].get('id') != transition_id
        or _actor(current_transition[1]) != 'github-actions[bot]'
    ):
        return None
    acknowledged_transition = _transition(events, 1) if stage == 2 else current_transition
    acknowledged_since = acknowledged_transition[0] if acknowledged_transition is not None else current_transition[0]
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if _closed_since(timeline, current_transition[0]) or _acknowledged(timeline, acknowledged_since, recipients):
        return None
    return _notice(current, kind, stage, current_transition, recipients)


def _finish_delivered_escalation(client: GitHubClient, repo: str, number: int) -> None:
    """Finish a terminal delivery while preserving its dormant marker."""
    _add_labels(client, repo, number, [_ESCALATED_LABEL])
    _remove_label(client, repo, number, _ACTION_LABEL)
    _remove_label(client, repo, number, _PINGED_LABEL)
    _remove_label(client, repo, number, _DELIVERED_LABEL)


def _reconcile_item(
    client: GitHubClient,
    repo: str,
    number: int,
    *,
    now: dt.datetime,
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
    if _DELIVERED_LABEL in labels:
        # The explicit receipt is written only after channel delivery. Removing
        # action first makes a retry safe: cleanup cannot post it again.
        _finish_delivered_escalation(client, repo, number)
        return f'#{number}: finished delivered channel escalation', None
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
    recipients = _ensure_recipients(client, repo, current)
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if _closed_since(timeline, transition_at) or _acknowledged(timeline, acknowledged_since, recipients):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer acknowledged the request', None
    # Stage 2 is the existing durable "terminal Slack delivery pending" state.
    # Keeping that meaning makes the channel cutover safe for in-flight items.
    if current_stage == 2:
        notice = _notice_if_current(
            client,
            repo,
            number,
            'escalation',
            current_stage,
            _transition_id(transition),
            recipients,
        )
        return (f'#{number}: queued channel escalation', notice) if notice is not None else None
    if now - transition_at < _SLA:
        return None
    if current_stage == 0:
        notice = _notice_if_current(
            client,
            repo,
            number,
            'reminder',
            current_stage,
            _transition_id(transition),
            recipients,
        )
        return (f'#{number}: queued channel reminder', notice) if notice is not None else None
    notice = _notice_if_current(
        client,
        repo,
        number,
        'escalation',
        current_stage,
        _transition_id(transition),
        recipients,
    )
    return (f'#{number}: queued channel escalation', notice) if notice is not None else None


def _sweep_escalated_item(client: GitHubClient, repo: str, number: int) -> str | None:
    """Wake or retire one dormant escalated item."""
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    if _ACTION_LABEL in labels or _ESCALATED_LABEL not in labels:
        return None
    if _DELIVERED_LABEL in labels:
        _remove_label(client, repo, number, _DELIVERED_LABEL)
        labels.remove(_DELIVERED_LABEL)
    if _PINGED_LABEL in labels:
        _remove_label(client, repo, number, _PINGED_LABEL)
        labels.remove(_PINGED_LABEL)
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


def reconcile(
    client: GitHubClient, repo: str, *, now: dt.datetime, notices: list[Notice] | None = None
) -> tuple[list[str], list[str]]:
    """Advance a bounded batch of active attention requests.

    Per-item failures are returned rather than raised so that notices queued
    by healthy items always reach the Slack delivery job.
    """
    ensure_labels(client, repo)
    slot = int(now.timestamp()) // int(_SLA.total_seconds() / 12)
    closed = _rotated_search(
        client,
        f'repo:{repo} is:closed label:"{_ACTION_LABEL}"',
        order='asc',
        limit=_CLOSED_CLEANUP_LIMIT,
        slot=slot,
    )
    active = _rotated_search(
        client,
        f'repo:{repo} is:open label:"{_ACTION_LABEL}"',
        order='asc',
        limit=_ACTIVE_OPEN_LIMIT,
        slot=slot,
    )
    items = [*closed, *active]
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
    if len(closed) == _CLOSED_CLEANUP_LIMIT or len(active) == _ACTIVE_OPEN_LIMIT:
        lines.append('additional attention items remain for a later rotated batch')
    encoded_escalated = urllib.parse.quote(_ESCALATED_LABEL, safe='')
    dormant = cast(
        list[dict[str, Any]],
        client.get(
            # state=all so a dormant item closed while escalated still sheds
            # its marker instead of carrying it forever.
            f'/repos/{repo}/issues?state=all&labels={encoded_escalated}'
            # Recent-first ensures that renewed activity on an old escalated
            # issue cannot sit behind the oldest 25 dormant items.
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


def _write_notices(repo: str, notices: Sequence[Notice]) -> None:
    if output_path := os.environ.get('GITHUB_OUTPUT'):
        reasons = {
            'reminder': 'no maintainer has acted for three days',
            'escalation': 'the previous reminder has had no maintainer response for three more days',
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
                    '*Expected action:* Open each item and make its next maintainer decision there. Reply, review, '
                    'merge, close, or request changes as appropriate. If no work is needed, say so briefly. Do not '
                    'remove the attention labels; the monitor clears them after maintainer activity.',
                ]
            )
        }
        refs = [
            NoticeRef(
                number=notice['number'],
                expected_stage=notice['expected_stage'],
                transition_id=notice['transition_id'],
                recipients=notice['recipients'],
            )
            for notice in notices
        ]
        with Path(output_path).open('a', encoding='utf-8') as output:
            output.write(f'has_notices={str(bool(notices)).lower()}\n')
            output.write(f'notice_items={json.dumps(refs, separators=(",", ":"))}\n')
            output.write(f'slack_payload={json.dumps(payload, separators=(",", ":"))}\n')


_LOGIN_PATTERN = re.compile(r'(?=.{1,39}\Z)[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?')


def _notice_refs(loaded: object) -> list[NoticeRef]:
    if not isinstance(loaded, Mapping):
        raise ValueError('Notices must contain only an items list')
    data = cast(Mapping[str, object], loaded)
    if set(data) != {'items'} or not isinstance(data['items'], list):
        raise ValueError('Notices must contain only an items list')
    values = cast(list[object], data['items'])
    notices: list[NoticeRef] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError('Notice has an invalid shape')
        notice = cast(Mapping[str, object], value)
        if set(notice) != {'number', 'expected_stage', 'transition_id', 'recipients'}:
            raise ValueError('Notice has an invalid shape')
        number = notice['number']
        stage = notice['expected_stage']
        transition_id = notice['transition_id']
        recipients = notice['recipients']
        recipient_values = cast(list[object], recipients) if isinstance(recipients, list) else []
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number < 1
            or not isinstance(stage, int)
            or isinstance(stage, bool)
            or stage not in {0, 1, 2}
            or not isinstance(transition_id, (int, str))
            or isinstance(transition_id, bool)
            or (isinstance(transition_id, int) and transition_id < 1)
            or (isinstance(transition_id, str) and not 1 <= len(transition_id) <= 100)
            or not isinstance(recipients, list)
            or not 1 <= len(recipient_values) <= 10
            or any(not isinstance(login, str) or not _LOGIN_PATTERN.fullmatch(login) for login in recipient_values)
            or len({cast(str, login).casefold() for login in recipient_values}) != len(recipient_values)
        ):
            raise ValueError('Notice has invalid values')
        notices.append(
            NoticeRef(
                number=number,
                expected_stage=cast(Literal[0, 1, 2], stage),
                transition_id=transition_id,
                recipients=cast(list[str], recipient_values),
            )
        )
    if len(notices) > _RECONCILE_LIMIT or len({notice['number'] for notice in notices}) != len(notices):
        raise ValueError('Notices must be unique and within the batch limit')
    return notices


def prepare_notices(client: GitHubClient, repo: str, notices: Sequence[NoticeRef]) -> list[Notice]:
    """Revalidate notices immediately before their channel delivery."""
    prepared: list[Notice] = []
    for notice in notices:
        stage = notice['expected_stage']
        kind: Literal['reminder', 'escalation'] = 'reminder' if stage == 0 else 'escalation'
        if live := _notice_if_current(
            client,
            repo,
            notice['number'],
            kind,
            stage,
            notice['transition_id'],
            notice['recipients'],
        ):
            prepared.append(live)
    return prepared


def _closed_since(timeline: Sequence[dict[str, Any]], since: dt.datetime) -> bool:
    return any(
        event.get('event') == 'closed' and (event_time := _event_time(event)) is not None and event_time >= since
        for event in timeline
    )


def _finalize_notice(
    client: GitHubClient,
    repo: str,
    notice: NoticeRef,
) -> str | None:
    number = notice['number']
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    stage = _stage(labels)
    if current.get('state') != 'open' or _ACTION_LABEL not in labels or stage != notice['expected_stage']:
        return None
    maintainers = _maintainer_assignees(client, repo, current)
    if not {login.casefold() for login in notice['recipients']} <= {login.casefold() for login in maintainers}:
        return None
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    transition = _transition(events, stage)
    if (
        transition is None
        or transition[1].get('id') != notice['transition_id']
        or _actor(transition[1]) != 'github-actions[bot]'
    ):
        return None
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if _closed_since(timeline, transition[0]) or _acknowledged(timeline, transition[0], notice['recipients']):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer activity completed the delivered notice'

    kind: Literal['reminder', 'escalation'] = 'reminder' if stage == 0 else 'escalation'
    if (
        _notice_if_current(
            client,
            repo,
            number,
            kind,
            stage,
            _transition_id(transition),
            notice['recipients'],
        )
        is None
    ):
        return None

    if stage == 0:
        _advance_stage(client, repo, number, labels, 1)
    else:
        # Record delivery before terminal cleanup so a later GitHub failure
        # cannot make reconciliation post the escalation again.
        _add_labels(client, repo, number, [_DELIVERED_LABEL])
        _finish_delivered_escalation(client, repo, number)

    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    completed_labels = labels | ({_PINGED_LABEL} if stage == 0 else {_ESCALATED_LABEL, _DELIVERED_LABEL})
    if _closed_since(timeline, transition[0]) or _acknowledged(timeline, transition[0], notice['recipients']):
        _complete(client, repo, number, completed_labels)
        return f'#{number}: maintainer activity completed the delivered notice'
    return f'#{number}: recorded channel {kind}'


def finalize_notices(client: GitHubClient, repo: str, notices: Sequence[NoticeRef]) -> list[str]:
    """Advance attention state only after the channel delivery succeeds."""
    lines: list[str] = []
    failures: list[str] = []
    for notice in notices:
        number = notice['number']
        try:
            if line := _finalize_notice(client, repo, notice):
                lines.append(line)
        except (urllib.error.HTTPError, RuntimeError) as exc:
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
    parser.add_argument('mode', choices=['snapshot', 'apply', 'reconcile', 'prepare', 'finalize'])
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
    elif args.mode == 'prepare':
        source = os.environ.get('ATTENTION_NOTICES')
        if source is None:
            parser.error('ATTENTION_NOTICES is required')
        notices = prepare_notices(client, repo, _notice_refs(json.loads(source)))
        _write_notices(repo, notices)
        lines = [f'prepared {len(notices)} current attention notice(s)']
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
