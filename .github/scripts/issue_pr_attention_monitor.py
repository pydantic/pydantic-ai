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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from typing_extensions import TypedDict

_API = 'https://api.github.com'
_SLA = dt.timedelta(days=3)
_CANDIDATE_LIMIT = 10
_RECONCILE_LIMIT = 25
_RESPONSE_LIMIT = 5_000_000
_SNAPSHOT_LIMIT = 80_000
_OWNER = 'adtyavrdhn'
_ESCALATION_OWNER = 'DouweM'
_ACTION_LABEL = 'needs-maintainer-action'
_PINGED_LABEL = 'attention-pinged'
_ESCALATED_LABEL = 'attention-escalated'
_STAGE_LABELS = (_PINGED_LABEL, _ESCALATED_LABEL)
_LABELS = {
    _ACTION_LABEL: ('d4c5f9', 'The next meaningful action must come from a maintainer'),
    _PINGED_LABEL: ('fbca04', 'The assigned maintainer has received one reminder'),
    _ESCALATED_LABEL: ('d93f0b', 'The maintainer attention request has been escalated after one reminder'),
}


class Decision(TypedDict):
    """The complete model-controlled surface."""

    item_number: int
    next_actor: Literal['maintainer', 'contributor', 'automation', 'none', 'uncertain']
    confidence: Literal['high', 'medium', 'low']


class Candidate(TypedDict):
    """Trusted snapshot fields needed to revalidate a model decision."""

    number: int
    updated_at: str


class GitHubClient:
    """Small GitHub REST client with bounded response parsing."""

    def __init__(self, token: str) -> None:
        self._token = token

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

    def last_page(self, path: str) -> list[dict[str, Any]]:
        return self.last_pages(path)


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
        reviews = client.last_page(f'/repos/{repo}/pulls/{number}/reviews')
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
    excluded = f'-label:"{_ACTION_LABEL}"'
    query = urllib.parse.quote_plus(f'repo:{repo} is:open updated:<{before} {excluded}')
    first = cast(dict[str, Any], client.get(f'/search/issues?q={query}&sort=updated&order=asc&per_page=1'))
    total = min(int(first.get('total_count') or 0), 1_000)
    if not total:
        return []
    pages = math.ceil(total / _CANDIDATE_LIMIT)
    slot = int(now.timestamp()) // int(_SLA.total_seconds() / 12)
    page = slot % pages + 1
    result = cast(
        dict[str, Any],
        client.get(f'/search/issues?q={query}&sort=updated&order=asc&per_page={_CANDIDATE_LIMIT}&page={page}'),
    )
    return cast(list[dict[str, Any]], result.get('items') or [])


def build_snapshot(client: GitHubClient, repo: str, *, now: dt.datetime) -> dict[str, object]:
    """Build the bounded public input consumed by the sandboxed agent."""
    cutoff = now - _SLA
    candidates: list[dict[str, object]] = []
    for result in _candidate_page(client, repo, now=now):
        number = int(result['number'])
        current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        labels = _labels(current)
        updated_at = str(current['updated_at'])
        if current.get('state') != 'open' or _parse_time(updated_at) > cutoff or _ACTION_LABEL in labels:
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
    if len(json.dumps(snapshot, indent=2).encode()) > _SNAPSHOT_LIMIT:
        raise RuntimeError(f'Attention snapshot exceeds {_SNAPSHOT_LIMIT} bytes')
    return snapshot


def write_snapshot(client: GitHubClient, repo: str, path: str, *, now: dt.datetime) -> list[str]:
    """Write one immutable, size-bounded candidate snapshot."""
    snapshot = build_snapshot(client, repo, now=now)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
    candidates = cast(list[object], snapshot['candidates'])
    return [f'wrote {len(candidates)} attention candidate(s)']


def _snapshot_candidates(path: str) -> dict[int, Candidate]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Snapshot must contain a candidates list')
    data = cast(Mapping[str, object], loaded)
    raw_candidates = data.get('candidates')
    if not isinstance(raw_candidates, list):
        raise ValueError('Snapshot must contain a candidates list')
    candidates: dict[int, Candidate] = {}
    for value in cast(list[object], raw_candidates):
        if not isinstance(value, Mapping):
            raise ValueError('Snapshot candidate must be an object')
        candidate = cast(Mapping[str, object], value)
        number = candidate.get('number')
        updated_at = candidate.get('updated_at')
        if not isinstance(number, int) or number < 1 or number in candidates or not isinstance(updated_at, str):
            raise ValueError('Snapshot candidates must have unique positive numbers and timestamps')
        candidates[number] = Candidate(number=number, updated_at=updated_at)
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
                or str(current.get('updated_at')) != candidates[number]['updated_at']
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
            for label in labels.intersection(_STAGE_LABELS):
                _remove_label(client, repo, number, label)
            _add_labels(client, repo, number, [_ACTION_LABEL])
            _ensure_owner(client, repo, number, current)
            lines.append(f'#{number}: assigned @{_OWNER} and requested maintainer attention')
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('Failed to apply attention: ' + '; '.join(failures))
    return lines


def _ensure_owner(client: GitHubClient, repo: str, number: int, item: Mapping[str, Any]) -> None:
    assignees = {str(value['login']).casefold() for value in item.get('assignees', [])}
    if _OWNER.casefold() in assignees:
        return
    assigned = cast(dict[str, Any], client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [_OWNER]}))
    if _OWNER.casefold() not in {str(value['login']).casefold() for value in assigned.get('assignees', [])}:
        raise RuntimeError(f'GitHub did not assign @{_OWNER}')


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


def _reminder(stage: Literal[1, 2]) -> str:
    if stage == 1:
        return f'@{_OWNER} this still needs a maintainer decision. Could you take a look?'
    return f'@{_ESCALATION_OWNER} this still needs a decision after a reminder to @{_OWNER}.'


def _event_time(event: Mapping[str, Any]) -> dt.datetime | None:
    value = event.get('created_at') or event.get('submitted_at')
    return _parse_time(str(value)) if value else None


def _transition(
    timeline: Sequence[dict[str, Any]], stage: Literal[0, 1, 2]
) -> tuple[dt.datetime, dict[str, Any]] | None:
    label = _ACTION_LABEL if stage == 0 else _STAGE_LABELS[stage - 1]
    transitions = [
        (time, event)
        for event in timeline
        if event.get('event') == 'labeled'
        and isinstance(event.get('label'), Mapping)
        and cast(Mapping[str, object], event['label']).get('name') == label
        and (time := _event_time(event)) is not None
    ]
    return max(transitions, key=lambda value: value[0], default=None)


def _actor(event: Mapping[str, Any]) -> str:
    value = event.get('actor') or event.get('user')
    return str(cast(Mapping[str, object], value).get('login') or '') if isinstance(value, Mapping) else ''


def _acknowledged(timeline: Sequence[dict[str, Any]], since: dt.datetime) -> bool:
    owners = {_OWNER.casefold(), _ESCALATION_OWNER.casefold()}
    return any(
        (_actor(event).casefold() in owners or event.get('author_association') in {'MEMBER', 'OWNER'})
        and event.get('event') in {'commented', 'reviewed'}
        and (event_time := _event_time(event)) is not None
        and event_time >= since
        for event in timeline
    )


def _fixed_comment_exists(timeline: Sequence[dict[str, Any]], body: str, since: dt.datetime) -> bool:
    return any(
        _actor(event) == 'github-actions[bot]'
        and event.get('event') == 'commented'
        and event.get('body') == body
        and (event_time := _event_time(event)) is not None
        and event_time >= since
        for event in timeline
    )


def _complete(client: GitHubClient, repo: str, number: int, labels: set[str]) -> None:
    _remove_label(client, repo, number, _ACTION_LABEL)
    for label in labels.intersection(_STAGE_LABELS):
        _remove_label(client, repo, number, label)


def _reconcile_item(client: GitHubClient, repo: str, number: int, *, now: dt.datetime) -> str | None:
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    if current.get('state') != 'open' or _ACTION_LABEL not in labels:
        return None
    current_stage = _stage(labels)
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=3)
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    transition = _transition(events, current_stage)
    if transition is None:
        raise RuntimeError('Could not find the current attention transition')
    transition_at, transition_event = transition
    if _actor(transition_event) != 'github-actions[bot]':
        _complete(client, repo, number, labels)
        return f'#{number}: removed a foreign attention transition'
    _ensure_owner(client, repo, number, current)
    if _acknowledged(timeline, transition_at):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer acknowledged the request'
    if current_stage > 0:
        current_body = _reminder(cast(Literal[1, 2], current_stage))
        if not _fixed_comment_exists(timeline, current_body, transition_at):
            client.post(f'/repos/{repo}/issues/{number}/comments', {'body': current_body})
            if current_stage == 2:
                _remove_label(client, repo, number, _ACTION_LABEL)
            return f'#{number}: restored reminder {current_stage}'
    if current_stage == 2:
        _remove_label(client, repo, number, _ACTION_LABEL)
        return f'#{number}: completed terminal escalation'
    if now - transition_at < _SLA:
        return None
    next_stage = current_stage + 1
    _advance_stage(client, repo, number, labels, next_stage)
    client.post(f'/repos/{repo}/issues/{number}/comments', {'body': _reminder(next_stage)})
    if next_stage == 2:
        _remove_label(client, repo, number, _ACTION_LABEL)
    return f'#{number}: posted reminder {next_stage}'


def reconcile(client: GitHubClient, repo: str, *, now: dt.datetime) -> list[str]:
    """Advance a bounded batch of active attention requests."""
    ensure_labels(client, repo)
    encoded = urllib.parse.quote(_ACTION_LABEL, safe='')
    items = cast(
        list[dict[str, Any]],
        client.get(
            f'/repos/{repo}/issues?state=open&labels={encoded}&sort=updated&direction=asc&per_page={_RECONCILE_LIMIT}'
        ),
    )
    lines: list[str] = []
    failures: list[str] = []
    for item in items:
        number = int(item['number'])
        try:
            if line := _reconcile_item(client, repo, number, now=now):
                lines.append(line)
        except (urllib.error.HTTPError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if len(items) == _RECONCILE_LIMIT:
        lines.append('additional attention items remain for a later rotated batch')
    if failures:
        raise RuntimeError('Failed to reconcile attention: ' + '; '.join(failures))
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
    parser.add_argument('mode', choices=['snapshot', 'apply', 'reconcile'])
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
    if args.mode == 'snapshot':
        lines = write_snapshot(client, repo, args.snapshot_path, now=now)
    elif args.mode == 'apply':
        if not args.agent_output:
            parser.error('--agent-output is required')
        lines = apply_decisions(client, repo, args.agent_output, args.snapshot_path)
    else:
        lines = reconcile(client, repo, now=now)
    _write_summary(lines)
    for line in lines:
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
