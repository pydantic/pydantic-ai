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
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path

# Stdlib-only imports: production invokes this script with the runner's bare
# `python`, which has no third-party packages installed. The repo-wide ban on
# `typing.TypedDict` exists for pydantic validation on Python 3.10/3.11, and
# this script uses no pydantic.
from typing import Any, Literal, TypedDict, cast  # noqa: TID251

_API = 'https://api.github.com'
_SLA = dt.timedelta(days=3)
_RESURFACE_AFTER = dt.timedelta(days=7)
_RECENT_ACTIVITY_WINDOW = dt.timedelta(days=45)
_CANDIDATE_LIMIT = 10
_RECENT_CANDIDATE_LIMIT = _CANDIDATE_LIMIT // 2
_BACKLOG_CANDIDATE_LIMIT = _CANDIDATE_LIMIT - _RECENT_CANDIDATE_LIMIT
_RECONCILE_LIMIT = 25
_ACTIVE_OPEN_LIMIT = 20
_CLOSED_CLEANUP_LIMIT = _RECONCILE_LIMIT - _ACTIVE_OPEN_LIMIT
_EVENT_PAGE_LIMIT = 10
_COMMENT_PAGE_LIMIT = 10
# `admin`/`write`/`read`/`none` are the only values the permission field returns;
# `maintain` and `triage` appear in `role_name` and collapse to `write`/`read` here.
_MAINTAINER_PERMISSIONS = frozenset({'admin', 'maintain', 'write'})
# Probing a discussion costs one request per distinct participant, which is
# unbounded in principle. Each sweep gets its own quota so a busy item cannot
# starve later ones, under a run-wide ceiling that keeps the whole pass inside
# the token's hourly rate limit. See `_MaintainerProbe` and `maintainer_login`.
_ITEM_PROBE_LIMIT = 40
_RUN_PROBE_LIMIT = 400
_RESPONSE_LIMIT = 5_000_000
_SNAPSHOT_LIMIT = 80_000
_WEEKLY_ITEM_LIMIT = 3
_LEGACY_ITEM_LIMIT = 2
_WEEKLY_TEXT_LIMIT = 30_000
REPOSITORIES = frozenset({'pydantic/pydantic-ai', 'pydantic/pydantic-ai-harness'})
MAINTAINER_OWNERS = ('adtyavrdhn', 'dsfaccini', 'DouweM', 'mpfaffenberger')
ROUTING_RECOVERY_EPOCH = '2026-08-18'
_MAINTAINER_NAMES = {
    'adtyavrdhn': 'Aditya',
    'dsfaccini': 'David SF',
    'DouweM': 'Douwe',
    'mpfaffenberger': 'Mike',
}
_FALLBACK_OWNER = MAINTAINER_OWNERS[0]
_ACTION_LABEL = 'needs-maintainer-action'
_PINGED_LABEL = 'attention-pinged'
_ESCALATED_LABEL = 'attention-escalated'
_DELIVERED_LABEL = 'attention-delivered'
_STAGE_LABELS = (_PINGED_LABEL, _ESCALATED_LABEL)
_LIFECYCLE_LABELS = (*_STAGE_LABELS, _DELIVERED_LABEL)
_LABELS = {
    _ACTION_LABEL: ('d4c5f9', 'The next meaningful action must come from a maintainer'),
    _PINGED_LABEL: ('fbca04', 'The assigned maintainer has received one reminder'),
    _ESCALATED_LABEL: ('d93f0b', 'The maintainer attention request is cooling down after escalation'),
    _DELIVERED_LABEL: ('ededed', 'A delivered channel escalation is waiting for GitHub state cleanup'),
}
_SLACK_MENTION = re.compile(r'<@[UW][A-Z0-9]+>')
_SEARCH_SUMMARY_QUERY = """
query AttentionSearch($query: String!, $first: Int!) {
  search(query: $query, type: ISSUE, first: $first) {
    issueCount
    nodes {
      ... on Issue { number createdAt }
      ... on PullRequest { number createdAt }
    }
  }
}
"""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never forward a GitHub bearer token through an HTTP redirect."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


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
    status: str


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
        self._maintainers: dict[tuple[str, str], str | None] = {}
        self._probes = 0

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
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
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

    def first_pages(self, path: str, *, count: int) -> tuple[list[dict[str, Any]], bool]:
        """Return up to `count` oldest pages, and whether that was all of them.

        A longer collection is truncated rather than refused, so a huge thread
        costs a bounded prefix instead of aborting the run. The flag lets the
        caller tell "nobody was there" from "we did not get to look".
        """
        separator = '&' if '?' in path else '?'
        page_path = f'{path}{separator}per_page=100&page=1'
        entries: list[dict[str, Any]] = []
        for _ in range(count):
            values, links = self._request('GET', page_path)
            entries.extend(cast(list[dict[str, Any]], values))
            if not (page_path := _link_path(links, 'next')):
                return entries, True
        return entries, False

    def knows_maintainer(self, repo: str, login: str) -> bool:
        """Whether `maintainer_login` can answer for `login` without a request."""
        return (repo, login.casefold()) in self._maintainers

    def spend_probe(self) -> bool:
        """Claim one of the run's speculative lookups, or report it unavailable."""
        if self._probes >= _RUN_PROBE_LIMIT:
            return False
        self._probes += 1
        return True

    def maintainer_login(self, repo: str, login: str, *, refresh: bool = False) -> str | None:
        """Return `login` when it can push to `repo`, resolved one user at a time.

        The collaborator *list* endpoint looks cheaper but is wrong here: it only
        reports collaborators the caller can see, and the workflow token cannot
        see organization members whose membership is private. Almost every
        maintainer on this repository is such a member, so the list silently
        demotes them to non-maintainers and every item falls to the fallback
        owner. This per-user endpoint reports them regardless of visibility.

        The answer is always exact. Speculative sweeps over a discussion go
        through `_MaintainerProbe`, which rations how many *new* logins they may
        resolve; a login already in the cache costs nothing and is never
        rationed.
        """
        key = (repo, login.casefold())
        if refresh or key not in self._maintainers:
            encoded = urllib.parse.quote(login, safe='')
            try:
                permission = cast(
                    Mapping[str, object], self.get(f'/repos/{repo}/collaborators/{encoded}/permission')
                ).get('permission')
            except urllib.error.HTTPError as exc:
                exc.close()
                if exc.code != 404:
                    raise
                permission = None
            self._maintainers[key] = login if permission in _MAINTAINER_PERMISSIONS else None
        return self._maintainers[key]


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
    cutoff_date = (now - _SLA).date()
    # An escalated item cools down outside classification. Reconciliation
    # either wakes it after new activity or returns it to the active queue.
    excluded = f'-label:"{_ACTION_LABEL}" -label:"{_ESCALATED_LABEL}"'
    base_query = f'repo:{repo} is:open {excluded}'
    slot = int(now.timestamp()) // int(_SLA.total_seconds() / 12)
    recent_after = (now - _RECENT_ACTIVITY_WINDOW).date()
    stale_through = cutoff_date - dt.timedelta(days=1)
    recent = _rotated_search(
        client,
        # GitHub Search does not intersect repeated `updated:` qualifiers; a
        # single range is required or the lower bound silently wins.
        f'{base_query} updated:{recent_after.isoformat()}..{stale_through.isoformat()}',
        order='desc',
        limit=_RECENT_CANDIDATE_LIMIT,
        slot=slot,
    )
    backlog = _rotated_search(
        client,
        f'{base_query} updated:<{recent_after.isoformat()}',
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
    return sorted(
        (
            maintainer
            for assignee in item.get('assignees', [])
            if (login := str(assignee['login'])) and (maintainer := client.maintainer_login(repo, login))
        ),
        key=str.casefold,
    )


class _MaintainerProbe:
    """One deduplicated maintainer sweep over a single item's participants.

    Each sweep carries its own quota, so a discussion full of community logins
    cannot spend the capacity the next item needs.
    """

    def __init__(self, client: GitHubClient, repo: str) -> None:
        self._client = client
        self._repo = repo
        self._seen: set[str] = set()
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """Whether *this* sweep left a participant unchecked, so absence proves nothing."""
        return self._exhausted

    def login(self, login: str) -> str | None:
        if not login:
            return None
        # A login the run already resolved is free, so it neither consumes a
        # quota nor makes this sweep inconclusive.
        if self._client.knows_maintainer(self._repo, login):
            return self._client.maintainer_login(self._repo, login)
        if (key := login.casefold()) not in self._seen and len(self._seen) >= _ITEM_PROBE_LIMIT:
            self._exhausted = True
            return None
        if not self._client.spend_probe():
            self._exhausted = True
            return None
        self._seen.add(key)
        return self._client.maintainer_login(self._repo, login)

    def entry(self, entry: Mapping[str, Any]) -> str | None:
        return self.login(_login(entry))


def _discussion(client: GitHubClient, repo: str, item: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Return an item's replies oldest first, across every surface a maintainer uses.

    On a pull request most maintainer engagement arrives as a review or a code
    comment, neither of which appears under the issue comments endpoint.
    """
    number = int(item['number'])
    paths = [f'/repos/{repo}/issues/{number}/comments']
    if 'pull_request' in item:
        paths += [f'/repos/{repo}/pulls/{number}/comments', f'/repos/{repo}/pulls/{number}/reviews']
    entries: list[dict[str, Any]] = []
    complete = True
    for path in paths:
        page, whole = client.first_pages(path, count=_COMMENT_PAGE_LIMIT)
        entries.extend(page)
        complete = complete and whole
    return sorted(entries, key=lambda entry: str(entry.get('created_at') or entry.get('submitted_at') or '')), complete


def _first_maintainer_in_discussion(
    client: GitHubClient, repo: str, item: Mapping[str, Any]
) -> tuple[str | None, bool]:
    """Return the first current maintainer who opened or joined an issue or PR.

    A maintainer's own issue or PR stays theirs: the author is checked before
    anyone who replied later.

    The second value says whether a `None` is trustworthy. A truncated thread or
    a spent probe quota means some participant went unchecked, and padding a
    discussion with throwaway accounts must not be a way to take an item off its
    real owner, so callers leave ownership alone rather than read that as
    nobody being there.
    """
    probe = _MaintainerProbe(client, repo)
    if author := probe.entry(item):
        return author, True
    entries, complete = _discussion(client, repo, item)
    for entry in entries:
        if login := probe.entry(entry):
            return login, True
    return None, complete and not probe.exhausted


def _ensure_recipients(
    client: GitHubClient,
    repo: str,
    item: Mapping[str, Any],
) -> list[str] | None:
    """Return who to notify, or None when ownership could not be decided."""
    number = int(item['number'])
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    if current.get('state') != 'open' or _ACTION_LABEL not in _labels(current):
        raise RuntimeError('Attention state changed during owner selection')
    # Whoever a human put on the item owns it: the monitor never reassigns
    # around an explicit decision. Its own fallback assignment is a placeholder
    # rather than a decision, so it steps aside once a real owner turns up.
    current_maintainers = _maintainer_assignees(client, repo, current)
    logins = [login.casefold() for login in current_maintainers]
    if current_maintainers and logins != [_FALLBACK_OWNER.casefold()]:
        return current_maintainers

    found, conclusive = _first_maintainer_in_discussion(client, repo, current)
    if found is None and not conclusive:
        return None
    owner = found or _FALLBACK_OWNER
    if logins == [owner.casefold()]:
        return current_maintainers
    client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [owner]})
    if current_maintainers:
        client.delete(f'/repos/{repo}/issues/{number}/assignees', {'assignees': current_maintainers})

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
            if recipients is None:
                lines.append(f'#{number}: deferred until its owner can be identified')
                continue
            mentions = ' '.join(f'@{login}' for login in recipients)
            lines.append(f'#{number}: requested maintainer attention from {mentions}')
        except (urllib.error.URLError, RuntimeError) as exc:
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


def _label_transition(timeline: Sequence[dict[str, Any]], label: str) -> tuple[dt.datetime, dict[str, Any]] | None:
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


def _transition(
    timeline: Sequence[dict[str, Any]], stage: Literal[0, 1, 2]
) -> tuple[dt.datetime, dict[str, Any]] | None:
    return _label_transition(timeline, _ACTION_LABEL if stage == 0 else _STAGE_LABELS[stage - 1])


def _actor(event: Mapping[str, Any]) -> str:
    value = event.get('actor') or event.get('user')
    return str(cast(Mapping[str, object], value).get('login') or '') if isinstance(value, Mapping) else ''


# `mentioned` and `subscribed` can be generated as activity side effects, so
# they must never count as acknowledgement.
_REPLY_EVENTS = frozenset({'commented', 'reviewed', 'line-commented'})
_NON_ACK_EVENTS = frozenset({'mentioned', 'subscribed'})
_ACK_ASSOCIATIONS = frozenset({'MEMBER', 'OWNER', 'COLLABORATOR'})


def structured_reply(event: Mapping[str, Any]) -> tuple[str, dt.datetime] | None:
    """Return trusted actor/time metadata for a GitHub reply or review event."""
    if event.get('event') not in _REPLY_EVENTS:
        return None
    actor = _actor(event)
    when = _event_time(event)
    return (actor, when) if actor and when is not None else None


def _acknowledged(
    client: GitHubClient,
    repo: str,
    timeline: Sequence[dict[str, Any]],
    since: dt.datetime,
    recipients: Sequence[str],
) -> bool:
    recipient_logins = {login.casefold() for login in recipients}
    probe = _MaintainerProbe(client, repo)

    def acknowledges(event: Mapping[str, Any]) -> bool:
        actor = _actor(event)
        if actor.casefold() in recipient_logins:
            return True
        if structured_reply(event) is None:
            return False
        # `author_association` is computed for the caller, so it reports a
        # maintainer whose organization membership is private as CONTRIBUTOR.
        # Confirm with the permission lookup rather than ignoring their reply
        # and reminding them about an item they just answered.
        return event.get('author_association') in _ACK_ASSOCIATIONS or bool(probe.login(actor))

    return any(
        (event_time := _event_time(event)) is not None
        and event_time >= since
        and event.get('event') not in _NON_ACK_EVENTS
        and acknowledges(event)
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


def _age(now: dt.datetime, then: dt.datetime) -> str:
    hours = max(0, int((now - then).total_seconds()) // 3600)
    return f'{hours}h ago' if hours < 48 else f'{hours // 24}d ago'


def _is_bot(entry: Mapping[str, Any]) -> bool:
    value = entry.get('actor') or entry.get('user')
    return isinstance(value, Mapping) and cast(Mapping[str, object], value).get('type') == 'Bot'


def _role(probe: _MaintainerProbe, item: Mapping[str, Any], event: Mapping[str, Any]) -> str:
    login = _actor(event)
    if _is_bot(event):
        return 'bot'
    if login.casefold() == _login(item).casefold():
        return 'author'
    return 'maintainer' if probe.login(login) else 'contributor'


def _status(
    client: GitHubClient,
    repo: str,
    item: Mapping[str, Any],
    timeline: Sequence[dict[str, Any]],
    *,
    now: dt.datetime,
) -> str:
    """Say what the item is waiting on, using only structured GitHub metadata.

    Deliberately not a written summary: the channel report must stay free of
    issue and PR prose, which is attacker-controlled text.
    """
    probe = _MaintainerProbe(client, repo)
    parts = ['pull request' if 'pull_request' in item else 'issue']
    if opened := item.get('created_at'):
        parts.append(f'opened by @{_login(item) or "unknown"} {_age(now, _parse_time(str(opened)))}')
    # `comments` is GitHub's own total. The timeline holds only the newest pages,
    # so counting it would understate a long-lived thread.
    # It counts issue comments only, so a PR carrying nothing but reviews reads
    # as zero; the reply clause below is what shows that activity.
    if comments := int(item.get('comments') or 0):
        parts.append(f'{comments} comment{"" if comments == 1 else "s"}')
    replies = [event for event in timeline if structured_reply(event) is not None]
    if replies:
        last = replies[-1]
        when = cast(dt.datetime, _event_time(last))
        parts.append(f'last from @{_actor(last)} {_age(now, when)} ({_role(probe, item, last)})')
    elif not comments:
        parts.append('no replies yet')
    # Asked over the whole discussion rather than the recent timeline: claiming
    # nobody has looked at an item a maintainer answered last year is worse than
    # saying nothing.
    engaged, conclusive = _first_maintainer_in_discussion(client, repo, item)
    if engaged is None and conclusive:
        parts.append('going stale: no maintainer has touched it')
    return ' · '.join(parts)


def _notice(
    client: GitHubClient,
    repo: str,
    item: Mapping[str, Any],
    kind: Literal['reminder', 'escalation'],
    stage: Literal[0, 1, 2],
    transition: tuple[dt.datetime, dict[str, Any]],
    recipients: Sequence[str],
    timeline: Sequence[dict[str, Any]],
    *,
    now: dt.datetime,
) -> Notice:
    return Notice(
        number=int(item['number']),
        kind=kind,
        expected_stage=stage,
        transition_id=_transition_id(transition),
        title=str(item.get('title') or '')[:300],
        recipients=list(recipients),
        status=_status(client, repo, item, timeline, now=now),
    )


def _notice_if_current(
    client: GitHubClient,
    repo: str,
    number: int,
    kind: Literal['reminder', 'escalation'],
    stage: Literal[0, 1, 2],
    transition_id: int | str,
    recipients: Sequence[str],
    *,
    now: dt.datetime,
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
        or {login.casefold() for login in recipients} != {login.casefold() for login in maintainers}
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
    if _closed_since(timeline, current_transition[0]) or _acknowledged(
        client, repo, timeline, acknowledged_since, recipients
    ):
        return None
    return _notice(client, repo, current, kind, stage, current_transition, recipients, timeline, now=now)


def _finish_delivered_escalation(client: GitHubClient, repo: str, number: int, *, new_delivery: bool = False) -> None:
    """Finish an escalation delivery while preserving its cooldown marker."""
    labels = [_ESCALATED_LABEL, _DELIVERED_LABEL] if new_delivery else [_ESCALATED_LABEL]
    _add_labels(client, repo, number, labels)
    _remove_label(client, repo, number, _ACTION_LABEL)
    _remove_label(client, repo, number, _PINGED_LABEL)
    _remove_label(client, repo, number, _DELIVERED_LABEL)


def _valid_delivery_receipt(
    events: Sequence[dict[str, Any]],
    transition: tuple[dt.datetime, dict[str, Any]],
) -> bool:
    receipt = _label_transition(events, _DELIVERED_LABEL)
    if receipt is None or _actor(receipt[1]) != 'github-actions[bot]':
        return False
    indexes = {id(event): index for index, event in enumerate(events)}
    return (receipt[0], indexes[id(receipt[1])]) > (transition[0], indexes[id(transition[1])])


def _finish_delivery_receipt(
    client: GitHubClient,
    repo: str,
    number: int,
    labels: set[str],
    events: Sequence[dict[str, Any]],
    transition: tuple[dt.datetime, dict[str, Any]],
) -> bool:
    if _DELIVERED_LABEL not in labels:
        return False
    if _valid_delivery_receipt(events, transition):
        _finish_delivered_escalation(client, repo, number)
        return True
    _remove_label(client, repo, number, _DELIVERED_LABEL)
    labels.remove(_DELIVERED_LABEL)
    return False


def _effective_stage(
    client: GitHubClient, repo: str, number: int, labels: set[str], events: Sequence[dict[str, Any]]
) -> Literal[0, 1, 2]:
    stage = _stage(labels)
    if stage != 2:
        return stage
    resurfaced = _transition(events, 0)
    escalated = _transition(events, 2)
    if resurfaced is None or escalated is None or resurfaced[0] <= escalated[0]:
        return stage
    # A resurface that added the action label but failed to remove the
    # escalation marker would re-enter stage 2 and queue a duplicate
    # escalation from the old transition. The newer action label is
    # authoritative: shed the stale marker so its label event starts the
    # restarted SLA instead.
    _remove_label(client, repo, number, _ESCALATED_LABEL)
    labels.discard(_ESCALATED_LABEL)
    return _stage(labels)


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
    events = client.last_pages(f'/repos/{repo}/issues/{number}/events', count=_EVENT_PAGE_LIMIT)
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    current_stage = _effective_stage(client, repo, number, labels, events)
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
    if _finish_delivery_receipt(client, repo, number, labels, events, transition):
        return f'#{number}: finished delivered channel escalation', None
    current_stage_label = _STAGE_LABELS[current_stage - 1] if current_stage else None
    for label in labels.intersection(_STAGE_LABELS):
        if label != current_stage_label:
            _remove_label(client, repo, number, label)
    maintainers = _maintainer_assignees(client, repo, current)
    reminder_transition = _transition(events, 1) if current_stage == 2 else None
    acknowledged_since = reminder_transition[0] if reminder_transition is not None else transition_at
    if _acknowledged(client, repo, timeline, acknowledged_since, maintainers or [_FALLBACK_OWNER]):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer acknowledged the request', None
    recipients = _ensure_recipients(client, repo, current)
    if recipients is None:
        return None
    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    if _closed_since(timeline, transition_at) or _acknowledged(client, repo, timeline, acknowledged_since, recipients):
        _complete(client, repo, number, labels)
        return f'#{number}: maintainer acknowledged the request', None
    # Stage 2 is the existing durable "terminal Slack delivery pending" state.
    # Keeping that meaning makes the channel cutover safe for in-flight items.
    if current_stage != 2 and now - transition_at < _SLA:
        return None
    kind: Literal['reminder', 'escalation'] = 'reminder' if current_stage == 0 else 'escalation'
    notice = _notice_if_current(
        client,
        repo,
        number,
        kind,
        current_stage,
        _transition_id(transition),
        recipients,
        now=now,
    )
    return (f'#{number}: queued channel {kind}', notice) if notice is not None else None


def _sweep_escalated_item(client: GitHubClient, repo: str, number: int, *, now: dt.datetime) -> str | None:
    """Wake, recycle, or retire one escalated item."""
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
    if now - transition[0] >= _RESURFACE_AFTER:
        # Add the active marker first so a partial GitHub failure cannot leave
        # unresolved work in neither state. Its label event starts a fresh SLA.
        _add_labels(client, repo, number, [_ACTION_LABEL])
        _remove_label(client, repo, number, _ESCALATED_LABEL)
        reactivated = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        _ensure_recipients(client, repo, reactivated)
        return f'#{number}: returned unresolved attention to the active queue'
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
    processed = {int(item['number']) for item in items}
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
        except (urllib.error.URLError, RuntimeError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if len(closed) == _CLOSED_CLEANUP_LIMIT or len(active) == _ACTIVE_OPEN_LIMIT:
        lines.append('additional attention items remain for a later rotated batch')
    dormant = _rotated_search(
        client,
        # No is:open qualifier so a dormant item closed while escalated still
        # sheds its marker instead of carrying it forever.
        f'repo:{repo} label:"{_ESCALATED_LABEL}"',
        # Recent-first keeps renewed activity on an old escalated issue from
        # sitting behind the oldest dormant items, while slot rotation still
        # reaches every page so a full page of items inside the cooldown
        # cannot strand older, already-eligible escalations indefinitely.
        order='desc',
        limit=_RECONCILE_LIMIT,
        slot=slot,
    )
    for item in dormant:
        number = int(item['number'])
        if number in processed or _ACTION_LABEL in _labels(item):
            continue
        try:
            if line := _sweep_escalated_item(client, repo, number, now=now):
                lines.append(line)
        except (urllib.error.URLError, RuntimeError, ValueError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    return lines, failures


def _slack_escape(value: str) -> str:
    normalized = ' '.join(value.split())
    normalized = ''.join(character for character in normalized if unicodedata.category(character) != 'Cf')
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
                f'owner {owners}\n'
                f'      {_slack_escape(notice["status"])}\n'
                f'      why: {reasons[notice["kind"]]}'
            )
        payload = {
            'text': '\n'.join(
                [
                    f'<!channel> *Maintainer attention requested in {_slack_escape(repo)}*',
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


def _search_summary(client: GitHubClient, query: str, *, first: int) -> tuple[int, list[dict[str, Any]]]:
    """Return a bounded GraphQL search page without REST Search burst limits."""
    result = client.post('/graphql', {'query': _SEARCH_SUMMARY_QUERY, 'variables': {'query': query, 'first': first}})
    if not isinstance(result, Mapping):
        raise RuntimeError('GitHub rejected the attention search')
    response = cast(Mapping[str, object], result)
    if response.get('errors'):
        raise RuntimeError('GitHub rejected the attention search')
    data = response.get('data')
    search = cast(Mapping[str, object], data).get('search') if isinstance(data, Mapping) else None
    if not isinstance(search, Mapping):
        raise RuntimeError('GitHub returned a malformed attention search')
    search_data = cast(Mapping[str, object], search)
    count = search_data.get('issueCount')
    nodes = search_data.get('nodes')
    if not isinstance(count, int) or isinstance(count, bool) or not isinstance(nodes, list):
        raise RuntimeError('GitHub returned a malformed attention search')
    values: list[dict[str, Any]] = []
    for node in cast(list[object], nodes):
        if not isinstance(node, Mapping):
            raise RuntimeError('GitHub returned a malformed attention search item')
        item = cast(Mapping[str, object], node)
        number = item.get('number')
        if not isinstance(number, int) or isinstance(number, bool):
            raise RuntimeError('GitHub returned a malformed attention search item')
        values.append({'number': item['number'], 'created_at': item.get('createdAt')})
    return count, values


def _search_count(client: GitHubClient, query: str) -> int:
    return _search_summary(client, query, first=1)[0]


def slack_mentions(value: str, required_owner: str) -> dict[str, str]:
    """Validate the fixed maintainer mention mapping owned by repository configuration."""
    loaded: object = json.loads(value)
    if not isinstance(loaded, Mapping):
        raise ValueError('Slack mention mapping must be an object')
    mentions = {str(key): str(mention) for key, mention in cast(Mapping[object, object], loaded).items()}
    if (
        required_owner not in mentions
        or not set(mentions) <= set(MAINTAINER_OWNERS)
        or any(_SLACK_MENTION.fullmatch(mention) is None for mention in mentions.values())
    ):
        raise ValueError('Slack mention mapping must contain the selected owner and no unknown owners')
    return mentions


def _qualified_routing_owners(client: GitHubClient, repo: str) -> tuple[str, ...]:
    return tuple(owner for owner in MAINTAINER_OWNERS if client.maintainer_login(repo, owner, refresh=True) is not None)


def _unowned_query(
    repo: str,
    owners: Sequence[str],
    *,
    lane: Literal['recent', 'legacy', 'draft'],
) -> str:
    exclusions = ' '.join(f'-assignee:{owner}' for owner in owners)
    if lane == 'recent':
        return f'repo:{repo} is:open created:>={ROUTING_RECOVERY_EPOCH} -draft:true {exclusions}'
    if lane == 'legacy':
        return f'repo:{repo} is:open created:<{ROUTING_RECOVERY_EPOCH} -draft:true {exclusions}'
    return f'repo:{repo} is:pr is:open draft:true {exclusions}'


def _unowned_snapshot(
    client: GitHubClient, repo: str, owners: Sequence[str]
) -> tuple[int, tuple[int, dt.datetime] | None]:
    query = _unowned_query(repo, owners, lane='recent')
    total, items = _search_summary(client, f'{query} sort:created-asc', first=1)
    if not items:
        return total, None
    return total, (int(items[0]['number']), _parse_time(str(items[0]['created_at'])))


def census(client: GitHubClient, repo: str, *, now: dt.datetime, urgent_mention: str | None = None) -> str:
    """Build one daily heartbeat for the queues that need prompt maintainer action."""
    active = _search_count(client, f'repo:{repo} is:open label:"{_ACTION_LABEL}"')
    cooling = _search_count(client, f'repo:{repo} is:open label:"{_ESCALATED_LABEL}"')
    owners = _qualified_routing_owners(client, repo)
    recent_unowned, oldest = _unowned_snapshot(client, repo, owners)
    # Counts and item numbers only: the heartbeat must stay free of issue and PR
    # prose, which is attacker-controlled text.
    oldest_age = max(0, (now - oldest[1]).days) if oldest else 0
    breach = recent_unowned > 100 or (oldest is not None and now - oldest[1] > dt.timedelta(days=1))
    if breach and (urgent_mention is None or _SLACK_MENTION.fullmatch(urgent_mention) is None):
        raise ValueError('A valid Aditya Slack mention is required for an intake breach')
    prefix = f'{urgent_mention} :rotating_light:' if breach else ':telescope:'
    recent = (
        f'{recent_unowned} post-rollout without designated owner; oldest #{oldest[0]} opened {oldest_age}d ago'
        if oldest
        else '0 post-rollout without designated owner'
    )
    saturation = ' — recovery search saturated' if recent_unowned > 100 else ''
    return (
        f'{prefix} Attention coverage for {_slack_escape(repo)} — '
        f'queue: {active} active, {cooling} cooling; intake: {recent}{saturation}. '
        'The Monday digest covers assigned, legacy, and draft work.'
    )


def _weekly_status(
    item: Mapping[str, Any], timeline: Sequence[dict[str, Any]], owner: str | None, *, now: dt.datetime
) -> str:
    """Describe recent interaction without scanning or interpreting discussion prose."""
    parts = ['pull request' if 'pull_request' in item else 'issue']
    if opened := item.get('created_at'):
        parts.append(f'opened by @{_login(item) or "unknown"} {_age(now, _parse_time(str(opened)))}')
    if comments := int(item.get('comments') or 0):
        parts.append(f'{comments} issue comment{"" if comments == 1 else "s"}')
    replies = [event for event in timeline if structured_reply(event) is not None]
    if replies:
        last = replies[-1]
        parts.append(f'last reply/review @{_actor(last)} {_age(now, cast(dt.datetime, _event_time(last)))}')
    if owner is not None:
        owner_replies = [event for event in replies if _actor(event).casefold() == owner.casefold()]
        if owner_replies:
            last_owner = owner_replies[-1]
            parts.append(f'owner replied/reviewed {_age(now, cast(dt.datetime, _event_time(last_owner)))}')
        else:
            parts.append('no owner reply/review in recent history')
    return ' · '.join(parts)


def _weekly_items(
    client: GitHubClient,
    repo: str,
    owner: str,
    matches: Sequence[dict[str, Any]],
    seen: set[int],
    *,
    attention_only: bool,
    limit: int,
    now: dt.datetime,
) -> list[str]:
    if not limit:
        return []
    lines: list[str] = []
    for match in matches:
        number = int(match['number'])
        if number in seen:
            continue
        current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        assignees = {str(value.get('login') or '').casefold() for value in current.get('assignees', [])}
        labels = _labels(current)
        if (
            str(current.get('state') or '').casefold() != 'open'
            or owner.casefold() not in assignees
            or (attention_only and _ACTION_LABEL not in labels)
        ):
            continue
        timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline?per_page=100')
        updated = _age(now, _parse_time(str(current['updated_at'])))
        title = _slack_escape(str(current.get('title') or ''))[:120]
        status = _slack_escape(_weekly_status(current, timeline, owner, now=now))
        label = f'#{number} {title}'.rstrip()
        phrase = (
            'channel escalation cooling'
            if _ESCALATED_LABEL in labels
            else ('awaiting maintainer action' if _ACTION_LABEL in labels else 'assigned')
        )
        lines.append(f'• <https://github.com/{repo}/issues/{number}|{label}> — {phrase} · updated {updated} · {status}')
        seen.add(number)
        if len(lines) == limit:
            break
    return lines


def _legacy_items(
    client: GitHubClient,
    repo: str,
    matches: Sequence[dict[str, Any]],
    owners: Sequence[str],
    *,
    now: dt.datetime,
) -> list[str]:
    lines: list[str] = []
    owner_keys = {owner.casefold() for owner in owners}
    for match in matches:
        number = int(match['number'])
        current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        assignees = {str(value.get('login') or '').casefold() for value in current.get('assignees', [])}
        updated_at = _parse_time(str(current['updated_at']))
        if (
            str(current.get('state') or '').casefold() != 'open'
            or assignees.intersection(owner_keys)
            or current.get('draft') is True
            or _parse_time(str(current['created_at'])).date() >= dt.date.fromisoformat(ROUTING_RECOVERY_EPOCH)
        ):
            continue
        timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline?per_page=100')
        title = _slack_escape(str(current.get('title') or ''))[:120]
        status = _slack_escape(_weekly_status(current, timeline, None, now=now))
        label = f'#{number} {title}'.rstrip()
        lines.append(
            f'• <https://github.com/{repo}/issues/{number}|{label}> — updated {_age(now, updated_at)} · {status}'
        )
    return lines


def weekly_digest(client: GitHubClient, repo: str, *, now: dt.datetime) -> str:
    """Build a bounded Monday view of every ownership lane."""
    if repo not in REPOSITORIES:
        raise ValueError(f'Unsupported repository: {repo}')
    lines = [f':spiral_calendar_pad: *Monday maintainer queues — {_slack_escape(repo)}* · {now.date().isoformat()}']
    owners = _qualified_routing_owners(client, repo)
    for owner in MAINTAINER_OWNERS:
        name = _MAINTAINER_NAMES[owner]
        if owner not in owners:
            total = _search_count(client, f'repo:{repo} is:open assignee:{owner}')
            noun = 'assignment' if total == 1 else 'assignments'
            verb = 'needs' if total == 1 else 'need'
            lines.extend(
                ['', f'*{name}* (`{owner}`) — not a current designated owner · {total} {noun} {verb} rerouting']
            )
            query = urllib.parse.quote_plus(f'repo:{repo} is:open assignee:{owner}')
            lines.append(f'<https://github.com/{repo}/issues?q={query}|View all {total}>')
            continue
        base = f'repo:{repo} is:open assignee:{owner}'
        total, assigned = _search_summary(client, f'{base} sort:updated-asc', first=_WEEKLY_ITEM_LIMIT * 2)
        if not total:
            lines.extend(['', f'*{name}* (`{owner}`) — clear'])
            continue
        awaiting, attention = _search_summary(
            client,
            f'{base} label:"{_ACTION_LABEL}" sort:updated-asc',
            first=_WEEKLY_ITEM_LIMIT,
        )
        lines.extend(['', f'*{name}* (`{owner}`) — {total} open assigned · {awaiting} awaiting action'])
        seen: set[int] = set()
        details = _weekly_items(
            client, repo, owner, attention, seen, attention_only=True, limit=_WEEKLY_ITEM_LIMIT, now=now
        )
        details.extend(
            _weekly_items(
                client,
                repo,
                owner,
                assigned,
                seen,
                attention_only=False,
                limit=_WEEKLY_ITEM_LIMIT - len(details),
                now=now,
            )
        )
        lines.extend(details)
        query = urllib.parse.quote_plus(f'repo:{repo} is:open assignee:{owner}')
        lines.append(f'<https://github.com/{repo}/issues?q={query}|View all {total}>')
    recent_query = _unowned_query(repo, owners, lane='recent')
    legacy_query = _unowned_query(repo, owners, lane='legacy')
    draft_query = _unowned_query(repo, owners, lane='draft')
    recent_total = _search_count(client, recent_query)
    legacy_total, legacy_matches = _search_summary(
        client, f'{legacy_query} sort:updated-desc', first=_LEGACY_ITEM_LIMIT
    )
    draft_total = _search_count(client, draft_query)
    lines.extend(
        [
            '',
            f'*Unassigned queues* — {recent_total} post-rollout · {legacy_total} legacy · {draft_total} drafts '
            'without a designated owner',
        ]
    )
    legacy_lines = _legacy_items(client, repo, legacy_matches, owners, now=now)
    if legacy_lines:
        lines.extend(['Recently updated legacy items:', *legacy_lines])
    encoded_recent = urllib.parse.quote_plus(recent_query)
    encoded_legacy = urllib.parse.quote_plus(legacy_query)
    encoded_drafts = urllib.parse.quote_plus(draft_query)
    lines.append(
        f'<https://github.com/{repo}/issues?q={encoded_recent}|View post-rollout> · '
        f'<https://github.com/{repo}/issues?q={encoded_legacy}|View legacy> · '
        f'<https://github.com/{repo}/issues?q={encoded_drafts}|View drafts>'
    )
    text = '\n'.join(lines)
    if len(text.encode()) > _WEEKLY_TEXT_LIMIT:
        raise RuntimeError('Weekly digest exceeds the Slack payload limit')
    return text


def _write_slack_payload(text: str) -> None:
    if output_path := os.environ.get('GITHUB_OUTPUT'):
        with Path(output_path).open('a', encoding='utf-8') as output:
            output.write(f'slack_payload={json.dumps({"text": text}, separators=(",", ":"))}\n')


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


def prepare_notices(client: GitHubClient, repo: str, notices: Sequence[NoticeRef], *, now: dt.datetime) -> list[Notice]:
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
            now=now,
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
    *,
    now: dt.datetime,
) -> str | None:
    number = notice['number']
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    labels = _labels(current)
    stage = _stage(labels)
    if current.get('state') != 'open' or _ACTION_LABEL not in labels or stage != notice['expected_stage']:
        return None
    maintainers = _maintainer_assignees(client, repo, current)
    if {login.casefold() for login in notice['recipients']} != {login.casefold() for login in maintainers}:
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
    if _closed_since(timeline, transition[0]) or _acknowledged(
        client, repo, timeline, transition[0], notice['recipients']
    ):
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
            now=now,
        )
        is None
    ):
        return None

    if stage == 0:
        _advance_stage(client, repo, number, labels, 1)
    else:
        # Record delivery before terminal cleanup so a later GitHub failure
        # cannot make reconciliation post the escalation again.
        _finish_delivered_escalation(client, repo, number, new_delivery=True)

    timeline = client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=3)
    completed_labels = labels | ({_PINGED_LABEL} if stage == 0 else {_ESCALATED_LABEL, _DELIVERED_LABEL})
    if _closed_since(timeline, transition[0]) or _acknowledged(
        client, repo, timeline, transition[0], notice['recipients']
    ):
        _complete(client, repo, number, completed_labels)
        return f'#{number}: maintainer activity completed the delivered notice'
    return f'#{number}: recorded channel {kind}'


def finalize_notices(client: GitHubClient, repo: str, notices: Sequence[NoticeRef], *, now: dt.datetime) -> list[str]:
    """Advance attention state only after the channel delivery succeeds."""
    lines: list[str] = []
    failures: list[str] = []
    for notice in notices:
        number = notice['number']
        try:
            if line := _finalize_notice(client, repo, notice, now=now):
                lines.append(line)
        except (urllib.error.URLError, RuntimeError) as exc:
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
    parser.add_argument('mode', choices=['snapshot', 'apply', 'reconcile', 'prepare', 'finalize', 'census', 'weekly'])
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
        notices = prepare_notices(client, repo, _notice_refs(json.loads(source)), now=now)
        _write_notices(repo, notices)
        lines = [f'prepared {len(notices)} current attention notice(s)']
    elif args.mode == 'census':
        mention = None
        if raw_mentions := os.environ.get('PYDANTIC_AI_TRIAGE_SLACK_MENTIONS'):
            try:
                mention = slack_mentions(raw_mentions, _FALLBACK_OWNER)[_FALLBACK_OWNER]
            except ValueError:
                pass
        coverage = census(client, repo, now=now, urgent_mention=mention)
        _write_slack_payload(coverage)
        lines = [coverage]
    elif args.mode == 'weekly':
        report = weekly_digest(client, repo, now=now)
        _write_slack_payload(report)
        lines = [report]
    else:
        source = os.environ.get('ATTENTION_NOTICES')
        if source is None:
            parser.error('ATTENTION_NOTICES is required')
        lines = finalize_notices(client, repo, _notice_refs(json.loads(source)), now=now)
    _write_summary(lines + [f'failed: {failure}' for failure in failures])
    for line in lines:
        print(line)
    for failure in failures:
        print(f'failed: {failure}', file=sys.stderr)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
