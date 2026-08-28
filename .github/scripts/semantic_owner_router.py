#!/usr/bin/env python3
"""Deterministically assign open items to their semantic maintainer owners.

Issues enter routing only once triage has applied a priority label
(`p:1-highest` or `p:2-high`); everything else stays unassigned, on the
triage automation's plate. The one exception is community pressure: an item
ignored for two weeks while people kept commenting or reacting may also be
assigned. The workflow only polls on a schedule, so a freshly labeled issue
is picked up on the next run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypedDict, cast  # noqa: TID251

import issue_pr_attention_monitor as attention

_REPOSITORIES = attention.REPOSITORIES
_OWNERS = frozenset(attention.MAINTAINER_OWNERS)
_MANUAL_OWNER = 'adtyavrdhn'
_RECOVERY_EPOCH = attention.ROUTING_RECOVERY_EPOCH
_PRIORITY_LABELS = frozenset(attention.PRIORITY_GATE_LABELS)
_RECENT_BATCH_LIMIT = 3
_COMMUNITY_BATCH_LIMIT = 3
_COMMUNITY_IGNORED_DAYS = 14
_COMMUNITY_MIN_INTERACTIONS = 3
_FILE_LIMIT = 100
_ASSIGNEE_LIMIT = 10
_MAX_ITEM_NUMBER = 2_147_483_647
_ITEM_QUERY = """
query RoutingItem($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issueOrPullRequest(number: $number) {
      __typename
      ... on Issue {
        number state createdAt
        comments { totalCount }
        reactions { totalCount }
        timelineItems(itemTypes: [UNASSIGNED_EVENT], last: 10) {
          nodes { ... on UnassignedEvent { createdAt } }
        }
        labels(first: 50) { nodes { name } pageInfo { hasNextPage } }
        assignees(first: 10) { nodes { login } pageInfo { hasNextPage } }
      }
      ... on PullRequest {
        number state isDraft changedFiles
        author { login }
        labels(first: 50) { nodes { name } pageInfo { hasNextPage } }
        assignees(first: 10) { nodes { login } pageInfo { hasNextPage } }
        files(first: 100) { nodes { path } pageInfo { hasNextPage } }
      }
    }
  }
}
"""
_SEARCH_QUERY = """
query RoutingRecovery($query: String!) {
  search(query: $query, type: ISSUE, first: 100) {
    nodes {
      ... on Issue { number }
      ... on PullRequest { number }
    }
  }
}
"""


@dataclass(frozen=True)
class Rule:
    """A code-reviewed owner signal and its canonical evidence string."""

    owner: str
    labels: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()


_UI_LABELS = ('AG-UI', 'UI adapters', 'area:ui-adapters', 'vercel-ai', 'web-ui')
_UI_PATHS = (
    'pydantic_ai_slim/pydantic_ai/ui/',
    'docs/ui/',
    'docs/api/ui/',
    'docs/examples/ag-ui.md',
    'examples/pydantic_ai_examples/ag_ui/',
)
_RULES: dict[str, tuple[Rule, ...]] = {
    'pydantic/pydantic-ai': (
        Rule(
            'adtyavrdhn',
            ('streaming', 'run_stream'),
            (
                'pydantic_ai_slim/pydantic_ai/realtime/',
                'pydantic_ai_slim/pydantic_ai/_cancel.py',
            ),
        ),
        Rule(
            'dsfaccini',
            (
                'model issue',
                'model settings',
                'MCP',
                'message-history',
                'cross-model-provider-mapping',
                'provider-parity',
            ),
            (
                'pydantic_ai_slim/pydantic_ai/models/',
                'pydantic_ai_slim/pydantic_ai/providers/',
                'pydantic_ai_slim/pydantic_ai/profiles/',
                'pydantic_ai_slim/pydantic_ai/messages.py',
                'pydantic_ai_slim/pydantic_ai/mcp.py',
                'pydantic_ai_slim/pydantic_ai/_mcp.py',
                'pydantic_ai_slim/pydantic_ai/_mcp_compat.py',
            ),
        ),
        # UI protocols are more specific than cross-cutting signals such as
        # streaming, so a streaming AG-UI/Vercel item remains David's.
        Rule(
            'dsfaccini',
            _UI_LABELS,
            _UI_PATHS,
        ),
        Rule(
            'DouweM',
            ('durable exec', 'temporal', 'DBOS', 'deferred-tools'),
            (
                'pydantic_ai_slim/pydantic_ai/durable_exec/',
                'pydantic_ai_slim/pydantic_ai/capabilities/',
                'pydantic_ai_slim/pydantic_ai/_deferred.py',
                'pydantic_ai_slim/pydantic_ai/_enqueue.py',
            ),
        ),
    ),
    # Temporarily empty: all harness intake goes to the default owner below.
    'pydantic/pydantic-ai-harness': (),
}
# Repos where one maintainer currently owns all intake. Harness has no triage
# labeler, so its issues skip the priority gate and route straight here.
_DEFAULT_OWNERS = {'pydantic/pydantic-ai-harness': 'mpfaffenberger'}
# Repos whose issues are triaged and priority-labeled; only these apply the
# priority gate before assignment.
_GATED_REPOS = frozenset({'pydantic/pydantic-ai'})


class Decision(TypedDict):
    """One deterministic assignment decision."""

    number: int
    owner: str
    evidence: str


class Selection(TypedDict):
    """One selection result, including no-op outcomes."""

    number: int
    decision: Decision | None
    status: str


def _repository(value: str) -> str:
    if value not in _REPOSITORIES:
        raise ValueError('repository is not allowlisted')
    return value


def _item_number(value: object) -> int:
    if type(value) is not int or not 1 <= value <= _MAX_ITEM_NUMBER:
        raise ValueError('item number must be a bounded positive integer')
    return value


def _labels(item: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for entry in item.get('labels', []):
        if not isinstance(entry, Mapping):
            raise ValueError('GitHub returned a malformed label')
        name = cast(Mapping[str, object], entry).get('name')
        if not isinstance(name, str):
            raise ValueError('GitHub returned a malformed label')
        if name.isascii() and not any(character.isspace() and character != ' ' for character in name):
            values.add(name.casefold())
    return values


def _graphql_time(value: object) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _recently_unassigned(item: Mapping[str, Any]) -> bool:
    """Whether a human took an assignee off this issue inside the back-off window.

    An unassignment means "leave this alone": whoever removed the assignee has
    looked at the item, and routing must not redo what they undid. Malformed
    timeline data counts as recent, failing toward not assigning.
    """
    now = dt.datetime.now(dt.timezone.utc)
    window = dt.timedelta(days=attention.ROUTING_UNASSIGN_BACKOFF_DAYS)
    timeline = item.get('timelineItems')
    if not isinstance(timeline, Mapping):
        return True
    for node in _connection_nodes(cast(Mapping[str, object], timeline)):
        removed_at = (
            _graphql_time(cast(Mapping[str, object], node).get('createdAt')) if isinstance(node, Mapping) else None
        )
        if removed_at is None or now - removed_at < window:
            return True
    return False


def _community_backed(item: Mapping[str, Any]) -> bool:
    """True when the item sat ignored for two weeks while people kept engaging."""
    now = dt.datetime.now(dt.timezone.utc)
    created_at = _graphql_time(item.get('createdAt'))
    if created_at is None or now - created_at < dt.timedelta(days=_COMMUNITY_IGNORED_DAYS):
        return False
    interactions = 0
    for field in ('comments', 'reactions'):
        value = item.get(field)
        count = cast(Mapping[str, object], value).get('totalCount') if isinstance(value, Mapping) else None
        if type(count) is not int or count < 0:
            return False
        interactions += count
    return interactions > _COMMUNITY_MIN_INTERACTIONS


def _valid_path(value: str) -> bool:
    if not value or len(value) > 300 or not value.isascii() or '\\' in value or value.startswith('/'):
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and all(part not in {'', '.', '..'} for part in parts)


def _path_rule(repo: str, filename: str) -> tuple[str, str] | None:
    matches: list[tuple[int, str, str]] = []
    for rule in _RULES[repo]:
        for prefix in rule.paths:
            if filename == prefix or (prefix.endswith('/') and filename.startswith(prefix)):
                matches.append((len(prefix), rule.owner, prefix))
    if not matches:
        return None
    longest = max(length for length, _, _ in matches)
    owners = {(owner, prefix) for length, owner, prefix in matches if length == longest}
    if len({owner for owner, _ in owners}) != 1:
        return None
    owner, prefix = min(owners)
    return owner, f'path:{prefix}'


def _route(repo: str, labels: set[str], filenames: Sequence[str] | None) -> tuple[str, str]:
    signals: set[tuple[str, str]] = set()
    for rule in _RULES[repo]:
        for label in rule.labels:
            if label.casefold() in labels:
                signals.add((rule.owner, f'label:{label}'))
    if filenames is not None:
        for filename in filenames:
            if not _valid_path(filename):
                return _MANUAL_OWNER, 'manual:invalid-file-list'
            if signal := _path_rule(repo, filename):
                signals.add(signal)
            elif not _neutral_path(filename):
                if default := _DEFAULT_OWNERS.get(repo):
                    return default, 'default:repo-intake'
                return _MANUAL_OWNER, 'manual:unowned-production-path'
    has_ui_signal = any(
        owner == 'dsfaccini'
        and (
            evidence in {f'path:{path}' for path in _UI_PATHS} or evidence in {f'label:{label}' for label in _UI_LABELS}
        )
        for owner, evidence in signals
    )
    if has_ui_signal:
        signals -= {('adtyavrdhn', 'label:streaming'), ('adtyavrdhn', 'label:run_stream')}
    owners = {owner for owner, _ in signals}
    if len(owners) != 1:
        if not owners and (default := _DEFAULT_OWNERS.get(repo)):
            return default, 'default:repo-intake'
        return _MANUAL_OWNER, 'manual:conflict-or-unknown'
    owner = owners.pop()
    evidence = min(evidence for signal_owner, evidence in signals if signal_owner == owner)
    return owner, evidence


def _neutral_path(filename: str) -> bool:
    return (
        filename.startswith(('tests/', 'docs/', 'examples/', '.github/'))
        or '/tests/' in filename
        or filename.endswith(('.md', '.rst', '.lock'))
    )


def _maintainer_assignees(
    client: attention.GitHubClient,
    repo: str,
    item: Mapping[str, Any],
    *,
    refresh: bool = False,
) -> list[str]:
    maintainers: list[str] = []
    for entry in item.get('assignees', []):
        if not isinstance(entry, Mapping):
            raise ValueError('GitHub returned a malformed assignee')
        login = cast(Mapping[str, object], entry).get('login')
        if not isinstance(login, str) or not login:
            raise ValueError('GitHub returned a malformed assignee')
        if maintainer := client.maintainer_login(repo, login, refresh=refresh):
            maintainers.append(maintainer)
    return sorted(set(maintainers), key=str.casefold)


def _decision(client: attention.GitHubClient, repo: str, number: int, owner: str, evidence: str) -> Decision:
    """Keep routing actionable when a configured semantic owner is unavailable."""
    if client.maintainer_login(repo, owner, refresh=True) is not None:
        return Decision(number=number, owner=owner, evidence=evidence)
    if owner != _MANUAL_OWNER and client.maintainer_login(repo, _MANUAL_OWNER, refresh=True) is not None:
        return Decision(number=number, owner=_MANUAL_OWNER, evidence=f'manual:unavailable-owner:{owner}')
    raise RuntimeError('manual routing owner lacks maintainer permission')


def _connection_nodes(value: object) -> list[object]:
    if not isinstance(value, Mapping):
        return []
    nodes = cast(Mapping[str, object], value).get('nodes')
    return cast(list[object], nodes) if isinstance(nodes, list) else []


def _connection_complete(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    page_info = cast(Mapping[str, object], value).get('pageInfo')
    return isinstance(page_info, Mapping) and cast(Mapping[str, object], page_info).get('hasNextPage') is False


def _fetch_item(client: attention.GitHubClient, repo: str, number: int) -> Mapping[str, Any] | None:
    owner, name = repo.split('/', 1)
    result = client.post(
        '/graphql',
        {'query': _ITEM_QUERY, 'variables': {'owner': owner, 'name': name, 'number': number}},
    )
    if not isinstance(result, Mapping):
        raise RuntimeError('GitHub rejected the routing metadata query')
    response = cast(Mapping[str, object], result)
    if response.get('errors'):
        raise RuntimeError('GitHub rejected the routing metadata query')
    data = response.get('data')
    if not isinstance(data, Mapping):
        raise RuntimeError('GitHub returned invalid routing metadata')
    repository = cast(Mapping[str, object], data).get('repository')
    if not isinstance(repository, Mapping):
        return None
    value = cast(Mapping[str, object], repository).get('issueOrPullRequest')
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else None


def _pull_request_draft_status(item: Mapping[str, Any]) -> str | None:
    value = item.get('isDraft')
    if type(value) is not bool:
        return 'invalid-draft-state'
    return 'draft' if value else None


def _pull_request_precedence(
    client: attention.GitHubClient,
    repo: str,
    number: int,
    item: Mapping[str, Any],
) -> Selection | None:
    if draft_status := _pull_request_draft_status(item):
        return Selection(number=number, decision=None, status=draft_status)
    author = item.get('author')
    author_login = cast(Mapping[str, object], author).get('login') if isinstance(author, Mapping) else None
    if isinstance(author_login, str):
        key = author_login.casefold()
        owner = next((candidate for candidate in _OWNERS if candidate.casefold() == key), None)
        if owner is not None and client.maintainer_login(repo, owner, refresh=True) is not None:
            return Selection(
                number=number,
                decision=Decision(number=number, owner=owner, evidence=f'author:{owner}'),
                status='route',
            )
    return None


def _issue_gate(repo: str, item: Mapping[str, Any], normalized: Mapping[str, Any], number: int) -> Selection | None:
    """Decide whether an issue may be routed at all; None means proceed."""
    # A human unassignment means "leave this alone", whatever the labels say:
    # without the back-off, a p:1 issue would be re-assigned to the same owner
    # six hours after a maintainer removed them.
    if _recently_unassigned(item):
        return Selection(number=number, decision=None, status='recently-unassigned')
    # A gate label missing from a truncated first page counts as absent, which
    # fails toward leaving the issue unassigned.
    if repo in _GATED_REPOS and not _labels(normalized) & _PRIORITY_LABELS and not _community_backed(item):
        return Selection(number=number, decision=None, status='awaiting-triage')
    return None


def decision_for(client: attention.GitHubClient, repo: str, number: int) -> Selection:
    """Refetch one item and make a deterministic, fail-closed decision."""
    repo = _repository(repo)
    number = _item_number(number)
    item = _fetch_item(client, repo, number)
    if item is None or item.get('state') != 'OPEN':
        return Selection(number=number, decision=None, status='closed')
    if item.get('number') != number or item.get('__typename') not in {'Issue', 'PullRequest'}:
        raise RuntimeError('GitHub returned mismatched routing metadata')
    labels = item.get('labels')
    assignees = item.get('assignees')
    if not _connection_complete(assignees):
        return Selection(number=number, decision=None, status='incomplete-assignees')
    normalized = {
        'labels': _connection_nodes(labels),
        'assignees': _connection_nodes(assignees),
    }
    is_pull_request = item.get('__typename') == 'PullRequest'
    if not is_pull_request and (gated := _issue_gate(repo, item, normalized, number)) is not None:
        return gated
    if _maintainer_assignees(client, repo, normalized):
        return Selection(number=number, decision=None, status='maintainer-present')
    if len(normalized['assignees']) >= _ASSIGNEE_LIMIT:
        return Selection(number=number, decision=None, status='assignee-capacity')
    if is_pull_request and (precedence := _pull_request_precedence(client, repo, number, item)) is not None:
        return precedence
    filenames: list[str] | None = None
    if is_pull_request:
        changed_files = item.get('changedFiles')
        files = item.get('files')
        entries = _connection_nodes(files)
        page_info = cast(Mapping[str, object], files).get('pageInfo') if isinstance(files, Mapping) else None
        filenames = []
        complete = (
            type(changed_files) is int
            and 0 <= changed_files <= _FILE_LIMIT
            and len(entries) == changed_files
            and isinstance(page_info, Mapping)
            and cast(Mapping[str, object], page_info).get('hasNextPage') is False
        )
        if complete:
            for entry in entries:
                path = cast(Mapping[str, object], entry).get('path') if isinstance(entry, Mapping) else None
                if not isinstance(path, str):
                    complete = False
                    break
                filenames.append(path)
        if not complete:
            return Selection(
                number=number,
                decision=_decision(client, repo, number, _MANUAL_OWNER, 'manual:incomplete-file-list'),
                status='route',
            )
    if not _connection_complete(labels):
        return Selection(
            number=number,
            decision=_decision(client, repo, number, _MANUAL_OWNER, 'manual:incomplete-labels'),
            status='route',
        )
    owner, evidence = _route(repo, _labels(normalized), filenames)
    return Selection(
        number=number,
        decision=_decision(client, repo, number, owner, evidence),
        status='route',
    )


def _search_numbers(client: attention.GitHubClient, query: str) -> list[int]:
    result = client.post('/graphql', {'query': _SEARCH_QUERY, 'variables': {'query': query}})
    if not isinstance(result, Mapping):
        raise RuntimeError('GitHub rejected the recovery query')
    response = cast(Mapping[str, object], result)
    if response.get('errors'):
        raise RuntimeError('GitHub rejected the recovery query')
    data = response.get('data')
    search = cast(Mapping[str, object], data).get('search') if isinstance(data, Mapping) else None
    if not isinstance(search, Mapping):
        raise RuntimeError('GitHub returned invalid recovery metadata')
    search_data = cast(Mapping[str, object], search)
    if not isinstance(search_data.get('nodes'), list):
        raise RuntimeError('GitHub returned invalid recovery metadata')
    numbers: list[int] = []
    for entry in _connection_nodes(search_data):
        if not isinstance(entry, Mapping):
            raise RuntimeError('GitHub returned invalid recovery metadata')
        numbers.append(_item_number(cast(Mapping[str, object], entry).get('number')))
    return numbers


def _qualified_owners(client: attention.GitHubClient, repo: str) -> tuple[str, ...]:
    return tuple(
        owner
        for owner in sorted(_OWNERS, key=str.casefold)
        if client.maintainer_login(repo, owner, refresh=True) is not None
    )


def _gated_numbers(client: attention.GitHubClient, repo: str, qualified: Sequence[str]) -> list[int]:
    """List candidates, priority-labeled issues before pull requests."""
    negatives = ' '.join(f'-assignee:{owner}' for owner in qualified)
    if repo in _GATED_REPOS:
        priorities = ','.join(f'"{label}"' for label in sorted(_PRIORITY_LABELS))
        issues = f'repo:{repo} is:open is:issue label:{priorities} {negatives} sort:created-asc'
    else:
        issues = f'repo:{repo} is:open is:issue {negatives} sort:created-asc'
    pulls = f'repo:{repo} is:open is:pr -draft:true created:>={_RECOVERY_EPOCH} {negatives} sort:created-asc'
    return list(dict.fromkeys(_search_numbers(client, issues) + _search_numbers(client, pulls)))


def _community_numbers(client: attention.GitHubClient, repo: str) -> list[int]:
    """List unassigned items ignored for two weeks despite community interactions."""
    # `created:` has date granularity, so search one day wide of the threshold and
    # let `_community_backed` apply the precise two-week check per item.
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=_COMMUNITY_IGNORED_DAYS - 1)).date().isoformat()
    query = (
        f'repo:{repo} is:open -draft:true created:<{cutoff} no:assignee '
        f'interactions:>{_COMMUNITY_MIN_INTERACTIONS} sort:updated-desc'
    )
    return _search_numbers(client, query)


def _select_numbers(
    client: attention.GitHubClient,
    repo: str,
    numbers: Sequence[int],
    *,
    limit: int,
) -> list[Selection]:
    selected: list[Selection] = []
    for number in numbers:
        selection = decision_for(client, repo, number)
        if selection['decision'] is not None:
            selected.append(selection)
            if len(selected) == limit:
                break
    return selected


def select_batch(
    client: attention.GitHubClient,
    repo: str,
    *,
    community_recovery: bool = False,
) -> list[Selection]:
    """Select a bounded gated batch, or a community batch when the gate is quiet."""
    repo = _repository(repo)
    qualified = _qualified_owners(client, repo)
    gated = _select_numbers(client, repo, _gated_numbers(client, repo, qualified), limit=_RECENT_BATCH_LIMIT)
    if gated or not community_recovery:
        return gated
    return _select_numbers(client, repo, _community_numbers(client, repo), limit=_COMMUNITY_BATCH_LIMIT)


def assign(client: attention.GitHubClient, repo: str, expected: Decision) -> bool:
    """Recompute under concurrency, then add one currently qualified owner."""
    repo = _repository(repo)
    if expected['owner'] not in _OWNERS:
        raise ValueError('owner is not allowlisted')
    current = decision_for(client, repo, expected['number'])
    if current['decision'] is None:
        return False
    if current['decision'] != expected:
        raise RuntimeError('routing evidence changed before assignment')
    if client.maintainer_login(repo, expected['owner'], refresh=True) is None:
        raise RuntimeError('selected owner no longer has maintainer permission')
    client.post(
        f'/repos/{repo}/issues/{expected["number"]}/assignees',
        {'assignees': [expected['owner']]},
    )
    assigned = _fetch_item(client, repo, expected['number'])
    if assigned is None:
        raise RuntimeError('assigned item disappeared')
    assignees = assigned.get('assignees')
    if not _connection_complete(assignees):
        raise RuntimeError('GitHub returned incomplete assignees after assignment')
    assigned_maintainers = _maintainer_assignees(
        client,
        repo,
        {'assignees': _connection_nodes(assignees)},
        refresh=True,
    )
    expected_key = expected['owner'].casefold()
    other_maintainers = [login for login in assigned_maintainers if login.casefold() != expected_key]
    if other_maintainers:
        raise RuntimeError('a concurrent maintainer assignment was detected after routing')
    if expected_key not in {login.casefold() for login in assigned_maintainers}:
        raise RuntimeError('GitHub did not apply the selected owner')
    return True


def _routing_reason(decision: Decision, owner_display: str) -> str:
    evidence = decision['evidence']
    owner = decision['owner']
    if evidence == f'author:{owner}':
        return f'{owner_display} authored this pull request.'
    source, separator, detail = evidence.partition(':')
    if separator and detail and source in {'label', 'path'}:
        return f'Matched ownership {source} `{detail}`.'
    if evidence == 'default:repo-intake':
        return 'All intake for this repository is currently routed to one owner.'
    if evidence.startswith('manual:'):
        return 'Automatic routing could not determine an available semantic owner, so this needs manual triage.'
    return 'Matched the semantic ownership policy.'


def _slack_payload(
    repo: str,
    item_type: Literal['Issue', 'PullRequest'],
    decision: Decision,
) -> str:
    """Build one canonical Slack assignment record.

    Deliberately ping-free: GitHub's own assignment notification alerts the
    owner, and the Slack interrupt is reserved for the reminder that fires
    once an assigned priority issue sits quiet past its window.
    """
    repo = _repository(repo)
    owner = decision['owner']
    if item_type == 'Issue':
        kind, path = 'Issue', 'issues'
    else:
        kind, path = 'Pull request', 'pull'
    number = decision['number']
    item = f'<https://github.com/{repo}/{path}/{number}|{repo}#{number}>'
    text = f'Routing intent: {kind} {item} → {owner}\nWhy: {_routing_reason(decision, owner)}'
    return json.dumps({'text': text}, separators=(',', ':'))


def prepare_current(
    client: attention.GitHubClient,
    repo: str,
    expected: Decision,
) -> str | None:
    """Build a notice only while the selected route still matches GitHub."""
    repo = _repository(repo)
    item = _fetch_item(client, repo, expected['number'])
    if item is None:
        return None
    item_type = item.get('__typename')
    if item_type not in ('Issue', 'PullRequest'):
        raise RuntimeError('GitHub returned invalid routing metadata')
    current = decision_for(client, repo, expected['number'])
    if current['decision'] != expected:
        return None
    return _slack_payload(repo, item_type, expected)


def _output(values: Mapping[str, object]) -> None:
    if path := os.environ.get('GITHUB_OUTPUT'):
        with Path(path).open('a', encoding='utf-8') as output:
            for key, value in values.items():
                output.write(f'{key}={value}\n')


def _summary(line: str) -> None:
    print(line)
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(path).open('a', encoding='utf-8') as summary:
            summary.write(f'## Semantic owner routing\n\n- {line}\n')


def main() -> int:
    """Run the select, prepare, or assign workflow phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['select', 'assign', 'prepare'])
    parser.add_argument('--number', type=int)
    parser.add_argument('--owner')
    parser.add_argument('--evidence')
    args = parser.parse_args()
    repo = os.environ.get('GITHUB_REPOSITORY', '')
    try:
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if not token:
            raise ValueError('GITHUB_TOKEN is required')
        client = attention.GitHubClient(token)
        if args.mode == 'prepare':
            if args.number is None or args.owner is None or args.evidence is None:
                parser.error('prepare requires --number, --owner, and --evidence')
            expected = Decision(number=_item_number(args.number), owner=args.owner, evidence=args.evidence)
            payload = prepare_current(client, repo, expected)
            _output({'should_notify': str(payload is not None).lower(), 'slack_payload': payload or ''})
            _summary(f'#{args.number}: ' + ('prepared routing intent' if payload else 'route changed'))
            return 0
        if args.mode == 'select':
            selected = select_batch(
                client,
                repo,
                community_recovery=os.environ.get('ROUTING_COMMUNITY_RECOVERY') == 'true',
            )
            decisions = [selection['decision'] for selection in selected if selection['decision'] is not None]
            first = selected[0] if selected else Selection(number=0, decision=None, status='nothing-to-route')
            _output(
                {
                    'should_assign': str(bool(decisions)).lower(),
                    'routes': json.dumps(decisions, separators=(',', ':')),
                }
            )
            if decisions:
                _summary(', '.join(f'#{route["number"]}' for route in decisions) + ': route')
            else:
                _summary(f'#{first["number"]}: {first["status"]}' if first['number'] else first['status'])
            return 0
        if args.number is None or args.owner is None or args.evidence is None:
            parser.error('assign requires --number, --owner, and --evidence')
        expected = Decision(number=_item_number(args.number), owner=args.owner, evidence=args.evidence)
        did_assign = assign(client, repo, expected)
        _output(
            {
                'did_assign': str(did_assign).lower(),
                'number': expected['number'],
                'owner': expected['owner'],
                'evidence': expected['evidence'],
            }
        )
        _summary(f'#{expected["number"]}: ' + ('assigned' if did_assign else 'already owned'))
        return 0
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        error = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            error += f' {exc.code}'
        print(f'owner routing failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
