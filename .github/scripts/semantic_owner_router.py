#!/usr/bin/env python3
"""Deterministically assign open items to their semantic maintainer owners."""

from __future__ import annotations

import argparse
import json
import os
import re
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
_LEGACY_BATCH_LIMIT = 6
_FILE_LIMIT = 100
_ASSIGNEE_LIMIT = 10
_PARTICIPATION_TIMELINE_PAGES = 2
_MAX_ITEM_NUMBER = 2_147_483_647
_ITEM_QUERY = """
query RoutingItem($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issueOrPullRequest(number: $number) {
      __typename
      ... on Issue {
        number state
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
    'pydantic/pydantic-ai-harness': (
        Rule(
            'adtyavrdhn',
            ('cap:code-mode', 'cap:acp', 'upstream-compat'),
            (
                'pydantic_ai_harness/code_mode/',
                'pydantic_ai_harness/acp/',
                'pydantic_ai_harness/runtime_authoring/',
            ),
        ),
        Rule('dsfaccini', ('cap:compaction',), ('pydantic_ai_harness/compaction/',)),
        Rule('DouweM', ('durable-exec', 'cap:step-persistence'), ('pydantic_ai_harness/step_persistence/',)),
    ),
}
_PARTICIPATION_OWNERS = frozenset({'adtyavrdhn', 'dsfaccini', 'DouweM'})


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


def event_number(issue_value: str | None, pull_request_value: str | None) -> int | None:
    """Validate the runner-projected issue or pull request number."""
    values = [value for value in (issue_value, pull_request_value) if value]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError('GitHub event must identify exactly one issue or pull request')
    value = values[0]
    if re.fullmatch(r'[1-9][0-9]{0,9}', value) is None:
        raise ValueError('item number must be a bounded positive integer')
    return _item_number(int(value))


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


def _participant_owner(login: str) -> str | None:
    key = login.casefold()
    return next((owner for owner in _PARTICIPATION_OWNERS if owner.casefold() == key), None)


def _latest_participant(
    client: attention.GitHubClient,
    repo: str,
    number: int,
    owners: Sequence[str],
) -> str | None:
    """Return the latest configured owner visible in the bounded timeline."""
    allowed = {owner.casefold(): owner for owner in owners}
    latest: str | None = None
    for event in client.last_pages(f'/repos/{repo}/issues/{number}/timeline', count=_PARTICIPATION_TIMELINE_PAGES):
        reply = attention.structured_reply(event)
        if reply is not None and (owner := allowed.get(reply[0].casefold())) is not None:
            latest = owner
    return latest


def _participant_from_evidence(decision: Decision) -> str | None:
    expected = f'participant:{decision["owner"]}'
    return decision['owner'] if decision['evidence'] == expected else None


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


def _participant_decision(
    client: attention.GitHubClient,
    repo: str,
    number: int,
    login: str | None,
) -> Decision | None:
    if not login or (owner := _participant_owner(login)) is None:
        return None
    qualified = tuple(
        candidate
        for candidate in _PARTICIPATION_OWNERS
        if client.maintainer_login(repo, candidate, refresh=True) is not None
    )
    if _latest_participant(client, repo, number, qualified) != owner:
        return None
    return _decision(client, repo, number, owner, f'participant:{owner}')


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


def decision_for(
    client: attention.GitHubClient,
    repo: str,
    number: int,
    *,
    participant_login: str | None = None,
) -> Selection:
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
    if _maintainer_assignees(client, repo, normalized):
        return Selection(number=number, decision=None, status='maintainer-present')
    if len(normalized['assignees']) >= _ASSIGNEE_LIMIT:
        return Selection(number=number, decision=None, status='assignee-capacity')
    is_pull_request = item.get('__typename') == 'PullRequest'
    if is_pull_request and (precedence := _pull_request_precedence(client, repo, number, item)) is not None:
        return precedence
    participant = _participant_decision(client, repo, number, participant_login)
    if participant is not None:
        return Selection(number=number, decision=participant, status='route')
    if participant_login:
        return Selection(number=number, decision=None, status='superseded-maintainer-response')
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


def _recovery_numbers(
    client: attention.GitHubClient,
    repo: str,
    qualified: Sequence[str],
    *,
    legacy: bool = False,
) -> list[int]:
    negatives = ' '.join(f'-assignee:{owner}' for owner in qualified)
    created = f'created:<{_RECOVERY_EPOCH}' if legacy else f'created:>={_RECOVERY_EPOCH}'
    order = 'updated-desc' if legacy else 'created-asc'
    assignees = 'no:assignee' if legacy else negatives
    query = f'repo:{repo} is:open {created} -draft:true {assignees} sort:{order}'
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
    issue_number: str | None,
    pull_request_number: str | None,
    participant_login: str | None = None,
    *,
    legacy_recovery: bool = False,
) -> list[Selection]:
    """Select an event item, one recent recovery, or a bounded legacy batch."""
    repo = _repository(repo)
    if number := event_number(issue_number, pull_request_number):
        if participant_login:
            owner = _participant_owner(participant_login)
            if owner is None or client.maintainer_login(repo, owner, refresh=True) is None:
                return [Selection(number=number, decision=None, status='non-maintainer-response')]
        return [decision_for(client, repo, number, participant_login=participant_login)]
    qualified = _qualified_owners(client, repo)
    recent = _select_numbers(client, repo, _recovery_numbers(client, repo, qualified), limit=1)
    if recent or not legacy_recovery:
        return recent
    return _select_numbers(
        client,
        repo,
        _recovery_numbers(client, repo, qualified, legacy=True),
        limit=_LEGACY_BATCH_LIMIT,
    )


def select(
    client: attention.GitHubClient,
    repo: str,
    issue_number: str | None,
    pull_request_number: str | None,
    participant_login: str | None = None,
) -> Selection:
    """Select exactly the event item or one recent recovery candidate."""
    selected = select_batch(client, repo, issue_number, pull_request_number, participant_login)
    return selected[0] if selected else Selection(number=0, decision=None, status='nothing-to-route')


def assign(client: attention.GitHubClient, repo: str, expected: Decision) -> bool:
    """Recompute under concurrency, then add one currently qualified owner."""
    repo = _repository(repo)
    if expected['owner'] not in _OWNERS:
        raise ValueError('owner is not allowlisted')
    current = decision_for(
        client,
        repo,
        expected['number'],
        participant_login=_participant_from_evidence(expected),
    )
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


def _routing_reason(decision: Decision, mention: str) -> str:
    evidence = decision['evidence']
    owner = decision['owner']
    if evidence == f'participant:{owner}':
        return f'{mention} was the most recent qualified maintainer to participate.'
    if evidence == f'author:{owner}':
        return f'{mention} authored this pull request.'
    source, separator, detail = evidence.partition(':')
    if separator and detail and source in {'label', 'path'}:
        return f'Matched ownership {source} `{detail}`.'
    if evidence.startswith('manual:'):
        return 'Automatic routing could not determine an available semantic owner, so this needs manual triage.'
    return 'Matched the semantic ownership policy.'


def _slack_payload(
    repo: str,
    item_type: Literal['Issue', 'PullRequest'],
    decision: Decision,
    mentions_value: str,
) -> str:
    """Build one canonical Slack assignment notice."""
    repo = _repository(repo)
    mentions = attention.slack_mentions(mentions_value, decision['owner'])
    mention = mentions[decision['owner']]
    if item_type == 'Issue':
        kind, path = 'Issue', 'issues'
    else:
        kind, path = 'Pull request', 'pull'
    number = decision['number']
    item = f'<https://github.com/{repo}/{path}/{number}|{repo}#{number}>'
    text = f'Routing intent: {kind} {item} → {mention}\nWhy: {_routing_reason(decision, mention)}'
    return json.dumps({'text': text}, separators=(',', ':'))


def prepare_current(
    client: attention.GitHubClient,
    repo: str,
    expected: Decision,
    mentions_value: str,
) -> str | None:
    """Build a notice only while the selected route still matches GitHub."""
    repo = _repository(repo)
    item = _fetch_item(client, repo, expected['number'])
    if item is None:
        return None
    item_type = item.get('__typename')
    if item_type not in ('Issue', 'PullRequest'):
        raise RuntimeError('GitHub returned invalid routing metadata')
    current = decision_for(
        client,
        repo,
        expected['number'],
        participant_login=_participant_from_evidence(expected),
    )
    if current['decision'] != expected:
        return None
    return _slack_payload(repo, item_type, expected, mentions_value)


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
            payload = prepare_current(
                client,
                repo,
                expected,
                os.environ['PYDANTIC_AI_TRIAGE_SLACK_MENTIONS'],
            )
            _output({'should_notify': str(payload is not None).lower(), 'slack_payload': payload or ''})
            _summary(f'#{args.number}: ' + ('prepared routing intent' if payload else 'route changed'))
            return 0
        if args.mode == 'select':
            selected = select_batch(
                client,
                repo,
                os.environ.get('ROUTING_ISSUE_NUMBER'),
                os.environ.get('ROUTING_PULL_REQUEST_NUMBER'),
                os.environ.get('ROUTING_PARTICIPANT_LOGIN'),
                legacy_recovery=os.environ.get('ROUTING_LEGACY_RECOVERY') == 'true',
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
