#!/usr/bin/env python3
"""Deterministically assign one open item to its semantic maintainer owner."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict, cast  # noqa: TID251

import issue_pr_attention_monitor as attention

_REPOSITORIES = attention.REPOSITORIES
_OWNERS = frozenset(attention.MAINTAINER_OWNERS)
_MANUAL_OWNER = 'adtyavrdhn'
# Everything before this rollout watermark was handled by the one-time manual
# audit. Keeping it fixed makes later outages recoverable without draining years
# of historical backlog into the triage channel.
_RECOVERY_EPOCH = attention.ROUTING_RECOVERY_EPOCH
_FILE_LIMIT = 100
_ASSIGNEE_LIMIT = 10
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


_RULES: dict[str, tuple[Rule, ...]] = {
    'pydantic/pydantic-ai': (
        Rule(
            'adtyavrdhn',
            ('streaming', 'run_stream', 'AG-UI', 'UI adapters'),
            (
                'pydantic_ai_slim/pydantic_ai/ui/',
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
_ROUTE_OWNERS = frozenset({_MANUAL_OWNER, *(rule.owner for rules in _RULES.values() for rule in rules)})
_EVIDENCE = frozenset(
    {
        'manual:conflict-or-unknown',
        'manual:incomplete-file-list',
        'manual:incomplete-labels',
        'manual:invalid-file-list',
        'manual:unowned-production-path',
        *(f'manual:unavailable-owner:{owner}' for owner in _ROUTE_OWNERS if owner != _MANUAL_OWNER),
        *(f'label:{label}' for rules in _RULES.values() for rule in rules for label in rule.labels),
        *(f'path:{path}' for rules in _RULES.values() for rule in rules for path in rule.paths),
    }
)


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
    if _maintainer_assignees(client, repo, normalized):
        return Selection(number=number, decision=None, status='maintainer-present')
    if len(normalized['assignees']) >= _ASSIGNEE_LIMIT:
        return Selection(number=number, decision=None, status='assignee-capacity')
    filenames: list[str] | None = None
    if item.get('__typename') == 'PullRequest':
        is_draft = item.get('isDraft')
        if type(is_draft) is not bool:
            return Selection(number=number, decision=None, status='invalid-draft-state')
        if is_draft:
            return Selection(number=number, decision=None, status='draft')
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


def _recovery_numbers(client: attention.GitHubClient, repo: str) -> list[int]:
    qualified = (owner for owner in _OWNERS if client.maintainer_login(repo, owner, refresh=True) is not None)
    negatives = ' '.join(f'-assignee:{owner}' for owner in sorted(qualified, key=str.casefold))
    query = f'repo:{repo} is:open created:>={_RECOVERY_EPOCH} -draft:true {negatives} sort:created-asc'
    result = client.post('/graphql', {'query': _SEARCH_QUERY, 'variables': {'query': query}})
    if not isinstance(result, Mapping):
        raise RuntimeError('GitHub rejected the recovery query')
    response = cast(Mapping[str, object], result)
    if response.get('errors'):
        raise RuntimeError('GitHub rejected the recovery query')
    data = response.get('data')
    search = cast(Mapping[str, object], data).get('search') if isinstance(data, Mapping) else None
    numbers: list[int] = []
    for entry in _connection_nodes(search):
        if isinstance(entry, Mapping):
            try:
                numbers.append(_item_number(cast(Mapping[str, object], entry).get('number')))
            except ValueError:
                continue
    return numbers


def select(
    client: attention.GitHubClient,
    repo: str,
    issue_number: str | None,
    pull_request_number: str | None,
) -> Selection:
    """Select exactly the event item or one recovery candidate."""
    repo = _repository(repo)
    if number := event_number(issue_number, pull_request_number):
        return decision_for(client, repo, number)
    for number in _recovery_numbers(client, repo):
        selection = decision_for(client, repo, number)
        if selection['decision'] is not None:
            return selection
    return Selection(number=0, decision=None, status='nothing-to-route')


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


def parse_mentions(value: str, owner: str) -> dict[str, str]:
    """Validate the caller-owned mention needed by this decision."""
    if owner not in _ROUTE_OWNERS:
        raise ValueError('notification owner is not routable')
    return attention.slack_mentions(value, owner)


def notify(repo: str, decision: Decision, mentions_value: str, webhook: str) -> None:
    """Send a constant-only Slack assignment notice without redirects."""
    repo = _repository(repo)
    if decision['owner'] not in _OWNERS or decision['evidence'] not in _EVIDENCE:
        raise ValueError('notification decision is not canonical')
    mentions = parse_mentions(mentions_value, decision['owner'])
    parsed = urllib.parse.urlparse(webhook)
    if (
        parsed.scheme != 'https'
        or parsed.hostname != 'hooks.slack.com'
        or parsed.port is not None
        or not parsed.path.startswith('/services/')
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError('Slack webhook URL is invalid')
    text = f'Routing intent: {repo}#{decision["number"]} → {mentions[decision["owner"]]}\nWhy: {decision["evidence"]}'
    request = urllib.request.Request(
        webhook,
        data=json.dumps({'text': text}).encode(),
        method='POST',
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.build_opener(attention.NoRedirect).open(request, timeout=10) as response:
            body = response.read(3)
            if response.status != 200 or body != b'ok':
                raise RuntimeError('Slack rejected the notification')
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise RuntimeError(f'Slack notification failed with HTTP {code}') from None
    except urllib.error.URLError:
        raise RuntimeError('Slack notification failed at the network boundary') from None


def notify_current(
    client: attention.GitHubClient,
    repo: str,
    expected: Decision,
    mentions_value: str,
    webhook: str,
) -> bool:
    """Notify only while the selected route still matches current GitHub state."""
    current = decision_for(client, repo, expected['number'])
    if current['decision'] != expected:
        return False
    notify(repo, expected, mentions_value, webhook)
    return True


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
    """Run the select, assign, or notify workflow phase."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['select', 'assign', 'notify'])
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
        if args.mode == 'notify':
            if args.number is None or args.owner is None or args.evidence is None:
                parser.error('notify requires --number, --owner, and --evidence')
            expected = Decision(number=_item_number(args.number), owner=args.owner, evidence=args.evidence)
            did_notify = notify_current(
                client,
                repo,
                expected,
                os.environ['PYDANTIC_AI_TRIAGE_SLACK_MENTIONS'],
                os.environ['PYDANTIC_AI_TRIAGE_SLACK_WEBHOOK_URL'],
            )
            _output({'did_notify': str(did_notify).lower()})
            _summary(f'#{args.number}: ' + ('notified routing intent' if did_notify else 'route changed'))
            return 0
        if args.mode == 'select':
            selected = select(
                client,
                repo,
                os.environ.get('ROUTING_ISSUE_NUMBER'),
                os.environ.get('ROUTING_PULL_REQUEST_NUMBER'),
            )
            decision = selected['decision']
            _output(
                {
                    'should_assign': str(decision is not None).lower(),
                    'number': selected['number'],
                    'owner': decision['owner'] if decision else '',
                    'evidence': decision['evidence'] if decision else '',
                }
            )
            _summary(f'#{selected["number"]}: {selected["status"]}' if selected['number'] else selected['status'])
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
    except (KeyError, OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            exc.close()
        print(f'owner routing failed: {type(exc).__name__}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
