#!/usr/bin/env python3
"""Deterministically assign open items to their semantic maintainer owners.

Issues enter routing only once triage has applied a priority label
(`p:1-highest` or `p:2-high`); everything else stays unassigned, on the
triage automation's plate. The one exception is community pressure: the
weekly community-demand sweep judges old-but-active unassigned issues and
applies `community-backed`, which also opens the gate.

Pull requests are never triaged on gated repositories — a human assigns one
when an issue warrants it. Ungated repositories blanket-route new intake,
issues and pull requests alike, to their default owner.
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
from pathlib import Path
from typing import Any, Literal, TypedDict, cast  # noqa: TID251

import issue_pr_attention_monitor as attention

try:
    from triage_telemetry import emit as _emit_event
except ImportError:  # sparse checkouts that omit the telemetry module stay silent
    # Emission is optional everywhere; every workflow that only reads or writes
    # GitHub state must keep working without the telemetry file on disk.
    def _emit_event(name: str, **attributes: object) -> None:
        return


_REPOSITORIES = attention.REPOSITORIES
_OWNERS = frozenset(attention.MAINTAINER_OWNERS)
_MANUAL_OWNER = 'adtyavrdhn'
_RECOVERY_EPOCH = attention.ROUTING_RECOVERY_EPOCH
_PRIORITY_LABELS = frozenset(attention.PRIORITY_GATE_LABELS)
_RECENT_BATCH_LIMIT = 3
_COMMUNITY_BATCH_LIMIT = 3
_COMMUNITY_LABEL = attention.COMMUNITY_LABEL
_ASSIGNEE_LIMIT = 10
_MAX_ITEM_NUMBER = 2_147_483_647
# Must match the `last:` on both `timelineItems` connections below.
_UNASSIGNED_EVENT_PAGE = 10
_ITEM_QUERY = """
query RoutingItem($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issueOrPullRequest(number: $number) {
      __typename
      ... on Issue {
        number state
        timelineItems(itemTypes: [UNASSIGNED_EVENT], last: 10) {
          nodes { ... on UnassignedEvent { createdAt actor { __typename } } }
        }
        labels(first: 50) { nodes { name } pageInfo { hasNextPage } }
        assignees(first: 10) { nodes { login } pageInfo { hasNextPage } }
      }
      ... on PullRequest {
        number state isDraft
        author { login }
        timelineItems(itemTypes: [UNASSIGNED_EVENT], last: 10) {
          nodes { ... on UnassignedEvent { createdAt actor { __typename } } }
        }
        labels(first: 50) { nodes { name } pageInfo { hasNextPage } }
        assignees(first: 10) { nodes { login } pageInfo { hasNextPage } }
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


_UI_LABELS = ('AG-UI', 'UI adapters', 'area:ui-adapters', 'vercel-ai', 'web-ui')
_RULES: dict[str, tuple[Rule, ...]] = {
    'pydantic/pydantic-ai': (
        Rule('adtyavrdhn', ('streaming', 'run_stream')),
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
        ),
        # UI protocols are more specific than cross-cutting signals such as
        # streaming, so a streaming AG-UI/Vercel item remains David's.
        Rule('dsfaccini', _UI_LABELS),
        Rule('DouweM', ('durable exec', 'temporal', 'DBOS', 'deferred-tools')),
    ),
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


def _nested_str(value: object, *keys: str) -> str | None:
    """Walk `keys` through nested mappings, returning the final string if present."""
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = cast(Mapping[str, object], value).get(key)
    return value if isinstance(value, str) else None


def _recently_unassigned(item: Mapping[str, Any]) -> bool:
    """Whether a human took an assignee off this issue inside the back-off window.

    An unassignment means "leave this alone": whoever removed the assignee has
    looked at the item, and routing must not redo what they undid. Malformed
    timeline data counts as recent, failing toward not assigning.
    """
    now = dt.datetime.now(dt.timezone.utc)
    window = dt.timedelta(days=attention.ROUTING_UNASSIGN_BACKOFF_DAYS)
    timeline = item.get('timelineItems')
    if not isinstance(timeline, Mapping) or not isinstance(cast(Mapping[str, object], timeline).get('nodes'), list):
        return True
    nodes = _connection_nodes(cast(Mapping[str, object], timeline))
    for node in nodes:
        # A bot removing an assignee (sweeps, placeholder swaps) is cleanup,
        # not a decision. GraphQL's `__typename` marks app accounts as `Bot`;
        # a missing actor (deleted account) counts as human.
        if _nested_str(node, 'actor', '__typename') == 'Bot':
            continue
        removed_at = _graphql_time(_nested_str(node, 'createdAt'))
        if removed_at is None or now - removed_at < window:
            return True
    # A full page whose oldest event is still inside the window may hide an
    # older human removal behind bot cleanup: truncation fails toward backing
    # off, like everywhere else in this stack. `nodes[0]` is the oldest
    # fetched (`last:` pages are chronological) and must match `_ITEM_QUERY`.
    if len(nodes) == _UNASSIGNED_EVENT_PAGE:
        oldest = _graphql_time(_nested_str(nodes[0], 'createdAt'))
        return oldest is None or now - oldest < window
    return False


def _route(repo: str, labels: set[str]) -> tuple[str, str]:
    signals: set[tuple[str, str]] = set()
    for rule in _RULES[repo]:
        for label in rule.labels:
            if label.casefold() in labels:
                signals.add((rule.owner, f'label:{label}'))
    has_ui_signal = any(
        owner == 'dsfaccini' and evidence in {f'label:{label}' for label in _UI_LABELS} for owner, evidence in signals
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
            # A maintainer's own pull request is already their responsibility:
            # no assignment and no ping, not even to the author.
            return Selection(number=number, decision=None, status='maintainer-author')
    return None


def _issue_gate(repo: str, normalized: Mapping[str, Any], number: int) -> Selection | None:
    """Decide whether an issue may be routed at all; None means proceed."""
    # A gate label missing from a truncated first page counts as absent, which
    # fails toward leaving the item unassigned. `community-backed` (a judged
    # community-demand verdict, see `community_demand.py`) opens the gate too.
    if repo in _GATED_REPOS and not _labels(normalized) & (_PRIORITY_LABELS | {_COMMUNITY_LABEL}):
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
    is_pull_request = item.get('__typename') == 'PullRequest'
    # Pull requests are never triaged on gated repositories: a human assigns
    # one when an issue warrants it. Only ungated blanket-intake repositories
    # route pull requests, to their default owner.
    if is_pull_request and repo in _GATED_REPOS:
        return Selection(number=number, decision=None, status='pull-request')
    labels = item.get('labels')
    assignees = item.get('assignees')
    if not _connection_complete(assignees):
        return Selection(number=number, decision=None, status='incomplete-assignees')
    normalized = {
        'labels': _connection_nodes(labels),
        'assignees': _connection_nodes(assignees),
    }
    # The back-off covers pull requests too: a human unassignment means "leave
    # this alone", and without it the same owner would be re-assigned six
    # hours after a maintainer removed them.
    if _recently_unassigned(item):
        return Selection(number=number, decision=None, status='recently-unassigned')
    if (gated := _issue_gate(repo, normalized, number)) is not None:
        return gated
    if _maintainer_assignees(client, repo, normalized):
        return Selection(number=number, decision=None, status='maintainer-present')
    if len(normalized['assignees']) >= _ASSIGNEE_LIMIT:
        return Selection(number=number, decision=None, status='assignee-capacity')
    if is_pull_request and (precedence := _pull_request_precedence(client, repo, number, item)) is not None:
        return precedence
    if not _connection_complete(labels):
        return Selection(
            number=number,
            decision=_decision(client, repo, number, _MANUAL_OWNER, 'manual:incomplete-labels'),
            status='route',
        )
    owner, evidence = _route(repo, _labels(normalized))
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
    """List routing candidates.

    Gated repos search priority-labeled issues only; ungated repos search
    all new intake, issues before pull requests.
    """
    negatives = ' '.join(f'-assignee:{owner}' for owner in qualified)
    if repo in _GATED_REPOS:
        # Pull requests are never triaged on gated repositories, so the sweep
        # does not search them at all.
        priorities = ','.join(f'"{label}"' for label in sorted(_PRIORITY_LABELS))
        issues = f'repo:{repo} is:open is:issue label:{priorities} {negatives} sort:created-asc'
        return _search_numbers(client, issues)
    # Blanket intake covers new items going forward, not the backlog.
    issues = f'repo:{repo} is:open is:issue created:>={_RECOVERY_EPOCH} {negatives} sort:created-asc'
    pulls = f'repo:{repo} is:open is:pr -draft:true created:>={_RECOVERY_EPOCH} {negatives} sort:created-asc'
    return list(dict.fromkeys(_search_numbers(client, issues) + _search_numbers(client, pulls)))


def _community_numbers(client: attention.GitHubClient, repo: str) -> list[int]:
    """List unassigned items the triage agent judged to have genuine community demand."""
    query = f'repo:{repo} is:open is:issue no:assignee label:"{_COMMUNITY_LABEL}" sort:updated-desc'
    return _search_numbers(client, query)


def _select_numbers(
    client: attention.GitHubClient,
    repo: str,
    numbers: Sequence[int],
    *,
    limit: int,
    lane: str,
) -> list[Selection]:
    selected: list[Selection] = []
    for number in numbers:
        selection = decision_for(client, repo, number)
        decision = selection['decision']
        _emit_event(
            'router.decision',
            repo=repo,
            lane=lane,
            number=number,
            status=selection['status'],
            owner=decision['owner'] if decision else None,
            evidence=decision['evidence'] if decision else None,
        )
        if decision is not None:
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
    gated_numbers = _gated_numbers(client, repo, qualified)
    gated = _select_numbers(client, repo, gated_numbers, limit=_RECENT_BATCH_LIMIT, lane='gate')
    _emit_event('router.sweep', repo=repo, lane='gate', candidates=len(gated_numbers), selected=len(gated))
    # The community-demand judge runs only on gated repos, so the community
    # lane must not run elsewhere: on an ungated repo it would sweep backlog
    # items past the new-intake epoch on a hand-applied label.
    if gated or not community_recovery or repo not in _GATED_REPOS:
        return gated
    community_numbers = _community_numbers(client, repo)
    community = _select_numbers(client, repo, community_numbers, limit=_COMMUNITY_BATCH_LIMIT, lane='community')
    _emit_event('router.sweep', repo=repo, lane='community', candidates=len(community_numbers), selected=len(community))
    return community


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


def _routing_reason(decision: Decision) -> str:
    evidence = decision['evidence']
    source, separator, detail = evidence.partition(':')
    if separator and detail and source == 'label':
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
    mentions_value: str,
) -> str:
    """Build one canonical Slack assignment notice."""
    repo = _repository(repo)
    if decision['evidence'] == 'default:repo-intake':
        # Blanket intake routing would ping the same person on every drained
        # item; the channel record keeps the plain name and GitHub's own
        # assignment notification does the alerting.
        mention = decision['owner']
    else:
        mention = attention.slack_mentions(mentions_value, decision['owner'])[decision['owner']]
    if item_type == 'Issue':
        kind, path = 'Issue', 'issues'
    else:
        kind, path = 'Pull request', 'pull'
    number = decision['number']
    item = f'<https://github.com/{repo}/{path}/{number}|{repo}#{number}>'
    text = f'Routing intent: {kind} {item} → {mention}\nWhy: {_routing_reason(decision)}'
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
    current = decision_for(client, repo, expected['number'])
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
                community_recovery=os.environ.get('ROUTING_COMMUNITY_RECOVERY') == 'true',
            )
            decisions = [selection['decision'] for selection in selected if selection['decision'] is not None]
            _output(
                {
                    'should_assign': str(bool(decisions)).lower(),
                    'routes': json.dumps(decisions, separators=(',', ':')),
                }
            )
            if decisions:
                _summary(', '.join(f'#{route["number"]}' for route in decisions) + ': route')
            else:
                _summary('nothing-to-route')
            return 0
        if args.number is None or args.owner is None or args.evidence is None:
            parser.error('assign requires --number, --owner, and --evidence')
        expected = Decision(number=_item_number(args.number), owner=args.owner, evidence=args.evidence)
        did_assign = assign(client, repo, expected)
        _emit_event(
            'router.assigned',
            repo=repo,
            number=expected['number'],
            owner=expected['owner'],
            evidence=expected['evidence'],
            did_assign=did_assign,
        )
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
