#!/usr/bin/env python3
"""Assign new issues and pull requests to a semantic maintainer owner."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
from collections.abc import Mapping, Sequence
from pathlib import Path

# This script runs under bare Python in Actions and intentionally shares the
# attention monitor's bounded GitHub client and permission checks.
from typing import Any, Literal, TypedDict, cast  # noqa: TID251

import issue_pr_attention_monitor as attention

_CANDIDATE_LIMIT = 10
_SNAPSHOT_LIMIT = 80_000
_FILE_LIMIT = 30

Route = Literal[
    'aditya-streaming-runtime',
    'david-model-integrations',
    'douwe-durable-architecture',
    'mike-tools-harness',
    'aditya-manual-route',
]


class RoutePolicy(TypedDict):
    """The fixed owner and channel explanation for one semantic route."""

    owner: str
    basis: str


_ROUTES: dict[Route, RoutePolicy] = {
    'aditya-streaming-runtime': {
        'owner': 'adtyavrdhn',
        'basis': 'streaming, cancellation, UI protocols, or CodeMode runtime',
    },
    'david-model-integrations': {
        'owner': 'dsfaccini',
        'basis': 'model/provider adapters, message mapping, compaction, or compatibility',
    },
    'douwe-durable-architecture': {
        'owner': 'DouweM',
        'basis': 'durable execution, deferred work, capability lifecycle, or identity',
    },
    'mike-tools-harness': {
        'owner': 'mpfaffenberger',
        'basis': 'tools, TestModel, general Harness capabilities, or contributor APIs',
    },
    'aditya-manual-route': {
        'owner': 'adtyavrdhn',
        'basis': 'no specialist route was clear; manual routing is required',
    },
}


def _labels(item: Mapping[str, Any]) -> set[str]:
    return {str(label['name']) for label in item.get('labels', [])}


def _maintainer_assignees(client: attention.GitHubClient, repo: str, item: Mapping[str, Any]) -> list[str]:
    return sorted(
        (
            maintainer
            for assignee in item.get('assignees', [])
            if (login := str(assignee['login'])) and (maintainer := client.maintainer_login(repo, login))
        ),
        key=str.casefold,
    )


def _slack_escape(value: str) -> str:
    normalized = value.replace('\r\n', '\n').replace('\r', '\n').replace('\n', ' ')
    return normalized.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class RouteDecision(TypedDict):
    """The complete model-controlled surface."""

    item_number: int
    route: Route


class AssignmentNotice(TypedDict):
    """One host-built Slack assignment line."""

    number: int
    title: str
    owner: str
    basis: str


def _event_number(path: str | None) -> int | None:
    if not path:
        return None
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('GitHub event must be an object')
    event = cast(Mapping[str, object], loaded)
    for key in ('issue', 'pull_request'):
        value = event.get(key)
        if isinstance(value, Mapping):
            number = cast(Mapping[str, object], value).get('number')
            if isinstance(number, int) and number > 0:
                return number
    return None


def _unassigned_page(client: attention.GitHubClient, repo: str) -> list[dict[str, Any]]:
    query = urllib.parse.quote_plus(f'repo:{repo} is:open no:assignee')
    result = cast(
        dict[str, Any],
        client.get(f'/search/issues?q={query}&sort=created&order=asc&per_page={_CANDIDATE_LIMIT}'),
    )
    return cast(list[dict[str, Any]], result.get('items') or [])


def _candidate(client: attention.GitHubClient, repo: str, number: int) -> dict[str, object] | None:
    current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
    if current.get('state') != 'open' or _maintainer_assignees(client, repo, current):
        return None
    files: list[str] = []
    if 'pull_request' in current:
        changed = cast(list[dict[str, Any]], client.get(f'/repos/{repo}/pulls/{number}/files?per_page=100'))
        files = [str(value.get('filename') or '')[:300] for value in changed[:_FILE_LIMIT]]
    user = current.get('user')
    author = str(cast(Mapping[str, object], user).get('login') or '') if isinstance(user, Mapping) else ''
    return {
        'number': number,
        'kind': 'pull_request' if 'pull_request' in current else 'issue',
        'title': str(current.get('title') or '')[:300],
        'body': str(current.get('body') or '')[:3_000],
        'author': author,
        'updated_at': str(current.get('updated_at') or ''),
        'labels': sorted(_labels(current)),
        'files': files,
    }


def build_snapshot(
    client: attention.GitHubClient,
    repo: str,
    *,
    event_path: str | None = None,
) -> dict[str, object]:
    """Build one exact event candidate or a bounded unassigned safety batch."""
    if number := _event_number(event_path):
        numbers = [number]
    else:
        numbers = [int(value['number']) for value in _unassigned_page(client, repo)]
    candidates = [candidate for number in numbers if (candidate := _candidate(client, repo, number)) is not None]
    snapshot: dict[str, object] = {'candidates': candidates}
    if len(json.dumps(snapshot, indent=2, ensure_ascii=False).encode()) > _SNAPSHOT_LIMIT:
        raise RuntimeError(f'Owner-routing snapshot exceeds {_SNAPSHOT_LIMIT} bytes')
    return snapshot


def write_snapshot(
    client: attention.GitHubClient,
    repo: str,
    path: str,
    *,
    event_path: str | None = None,
) -> list[str]:
    """Write the immutable input consumed by the sandboxed router."""
    snapshot = build_snapshot(client, repo, event_path=event_path)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    candidates = cast(list[object], snapshot['candidates'])
    return [f'wrote {len(candidates)} semantic owner candidate(s)']


def _snapshot_candidates(path: str) -> dict[int, str]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Snapshot must contain a candidates list')
    values = cast(Mapping[str, object], loaded).get('candidates')
    if not isinstance(values, list):
        raise ValueError('Snapshot must contain a candidates list')
    candidates: dict[int, str] = {}
    for value in cast(list[object], values):
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


def parse_decisions(path: str) -> list[RouteDecision]:
    """Parse and validate the bounded semantic route outputs."""
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Agent output must contain an items list')
    values = cast(Mapping[str, object], loaded).get('items')
    if not isinstance(values, list):
        raise ValueError('Agent output must contain an items list')
    decisions: list[RouteDecision] = []
    for value in cast(list[object], values):
        if not isinstance(value, Mapping):
            continue
        decision = cast(Mapping[str, object], value)
        if decision.get('type') != 'route_maintainer_owner':
            continue
        number = decision.get('item_number')
        route = decision.get('route')
        if not isinstance(number, str) or re.fullmatch(r'[1-9][0-9]*', number) is None:
            raise ValueError('Decision item_number must be a positive decimal string')
        if route not in _ROUTES:
            raise ValueError(f'Invalid semantic route: {route!r}')
        decisions.append(RouteDecision(item_number=int(number), route=route))
    numbers = [decision['item_number'] for decision in decisions]
    if len(numbers) > _CANDIDATE_LIMIT or len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains too many or duplicate decisions')
    return decisions


_SLACK_MENTION = re.compile(r'<@[UW][A-Z0-9]+>')


def parse_slack_mentions(value: str) -> dict[str, str]:
    """Validate the caller-owned GitHub-login to Slack-member mapping."""
    loaded: object = json.loads(value)
    if not isinstance(loaded, Mapping):
        raise ValueError('Slack mention mapping must be an object')
    mentions = {str(key): str(mention) for key, mention in cast(Mapping[object, object], loaded).items()}
    owners = {policy['owner'] for policy in _ROUTES.values()}
    if set(mentions) != owners or any(_SLACK_MENTION.fullmatch(mention) is None for mention in mentions.values()):
        raise ValueError('Slack mention mapping must contain one valid member mention for every semantic owner')
    return mentions


def apply_routes(
    client: attention.GitHubClient,
    repo: str,
    output_path: str,
    snapshot_path: str,
) -> tuple[list[str], list[AssignmentNotice]]:
    """Revalidate every route, preserve human ownership, then assign."""
    candidates = _snapshot_candidates(snapshot_path)
    decisions = parse_decisions(output_path)
    unknown = {decision['item_number'] for decision in decisions} - candidates.keys()
    if unknown:
        raise ValueError(f'Agent output contains numbers outside the snapshot: {sorted(unknown)}')
    if {decision['item_number'] for decision in decisions} != candidates.keys():
        raise ValueError('Agent output must route every snapshot candidate exactly once')
    lines: list[str] = []
    notices: list[AssignmentNotice] = []
    failures: list[str] = []
    for decision in decisions:
        number = decision['item_number']
        policy = _ROUTES[decision['route']]
        try:
            current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if current.get('state') != 'open' or str(current.get('updated_at') or '') != candidates[number]:
                lines.append(f'#{number}: skipped because the item changed after routing')
                continue
            if maintainers := _maintainer_assignees(client, repo, current):
                lines.append(
                    f'#{number}: kept existing maintainer owner {" ".join(f"@{login}" for login in maintainers)}'
                )
                continue
            owner = policy['owner']
            client.post(f'/repos/{repo}/issues/{number}/assignees', {'assignees': [owner]})
            assigned = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            assigned_maintainers = _maintainer_assignees(client, repo, assigned)
            if assigned.get('state') != 'open' or owner.casefold() not in {
                login.casefold() for login in assigned_maintainers
            }:
                raise RuntimeError(f'GitHub did not assign @{owner}')
            lines.append(f'#{number}: routed to @{owner} for {policy["basis"]}')
            notices.append(
                AssignmentNotice(
                    number=number,
                    title=str(current.get('title') or '')[:300],
                    owner=owner,
                    basis=policy['basis'],
                )
            )
        except (urllib.error.URLError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('Failed to apply semantic owner routes: ' + '; '.join(failures))
    return lines, notices


def write_notifications(repo: str, notices: Sequence[AssignmentNotice], mentions: Mapping[str, str]) -> None:
    """Write one fixed Slack digest for assignments made by this run."""
    if not (output_path := os.environ.get('GITHUB_OUTPUT')):
        return
    details = [
        f'• <https://github.com/{repo}/issues/{notice["number"]}|#{notice["number"]} '
        f'{_slack_escape(notice["title"]) or "(untitled)"}> → {mentions[notice["owner"]]}'
        f'\n      why: {notice["basis"]}'
        for notice in notices
    ]
    payload = {'text': f':label: Semantic owner routing for {repo}\n' + '\n'.join(details)}
    with Path(output_path).open('a', encoding='utf-8') as output:
        output.write(f'has_assignments={str(bool(notices)).lower()}\n')
        output.write(f'slack_payload={json.dumps(payload, separators=(",", ":"))}\n')


def _write_summary(lines: Sequence[str]) -> None:
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(path).open('a', encoding='utf-8') as summary:
            summary.write('## Semantic owner routing\n\n')
            summary.write('\n'.join(f'- {line}' for line in lines) or '- No changes')
            summary.write('\n')


def main() -> int:
    """Build an owner snapshot or apply validated semantic routes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'apply'])
    parser.add_argument('--snapshot-path', default='owner-routing-candidates.json')
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token:
        print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
        return 1
    client = attention.GitHubClient(token)
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    if args.mode == 'snapshot':
        lines = write_snapshot(client, repo, args.snapshot_path, event_path=os.environ.get('GITHUB_EVENT_PATH'))
    else:
        if not args.agent_output:
            parser.error('--agent-output is required')
        mentions_value = os.environ.get('PYDANTIC_AI_TRIAGE_SLACK_MENTIONS')
        if mentions_value is None:
            parser.error('PYDANTIC_AI_TRIAGE_SLACK_MENTIONS is required')
        mentions = parse_slack_mentions(mentions_value)
        lines, notices = apply_routes(client, repo, args.agent_output, args.snapshot_path)
        write_notifications(repo, notices, mentions)
    _write_summary(lines)
    for line in lines:
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
