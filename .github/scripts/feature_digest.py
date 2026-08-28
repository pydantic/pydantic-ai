#!/usr/bin/env python3
"""Weekly feature digest: surface up to five unconsidered feature requests.

Deterministic code owns eligibility, validation, and every write; the agent only
ranks a bounded immutable snapshot. A surfaced feature gets the
`digest:considered` label so it is never surfaced again; a human removing that
label returns it to the pool. Community pressure on an ignored feature still
reaches maintainers through the routing community lane, so "considered" means
"will not proactively resurface", not "can never reach us".
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict, cast  # noqa: TID251

import issue_pr_attention_monitor as attention

REPO = 'pydantic/pydantic-ai'
FEATURE_LABEL = 'pydanty:feature'
MODEL_REQUEST_LABEL = 'pydanty:model-request'
CONSIDERED_LABEL = 'digest:considered'
_CONSIDERED_COLOR = 'c2e0c6'
_CONSIDERED_DESCRIPTION = 'Surfaced in a weekly feature digest; the digest will not surface it again'
_CANDIDATE_LIMIT = 25
_PICK_LIMIT = 5
_TITLE_LIMIT = 120
_EXCERPT_LIMIT = 600
_REASON_LIMIT = 240
_SNAPSHOT_LIMIT = 120_000
_MODEL_REQUEST_WINDOW_DAYS = 7
SNAPSHOT_PATH = 'feature-candidates.json'


class Pick(TypedDict):
    """One validated agent selection."""

    item_number: int
    reason: str


def _repository(value: str) -> str:
    if value != REPO:
        raise ValueError('repository is not allowlisted')
    return value


def _slack_escape(value: str) -> str:
    """Mirror the attention monitor's Slack sanitizer for untrusted text."""
    normalized = ' '.join(value.split())
    normalized = ''.join(character for character in normalized if unicodedata.category(character) != 'Cf')
    for character in '*_~`|\\':
        normalized = normalized.replace(character, '')
    normalized = ' '.join(normalized.split())
    return normalized.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _sanitize_reason(value: str) -> str:
    """Bound model-written text that was derived from untrusted issue bodies."""
    text = _slack_escape(value)[:_REASON_LIMIT].strip()
    return text or 'selected by the weekly review'


def _excerpt(value: object) -> str:
    if not isinstance(value, str):
        return ''
    return ' '.join(value.split())[:_EXCERPT_LIMIT]


def eligible_query() -> str:
    """Open, never-surfaced, unowned feature requests; model asks are counted separately."""
    return (
        f'repo:{REPO} is:open is:issue label:"{FEATURE_LABEL}" '
        f'-label:"{CONSIDERED_LABEL}" -label:"{MODEL_REQUEST_LABEL}" no:assignee no:milestone'
    )


def _search(client: attention.GitHubClient, query: str, *, sort: str, per_page: int) -> Mapping[str, Any]:
    encoded = urllib.parse.quote_plus(query)
    result = client.get(f'/search/issues?q={encoded}&sort={sort}&order=desc&per_page={per_page}')
    if not isinstance(result, Mapping):
        raise RuntimeError('GitHub returned a malformed search result')
    return cast(Mapping[str, Any], result)


def _model_request_count(client: attention.GitHubClient, *, now: dt.datetime) -> int:
    since = (now - dt.timedelta(days=_MODEL_REQUEST_WINDOW_DAYS)).date().isoformat()
    query = f'repo:{REPO} is:issue label:"{MODEL_REQUEST_LABEL}" created:>={since}'
    count = _search(client, query, sort='created', per_page=1).get('total_count')
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise RuntimeError('GitHub returned a malformed search count')
    return count


def build_snapshot(client: attention.GitHubClient, *, now: dt.datetime) -> dict[str, Any]:
    """Build one immutable, demand-ranked, size-bounded candidate snapshot."""
    result = _search(client, eligible_query(), sort='interactions', per_page=_CANDIDATE_LIMIT)
    items = result.get('items')
    if not isinstance(items, list):
        raise RuntimeError('GitHub returned a malformed search result')
    candidates: list[dict[str, Any]] = []
    for value in cast(list[object], items):
        if not isinstance(value, Mapping):
            raise RuntimeError('GitHub returned a malformed search item')
        item = cast(Mapping[str, Any], value)
        if 'pull_request' in item:
            continue
        number = item.get('number')
        title = item.get('title')
        updated_at = item.get('updated_at')
        created_at = item.get('created_at')
        if (
            not isinstance(number, int)
            or isinstance(number, bool)
            or number < 1
            or not isinstance(title, str)
            or not isinstance(updated_at, str)
            or not isinstance(created_at, str)
        ):
            raise RuntimeError('GitHub returned a malformed search item')
        comments = item.get('comments')
        reactions = item.get('reactions')
        reaction_count = (
            cast(Mapping[str, object], reactions).get('total_count') if isinstance(reactions, Mapping) else 0
        )
        candidates.append(
            {
                'number': number,
                'title': ' '.join(title.split())[: _TITLE_LIMIT * 2],
                'excerpt': _excerpt(item.get('body')),
                'created_at': created_at,
                'updated_at': updated_at,
                'comments': comments if isinstance(comments, int) and not isinstance(comments, bool) else 0,
                'reactions': reaction_count if isinstance(reaction_count, int) else 0,
            }
        )
    snapshot = {
        'repo': REPO,
        'generated_at': now.isoformat(),
        'model_requests_last_week': _model_request_count(client, now=now),
        'candidates': candidates,
    }
    if len(json.dumps(snapshot, indent=2, ensure_ascii=False).encode()) > _SNAPSHOT_LIMIT:
        raise RuntimeError(f'Feature snapshot exceeds {_SNAPSHOT_LIMIT} bytes')
    return snapshot


def write_snapshot(client: attention.GitHubClient, path: str, *, now: dt.datetime) -> list[str]:
    """Write the immutable candidate snapshot the agent and apply step both trust."""
    snapshot = build_snapshot(client, now=now)
    Path(path).write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    candidates = cast(list[object], snapshot['candidates'])
    return [f'wrote {len(candidates)} feature candidate(s)']


class _Snapshot(TypedDict):
    candidates: dict[int, dict[str, str]]
    model_requests_last_week: int


def _load_snapshot(path: str) -> _Snapshot:
    """Return the trusted candidate map (number -> snapshot title and timestamp)."""
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Snapshot must contain a candidates list')
    data = cast(Mapping[str, object], loaded)
    raw_candidates = data.get('candidates')
    model_requests = data.get('model_requests_last_week')
    if not isinstance(raw_candidates, list) or not isinstance(model_requests, int) or model_requests < 0:
        raise ValueError('Snapshot must contain a candidates list and a model-request count')
    candidates: dict[int, dict[str, str]] = {}
    for value in cast(list[object], raw_candidates):
        if not isinstance(value, Mapping):
            raise ValueError('Snapshot candidate must be an object')
        candidate = cast(Mapping[str, object], value)
        number = candidate.get('number')
        updated_at = candidate.get('updated_at')
        title = candidate.get('title')
        if (
            not isinstance(number, int)
            or number < 1
            or number in candidates
            or not isinstance(updated_at, str)
            or not isinstance(title, str)
        ):
            raise ValueError('Snapshot candidates must have unique positive numbers, titles, and timestamps')
        candidates[number] = {'updated_at': updated_at, 'title': title}
    if len(candidates) > _CANDIDATE_LIMIT:
        raise ValueError('Snapshot exceeds the candidate limit')
    return _Snapshot(candidates=candidates, model_requests_last_week=model_requests)


def _parse_picks(path: str) -> list[Pick]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Agent output must contain an items list')
    raw_items = cast(Mapping[str, object], loaded).get('items')
    if not isinstance(raw_items, list):
        raise ValueError('Agent output must contain an items list')
    picks: list[Pick] = []
    for value in cast(list[object], raw_items):
        if not isinstance(value, Mapping):
            continue
        pick = cast(Mapping[str, object], value)
        if pick.get('type') != 'record_feature_pick':
            continue
        number = pick.get('item_number')
        reason = pick.get('reason')
        if not isinstance(number, str) or re.fullmatch(r'[1-9][0-9]*', number) is None:
            raise ValueError('Pick item_number must be a positive decimal string')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('Pick reason must be a non-empty string')
        picks.append(Pick(item_number=int(number), reason=reason))
    numbers = [pick['item_number'] for pick in picks]
    if len(numbers) > _PICK_LIMIT or len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains too many or duplicate picks')
    return picks


def ensure_considered_label(client: attention.GitHubClient) -> None:
    """Create the `digest:considered` label if it does not exist yet."""
    encoded = urllib.parse.quote(CONSIDERED_LABEL, safe='')
    try:
        client.get(f'/repos/{REPO}/labels/{encoded}')
        return
    except urllib.error.HTTPError as exc:
        exc.close()
        if exc.code != 404:
            raise
    client.post(
        f'/repos/{REPO}/labels',
        {'name': CONSIDERED_LABEL, 'color': _CONSIDERED_COLOR, 'description': _CONSIDERED_DESCRIPTION},
    )


def _labels(item: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for entry in item.get('labels', []):
        if isinstance(entry, Mapping):
            name = cast(Mapping[str, object], entry).get('name')
            if isinstance(name, str):
                values.add(name)
    return values


def apply_picks(
    client: attention.GitHubClient,
    output_path: str,
    snapshot_path: str,
    *,
    now: dt.datetime,
) -> tuple[list[str], str | None]:
    """Revalidate the agent's picks, label them, and build the Slack digest."""
    snapshot = _load_snapshot(snapshot_path)
    picks = _parse_picks(output_path)
    unknown = {pick['item_number'] for pick in picks} - snapshot['candidates'].keys()
    if unknown:
        raise ValueError(f'Agent output contains numbers outside the snapshot: {sorted(unknown)}')
    ensure_considered_label(client)
    lines: list[str] = []
    bullets: list[str] = []
    for pick in picks:
        number = pick['item_number']
        current = cast(dict[str, Any], client.get(f'/repos/{REPO}/issues/{number}'))
        labels = _labels(current)
        if (
            str(current.get('state') or '').casefold() != 'open'
            or str(current.get('updated_at')) != snapshot['candidates'][number]['updated_at']
            or CONSIDERED_LABEL in labels
            or FEATURE_LABEL not in labels
        ):
            lines.append(f'#{number}: skipped because the item changed after selection')
            continue
        client.post(f'/repos/{REPO}/issues/{number}/labels', {'labels': [CONSIDERED_LABEL]})
        title = _slack_escape(snapshot['candidates'][number]['title'])[:_TITLE_LIMIT]
        reason = _sanitize_reason(pick['reason'])
        bullets.append(f'• <https://github.com/{REPO}/issues/{number}|#{number} {title}> — {reason}')
        lines.append(f'#{number}: surfaced in the weekly feature digest')
    if not bullets:
        return lines or ['no picks to surface'], None
    text_lines = [
        f':bulb: *Weekly feature digest — {REPO}* · {now.date().isoformat()}',
        *bullets,
    ]
    if model_requests := snapshot['model_requests_last_week']:
        noun = 'request' if model_requests == 1 else 'requests'
        text_lines.append(f'+ {model_requests} new model {noun} this week')
    text_lines.append(
        'React in the issue, milestone it, or close it as not planned — surfaced features are not shown again.'
    )
    return lines, '\n'.join(text_lines)


def _write_outputs(payload: str | None) -> None:
    if output_path := os.environ.get('GITHUB_OUTPUT'):
        with Path(output_path).open('a', encoding='utf-8') as output:
            output.write(f'should_post={str(payload is not None).lower()}\n')
            if payload is not None:
                output.write(f'slack_payload={json.dumps({"text": payload}, separators=(",", ":"))}\n')


def main() -> int:
    """Build the candidate snapshot or apply validated picks."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'apply'])
    parser.add_argument('--snapshot-path', default=SNAPSHOT_PATH)
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    try:
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if not token:
            raise ValueError('GITHUB_TOKEN or GH_TOKEN is required')
        _repository(os.environ.get('GITHUB_REPOSITORY', REPO))
        client = attention.GitHubClient(token)
        now = dt.datetime.now(dt.timezone.utc)
        if args.mode == 'snapshot':
            lines = write_snapshot(client, args.snapshot_path, now=now)
        else:
            if not args.agent_output:
                parser.error('--agent-output is required')
            lines, payload = apply_picks(client, args.agent_output, args.snapshot_path, now=now)
            _write_outputs(payload)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        error = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            error += f' {exc.code}'
        print(f'feature digest failed: {error}', file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
