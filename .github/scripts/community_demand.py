"""Judge community demand on old unassigned issues and label the genuine ones.

An issue that sits unassigned for weeks while people keep commenting either has
real users asking for it or an AI-generated pile-on, and raw interaction counts
cannot tell the two apart. `snapshot` writes a bounded candidate file carrying
the actual comment threads for the sandboxed agent to read; `apply` revalidates
the agent's verdicts against that snapshot and adds the `community-backed`
label only for genuine demand judged with high confidence. The label opens the
assignment gate in `semantic_owner_router` and the weekly reminder cadence in
`issue_pr_attention_monitor`; this script owns every GitHub write, the agent
only classifies.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
from collections.abc import Mapping
from pathlib import Path

# `typing.TypedDict` is fine here: nothing validates these shapes with pydantic.
from typing import Any, Literal, TypedDict, cast  # noqa: TID251

sys.path.insert(0, str(Path(__file__).parent))
import issue_pr_attention_monitor as attention

_IGNORED_DAYS = 14
_MIN_INTERACTIONS = 3
_CANDIDATE_LIMIT = 8
_COMMENT_LIMIT = 20
_COMMENT_TEXT_LIMIT = 700
_BODY_TEXT_LIMIT = 2_000
_SNAPSHOT_LIMIT = 120_000
# A human already made a call on these; community demand does not override it.
_EXCLUDED_LABELS = (attention.COMMUNITY_LABEL, *attention.PRIORITY_GATE_LABELS, 'unplanned', 'duplicate')


class Verdict(TypedDict):
    """One validated agent classification of a snapshot candidate."""

    item_number: int
    verdict: Literal['genuine', 'artificial', 'unclear']
    confidence: Literal['high', 'medium', 'low']


def _labels(item: Mapping[str, Any]) -> set[str]:
    return {str(label['name']) for label in item.get('labels', [])}


def _parse_time(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace('Z', '+00:00'))


def _candidate_numbers(client: attention.GitHubClient, repo: str, *, now: dt.datetime) -> list[int]:
    cutoff = (now - dt.timedelta(days=_IGNORED_DAYS)).date().isoformat()
    negatives = ' '.join(f'-label:"{label}"' for label in _EXCLUDED_LABELS)
    query = f'repo:{repo} is:open is:issue no:assignee created:<{cutoff} interactions:>{_MIN_INTERACTIONS} {negatives}'
    encoded = urllib.parse.urlencode({'q': query, 'sort': 'interactions', 'order': 'desc', 'per_page': 30})
    result = cast(dict[str, Any], client.get(f'/search/issues?{encoded}'))
    items = result.get('items')
    if not isinstance(items, list):
        raise RuntimeError('GitHub search returned no items list')
    return [int(cast(Mapping[str, Any], item)['number']) for item in cast(list[Any], items)]


def _thread(client: attention.GitHubClient, repo: str, number: int) -> list[dict[str, object]]:
    comments = client.last_pages(f'/repos/{repo}/issues/{number}/comments')
    thread: list[dict[str, object]] = []
    for comment in comments[-_COMMENT_LIMIT:]:
        author = cast(Mapping[str, Any], comment.get('user') or {})
        thread.append(
            {
                'author': str(author.get('login') or ''),
                'association': str(comment.get('author_association') or ''),
                'created_at': str(comment.get('created_at') or ''),
                'body': str(comment.get('body') or '')[:_COMMENT_TEXT_LIMIT],
            }
        )
    return thread


def build_snapshot(client: attention.GitHubClient, repo: str, *, now: dt.datetime) -> dict[str, object]:
    """Build the bounded public input consumed by the sandboxed agent."""
    cutoff = now - dt.timedelta(days=_IGNORED_DAYS)
    candidates: list[dict[str, object]] = []
    for number in _candidate_numbers(client, repo, now=now):
        current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
        labels = _labels(current)
        interactions = int(current.get('comments') or 0) + int(
            cast(Mapping[str, Any], current.get('reactions') or {}).get('total_count') or 0
        )
        if (
            current.get('state') != 'open'
            or 'pull_request' in current
            or current.get('assignees')
            or labels.intersection(_EXCLUDED_LABELS)
            or _parse_time(str(current['created_at'])) > cutoff
            or interactions <= _MIN_INTERACTIONS
        ):
            continue
        candidates.append(
            {
                'number': number,
                'title': str(current.get('title') or '')[:300],
                'body': str(current.get('body') or '')[:_BODY_TEXT_LIMIT],
                'created_at': str(current['created_at']),
                'updated_at': str(current['updated_at']),
                'labels': sorted(labels),
                'comment_count': int(current.get('comments') or 0),
                'reaction_count': int(cast(Mapping[str, Any], current.get('reactions') or {}).get('total_count') or 0),
                'recent_comments': _thread(client, repo, number),
            }
        )
        if len(candidates) == _CANDIDATE_LIMIT:
            break
    snapshot: dict[str, object] = {'generated_at': now.isoformat(), 'candidates': candidates}
    if len(json.dumps(snapshot, indent=2, ensure_ascii=False).encode()) > _SNAPSHOT_LIMIT:
        raise RuntimeError(f'Community snapshot exceeds {_SNAPSHOT_LIMIT} bytes')
    return snapshot


def write_snapshot(client: attention.GitHubClient, repo: str, path: str, *, now: dt.datetime) -> list[str]:
    """Write one immutable, size-bounded candidate snapshot."""
    snapshot = build_snapshot(client, repo, now=now)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    candidates = cast(list[object], snapshot['candidates'])
    return [f'wrote {len(candidates)} community demand candidate(s)']


def _snapshot_candidates(path: str) -> dict[int, str]:
    """Return the trusted candidate map (number -> snapshot updated_at)."""
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Snapshot must contain a candidates list')
    raw_candidates = cast(Mapping[str, object], loaded).get('candidates')
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


def _parse_verdicts(path: str) -> list[Verdict]:
    loaded: object = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(loaded, Mapping):
        raise ValueError('Agent output must contain an items list')
    raw_items = cast(Mapping[str, object], loaded).get('items')
    if not isinstance(raw_items, list):
        raise ValueError('Agent output must contain an items list')
    verdicts: list[Verdict] = []
    for value in cast(list[object], raw_items):
        if not isinstance(value, Mapping):
            continue
        entry = cast(Mapping[str, object], value)
        if entry.get('type') != 'record_community_verdict':
            continue
        number = entry.get('item_number')
        verdict = entry.get('verdict')
        confidence = entry.get('confidence')
        if not isinstance(number, str) or not number.isdecimal() or number.startswith('0'):
            raise ValueError('Verdict item_number must be a positive decimal string')
        if verdict not in {'genuine', 'artificial', 'unclear'}:
            raise ValueError(f'Invalid verdict: {verdict!r}')
        if confidence not in {'high', 'medium', 'low'}:
            raise ValueError(f'Invalid confidence: {confidence!r}')
        verdicts.append(
            Verdict(
                item_number=int(number),
                verdict=cast(Literal['genuine', 'artificial', 'unclear'], verdict),
                confidence=cast(Literal['high', 'medium', 'low'], confidence),
            )
        )
    numbers = [entry['item_number'] for entry in verdicts]
    if len(numbers) > _CANDIDATE_LIMIT or len(numbers) != len(set(numbers)):
        raise ValueError('Agent output contains too many or duplicate verdicts')
    return verdicts


def apply_verdicts(client: attention.GitHubClient, repo: str, output_path: str, snapshot_path: str) -> list[str]:
    """Revalidate allowlisted model verdicts, then label genuine demand."""
    candidates = _snapshot_candidates(snapshot_path)
    verdicts = _parse_verdicts(output_path)
    if {entry['item_number'] for entry in verdicts} != candidates.keys():
        raise ValueError('Agent output must classify every snapshot candidate exactly once')
    attention.ensure_labels(client, repo)
    lines: list[str] = []
    failures: list[str] = []
    for entry in verdicts:
        number = entry['item_number']
        try:
            current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if (
                current.get('state') != 'open'
                or str(current.get('updated_at')) != candidates[number]
                or current.get('assignees')
                or _labels(current).intersection(_EXCLUDED_LABELS)
            ):
                lines.append(f'#{number}: skipped because the issue changed after classification')
                continue
            if entry['verdict'] != 'genuine' or entry['confidence'] != 'high':
                lines.append(f'#{number}: left unlabeled ({entry["verdict"]}, {entry["confidence"]} confidence)')
                continue
            client.post(f'/repos/{repo}/issues/{number}/labels', {'labels': [attention.COMMUNITY_LABEL]})
            labeled = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if attention.COMMUNITY_LABEL not in _labels(labeled):
                raise RuntimeError('GitHub did not apply the community label')
            lines.append(f'#{number}: marked as genuine community demand')
        except (urllib.error.URLError, RuntimeError) as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            failures.append(f'#{number}: {type(exc).__name__}: {exc}')
    if failures:
        raise RuntimeError('Failed to apply community verdicts: ' + '; '.join(failures))
    return lines


def _write_summary(lines: list[str]) -> None:
    if path := os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(path).open('a', encoding='utf-8') as summary:
            for line in lines:
                summary.write(f'- {line}\n')


def main() -> int:
    """Build the candidate snapshot or apply validated verdicts."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'apply'])
    parser.add_argument('--snapshot-path', default='community-candidates.json')
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
    if not token:
        print('GITHUB_TOKEN or GH_TOKEN is required', file=sys.stderr)
        return 1
    client = attention.GitHubClient(token)
    repo = os.environ.get('GITHUB_REPOSITORY', 'pydantic/pydantic-ai')
    if args.mode == 'snapshot':
        lines = write_snapshot(client, repo, args.snapshot_path, now=dt.datetime.now(dt.timezone.utc))
    else:
        if not args.agent_output:
            parser.error('--agent-output is required')
        lines = apply_verdicts(client, repo, args.agent_output, args.snapshot_path)
    _write_summary(lines)
    for line in lines:
        print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
