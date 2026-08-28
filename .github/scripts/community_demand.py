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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

sys.path.insert(0, str(Path(__file__).parent))
import issue_pr_attention_monitor as attention
from triage_models import AgentItem, agent_items, item_labels, parse_time, snapshot_candidates

_IGNORED_DAYS = 14
_MIN_INTERACTIONS = 3
_CANDIDATE_LIMIT = 8
_COMMENT_LIMIT = 20
_COMMENT_TEXT_LIMIT = 700
_BODY_TEXT_LIMIT = 2_000
_SNAPSHOT_LIMIT = 120_000
# p:1/p:2 are already in the assignment lane, and `unplanned`/`duplicate` are
# human decisions demand must not override. p:3/p:4 stay eligible on purpose:
# they are the triage agent's pre-demand scores, and genuine demand is exactly
# the evidence that should promote one.
_EXCLUDED_LABELS = (attention.COMMUNITY_LABEL, *attention.PRIORITY_GATE_LABELS, 'unplanned', 'duplicate')


class Verdict(AgentItem):
    """One validated agent classification of a snapshot candidate."""

    verdict: Literal['genuine', 'artificial', 'unclear']
    confidence: Literal['high', 'medium', 'low']


def _candidate_numbers(client: attention.GitHubClient, repo: str, *, now: dt.datetime) -> list[int]:
    cutoff = (now - dt.timedelta(days=_IGNORED_DAYS)).date().isoformat()
    negatives = ' '.join(f'-label:"{label}"' for label in _EXCLUDED_LABELS)
    query = f'repo:{repo} is:open is:issue no:assignee created:<{cutoff} interactions:>{_MIN_INTERACTIONS} {negatives}'
    # Rotate through the eligible pool week by week: a non-genuine verdict
    # leaves no marker on the issue, so a fixed page would resubmit the same
    # judged candidates every run and starve everything behind them.
    slot = int(now.timestamp()) // int(dt.timedelta(days=7).total_seconds())
    matches = attention.rotated_search(client, query, order='desc', limit=30, slot=slot)
    return [int(match['number']) for match in matches]


def _thread(client: attention.GitHubClient, repo: str, number: int) -> list[dict[str, object]]:
    # Two pages: the final API page alone can hold a single comment.
    comments = client.last_pages(f'/repos/{repo}/issues/{number}/comments', count=2)
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
        labels = item_labels(current)
        interactions = int(current.get('comments') or 0) + int(
            cast(Mapping[str, Any], current.get('reactions') or {}).get('total_count') or 0
        )
        if (
            current.get('state') != 'open'
            or 'pull_request' in current
            or current.get('assignees')
            or labels.intersection(_EXCLUDED_LABELS)
            or parse_time(str(current['created_at'])) > cutoff
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
    # Shedding from the end keeps the sweep alive on a fat backlog instead of
    # failing every week on the same limit; the dropped tail rotates back in.
    while candidates and len(json.dumps(snapshot, indent=2, ensure_ascii=False).encode()) > _SNAPSHOT_LIMIT:
        candidates.pop()
    return snapshot


def write_snapshot(client: attention.GitHubClient, repo: str, path: str, *, now: dt.datetime) -> list[str]:
    """Write one immutable, size-bounded candidate snapshot."""
    snapshot = build_snapshot(client, repo, now=now)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding='utf-8')
    candidates = cast(list[object], snapshot['candidates'])
    return [f'wrote {len(candidates)} community demand candidate(s)']


def apply_verdicts(client: attention.GitHubClient, repo: str, output_path: str, snapshot_path: str) -> list[str]:
    """Revalidate allowlisted model verdicts, then label genuine demand."""
    candidates = snapshot_candidates(snapshot_path, limit=_CANDIDATE_LIMIT)
    verdicts = agent_items(output_path, Verdict, tag='record_community_verdict', limit=_CANDIDATE_LIMIT)
    if {entry.item_number for entry in verdicts} != candidates.keys():
        raise ValueError('Agent output must classify every snapshot candidate exactly once')
    attention.ensure_labels(client, repo)
    lines: list[str] = []
    failures: list[str] = []
    for entry in verdicts:
        number = entry.item_number
        try:
            current = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if (
                current.get('state') != 'open'
                or str(current.get('updated_at')) != candidates[number]
                or current.get('assignees')
                or item_labels(current).intersection(_EXCLUDED_LABELS)
            ):
                lines.append(f'#{number}: skipped because the issue changed after classification')
                continue
            if entry.verdict != 'genuine' or entry.confidence != 'high':
                lines.append(f'#{number}: left unlabeled ({entry.verdict}, {entry.confidence} confidence)')
                continue
            client.post(f'/repos/{repo}/issues/{number}/labels', {'labels': [attention.COMMUNITY_LABEL]})
            labeled = cast(dict[str, Any], client.get(f'/repos/{repo}/issues/{number}'))
            if attention.COMMUNITY_LABEL not in item_labels(labeled):
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
