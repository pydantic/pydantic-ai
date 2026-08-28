#!/usr/bin/env python3
"""Weekly feature digest: surface up to five unconsidered feature requests.

Deterministic code owns eligibility, validation, and every write; the agent only
ranks a bounded immutable snapshot. A surfaced feature gets the
`digest:considered` label — only after the Slack post succeeds, so a failed
delivery leaves the picks in the pool — and is never surfaced again; a human
removing that label returns it to the pool.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import unicodedata
import urllib.error
import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import issue_pr_attention_monitor as attention
from pydantic import BaseModel, Field, field_validator
from triage_models import AgentItem, agent_items, item_labels

REPO = 'pydantic/pydantic-ai'
# The triage agent applies `pydanty:feature`; older and human-triaged requests
# carry plain `feature`. Both pools are eligible.
FEATURE_LABELS = ('pydanty:feature', 'feature')
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


class Pick(AgentItem):
    """One validated agent selection."""

    reason: str

    @field_validator('reason')
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError('reason must be a non-empty string')
        return value


def _repository(value: str) -> str:
    if value != REPO:
        raise ValueError('repository is not allowlisted')
    return value


def _plain(value: str) -> str:
    """Collapse whitespace and drop every control/format character."""
    collapsed = ' '.join(value.split())
    return ''.join(character for character in collapsed if not unicodedata.category(character).startswith('C'))


def _slack_escape(value: str) -> str:
    normalized = _plain(value)
    for character in '*_~`|\\':
        normalized = normalized.replace(character, '')
    normalized = ' '.join(normalized.split())
    return normalized.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _sanitize_reason(value: str) -> str:
    """Bound model-written text that was derived from untrusted issue bodies.

    Links and channel-wide mention keywords are dropped, not escaped: a
    prompt-injected reason must not be able to make anyone click or ping.
    """
    # Filter after `_plain` so invisible characters cannot reassemble a
    # mention or link once they are stripped downstream.
    words = [
        word
        for word in _plain(value).split()
        if '://' not in word
        and not word.casefold().startswith('www.')
        and not any(mention in word.casefold() for mention in ('@channel', '@here', '@everyone'))
    ]
    text = _slack_escape(' '.join(words)[:_REASON_LIMIT]).strip()
    return text or 'selected by the weekly review'


def _excerpt(value: object) -> str:
    if not isinstance(value, str):
        return ''
    return _plain(value)[:_EXCERPT_LIMIT]


def eligible_query() -> str:
    """Open, never-surfaced, unowned feature requests; model asks are counted separately."""
    either_feature = ','.join(f'"{label}"' for label in FEATURE_LABELS)
    return (
        f'repo:{REPO} is:open is:issue label:{either_feature} '
        f'-label:"{CONSIDERED_LABEL}" -label:"{MODEL_REQUEST_LABEL}" no:assignee no:milestone'
    )


def _search(client: attention.GitHubClient, query: str, *, sort: str, per_page: int) -> Mapping[str, Any]:
    encoded = urllib.parse.quote_plus(query)
    result = client.get(f'/search/issues?q={encoded}&sort={sort}&order=desc&per_page={per_page}')
    if not isinstance(result, Mapping):
        raise RuntimeError('GitHub returned a malformed search result')
    return cast(Mapping[str, Any], result)


def _model_request_count(client: attention.GitHubClient, *, now: dt.datetime) -> int:
    # Full timestamp: truncating to a date widens the window and double-counts across consecutive runs.
    since = (now - dt.timedelta(days=_MODEL_REQUEST_WINDOW_DAYS)).replace(microsecond=0).isoformat()
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
                'title': _plain(title)[: _TITLE_LIMIT * 2],
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


class _Candidate(BaseModel):
    number: int = Field(ge=1, strict=True)
    updated_at: str
    title: str


class _SnapshotFile(BaseModel):
    candidates: list[_Candidate] = Field(max_length=_CANDIDATE_LIMIT)
    model_requests_last_week: int = Field(ge=0, strict=True)


def _load_snapshot(path: str) -> tuple[dict[int, _Candidate], int]:
    """Return the trusted candidate map (number -> snapshot title and timestamp)."""
    loaded = _SnapshotFile.model_validate_json(Path(path).read_text(encoding='utf-8'))
    candidates = {candidate.number: candidate for candidate in loaded.candidates}
    if len(candidates) != len(loaded.candidates):
        raise ValueError('Snapshot candidates must have unique numbers')
    return candidates, loaded.model_requests_last_week


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


def apply_picks(
    client: attention.GitHubClient,
    output_path: str,
    snapshot_path: str,
    *,
    now: dt.datetime,
) -> tuple[list[str], str | None, list[int]]:
    """Revalidate the agent's picks and build the Slack digest.

    Labeling happens in `finalize_picks`, only after the Slack post succeeds:
    a failed delivery must leave the picks in the pool, not consume them.
    """
    candidates, model_requests = _load_snapshot(snapshot_path)
    picks = agent_items(output_path, Pick, tag='record_feature_pick', limit=_PICK_LIMIT)
    unknown = {pick.item_number for pick in picks} - candidates.keys()
    if unknown:
        raise ValueError(f'Agent output contains numbers outside the snapshot: {sorted(unknown)}')
    lines: list[str] = []
    bullets: list[str] = []
    surfaced: list[int] = []
    for pick in picks:
        number = pick.item_number
        current = cast(dict[str, Any], client.get(f'/repos/{REPO}/issues/{number}'))
        labels = item_labels(current)
        if (
            str(current.get('state') or '').casefold() != 'open'
            or str(current.get('updated_at')) != candidates[number].updated_at
            or CONSIDERED_LABEL in labels
            or not labels.intersection(FEATURE_LABELS)
        ):
            lines.append(f'#{number}: skipped because the item changed after selection')
            continue
        title = _slack_escape(candidates[number].title[:_TITLE_LIMIT])
        reason = _sanitize_reason(pick.reason)
        bullets.append(f'• <https://github.com/{REPO}/issues/{number}|#{number} {title}> — {reason}')
        surfaced.append(number)
        lines.append(f'#{number}: surfaced in the weekly feature digest')
    if not bullets:
        return lines or ['no picks to surface'], None, []
    text_lines = [
        f':bulb: *Weekly feature digest — {REPO}* · {now.date().isoformat()}',
        *bullets,
    ]
    if model_requests:
        noun = 'request' if model_requests == 1 else 'requests'
        text_lines.append(f'+ {model_requests} new model {noun} this week')
    text_lines.append(
        'React in the issue, milestone it, or close it as not planned — surfaced features are not shown again.'
    )
    return lines, '\n'.join(text_lines), surfaced


def finalize_picks(client: attention.GitHubClient, numbers: list[int]) -> tuple[list[str], list[int]]:
    """Mark delivered picks `digest:considered`; runs only after the Slack post succeeded.

    One pick failing must not leave the rest unlabeled (they were all posted), so
    failures are collected per item and the caller turns them into a red run.
    """
    ensure_considered_label(client)
    lines: list[str] = []
    failed: list[int] = []
    for number in numbers:
        try:
            current = cast(dict[str, Any], client.get(f'/repos/{REPO}/issues/{number}'))
            # A pick closed since delivery is still labeled (labels apply to
            # closed issues): reopening must not surface it a second time.
            if CONSIDERED_LABEL in item_labels(current):
                lines.append(f'#{number}: already marked, not relabeled')
                continue
            client.post(f'/repos/{REPO}/issues/{number}/labels', {'labels': [CONSIDERED_LABEL]})
            lines.append(f'#{number}: marked considered')
        except (urllib.error.URLError, RuntimeError, ValueError) as exc:
            error = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                error += f' {exc.code}'
                exc.close()
            failed.append(number)
            lines.append(f'#{number}: not relabeled ({error}); it may be surfaced again')
    return lines, failed


def _picked_numbers(value: str) -> list[int]:
    loaded: object = json.loads(value)
    if not isinstance(loaded, list) or not all(type(item) is int and 0 < item for item in cast(list[object], loaded)):
        raise ValueError('DIGEST_PICKED must be a JSON list of positive issue numbers')
    numbers = cast(list[int], loaded)
    if len(numbers) > _PICK_LIMIT or len(numbers) != len(set(numbers)):
        raise ValueError('DIGEST_PICKED contains too many or duplicate numbers')
    return numbers


def _write_outputs(payload: str | None, surfaced: list[int]) -> None:
    if output_path := os.environ.get('GITHUB_OUTPUT'):
        with Path(output_path).open('a', encoding='utf-8') as output:
            output.write(f'should_post={str(payload is not None).lower()}\n')
            if payload is not None:
                output.write(f'slack_payload={json.dumps({"text": payload}, separators=(",", ":"))}\n')
                output.write(f'picked_numbers={json.dumps(surfaced, separators=(",", ":"))}\n')


def main() -> int:
    """Build the candidate snapshot or apply validated picks."""
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['snapshot', 'apply', 'finalize'])
    parser.add_argument('--snapshot-path', default=SNAPSHOT_PATH)
    parser.add_argument('--agent-output', default=os.environ.get('GH_AW_AGENT_OUTPUT'))
    args = parser.parse_args()
    failed: list[int] = []
    try:
        token = os.environ.get('GITHUB_TOKEN') or os.environ.get('GH_TOKEN')
        if not token:
            raise ValueError('GITHUB_TOKEN or GH_TOKEN is required')
        _repository(os.environ.get('GITHUB_REPOSITORY', REPO))
        client = attention.GitHubClient(token)
        now = dt.datetime.now(dt.timezone.utc)
        if args.mode == 'snapshot':
            lines = write_snapshot(client, args.snapshot_path, now=now)
        elif args.mode == 'finalize':
            picked = os.environ.get('DIGEST_PICKED')
            if picked is None:
                raise ValueError('DIGEST_PICKED is required')
            lines, failed = finalize_picks(client, _picked_numbers(picked))
        else:
            if not args.agent_output:
                parser.error('--agent-output is required')
            lines, payload, surfaced = apply_picks(client, args.agent_output, args.snapshot_path, now=now)
            _write_outputs(payload, surfaced)
    except (KeyError, OSError, ValueError, RuntimeError) as exc:
        error = type(exc).__name__
        if isinstance(exc, urllib.error.HTTPError):
            error += f' {exc.code}'
        print(f'feature digest failed: {error}', file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    if failed:
        print(f'feature digest failed: {len(failed)} delivered pick(s) not relabeled', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
