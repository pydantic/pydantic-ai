"""Repo memory proposals: what developers' agents proposed to write into shared repo memory, for a human to review.

clai2's `repo_propose_memory` tool emits a `memory proposal` span per proposal. Each (repo, file) becomes one proposal
(`kind: 'memory'`) carrying the latest proposed content next to the current shared file (`memory__clai2`), with every
proposing span as evidence. There is no distinct-developer gate: a human reviews every one. Exact duplicates of the
current file and anything that looks like it carries a credential are dropped; files over the size limit too.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from logfire.query_client import AsyncLogfireQueryClient
from pydantic import BaseModel, Field

from pydantic_ai import Agent

from . import __version__
from .fetch import _SESSIONS, IDENTITY, NOT_TEST, OVERLAP
from .llm_cache import run_cached
from .models import AppliesTo, Evidence, Proposal, contains_secrets

MEMORY_VARIABLE = 'memory__clai2'
MAX_BYTES = 8 * 1024
"""Per file, as `memory__clai2` accepts (design decision 2)."""

MEMORY_SQL = f"""
SELECT r.trace_id, r.span_id, r.start_timestamp,
       r.attributes->>'clai2.memory.scope' AS scope,
       r.attributes->>'clai2.memory.path' AS path,
       r.attributes->>'clai2.memory.content' AS content,
       r.attributes->>'clai2.memory.why' AS why,
       r.attributes->>'clai2.memory.base_sha' AS base_sha,
       {IDENTITY}
FROM records r
LEFT JOIN ({_SESSIONS}) s ON r.trace_id = s.trace_id
WHERE r.span_name = 'memory proposal'
  AND {NOT_TEST}
ORDER BY r.start_timestamp
"""


class MemorySpan(BaseModel):
    trace_id: str
    span_id: str
    timestamp: datetime = Field(validation_alias='start_timestamp')
    scope: str | None = None
    repo_slug: str | None = None
    path: str | None = None
    content: str | None = None
    why: str | None = None
    base_sha: str | None = None
    user_email: str | None = None
    host: str | None = None
    session_id: str | None = None
    team: str | None = None

    model_config = {'populate_by_name': True}

    @property
    def user(self) -> str:
        return self.user_email or f'host:{self.host}'


async def fetch_memory_spans(
    read_token: str, *, base_url: str, since: datetime, store_path: Path | None = None
) -> list[MemorySpan]:
    """`memory proposal` spans in the window, querying only what is newer than the stored watermark."""
    stored: dict[str, dict[str, Any]] = {}
    watermark: datetime | None = None
    if store_path and store_path.exists():
        data = json.loads(store_path.read_text())
        stored = {r['span_id']: r for r in data['rows']}
        watermark = datetime.fromisoformat(data['watermark']) if data.get('watermark') else None
    query_since = max(since, watermark - OVERLAP) if watermark else since
    async with AsyncLogfireQueryClient(read_token, base_url=base_url, timeout=120) as client:
        rows = (await client.query_json_rows(MEMORY_SQL, min_timestamp=query_since, limit=10_000))['rows']
    for r in rows:
        stored[r['span_id']] = r
    spans = [MemorySpan.model_validate(r) for r in stored.values()]
    spans = [s for s in spans if s.timestamp >= since]
    if store_path:
        latest = max((s.timestamp for s in spans), default=watermark)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(
            json.dumps(
                {
                    'watermark': latest.isoformat() if latest else None,
                    'rows': [s.model_dump(mode='json', by_alias=True) for s in spans],
                }
            )
        )
    return sorted(spans, key=lambda s: s.timestamp)


def load_fixture(path: Path) -> list[MemorySpan]:
    """Spans as `fetch_memory_spans` returns them (`start_timestamp` or `timestamp`), for offline tests."""
    return [MemorySpan.model_validate(r) for r in json.loads(path.read_text())]


def shared_files(value: dict[str, Any] | None) -> dict[tuple[str, str], str]:
    """`memory__clai2` (`{files: [{scope, applies_to: {repos}, path, content, ...}]}`) by (repo, path)."""
    files: dict[tuple[str, str], str] = {}
    for f in (value or {}).get('files', []):
        for repo in (f.get('applies_to') or {}).get('repos', []):
            if f.get('path') and isinstance(f.get('content'), str):
                files[(repo, f['path'])] = f['content']
    return files


def memory_id(repo_slug: str, path: str) -> str:
    """Stable per (repo, file): readable, with a short hash so different spellings can't collide."""
    slug = re.sub(r'[^a-z0-9]+', '-', f'{repo_slug} {path}'.lower()).strip('-')[:60]
    digest = hashlib.sha256(f'{repo_slug}\0{path}'.encode()).hexdigest()[:6]
    return f'memory-{slug}-{digest}'


REVIEW_INSTRUCTIONS = """\
A developer's coding agent proposed this file for a repository's SHARED memory, which every developer's agent working
in that repository will read. Repo memory should hold facts about the repository: its conventions, layout, commands,
pitfalls, decisions. Say whether this is instead a personal preference of one developer (how they like answers, their
own habits or tools, their tone or language), which belongs in their personal memory. `personal` true only when it
clearly is; `reason`: a few words.
"""


class _Review(BaseModel):
    personal: bool
    reason: str


@dataclass
class MemoryResult:
    proposals: list[Proposal]
    stale_reasons: dict[str, str]
    """Pending memory proposals that should go stale (e.g. the shared file now has exactly this content)."""
    dropped: dict[str, int]
    """Why spans were left out, with counts, for the run summary."""


async def mine_memory(
    spans: list[MemorySpan],
    *,
    current: dict[tuple[str, str], str],
    existing: list[Proposal],
    model: str | None,
    max_evidence: int = 20,
) -> MemoryResult:
    """One proposal per (repo, file) from the spans; `model=None` skips the personal-preference check."""
    dropped: defaultdict[str, int] = defaultdict(int)
    by_file: dict[tuple[str, str], list[MemorySpan]] = defaultdict(list)
    for span in spans:
        if span.scope not in (None, 'repo') or not span.repo_slug or not span.path or span.content is None:
            dropped['not a repo memory file'] += 1
        elif contains_secrets(span.content):
            dropped['looks like it contains a secret'] += 1
        elif len(span.content.encode()) > MAX_BYTES:
            dropped[f'over {MAX_BYTES // 1024} KB'] += 1
        else:
            by_file[(span.repo_slug, span.path)].append(span)

    old_by_id = {p.id: p for p in existing if p.kind == 'memory'}
    agent = (
        Agent(model, output_type=_Review, instructions=REVIEW_INSTRUCTIONS, name='fleet_miner_memory')
        if model
        else None
    )
    proposals: list[Proposal] = []
    stale_reasons: dict[str, str] = {}
    for (repo, path), file_spans in by_file.items():
        base_id = memory_id(repo, path)
        proposal_id = base_id
        # A reviewed (accepted or dismissed) proposal is never re-opened: spans proposing this file after it was
        # reviewed become a new revision of it.
        revision = 1
        while (old := old_by_id.get(proposal_id)) is not None and old.status in ('accepted', 'dismissed'):
            reviewed_at = old.accepted_at or max((e.timestamp for e in old.evidence), default=None)
            newer = [s for s in file_spans if reviewed_at is None or s.timestamp > reviewed_at]
            if not newer or newer[-1].content == old.content:
                break
            file_spans = newer
            revision += 1
            proposal_id = f'{base_id}-r{revision}'
        latest = file_spans[-1]
        base = current.get((repo, path))
        assert latest.content is not None
        if latest.content == base:
            dropped['same as the current shared file'] += len(file_spans)
            stale_reasons[proposal_id] = 'The shared file already has exactly this content.'
            continue
        users = {s.user for s in file_spans}
        review_flag = None
        if agent is not None:
            review = await run_cached(
                agent, f'Repository: {repo}\nFile: {path}\n\n{latest.content[:4000]}', output_type=_Review
            )
            review_flag = f'Looks like a personal preference: {review.reason}' if review.personal else None
        n = len(file_spans)
        proposals.append(
            Proposal(
                id=proposal_id,
                kind='memory',
                name=path,
                description=latest.why or f'Repo memory `{path}` for {repo}',
                text=latest.content,
                suggested_tier=None,
                rationale=f'Proposed {n} time{"" if n == 1 else "s"} by {len(users)} developer'
                f'{"" if len(users) == 1 else "s"}' + (f'. Why: {latest.why}' if latest.why else '.'),
                pattern=f'{repo}:{path}',
                distinct_users=len(users),
                sessions=len({s.session_id or s.trace_id for s in file_spans}),
                evidence=[
                    Evidence(trace_id=s.trace_id, span_id=s.span_id, timestamp=s.timestamp)
                    for s in reversed(file_spans[-max_evidence:])
                ],
                scope='repo',
                scope_reason='Repo memory applies to the repository it was proposed for.',
                applies_to=AppliesTo(repos=[repo]),
                repo_slug=repo,
                path=path,
                content=latest.content,
                base_content=base,
                base_sha=latest.base_sha,
                why=latest.why,
                proposed_by=latest.user_email,
                proposal_count=n,
                review_flag=review_flag,
                generated_by=f'fleet-miner {__version__}' + (f' / {model}' if model else ''),
            )
        )
    return MemoryResult(proposals, stale_reasons, dict(dropped))
