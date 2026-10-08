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
from typing import Any, Literal

from logfire.query_client import AsyncLogfireQueryClient
from pydantic import BaseModel, Field

from pydantic_ai import Agent

from . import __version__
from .fetch import _SESSIONS, IDENTITY, NOT_TEST, OVERLAP
from .llm_cache import run_cached
from .models import AppliesTo, Evidence, Proposal, contains_secrets, redact_secrets

MEMORY_VARIABLE = 'memory__clai2'
AGENT_VARIABLE = 'agent__clai2'
MAX_BYTES = 8 * 1024
"""Per file, as `memory__clai2` accepts (design decision 2)."""
MAX_FILES_PER_REPO = 20

SharedMode = Literal['review', 'corroborate', 'auto']
"""`agent__clai2.policy.memory.shared`: every proposal waits for an admin (`review`), is published once agents of
2+ developers propose equivalent content (`corroborate`), or is published right away (`auto`)."""


def shared_mode(agent_value: dict[str, Any] | None) -> SharedMode:
    mode = (((agent_value or {}).get('policy') or {}).get('memory') or {}).get('shared')
    return mode if mode in ('review', 'corroborate', 'auto') else 'review'


def normalized(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


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


EQUIVALENT_INSTRUCTIONS = """\
Two developers' coding agents proposed versions of the same shared repo memory file. Are they equivalent: do they
state the same facts and guidance, so either could be published without losing or changing anything that matters?
Wording, order and formatting differences don't count; a fact only one of them has, or a contradiction, does.
"""


class _Equivalent(BaseModel):
    equivalent: bool


@dataclass
class Publish:
    """A proposal the shared-memory mode publishes without an admin."""

    proposal_id: str
    source: Literal['auto', 'corroborated']
    corroborated_by: list[str]
    """Emails of the developers whose agents proposed equivalent content (`corroborated` only)."""


@dataclass
class MemoryResult:
    proposals: list[Proposal]
    stale_reasons: dict[str, str]
    """Pending memory proposals that should go stale (e.g. the shared file now has exactly this content)."""
    dropped: dict[str, int]
    """Why spans were left out, with counts, for the run summary."""
    publish: list[Publish]
    """What `mode` says to publish now; the caller writes `memory__clai2`, then marks these accepted."""


async def mine_memory(
    spans: list[MemorySpan],
    *,
    current: dict[tuple[str, str], str],
    existing: list[Proposal],
    model: str | None,
    mode: SharedMode = 'review',
    max_evidence: int = 20,
) -> MemoryResult:
    """One proposal per (repo, file) from the spans; `model=None` skips the LLM checks (personal preference, and
    equivalence beyond normalized text for `corroborate`).

    A file whose latest proposal fails the secret or size check is dropped in `review` mode. In `auto` and
    `corroborate` it stays a pending proposal with the reason (secrets redacted), so an admin sees why it wasn't
    published.
    """
    dropped: defaultdict[str, int] = defaultdict(int)
    by_file: dict[tuple[str, str], list[MemorySpan]] = defaultdict(list)
    failed: dict[tuple[str, str], tuple[MemorySpan, str]] = {}
    for span in spans:
        if span.scope not in (None, 'repo') or not span.repo_slug or not span.path or span.content is None:
            dropped['not a repo memory file'] += 1
            continue
        key = (span.repo_slug, span.path)
        problem = (
            'looks like it contains a secret'
            if contains_secrets(span.content)
            else f'over {MAX_BYTES // 1024} KB'
            if len(span.content.encode()) > MAX_BYTES
            else None
        )
        if problem:
            dropped[problem] += 1
            failed[key] = (span, problem)
        else:
            by_file[key].append(span)
            failed.pop(key, None)  # a later good proposal supersedes an earlier failed one

    old_by_id = {p.id: p for p in existing if p.kind == 'memory'}
    agent = (
        Agent(model, output_type=_Review, instructions=REVIEW_INSTRUCTIONS, name='fleet_miner_memory')
        if model
        else None
    )
    equivalence = (
        Agent(model, output_type=_Equivalent, instructions=EQUIVALENT_INSTRUCTIONS, name='fleet_miner_memory_same')
        if model
        else None
    )
    proposals: list[Proposal] = []
    stale_reasons: dict[str, str] = {}
    publish: list[Publish] = []
    if mode != 'review':
        # Spans that failed a check while being the latest for their file: pending, with the reason, never published.
        for (repo, path), (span, problem) in failed.items():
            by_file.pop((repo, path), None)
            assert span.content is not None
            proposal = _proposal(memory_id(repo, path), repo, path, [span], current.get((repo, path)), model, None, 1)
            content = redact_secrets(span.content)
            proposals.append(
                proposal.model_copy(
                    update={'content': content, 'text': content, 'status_reason': f'Not published: {problem}.'}
                )
            )
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
        review_flag = None
        if agent is not None:
            review = await run_cached(
                agent, f'Repository: {repo}\nFile: {path}\n\n{latest.content[:4000]}', output_type=_Review
            )
            review_flag = f'Looks like a personal preference: {review.reason}' if review.personal else None
        proposals.append(_proposal(proposal_id, repo, path, file_spans, base, model, review_flag, max_evidence))
        if mode == 'auto':
            publish.append(Publish(proposal_id, 'auto', []))
        elif mode == 'corroborate':
            # The latest proposal of each other developer, checked against the newest content.
            others: dict[str, MemorySpan] = {}
            for s in file_spans[:-1]:
                if s.user != latest.user:
                    others[s.user] = s
            agreeing = [latest.user]
            for user, s in others.items():
                assert s.content is not None
                same = normalized(s.content) == normalized(latest.content)
                if not same and equivalence is not None:
                    a, b = sorted([s.content, latest.content])
                    same = (
                        await run_cached(
                            equivalence, f'Version A:\n{a[:4000]}\n\nVersion B:\n{b[:4000]}', output_type=_Equivalent
                        )
                    ).equivalent
                if same:
                    agreeing.append(user)
            if len(agreeing) >= 2:
                emails = [u for u in agreeing if not u.startswith('host:')] or agreeing
                publish.append(Publish(proposal_id, 'corroborated', emails))
    return MemoryResult(proposals, stale_reasons, dict(dropped), publish)


def _proposal(
    proposal_id: str,
    repo: str,
    path: str,
    file_spans: list[MemorySpan],
    base: str | None,
    model: str | None,
    review_flag: str | None,
    max_evidence: int,
) -> Proposal:
    latest = file_spans[-1]
    assert latest.content is not None
    users = {s.user for s in file_spans}
    n = len(file_spans)
    return Proposal(
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


def with_published(
    value: dict[str, Any] | None, publish: list[Publish], proposals: dict[str, Proposal], now: datetime
) -> tuple[dict[str, Any], dict[str, str]]:
    """`memory__clai2` with these proposals' files written in (the shape the UI writes on accept), plus why any
    could not be (`{proposal_id: reason}`).

    A file replaces the entry for the same repo and path; clai2 takes the first entry per path, so it goes first.
    """
    files: list[dict[str, Any]] = list((value or {}).get('files', []))
    refused: dict[str, str] = {}
    for item in publish:
        p = proposals[item.proposal_id]
        assert p.repo_slug and p.path and p.content is not None

        def same_file(f: dict[str, Any]) -> bool:
            return f.get('path') == p.path and (f.get('applies_to') or {}).get('repos') == [p.repo_slug]

        rest = [f for f in files if not same_file(f)]
        paths = {f.get('path') for f in rest if p.repo_slug in ((f.get('applies_to') or {}).get('repos') or [])}
        if p.path not in paths and len(paths) >= MAX_FILES_PER_REPO:
            refused[p.id] = f'Not published: {p.repo_slug} already has {MAX_FILES_PER_REPO} shared files.'
            continue
        entry = {
            'scope': 'repo',
            'applies_to': {'repos': [p.repo_slug]},
            'path': p.path,
            'content': p.content,
            'source': item.source,
            'proposal_id': p.id,
            'proposed_by': p.proposed_by,
            'accepted_by': 'auto-publish',
            'accepted_at': now.isoformat(),
            **({'corroborated_by': item.corroborated_by} if item.corroborated_by else {}),
        }
        files = [entry, *rest]
    return {**(value or {}), 'files': files}, refused
