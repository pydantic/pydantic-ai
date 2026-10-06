"""Stages 2 and 3: group intents across the fleet, then draft a skill or instruction per recurring pattern."""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pydantic_ai import Agent, ModelRetry, RunContext

from . import __version__
from .models import Evidence, Facet, Proposal, ProposalKind, Tier, UserPrompt, strip_markup

CLUSTER_INSTRUCTIONS = """\
You are given intents extracted from prompts that many developers typed into their coding agents, plus the
patterns that were proposed in earlier runs. Group intents that express the SAME underlying request, even when
worded very differently ("babysit the PR until CI is green" and "keep an eye on checks and review comments and
iterate" are the same pattern). Prefer fewer, broader groups: variants of one workflow (e.g. "iterate on the PR
until CI is green", "iterate until the review bot is satisfied", "implement it, open a PR and keep iterating
until green") belong in ONE group. Leave out intents that match nothing else, and leave out throwaway test tasks
that ask for a specific artifact (e.g. "write FizzBuzz in Rust") rather than describing how the user wants work done.

For each group: write the shared pattern as one sentence, give a short kebab-case slug, and if it is the same
pattern as an earlier proposal, set `existing_id` to that proposal's id exactly as given (so it is not proposed twice); otherwise leave it null.
`confidence` is how sure you are that this is one coherent, reusable pattern (0-1).
"""

DRAFT_INSTRUCTIONS = """\
Several developers at one company typed the same kind of request into their coding agents. Turn it into something
pushed to every developer's agent, so nobody has to type it again.

Write clear, generalized guidance in your own words; do not copy their phrasing. Use what they typed as the measure
of how much context it needs: if they got by with one line, yours is about one line, not three paragraphs. You are
given the median length of their prompts: keep `text` close to that length, and never more than twice as long.

Decide the kind:
- `skill` when they asked for a sequence of steps: the steps, and at most one stop condition. No headings, no
  "Purpose" or "When to use" sections, no generic advice they didn't ask for (testing, linting, force-push warnings).
- `instruction` when it is a standing preference or rule.

`name`: a short kebab-case slug. `description`: one short sentence saying when it applies.
`suggested_tier`: `required` if nearly everyone would want it, `default_on` if broadly useful, `optional` if niche.
`rationale`: one or two sentences: how many people asked, and what it saves them.

Never include personal identifiers in any field: no people's names, GitHub usernames, handles or emails. Replace a
person with their role ("the requested reviewer", "the PR author"). Do keep the names of tools, bots and repository
conventions (e.g. Macroscope, douwebot, `SKIP=typecheck`): they are useful context for a company skill.
"""


class _Group(BaseModel):
    slug: str
    pattern: str
    span_ids: list[str]
    existing_id: str | None = None
    confidence: float = Field(ge=0, le=1)


class _Groups(BaseModel):
    groups: list[_Group]


class _Draft(BaseModel):
    kind: ProposalKind
    name: str
    description: str
    text: str
    suggested_tier: Tier
    rationale: str

    _strip_markup = field_validator('name', 'description', 'text', 'rationale')(strip_markup)


@dataclass
class Pattern:
    id: str
    pattern: str
    confidence: float
    prompts: list[UserPrompt]
    existing_id: str | None = None
    users: set[str] = field(default_factory=set[str])
    sessions: set[str] = field(default_factory=set[str])

    @property
    def score(self) -> float:
        # braindump's spread factor, over distinct users instead of distinct PRs.
        spread = {0: 0.0, 1: 0.6, 2: 0.85, 3: 0.95}.get(len(self.users), 1.0)
        return round(self.confidence * spread, 3)


async def find_patterns(
    prompts: list[UserPrompt], facets: dict[str, Facet], *, model: str, existing: list[Proposal]
) -> list[Pattern]:
    by_span = {p.span_id: p for p in prompts}
    items = [
        {'span_id': span_id, 'intent': f.intent, 'standing_request': f.standing_request}
        for span_id, f in facets.items()
        if f.intent and span_id in by_span
    ]
    if not items:
        return []
    earlier = [{'id': p.id, 'pattern': p.pattern, 'status': p.status} for p in existing]
    agent = Agent(model, output_type=_Groups, instructions=CLUSTER_INSTRUCTIONS, name='fleet_miner_cluster')
    result = await agent.run(
        f'Earlier proposals:\n{json.dumps(earlier, indent=2)}\n\nIntents:\n{json.dumps(items, indent=2)}'
    )
    known_ids = {p.id for p in existing}
    patterns: list[Pattern] = []
    for group in result.output.groups:
        # Only reuse an id the model was actually shown; anything else is a mangled or invented one.
        existing_id = group.existing_id if group.existing_id in known_ids else None
        group_prompts = [by_span[s] for s in dict.fromkeys(group.span_ids) if s in by_span]
        if len(group_prompts) < 2:
            continue
        pattern = Pattern(
            id=existing_id or group.slug,
            pattern=group.pattern,
            confidence=group.confidence,
            prompts=group_prompts,
            existing_id=existing_id,
        )
        for p in group_prompts:
            pattern.users.add(p.user or f'unknown:{p.session_id or p.trace_id}')
            pattern.sessions.add(p.session_id or p.trace_id)
        patterns.append(pattern)
    return sorted(patterns, key=lambda p: (len(p.users), p.score), reverse=True)


def save_patterns(path: Path, patterns: list[Pattern]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {'id': p.id, 'pattern': p.pattern, 'confidence': p.confidence, 'existing_id': p.existing_id,
         'span_ids': [u.span_id for u in p.prompts]}
        for p in patterns
    ]
    path.write_text(json.dumps(data, indent=2))


def load_patterns(path: Path, prompts: list[UserPrompt]) -> list[Pattern]:
    """Rebuild cached groups against the current prompts, so drafting can be re-run without re-clustering."""
    by_span = {p.span_id: p for p in prompts}
    patterns: list[Pattern] = []
    for item in json.loads(path.read_text()):
        group_prompts = [by_span[s] for s in item['span_ids'] if s in by_span]
        pattern = Pattern(
            id=item['id'], pattern=item['pattern'], confidence=item['confidence'],
            prompts=group_prompts, existing_id=item['existing_id'],
        )
        for p in group_prompts:
            pattern.users.add(p.user or f'unknown:{p.session_id or p.trace_id}')
            pattern.sessions.add(p.session_id or p.trace_id)
        patterns.append(pattern)
    return sorted(patterns, key=lambda p: (len(p.users), p.score), reverse=True)


_GENERIC_DOMAINS = {'example', 'test', 'localhost', 'gmail', 'googlemail', 'outlook', 'hotmail', 'yahoo', 'icloud', 'proton', 'protonmail', 'pydantic'}


_HOST_OWNER = re.compile(r"^([A-Za-z]+?)'?s-(?:MacBook|MBP|Mac|iMac|Laptop|PC|Desktop)", re.IGNORECASE)
_HANDLE_PATTERNS = (
    re.compile(r'(?<![\w.])@([A-Za-z0-9](?:[A-Za-z0-9-]{1,37}[A-Za-z0-9])?)\b'),
    re.compile(r'\bassign(?:ed|ee)?(?: it| the PR| this)? to @?([A-Za-z0-9-]{3,39})\b', re.IGNORECASE),
    re.compile(r'/(?:Users|home)/([A-Za-z0-9._-]{3,})/'),
)
_NOT_HANDLES = {
    *('main', 'master', 'yourself', 'me', 'the', 'them', 'reviewer', 'author'),
    # Products and bots that get @-mentioned or assigned to, which skills should keep naming.
    *('github', 'gitlab', 'claude', 'codex', 'copilot', 'devin', 'coderabbit', 'macroscope', 'douwebot', 'logfire'),
    *('pydantic', 'anthropic', 'openai', 'gemini'),
}


def personal_identifiers(prompts: list[UserPrompt]) -> set[str]:
    """Emails, their local parts and personal domains, and machine names of the people behind `prompts`."""
    found: set[str] = set()
    for p in prompts:
        user = (p.user or '').removeprefix('host:')
        if '@' in user:
            local, domain = user.split('@', 1)
            found |= {user, local, *local.replace('_', '.').split('.')}
            if (label := domain.split('.')[0]) not in _GENERIC_DOMAINS:
                found.add(label)
        elif match := _HOST_OWNER.match(user):
            found.add(match[1])  # "Janes-MacBook-Air.local" names Jane; "pydantic-ai" names nobody
        # Handles typed in the prompt itself: "@someone", "assign it to someone", "/Users/someone/".
        found |= {m for pattern in _HANDLE_PATTERNS for m in pattern.findall(p.text)}
    return {f for f in found if len(f) >= 4 and f.lower() not in _NOT_HANDLES}


def leaked_identifiers(draft: _Draft, identifiers: set[str]) -> set[str]:
    """Which identifiers appear as whole words in any drafted field (so `douwebot` does not count as a name)."""
    text = '\n'.join([draft.name, draft.description, draft.text, draft.rationale])
    return {i for i in identifiers if re.search(rf'(?<![\w-]){re.escape(i)}(?![\w-])', text, re.IGNORECASE)}


def _redact(value: str, identifiers: set[str]) -> str:
    for i in sorted(identifiers, key=len, reverse=True):
        value = re.sub(rf'(?<![\w-]){re.escape(i)}(?![\w-])', 'the requested person', value, flags=re.IGNORECASE)
    return value


_MIN_TARGET_CHARS = 60
"""Even a terse ask ("babysit it") needs a sentence once the conversation that gave it meaning is gone."""


def median_prompt_chars(pattern: Pattern) -> int:
    return max(_MIN_TARGET_CHARS, int(statistics.median(len(p.text) for p in pattern.prompts)))


@dataclass
class _DraftDeps:
    identifiers: set[str]
    target_chars: int

    @property
    def max_chars(self) -> int:
        return 2 * self.target_chars


async def draft_proposals(patterns: list[Pattern], *, model: str, max_evidence: int = 5) -> list[Proposal]:
    agent = Agent(
        model,
        deps_type=_DraftDeps,
        output_type=_Draft,
        instructions=DRAFT_INSTRUCTIONS,
        name='fleet_miner_draft',
        retries=3,
    )

    @agent.output_validator
    def check_draft(ctx: RunContext[_DraftDeps], draft: _Draft) -> _Draft:
        if leaked := leaked_identifiers(draft, ctx.deps.identifiers):
            raise ModelRetry(
                f'The draft contains personal identifiers ({", ".join(sorted(leaked))}). '
                'Replace each with the person\'s role, e.g. "the requested reviewer".'
            )
        # Length is a target, not a hard rule: after two tries, take what we have.
        if len(draft.text) > ctx.deps.max_chars and ctx.retry < 2:
            raise ModelRetry(
                f'`text` is {len(draft.text)} characters; the users typed ~{ctx.deps.target_chars}. '
                f'Shorten it to at most {ctx.deps.max_chars} characters.'
            )
        return draft

    async def one(pattern: Pattern) -> Proposal:
        # The drafter never sees who asked: developers are numbered, not named.
        numbers = {user: n for n, user in enumerate(dict.fromkeys(p.user for p in pattern.prompts), 1)}
        examples = [{'developer': numbers[p.user], 'prompt': p.text[:2000]} for p in pattern.prompts[:12]]
        identifiers = personal_identifiers(pattern.prompts)
        target = median_prompt_chars(pattern)
        result = await agent.run(
            f'Pattern: {pattern.pattern}\n'
            f'Asked by {len(pattern.users)} distinct developers across {len(pattern.sessions)} sessions.\n'
            f'Median prompt length: {target} characters. Target for `text`: about {target}, at most {2 * target}.\n'
            f'What they typed:\n{json.dumps(examples, indent=2)}',
            deps=_DraftDeps(identifiers=identifiers, target_chars=target),
        )
        draft = result.output
        if leaked := leaked_identifiers(draft, identifiers):  # pragma: no cover - only if retries ran out
            print(f'warning: redacted {len(leaked)} personal identifier(s) from `{pattern.id}`')
            draft = _Draft.model_validate({k: _redact(v, leaked) if isinstance(v, str) else v for k, v in draft})
        return Proposal(
            id=pattern.id,
            pattern=pattern.pattern,
            distinct_users=len(pattern.users),
            sessions=len(pattern.sessions),
            evidence=_evidence(pattern, max_evidence),
            score=pattern.score,
            generated_by=f'fleet-miner {__version__} / {model}',
            **draft.model_dump(),
        )

    return list(await asyncio.gather(*(one(p) for p in patterns)))


def _evidence(pattern: Pattern, limit: int) -> list[Evidence]:
    """One excerpt per user first, so the evidence shows the spread across people."""
    picked: list[UserPrompt] = []
    seen_users: set[str | None] = set()
    for p in pattern.prompts:
        if p.user not in seen_users:
            picked.append(p)
            seen_users.add(p.user)
    picked += [p for p in pattern.prompts if p not in picked]
    return [
        Evidence(
            user=p.user,
            trace_id=p.trace_id,
            span_id=p.span_id,
            session_id=p.session_id,
            timestamp=p.timestamp,
            excerpt=p.text[:500],
        )
        for p in picked[:limit]
    ]


MergeAction = Literal['new', 'updated', 'skipped', 'stale']


def merge(
    existing: list[Proposal], fresh: list[Proposal], *, stale_if_missing: bool = False
) -> tuple[list[Proposal], dict[str, MergeAction]]:
    """Never re-propose an accepted or dismissed id; refresh a pending (or stale) one's evidence and draft.

    With `stale_if_missing` (a full run), a pending proposal the run no longer qualifies becomes `stale`.
    """
    by_id = {p.id: p for p in existing}
    actions: dict[str, MergeAction] = {}
    for proposal in fresh:
        old = by_id.get(proposal.id)
        if old is None:
            by_id[proposal.id] = proposal
            actions[proposal.id] = 'new'
        elif old.status in ('pending', 'stale'):
            by_id[proposal.id] = proposal
            actions[proposal.id] = 'updated'
        else:
            actions[proposal.id] = 'skipped'
    if stale_if_missing:
        for id_, proposal in by_id.items():
            if id_ not in actions and proposal.status == 'pending':
                by_id[id_] = proposal.model_copy(update={'status': 'stale'})
                actions[id_] = 'stale'
    return list(by_id.values()), actions
