"""Stages 2 and 3: group intents across the fleet, then draft a skill or instruction per recurring pattern."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent

from .models import Evidence, Facet, Proposal, ProposalKind, Tier, UserPrompt

CLUSTER_INSTRUCTIONS = """\
You are given intents extracted from prompts that many developers typed into their coding agents, plus the
patterns that were proposed in earlier runs. Group intents that express the SAME underlying request, even when
worded very differently ("babysit the PR until CI is green" and "keep an eye on checks and review comments and
iterate" are the same pattern). Leave out intents that match nothing else.

For each group: write the shared pattern as one sentence, give a short kebab-case slug, and if it is the same
pattern as an earlier proposal, set `existing_id` to that proposal's id (so it is not proposed twice).
`confidence` is how sure you are that this is one coherent, reusable pattern (0-1).
"""

DRAFT_INSTRUCTIONS = """\
Several developers at one company independently asked their coding agents for the same thing. Turn that into a
change that is pushed to every developer's agent, so nobody has to ask again.

Decide the kind:
- `skill` when the pattern is a multi-step procedure (a workflow with steps, checks and a stop condition). Write
  `text` as the body of a SKILL.md: a short purpose line, when to use it, numbered steps, and when to stop. Make it
  concrete and tool-agnostic enough to work in any repository (e.g. use `gh` for GitHub).
- `instruction` when it is a short standing preference or rule. Write `text` as one to three sentences addressed
  to the agent.

`name` is a kebab-case slug, `description` a one-line catalog blurb saying when the agent should use it.
`suggested_tier`: `required` for something nearly everyone wants by default, `default_on` for broadly useful
but opinionated, `optional` for niche. `rationale`: two to four sentences citing how many people asked and why
pushing it down saves them time.
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
    patterns: list[Pattern] = []
    for group in result.output.groups:
        group_prompts = [by_span[s] for s in dict.fromkeys(group.span_ids) if s in by_span]
        if len(group_prompts) < 2:
            continue
        pattern = Pattern(
            id=group.existing_id or group.slug,
            pattern=group.pattern,
            confidence=group.confidence,
            prompts=group_prompts,
            existing_id=group.existing_id,
        )
        for p in group_prompts:
            pattern.users.add(p.user or f'unknown:{p.session_id or p.trace_id}')
            pattern.sessions.add(p.session_id or p.trace_id)
        patterns.append(pattern)
    return sorted(patterns, key=lambda p: (len(p.users), p.score), reverse=True)


async def draft_proposals(patterns: list[Pattern], *, model: str, max_evidence: int = 5) -> list[Proposal]:
    agent = Agent(model, output_type=_Draft, instructions=DRAFT_INSTRUCTIONS, name='fleet_miner_draft')

    async def one(pattern: Pattern) -> Proposal:
        examples = [{'user': p.user, 'prompt': p.text[:2000]} for p in pattern.prompts[:12]]
        result = await agent.run(
            f'Pattern: {pattern.pattern}\n'
            f'Asked by {len(pattern.users)} distinct developers across {len(pattern.sessions)} sessions.\n'
            f'What they typed:\n{json.dumps(examples, indent=2)}'
        )
        draft = result.output
        return Proposal(
            id=pattern.id,
            pattern=pattern.pattern,
            distinct_users=len(pattern.users),
            sessions=len(pattern.sessions),
            evidence=_evidence(pattern, max_evidence),
            score=pattern.score,
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


MergeAction = Literal['new', 'updated', 'skipped']


def merge(existing: list[Proposal], fresh: list[Proposal]) -> tuple[list[Proposal], dict[str, MergeAction]]:
    """Never re-propose an accepted or dismissed id; refresh a pending one's evidence and draft."""
    by_id = {p.id: p for p in existing}
    actions: dict[str, MergeAction] = {}
    for proposal in fresh:
        old = by_id.get(proposal.id)
        if old is None:
            by_id[proposal.id] = proposal
            actions[proposal.id] = 'new'
        elif old.status == 'pending':
            by_id[proposal.id] = proposal
            actions[proposal.id] = 'updated'
        else:
            actions[proposal.id] = 'skipped'
    return list(by_id.values()), actions
