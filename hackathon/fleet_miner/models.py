"""Data shapes: raw prompts in, `fleet_proposals__clai2` out (see the hackathon contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import re

from pydantic import BaseModel, Field, field_validator

Tier = Literal['required', 'default_on', 'optional']
ProposalKind = Literal['skill', 'instruction']
ProposalStatus = Literal['pending', 'accepted', 'dismissed', 'stale']
"""`stale`: was pending, but the latest run no longer finds the pattern often enough. Kept, not deleted."""


class UserPrompt(BaseModel):
    """One prompt a user typed into clai2, with the identity of who typed it."""

    trace_id: str
    span_id: str
    timestamp: datetime
    text: str
    user: str | None = None
    host: str | None = None
    session_id: str | None = None
    source: Literal['prompt_submitted', 'agent_run'] = 'prompt_submitted'


class Facet(BaseModel):
    """Stage 1 output for one prompt: the reusable intent behind it, if any."""

    span_id: str
    intent: str | None = Field(
        description='The generalized, user- and repo-independent request in one imperative sentence of at most 20 '
        'words, or null when the prompt is not reusable (task-specific details only, "yes", "continue", a typo fix).'
    )
    standing_request: bool = Field(
        description='True when the user asks for behavior the agent arguably should have done unprompted, '
        'or that the user likely asks for repeatedly (e.g. "keep watching CI until it passes").'
    )
    workflow: bool = Field(description='True when the intent is a multi-step procedure rather than a one-line preference.')


# Tool-call markup a model sometimes leaks into a text field (e.g. a trailing `</parameter> </invoke>`).
_MARKUP = re.compile(r'</?(?:antml:)?(?:parameter|invoke|function_calls|function_results)\b[^>]*>')


def strip_markup(value: str) -> str:
    return _MARKUP.sub('', value).strip()


class Evidence(BaseModel):
    user: str | None
    trace_id: str
    span_id: str
    session_id: str | None
    timestamp: datetime
    excerpt: str


class Proposal(BaseModel):
    id: str
    kind: ProposalKind
    name: str
    description: str
    text: str
    suggested_tier: Tier
    rationale: str
    pattern: str
    distinct_users: int
    sessions: int
    evidence: list[Evidence]
    status: ProposalStatus = 'pending'
    accepted_tier: Tier | None = None
    accepted_at: datetime | None = None
    source: Literal['fleet-miner'] = 'fleet-miner'
    generated_by: str | None = None
    """Miner version and drafting model, e.g. `fleet-miner 0.2 / gateway/anthropic:claude-sonnet-5-5`."""
    score: float | None = None
    """Hackathon extra: LLM confidence times the distinct-user spread factor (braindump's scoring)."""

    _strip_markup = field_validator('id', 'name', 'description', 'text', 'rationale', 'pattern')(strip_markup)


class Window(BaseModel):
    start: datetime
    end: datetime


class ProposalsDoc(BaseModel):
    """The value of the `fleet_proposals__clai2` managed variable."""

    generated_at: datetime
    window: Window
    proposals: list[Proposal] = []
