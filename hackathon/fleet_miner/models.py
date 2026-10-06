"""Data shapes: raw prompts in, `fleet_proposals__clai2` out (see the hackathon contract)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Tier = Literal['required', 'default_on', 'optional']
ProposalKind = Literal['skill', 'instruction']
ProposalStatus = Literal['pending', 'accepted', 'dismissed']


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
    score: float | None = None
    """Hackathon extra: LLM confidence times the distinct-user spread factor (braindump's scoring)."""


class Window(BaseModel):
    start: datetime
    end: datetime


class ProposalsDoc(BaseModel):
    """The value of the `fleet_proposals__clai2` managed variable."""

    generated_at: datetime
    window: Window
    proposals: list[Proposal] = []
