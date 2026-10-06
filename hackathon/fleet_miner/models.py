"""Data shapes: raw prompts in, `fleet_proposals__clai2` out (see the hackathon contract)."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Tier = Literal['required', 'default_on', 'optional']
ProposalKind = Literal['skill', 'instruction', 'policy']
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
    workflow: bool = Field(
        description='True when the intent is a multi-step procedure rather than a one-line preference.'
    )


# Tool-call markup a model sometimes leaks into a text field (e.g. a trailing `</parameter> </invoke>`).
_MARKUP = re.compile(r'</?(?:antml:)?(?:parameter|invoke|function_calls|function_results)\b[^>]*>')


def strip_markup(value: str) -> str:
    return _MARKUP.sub('', value).strip()


# Anything that looks like a credential, so a command line or excerpt never carries one into the variable.
_SECRETS = (
    re.compile(r'\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}'),  # OpenAI/Anthropic-style keys
    re.compile(r'\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}'),
    re.compile(r'\bpylf_[A-Za-z0-9_]{16,}|\bxox[abprs]-[A-Za-z0-9-]{10,}|\bAKIA[0-9A-Z]{16}\b'),
    re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}'),
    re.compile(r'\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'),  # JWTs
    re.compile(r'\b[A-Fa-f0-9]{40,}\b|\b[A-Za-z0-9+/]{48,}={0,2}'),  # long hex/base64 blobs
)
_SECRET_ASSIGNMENT = re.compile(
    r'(?i)\b([A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_KEY|PRIVATE_KEY|AUTH)[A-Z0-9_]*)'
    r'(\s*[=:]\s*|\s+)(["\']?)[^\s"\']{6,}\3'
)
_HOME = re.compile(r'(/Users/|/home/|C:\\Users\\)[^/\\\s]+')


def redact_secrets(value: str) -> str:
    """Replace credentials (and home-directory user names) with placeholders."""
    for pattern in _SECRETS:
        value = pattern.sub('<redacted>', value)
    value = _SECRET_ASSIGNMENT.sub(lambda m: m[0] if '<redacted>' in m[0] else f'{m[1]}{m[2]}<redacted>', value)
    return _HOME.sub(r'\1<user>', value)


def clean_text(value: str) -> str:
    return redact_secrets(strip_markup(value))


class Evidence(BaseModel):
    """A pointer into the traces, nothing more.

    Every clai2 process downloads every variable in the project, so the stored document must not carry who said what:
    no email, host or excerpt. The UI resolves the user and the text lazily from `trace_id`/`span_id`. `developer`
    is a pseudonymous number, stable within one document, so the UI can show "3 developers" and group evidence.
    (Older documents carried `user`, `session_id` and `excerpt`; those keys are dropped on load.)
    """

    trace_id: str
    span_id: str
    timestamp: datetime
    developer: int = 0


class PolicyMatch(BaseModel):
    tool: str
    command: str | None = None
    args: dict[str, str] | None = None


class PolicyRule(BaseModel):
    """One `agent__<agent>.policy.rules` entry (contract-policy.md)."""

    name: str
    description: str
    mode: Literal['observe', 'enforce'] = 'observe'
    action: Literal['deny', 'ask']
    match: PolicyMatch
    monty: str | None = None
    source: Literal['manual', 'fleet-miner'] = 'fleet-miner'
    proposal_id: str | None = None

    _clean = field_validator('name', 'description')(clean_text)


class McpAllow(BaseModel):
    """Hackathon extension: a policy proposal to add servers to `policy.mcp.allow` (no `rule`)."""

    allow: list[str]


class Proposal(BaseModel):
    id: str
    kind: ProposalKind
    name: str
    description: str
    text: str
    suggested_tier: Tier | None
    """`None` for a policy rule only one person's actions motivated: worth a look, not a rollout."""
    rationale: str
    pattern: str
    distinct_users: int
    sessions: int
    evidence: list[Evidence]
    status: ProposalStatus = 'pending'
    status_reason: str | None = None
    """Why a proposal was dismissed or marked stale by the miner, e.g. "test traffic"."""
    accepted_tier: Tier | None = None
    accepted_at: datetime | None = None
    source: Literal['fleet-miner'] = 'fleet-miner'
    generated_by: str | None = None
    """Miner version and drafting model, e.g. `fleet-miner 0.2 / gateway/anthropic:claude-sonnet-5-5`."""
    score: float | None = None
    """Hackathon extra: LLM confidence times the distinct-user spread factor (braindump's scoring)."""
    rule: PolicyRule | None = None
    """Set on `kind: 'policy'` proposals; always `mode: 'observe'` from the miner."""
    mcp: McpAllow | None = None

    _clean = field_validator('id', 'name', 'description', 'text', 'rationale', 'pattern')(clean_text)


class Window(BaseModel):
    start: datetime
    end: datetime


class ProposalsDoc(BaseModel):
    """The value of the `fleet_proposals__clai2` managed variable."""

    generated_at: datetime
    window: Window
    min_users: int | None = None
    """Distinct developers a pattern needed in the run that wrote this document."""
    proposals: list[Proposal] = []


def pseudonymize(doc: ProposalsDoc, users_by_span: dict[str, str]) -> None:
    """Number developers 1, 2, ... in order of first appearance across the whole document, in place."""
    numbers: dict[str, int] = {}
    for proposal in doc.proposals:
        for evidence in proposal.evidence:
            # Evidence whose span is outside this run's window keeps a number of its own rather than a guessed one.
            user = users_by_span.get(evidence.span_id, f'unknown:{evidence.span_id}')
            evidence.developer = numbers.setdefault(user, len(numbers) + 1)
