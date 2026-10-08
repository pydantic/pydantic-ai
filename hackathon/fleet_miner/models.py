"""Data shapes: raw prompts in, `fleet_proposals__clai2` out (see the hackathon contract)."""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Tier = Literal['required', 'default_on', 'optional']
ProposalKind = Literal['skill', 'instruction', 'policy', 'memory']
ProposalStatus = Literal['pending', 'accepted', 'dismissed', 'stale']
"""`stale`: not a current suggestion, with `status_reason` saying why: it was pending but no longer passes the gates,
or it passes them but ranks below the cap on pending suggestions (then `emerging` is true). Kept, not deleted."""


class UserPrompt(BaseModel):
    """One prompt a user typed into clai2, with the identity of who typed it."""

    trace_id: str
    span_id: str
    timestamp: datetime
    text: str
    user: str | None = None
    host: str | None = None
    session_id: str | None = None
    team: str | None = None
    repo_slug: str | None = None
    source: Literal['prompt_submitted', 'agent_run'] = 'prompt_submitted'
    route: str | None = None
    """clai2's `prompt submitted` route: `queued`/`run now`/`edited queued` were typed while the agent was working.
    `''` when looked up and absent, `None` when never looked up (prompts stored before the miner read it)."""


def span_of(item_id: str) -> str:
    """The prompt span an item id came from: `<span_id>#r<n>` (a rule extracted from it), or the older
    `<span_id>#pref<n>`, or the span id itself."""
    return item_id.split('#', 1)[0]


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


def contains_secrets(value: str) -> bool:
    """Whether text looks like it carries a credential (the patterns `redact_secrets` replaces, minus home paths and
    commit-sha-like hex, which are fine in a repo memory file)."""
    if any(p.search(value) for p in _SECRETS[:-1]) or _SECRET_ASSIGNMENT.search(value):
        return True
    return any(not re.fullmatch(r'[A-Fa-f0-9]+', m) for m in re.findall(r'\b[A-Za-z0-9+/]{48,}={0,2}', value))


def clean_text[T: str | None](value: T) -> T:
    return value if value is None else redact_secrets(strip_markup(value))  # pyright: ignore[reportReturnType]


class Evidence(BaseModel):
    """A pointer into the traces, plus who it came from.

    This is oversight for the organization, so people are identified: `email` is the developer's `user.email` when
    known, and `developer` numbers them consistently within one document (the doc-level `developers` map names each
    number, also for traces that lack `user.email`). The UI resolves the text lazily from `trace_id`/`span_id`. Never
    a token, key or credential. (Older documents carried `user`, `session_id` and `excerpt`; dropped on load.)
    """

    trace_id: str
    span_id: str
    timestamp: datetime
    developer: int = 0
    email: str | None = None
    origin: Literal['requested', 'unprompted'] | None = None
    """Policy evidence: whether the developer asked for this call in that turn or the one before, or the agent chose it."""
    target: Literal['protected', 'own'] | None = None
    """Policy evidence: whether the call hit a protected or shared target (main, a remote host, outside the workspace)."""


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


class AppliesTo(BaseModel):
    teams: list[str] = []
    repos: list[str] = []
    """`owner/name` slugs."""


class TrendPoint(BaseModel):
    """One day of a pattern: matching prompts (or, for policy, matching tool calls) and distinct developers."""

    date: date
    count: int
    users: int


def daily_trend(events: Iterable[tuple[datetime, str]], start: datetime, end: datetime) -> list[TrendPoint]:
    """Per-day counts over [start, end], zero-filled so a sparkline has every day."""
    days: dict[date, list[str]] = {}
    for timestamp, user in events:
        days.setdefault(timestamp.date(), []).append(user)
    points: list[TrendPoint] = []
    day = start.date()
    while day <= end.date():
        users = days.get(day, [])
        points.append(TrendPoint(date=day, count=len(users), users=len(set(users))))
        day += timedelta(days=1)
    return points


class Impact(BaseModel):
    """What changed since a proposal was accepted. Counts first: with a handful of developers, ratios mislead."""

    computed_at: datetime
    accepted_at: datetime
    days_before: float
    """Days of the mining window before acceptance."""
    days_after: float
    before_count: int
    """Skill/instruction: prompts matching the pattern before acceptance. Policy: tool calls matching the rule."""
    after_count: int
    before_per_day: float | None
    after_per_day: float | None
    users_with_item: int | None = None
    """Skill/instruction: distinct developers whose runs had it active (`clai2.fleet.active`) since acceptance."""
    follow_up_prompts_avoided_estimate: int | None = None
    """Skill/instruction: (before_per_day - after_per_day) * days_after, floored at 0. An estimate, not a count."""
    decisions: dict[str, int] | None = None
    """Policy: `policy decision` records for this rule since acceptance, by `clai2.policy.outcome`."""
    decision_users: int | None = None


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
    matching_calls: int | None = None
    """Policy proposals: flagged tool calls (unprompted, or on a protected target) the rule's glob matched."""
    suggested_instruction: str | None = None
    """Policy proposals for risky actions agents take unprompted: an instruction to pair with the rule
    ("Don't force-push unless the user asks")."""
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
    """LLM confidence times the distinct-user spread factor (braindump's scoring), plus a bonus for corrections."""
    value_reason: str | None = None
    """Skills and instructions: one line on why writing it down changes what agents do (or, once stale, why not)."""
    verified_prompts: int | None = None
    """Skills and instructions: prompts the validation step confirmed express this rule (users and sessions count
    only those)."""
    corrections: int | None = None
    """Of `verified_prompts`, how many corrected or steered the agent (typed mid-run, or "no, don't...")."""
    emerging: bool = False
    """Passes every gate but ranks below the cap on pending suggestions: `status` is `stale`, shown collapsed."""
    scope: Literal['organization', 'team', 'repo'] = 'organization'
    """Who it should apply to, measured from the teams and repos of its evidence (see `scope.py`)."""
    scope_reason: str | None = None
    """How the scope was decided: starts with "Measured:" or "LLM judgment (no repo or team data):"."""
    applies_to: AppliesTo = Field(default_factory=lambda: AppliesTo())
    """Suggested targets to pre-fill on Accept; empty lists mean the whole organization."""

    @field_validator('scope', mode='before')
    @classmethod
    def _company_is_organization(cls, value: object) -> object:
        return 'organization' if value == 'company' else value  # documents written before the rename

    trend: list[TrendPoint] | None = None
    """Pending proposals: per day over the window, for a sparkline."""
    impact: Impact | None = None
    """Accepted proposals: before vs after acceptance."""
    rule: PolicyRule | None = None
    """Set on `kind: 'policy'` proposals; always `mode: 'observe'` from the miner."""
    mcp: McpAllow | None = None
    # `kind: 'memory'`: a repo memory file a developer's agent proposed (`memory proposal` spans), written into
    # `memory__clai2` on accept. `text` holds the same content.
    repo_slug: str | None = None
    path: str | None = None
    content: str | None = None
    """The latest proposed full file content."""
    base_content: str | None = None
    """The file as `memory__clai2` (production) has it now; None for a new file."""
    base_sha: str | None = None
    """What the proposing agent saw as the current file, as it reported it."""
    why: str | None = None
    proposed_by: str | None = None
    """Email of whoever proposed the latest content."""
    proposal_count: int | None = None
    """How many `memory proposal` spans this file collected (all of them are evidence)."""
    review_flag: str | None = None
    """Set when a light check thinks this is a personal preference rather than a repo fact. Flags, never drops."""

    _clean = field_validator('id', 'name', 'description', 'text', 'rationale', 'pattern')(clean_text)


class Window(BaseModel):
    start: datetime
    end: datetime


class Developer(BaseModel):
    """Who a `developer` number in the document is: their `user.email` and machine, as far as either is known."""

    email: str | None = None
    host: str | None = None


class ProposalsDoc(BaseModel):
    """The value of the `fleet_proposals__clai2` managed variable."""

    generated_at: datetime
    window: Window
    min_users: int | None = None
    """Distinct developers a pattern needed in the run that wrote this document."""
    developers: dict[str, Developer] = {}
    """Evidence `developer` number (as a string key) -> who it is, so the UI labels every trace the same way."""
    proposals: list[Proposal] = []


def identify_developers(doc: ProposalsDoc, users_by_span: dict[str, str], hosts: dict[str, str]) -> None:
    """Number developers 1, 2, ... in order of first appearance across the whole document, name them, in place.

    `users_by_span` maps evidence spans to an email, or `host:<machine>` when no email is linked to it; `hosts` maps
    emails to their machine.
    """
    numbers: dict[str, int] = {}
    developers: dict[str, Developer] = {}
    for proposal in doc.proposals:
        for evidence in proposal.evidence:
            # Evidence whose span is outside this run's window keeps a number of its own rather than a guessed one.
            user = users_by_span.get(evidence.span_id, f'unknown:{evidence.span_id}')
            evidence.developer = numbers.setdefault(user, len(numbers) + 1)
            email = user if '@' in user and not user.startswith(('host:', 'unknown:')) else None
            evidence.email = email
            host = user.removeprefix('host:') if user.startswith('host:') else hosts.get(user)
            developers[str(evidence.developer)] = Developer(email=email, host=host)
    doc.developers = developers
