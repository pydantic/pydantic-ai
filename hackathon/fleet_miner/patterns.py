"""Stages 2 and 3: group intents across the fleet, then draft a skill or instruction per recurring pattern."""

from __future__ import annotations

import asyncio
import builtins
import json
import keyword
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pydantic_ai import Agent, ModelRetry, RunContext

from . import __version__
from .llm_cache import CACHE_DIR, run_cached
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

`name`: a short kebab-case slug.
`description`: for a skill, this one line is all the agent sees when deciding whether to load the skill, so write it
as an explicit trigger naming the concrete situation, starting with "Load whenever" or "Load when" (e.g. "Load
whenever you open or push to a pull request, to keep going until CI and review bots are green."), not a summary.
For an instruction, one short sentence saying when it applies.
`suggested_tier`: `required` if nearly everyone would want it, `default_on` if broadly useful, `optional` if niche.
`rationale`: one or two sentences: how many people asked, and what it saves them.

`scope`: `repo` when the request only makes sense in one codebase (it names a specific repository, its paths,
modules, scripts, CI jobs, branch conventions or tools unique to it), `company` when any developer on any repository
could use it. `scope_reason`: one short sentence saying which detail makes it repo-specific, or null for `company`.
Tools and bots used across many repos (gh, Macroscope, CI in general) do not make it repo-specific.

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
    scope: Literal['company', 'repo']
    scope_reason: str | None = None

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
    output = await run_cached(
        agent,
        f'Earlier proposals:\n{json.dumps(earlier, indent=2)}\n\nIntents:\n{json.dumps(items, indent=2)}',
        output_type=_Groups,
    )
    known_ids = {p.id for p in existing}
    patterns: list[Pattern] = []
    for group in output.groups:
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


def assign_stable_ids(
    patterns: list[Pattern],
    existing: list[Proposal],
    *,
    prior_spans: dict[str, set[str]] | None = None,
    taken: set[str] | None = None,
) -> None:
    """Give each cluster the id of the earlier proposal it continues, so a renamed cluster can't resurrect an
    accepted or dismissed pattern under a new id.

    A cluster continues an earlier proposal (of any status) when they share prompts: the proposal's evidence spans,
    plus the full span list cached from the run that drafted it (`prior_spans`). Failing that, the clustering model's
    intent match (`existing_id`) decides. Only a cluster that matches nothing gets a new id, never one already taken.
    """
    prior = [p for p in existing if p.kind in ('skill', 'instruction')]
    spans_of = {p.id: {e.span_id for e in p.evidence} | (prior_spans or {}).get(p.id, set()) for p in prior}
    taken = set(taken or ())
    for pattern in sorted(patterns, key=lambda p: len(p.prompts), reverse=True):
        spans = {u.span_id for u in pattern.prompts}
        overlaps = {pid: len(spans & s) for pid, s in spans_of.items() if pid not in taken}
        best = max(overlaps, key=lambda pid: overlaps[pid], default=None)
        if best is not None and overlaps[best] > 0:
            pattern.id = pattern.existing_id = best
        elif pattern.existing_id in spans_of and pattern.existing_id not in taken:
            pattern.id = pattern.existing_id
        else:
            pattern.existing_id = None
            base, n = pattern.id, 2
            while pattern.id in taken or pattern.id in spans_of or any(pattern.id == p.id for p in existing):
                pattern.id, n = f'{base}-{n}', n + 1
        taken.add(pattern.id)


SPANS_PATH = CACHE_DIR / 'pattern_spans.json'
"""Every prompt span ever assigned to each proposal id, across runs (clusters.json only has the latest run)."""


def load_prior_spans(path: Path = SPANS_PATH) -> dict[str, set[str]]:
    return {pid: set(spans) for pid, spans in json.loads(path.read_text()).items()} if path.exists() else {}


def record_pattern_spans(patterns: list[Pattern], path: Path = SPANS_PATH) -> dict[str, set[str]]:
    spans = load_prior_spans(path)
    for p in patterns:
        spans.setdefault(p.id, set()).update(u.span_id for u in p.prompts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({pid: sorted(s) for pid, s in spans.items()}))
    return spans


def save_patterns(path: Path, patterns: list[Pattern]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            'id': p.id,
            'pattern': p.pattern,
            'confidence': p.confidence,
            'existing_id': p.existing_id,
            'span_ids': [u.span_id for u in p.prompts],
        }
        for p in patterns
    ]
    path.write_text(json.dumps(data, indent=2))


ASSIGN_INSTRUCTIONS = """\
You keep groups of recurring requests that developers type into their coding agents up to date. You get the existing
groups (id, the shared request, size), intents typed since the groups were made, and earlier intents that are not in
any group yet. Put each new intent in the existing group that expresses the SAME request (even if worded differently),
or form new groups from new and earlier ungrouped intents that share a request. Leave out intents that match nothing.
Never put a throwaway test task (e.g. "write FizzBuzz in Rust") in a group.
"""


class _ExistingAssignment(BaseModel):
    group_id: str
    span_ids: list[str]


class _Assignments(BaseModel):
    to_existing: list[_ExistingAssignment]
    new_groups: list[_Group]


CLUSTERED_PATH = CACHE_DIR / 'clustered_spans.json'
"""Every intent span clustering has already seen, grouped or not: the incremental step only sends newer ones."""


def load_clustered_spans() -> set[str]:
    return set(json.loads(CLUSTERED_PATH.read_text())) if CLUSTERED_PATH.exists() else set()


def save_clustered_spans(spans: set[str]) -> None:
    CLUSTERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTERED_PATH.write_text(json.dumps(sorted(spans)))


async def update_patterns(
    prompts: list[UserPrompt],
    facets: dict[str, Facet],
    cached: list[Pattern],
    *,
    model: str,
    existing: list[Proposal],
    max_loners: int = 150,
) -> list[Pattern]:
    """Incremental clustering: place only intents typed since the last run, against the cached groups.

    The cost of a run then scales with what is new, not with everything in the window. With nothing new there is
    no LLM call at all.
    """
    by_span = {p.span_id: p for p in prompts}
    seen = load_clustered_spans()
    intents = {s: f for s, f in facets.items() if f.intent and s in by_span}
    new = [s for s in intents if s not in seen]
    if not new:
        return cached
    grouped = {u.span_id for p in cached for u in p.prompts}
    loners = sorted((s for s in intents if s in seen and s not in grouped), key=lambda s: by_span[s].timestamp)
    loners = loners[-max_loners:]
    agent = Agent(model, output_type=_Assignments, instructions=ASSIGN_INSTRUCTIONS, name='fleet_miner_assign')
    groups = [{'id': p.id, 'pattern': p.pattern, 'size': len(p.prompts)} for p in cached]
    output = await run_cached(
        agent,
        f'Existing groups:\n{json.dumps(groups, indent=2)}\n\n'
        f'New intents:\n{json.dumps([{"span_id": s, "intent": intents[s].intent} for s in new], indent=2)}\n\n'
        f'Earlier ungrouped intents:\n{json.dumps([{"span_id": s, "intent": intents[s].intent} for s in loners], indent=2)}',
        output_type=_Assignments,
    )
    by_id = {p.id: p for p in cached}
    allowed = set(new) | set(loners)
    for assignment in output.to_existing:
        if (pattern := by_id.get(assignment.group_id)) is None:
            continue
        for span_id in assignment.span_ids:
            if span_id in allowed and span_id not in grouped:
                pattern.prompts.append(by_span[span_id])
                grouped.add(span_id)
    fresh: list[Pattern] = []
    for group in output.new_groups:
        group_prompts = [by_span[s] for s in dict.fromkeys(group.span_ids) if s in allowed and s not in grouped]
        if len(group_prompts) < 2:
            continue
        grouped.update(p.span_id for p in group_prompts)
        fresh.append(Pattern(id=group.slug, pattern=group.pattern, confidence=group.confidence, prompts=group_prompts))
    assign_stable_ids(fresh, existing, prior_spans=load_prior_spans(), taken={p.id for p in cached})
    patterns = cached + fresh
    for pattern in patterns:
        pattern.users = {p.user or f'unknown:{p.session_id or p.trace_id}' for p in pattern.prompts}
        pattern.sessions = {p.session_id or p.trace_id for p in pattern.prompts}
    save_clustered_spans(seen | set(intents))
    return sorted(patterns, key=lambda p: (len(p.users), p.score), reverse=True)


def load_patterns(path: Path, prompts: list[UserPrompt]) -> list[Pattern]:
    """Rebuild cached groups against the current prompts, so drafting can be re-run without re-clustering."""
    by_span = {p.span_id: p for p in prompts}
    patterns: list[Pattern] = []
    for item in json.loads(path.read_text()):
        group_prompts = [by_span[s] for s in item['span_ids'] if s in by_span]
        pattern = Pattern(
            id=item['id'],
            pattern=item['pattern'],
            confidence=item['confidence'],
            prompts=group_prompts,
            existing_id=item['existing_id'],
        )
        for p in group_prompts:
            pattern.users.add(p.user or f'unknown:{p.session_id or p.trace_id}')
            pattern.sessions.add(p.session_id or p.trace_id)
        patterns.append(pattern)
    return sorted(patterns, key=lambda p: (len(p.users), p.score), reverse=True)


_GENERIC_DOMAINS = {
    'example',
    'test',
    'localhost',
    'gmail',
    'googlemail',
    'outlook',
    'hotmail',
    'yahoo',
    'icloud',
    'proton',
    'protonmail',
    'pydantic',
}


_HOST_OWNER = re.compile(r"^([A-Za-z]+?)'?s-(?:MacBook|MBP|Mac|iMac|Laptop|PC|Desktop)", re.IGNORECASE)
# Ordinary words that the handle patterns can pick up ("commits or code history" must never become "<person>").
_COMMON_WORDS = frozenset(
    """
    about above after again against also always another anything around back because before being below between both
    branch build change changes check code commit commits config context could data default diff does done down each
    error everything file files first from have help here history issue issues just keep know last like line lines
    look make many more most much must need never next note only other over please project pull push read really
    repo review same should since some something still such sure task tell than that their then there these thing
    things this those through time today update used using very want what when where which while will with without
    work would write your
    """.split()
)

_HANDLE_PATTERNS = (
    # "@someone" in prose, not a decorator (`@dataclass` on its own line) or an attribute (`@app.get(`).
    re.compile(r'(?<![\w.])@([A-Za-z0-9](?:[A-Za-z0-9-]{1,37}[A-Za-z0-9])?)(?![\w(-]|\.\w)'),
    re.compile(r'\bassign(?:ed|ee)?(?: it| the PR| this)? to @?([A-Za-z0-9-]{3,39})\b', re.IGNORECASE),
    re.compile(r'/(?:Users|home)/([A-Za-z0-9._-]{3,})/'),
    re.compile(r"\b([A-Z][a-z]{2,})'s (?:coding )?agent\b"),  # attribution lines like "(Claude, X's coding agent)"
    re.compile(r'\b(?:origin/)?([a-z]{3,})/[\w.-]+-\d{8}'),  # personal branch prefixes like name/topic-20261002
)
_NOT_HANDLES = {
    *('main', 'master', 'yourself', 'me', 'the', 'them', 'reviewer', 'author'),
    # Products and bots that get @-mentioned or assigned to, which skills should keep naming.
    *('github', 'gitlab', 'claude', 'codex', 'copilot', 'devin', 'coderabbit', 'macroscope', 'douwebot', 'logfire'),
    *('pydantic', 'anthropic', 'openai', 'gemini'),
    # Words that fill the "name" slot of the handle patterns without being anyone's name (branch prefixes etc.).
    *('agent', 'agents', 'feature', 'feat', 'fix', 'bugfix', 'hotfix', 'release', 'test', 'tests', 'chore', 'docs'),
    *('claude-code', 'origin', 'fork', 'upstream', 'user', 'users', 'home', 'tmp'),
    # Code that looks like a handle: Python keywords, builtins and common decorators.
    *keyword.kwlist,
    *(name.lower() for name in dir(builtins)),
    *('dataclass', 'classmethod', 'staticmethod', 'property', 'cached_property', 'contextmanager'),
    *('asynccontextmanager', 'functools', 'pytest', 'override', 'overload', 'abstractmethod', 'cache', 'mermaid'),
    *_COMMON_WORDS,
}


def _handles(pattern: re.Pattern[str], text: str) -> list[str]:
    # An "@name" that starts its line (after indentation) is a decorator, not a mention.
    return [
        m[1]
        for m in pattern.finditer(text)
        if not (m[0].startswith('@') and not text[text.rfind('\n', 0, m.start()) + 1 : m.start()].strip())
    ]


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
        found |= {m for pattern in _HANDLE_PATTERNS for m in _handles(pattern, p.text)}
    return {f for f in found if len(f) >= 4 and f.lower() not in _NOT_HANDLES and not f.isdigit()}


def leaked_identifiers_in(text: str, identifiers: set[str]) -> set[str]:
    """Which identifiers appear as whole words in `text` (so `douwebot` does not count as a name)."""
    return {i for i in identifiers if re.search(rf'(?<![\w-]){re.escape(i)}(?![\w-])', text, re.IGNORECASE)}


def leaked_identifiers(draft: _Draft, identifiers: set[str]) -> set[str]:
    return leaked_identifiers_in('\n'.join([draft.name, draft.description, draft.text, draft.rationale]), identifiers)


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
        draft = await run_cached(
            agent,
            f'Pattern: {pattern.pattern}\n'
            f'Asked by {len(pattern.users)} distinct developers across {len(pattern.sessions)} sessions.\n'
            f'Median prompt length: {target} characters. Target for `text`: about {target}, at most {2 * target}.\n'
            f'What they typed:\n{json.dumps(examples, indent=2)}',
            output_type=_Draft,
            deps=_DraftDeps(identifiers=identifiers, target_chars=target),
        )
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
    return [Evidence(trace_id=p.trace_id, span_id=p.span_id, timestamp=p.timestamp) for p in picked[:limit]]


MergeAction = Literal['new', 'updated', 'skipped', 'stale']


def merge(
    existing: list[Proposal], fresh: list[Proposal], *, stale_kinds: set[str] = frozenset()
) -> tuple[list[Proposal], dict[str, MergeAction]]:
    """Never re-propose an accepted or dismissed id; refresh a pending (or stale) one's evidence and draft.

    A pending proposal of a kind this run fully re-mined (`stale_kinds`) that it no longer qualifies becomes `stale`.
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
    for id_, proposal in by_id.items():
        if id_ not in actions and proposal.status == 'pending' and proposal.kind in stale_kinds:
            by_id[id_] = proposal.model_copy(update={'status': 'stale'})
            actions[id_] = 'stale'
    return list(by_id.values()), actions
