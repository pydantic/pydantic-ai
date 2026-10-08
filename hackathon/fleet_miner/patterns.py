"""Stages 2 to 4: cluster the extracted rules across the fleet, validate each cluster, gate and rank, then draft.

braindump's synthesize stage, ported to prompts: potential rules (see `extract.py`) are grouped by an LLM (the gateway
only serves the miner's Anthropic model, which has no embeddings, so no cosine clustering), each group is validated
(coherence, one rule, which prompts actually support it, and whether writing it down changes what an agent does),
then deterministic gates and ranking decide what becomes a suggestion.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import keyword
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from pydantic_ai import Agent, ModelRetry, RunContext

from . import __version__
from .extract import RULE_KEY, STEERING_ROUTES, PromptExtraction
from .llm_cache import CACHE_DIR, run_cached
from .models import Evidence, Proposal, Tier, UserPrompt, span_of, strip_markup
from .scope import measure_scope

CLUSTER_INSTRUCTIONS = """\
You are given potential rules for coding agents, each extracted from a prompt a developer typed into their agent, plus
the suggestions made in earlier runs. Group rules that express the SAME guidance, even when worded very differently
("babysit the PR until CI is green" and "keep fixing checks and review comments until they pass" are one rule). Prefer
fewer, broader groups: variants of one way of working belong together. Leave out rules that match nothing else.

For each group: write the shared rule as one sentence, give a short kebab-case slug, and if it is the same guidance as
an earlier suggestion, set `existing_id` to that suggestion's id exactly as given; otherwise leave it null.
`confidence` is how sure you are that this is one coherent rule (0-1).
"""

ASSIGN_INSTRUCTIONS = """\
You keep groups of recurring guidance that developers give their coding agents up to date. You get the existing
groups (id, the shared rule, size) and potential rules extracted since. Put each new rule in the existing group that
expresses the SAME guidance, even if worded differently. Leave out rules that match no group.
"""

UNPLACED_INSTRUCTIONS = """\
You are given potential rules for coding agents, each extracted from a prompt a developer typed into their agent, that
fit none of the known groups. Find the ones that express the SAME guidance, even when worded very differently, and
group them. Leave out rules that match nothing else.

For each group: write the shared rule as one sentence and give a short kebab-case slug. `confidence` is how sure you
are that this is one coherent rule (0-1).
"""

VALIDATE_INSTRUCTIONS = """\
You validate one cluster of potential rules for coding agents. Each was extracted from a prompt a developer typed into
their coding agent, shown with its source prompt. A validated rule is pushed to every developer's agent in the
company, so be strict: a vague or useless rule costs everyone.

1. `common_pattern`: what the cluster is about, in one sentence.
2. `rule`: ONE rule at the level of the pattern: drop the instances (names, features, the task at hand) but keep team
   conventions and named tools. Merge rephrasings and entity-specific copies into one rule, complementary halves into
   one rule with its condition, and pick one framing of inverses.
3. `supporting_ids`: the ids whose SOURCE PROMPT itself expresses this rule. Read each prompt, not only the rule
   extracted from it: a prompt that is merely near the topic, a one-off task ("fix this", "TF is this"), or a question
   does not support it. Be strict; few supporting prompts is fine.
4. `kind`: `instruction` for a rule, preference or convention. `skill` only when the supporting prompts spell out a
   multi-step procedure that developers had to explain; `steps` is then that procedure, generalized.
5. `changes_behaviour`: would a capable coding agent (Claude Code or Codex class) already do this unprompted, given
   the plain request? True only when writing it down changes what the agent does: a preference, a team convention, a
   specific procedure or tool, or a recurring correction of the agent. False for:
   - the task itself ("diagnose the screenshot and fix it", "resolve the merge conflicts", "review this PR"),
   - generic good practice an agent follows anyway,
   - anything narrow to one subject, feature or conversation thread. Questions about the same topic ("how does code
     mode work?") are not a shared need for different behaviour; the one exception is knowledge the agent repeatedly
     lacked and had to be told, shown by the prompts correcting it, and then the rule is that knowledge.
   `value_reason`: one line, what changes in the agent's behaviour, or why nothing does.
6. `cluster_coherence` (0-1): how closely the SUPPORTING prompts share this one rule (not the left-out ones).
   `confidence` (0-1), braindump's scale: high (0.8+) when the supporting prompts clearly converge on this rule;
   medium (0.5-0.8) when they are related but the right abstraction is uncertain; low (<0.5) when the rule is
   speculative. Whether it is worth writing down is `changes_behaviour`, not confidence; wording differences (one
   developer adds an example, another says "only") don't lower it.
Set `rule` to null and give `rejection_reason` when they are only superficially related, contradict each other, or
are specific to one task, feature or thread.
"""

DRAFT_INSTRUCTIONS = """\
Several developers at one company had to tell their coding agents the same thing. You get the validated rule, whether
it is an `instruction` or a `skill` (decided already), and what they typed. Turn it into text pushed to every
developer's agent, so nobody has to type it again.

Write clear, generalized guidance in your own words; do not copy their phrasing. Use what they typed as the measure
of how much context it needs: if they got by with one line, yours is about one line, not three paragraphs. You are
given the median length of their prompts: keep `text` close to that length, and never more than twice as long.
A skill's `text` is its steps, with at most one stop condition: no headings, no "Purpose" or "When to use" sections,
no generic advice they didn't ask for. An instruction's `text` is the rule.

`name`: a short kebab-case slug.
`description`: for a skill, this one line is all the agent sees when deciding whether to load the skill, so write it
as an explicit trigger naming the concrete situation, starting with "Load whenever" or "Load when" (e.g. "Load
whenever you open or push to a pull request, to keep going until CI and review bots are green."), not a summary.
For an instruction, one short sentence saying when it applies.
`suggested_tier`: `required` if nearly everyone would want it, `default_on` if broadly useful, `optional` if niche.
`rationale`: one or two sentences: how many people had to ask, and what it saves them.

`scope`: `repo` when the rule only makes sense in one codebase (it names a specific repository, its paths, modules,
scripts, CI jobs, branch conventions or tools unique to it), `organization` when any developer on any repository could
use it. `scope_reason`: one short sentence saying which detail makes it repo-specific, or null. Tools and bots used
across many repos (gh, Macroscope, CI in general) do not make it repo-specific.

Never include personal identifiers in any field: no people's names, GitHub usernames, handles or emails. Replace a
person with their role ("the requested reviewer", "the PR author"). Do keep the names of tools, bots and repository
conventions (e.g. Macroscope, douwebot, `SKIP=typecheck`): they are useful context for an organization-wide rule.
"""


class _Group(BaseModel):
    slug: str
    pattern: str
    member_ids: list[str]
    existing_id: str | None = None
    confidence: float = Field(ge=0, le=1)


class _Groups(BaseModel):
    groups: list[_Group]


class _Validation(BaseModel):
    common_pattern: str
    rule: str | None
    kind: Literal['instruction', 'skill'] = 'instruction'
    steps: list[str] = []
    supporting_ids: list[str] = []
    cluster_coherence: float = Field(ge=0, le=1)
    changes_behaviour: bool
    value_reason: str
    confidence: float = Field(ge=0, le=1)
    rejection_reason: str | None = None

    _strip_markup = field_validator('common_pattern', 'value_reason')(strip_markup)


class _Draft(BaseModel):
    name: str
    description: str
    text: str
    suggested_tier: Tier
    rationale: str
    scope: Literal['organization', 'repo']
    """Only a fallback: the miner measures scope from evidence whenever repo or team data exists."""
    scope_reason: str | None = None

    _strip_markup = field_validator('name', 'description', 'text', 'rationale')(strip_markup)


@dataclass(frozen=True)
class Item:
    """One thing to cluster: a potential rule extracted from one prompt."""

    id: str
    span_id: str
    rule: str
    motivation: str

    def item(self) -> dict[str, object]:
        return {'id': self.id, 'rule': self.rule, 'motivation': self.motivation}


def items_of(prompts: list[UserPrompt], extractions: dict[str, PromptExtraction]) -> dict[str, Item]:
    """Every potential rule of every actionable prompt in the window, keyed `<span_id>#r<n>`."""
    items: dict[str, Item] = {}
    for p in prompts:
        if (e := extractions.get(p.span_id)) is None or not e.is_actionable:
            continue
        for n, rule in enumerate(e.potential_rules):
            key = f'{p.span_id}{RULE_KEY}{n}'
            items[key] = Item(key, p.span_id, rule.generalization, rule.motivation)
    return items


@dataclass
class Pattern:
    id: str
    pattern: str
    confidence: float
    prompts: list[UserPrompt]
    """One per prompt span, even when several rules of one prompt are in the group."""
    existing_id: str | None = None
    users: set[str] = field(default_factory=set[str])
    sessions: set[str] = field(default_factory=set[str])
    item_ids: list[str] = field(default_factory=list[str])
    """The grouped items: `<span_id>#r<n>` rule ids."""

    @property
    def days(self) -> set[str]:
        return {p.timestamp.date().isoformat() for p in self.prompts}

    @property
    def latest(self) -> datetime:
        return max(p.timestamp for p in self.prompts)


SPREAD = {0: 0.0, 1: 0.6, 2: 0.85, 3: 0.95}
"""braindump's spread factor over unique PRs, here over distinct developers (4+ -> 1.0)."""


def _user(p: UserPrompt) -> str:
    return p.user or f'unknown:{p.session_id or p.trace_id}'


def _pattern(
    id: str,
    pattern: str,
    confidence: float,
    item_ids: list[str],
    by_span: dict[str, UserPrompt],
    existing_id: str | None = None,
) -> Pattern:
    item_ids = [i for i in dict.fromkeys(item_ids) if span_of(i) in by_span]
    prompts = [by_span[s] for s in dict.fromkeys(span_of(i) for i in item_ids)]
    return Pattern(
        id=id,
        pattern=pattern,
        confidence=confidence,
        prompts=prompts,
        existing_id=existing_id,
        users={_user(p) for p in prompts},
        sessions={p.session_id or p.trace_id for p in prompts},
        item_ids=item_ids,
    )


def _by_size(patterns: list[Pattern]) -> list[Pattern]:
    return sorted(patterns, key=lambda p: (len(p.users), len(p.prompts), p.id), reverse=True)


async def find_patterns(
    prompts: list[UserPrompt], extractions: dict[str, PromptExtraction], *, model: str, existing: list[Proposal]
) -> list[Pattern]:
    by_span = {p.span_id: p for p in prompts}
    items = [i.item() for i in items_of(prompts, extractions).values()]
    if not items:
        return []
    earlier = [
        {'id': p.id, 'rule': p.pattern, 'status': p.status} for p in existing if p.kind in ('skill', 'instruction')
    ]
    agent = Agent(model, output_type=_Groups, instructions=CLUSTER_INSTRUCTIONS, name='fleet_miner_cluster')
    output = await run_cached(
        agent,
        f'Earlier suggestions:\n{json.dumps(earlier, indent=2)}\n\nPotential rules:\n{json.dumps(items, indent=2)}',
        output_type=_Groups,
    )
    known_ids = {p.id for p in existing}
    patterns: list[Pattern] = []
    for group in output.groups:
        # Only reuse an id the model was actually shown; anything else is a mangled or invented one.
        existing_id = group.existing_id if group.existing_id in known_ids else None
        pattern = _pattern(
            existing_id or group.slug, group.pattern, group.confidence, group.member_ids, by_span, existing_id
        )
        if len(pattern.prompts) >= 2:
            patterns.append(pattern)
    return _by_size(patterns)


def assign_stable_ids(
    patterns: list[Pattern],
    existing: list[Proposal],
    *,
    prior_spans: dict[str, set[str]] | None = None,
    single_rule_spans: set[str] = frozenset(),  # pyright: ignore[reportArgumentType]
    taken: set[str] | None = None,
) -> None:
    """Give each cluster the id of the earlier proposal it continues, so a renamed cluster can't resurrect an
    accepted or dismissed pattern under a new id.

    In order: the earlier proposal sharing the most rule ids (`prior_spans` records every item ever grouped under an
    id); else the clustering model's match (`existing_id`); else, for proposals from before rules were extracted, the
    one sharing the most prompts, counting only prompts that yield a single rule (`single_rule_spans`: a prompt with
    four rules says nothing about which of them an old intent was) and needing 2+ of them and half the cluster. Only a
    cluster that matches nothing gets a new id, never one already taken.
    """
    prior = [p for p in existing if p.kind in ('skill', 'instruction')]
    items_of_id = {p.id: (prior_spans or {}).get(p.id, set()) for p in prior}
    spans_of = {p.id: {span_of(i) for i in items_of_id[p.id]} or {e.span_id for e in p.evidence} for p in prior}
    taken = set(taken or ())

    def best(scores: dict[str, int]) -> str | None:
        top = max(scores, key=lambda pid: (scores[pid], pid), default=None)
        return top if top is not None and scores[top] > 0 else None

    for pattern in sorted(patterns, key=lambda p: len(p.prompts), reverse=True):
        free = [pid for pid in spans_of if pid not in taken]
        by_items = best({pid: len(set(pattern.item_ids) & items_of_id[pid]) for pid in free})
        spans = {p.span_id for p in pattern.prompts} & single_rule_spans
        by_spans = best({pid: len(spans & spans_of[pid]) for pid in free})
        if by_items is not None:
            pattern.id = pattern.existing_id = by_items
        elif pattern.existing_id in free:
            pattern.id = pattern.existing_id
        elif by_spans is not None and (n := len(spans & spans_of[by_spans])) >= 2 and 2 * n >= len(pattern.prompts):
            pattern.id = pattern.existing_id = by_spans
        else:
            pattern.existing_id = None
            base, n = pattern.id, 2
            while pattern.id in taken or pattern.id in spans_of or any(pattern.id == p.id for p in existing):
                pattern.id, n = f'{base}-{n}', n + 1
        taken.add(pattern.id)


def single_rule_spans(extractions: dict[str, PromptExtraction]) -> set[str]:
    return {s for s, e in extractions.items() if len(e.potential_rules) == 1}


SPANS_PATH = CACHE_DIR / 'pattern_spans.json'
"""Every item id ever assigned to each proposal id, across runs (the clusters file only has the latest run).

Item ids are `<span_id>#r<n>` rule ids, or, from runs before rules were extracted, prompt span ids and
`<span_id>#pref<n>` (see `span_of`).
"""


def load_prior_spans(path: Path = SPANS_PATH) -> dict[str, set[str]]:
    return {pid: set(spans) for pid, spans in json.loads(path.read_text()).items()} if path.exists() else {}


def record_pattern_spans(patterns: list[Pattern], path: Path = SPANS_PATH) -> dict[str, set[str]]:
    spans = load_prior_spans(path)
    for p in patterns:
        spans.setdefault(p.id, set()).update(p.item_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({pid: sorted(s) for pid, s in spans.items()}))
    return spans


CLUSTERS_PATH = CACHE_DIR / 'rule_clusters.json'
"""The latest groups of rule ids. (`clusters.json` held intent groups, before rules were extracted.)"""


def save_patterns(patterns: list[Pattern], path: Path = CLUSTERS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            'id': p.id,
            'pattern': p.pattern,
            'confidence': p.confidence,
            'existing_id': p.existing_id,
            'item_ids': p.item_ids,
        }
        for p in patterns
    ]
    path.write_text(json.dumps(data, indent=2))


def load_patterns(prompts: list[UserPrompt], path: Path = CLUSTERS_PATH) -> list[Pattern]:
    """Rebuild cached groups against the current prompts, so a run can continue without re-clustering."""
    by_span = {p.span_id: p for p in prompts}
    return _by_size(
        [
            _pattern(item['id'], item['pattern'], item['confidence'], item['item_ids'], by_span, item['existing_id'])
            for item in json.loads(path.read_text())
        ]
    )


class _ExistingAssignment(BaseModel):
    group_id: str
    member_ids: list[str]


class _Placements(BaseModel):
    to_existing: list[_ExistingAssignment]


CLUSTERED_PATH = CACHE_DIR / 'rule_clustered.json'
"""Every rule id the incremental step has already offered to the existing groups, placed or not."""


def load_clustered_spans() -> set[str]:
    return set(json.loads(CLUSTERED_PATH.read_text())) if CLUSTERED_PATH.exists() else set()


def save_clustered_spans(ids: set[str]) -> None:
    CLUSTERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLUSTERED_PATH.write_text(json.dumps(sorted(ids)))


async def update_patterns(
    prompts: list[UserPrompt],
    extractions: dict[str, PromptExtraction],
    cached: list[Pattern],
    *,
    model: str,
    existing: list[Proposal],
    max_unplaced: int = 150,
) -> list[Pattern]:
    """Incremental clustering, so a run costs what is new rather than what is in the window.

    1. Rules never offered before are placed into the cached groups (one call, only when there are any).
    2. Every rule in no group ("unplaced", new or earlier) is clustered among the unplaced only, so a theme that is
       new since the last full re-cluster still becomes a pattern. They are few, and the call is cached by its exact
       input, so an unchanged set costs nothing.
    """
    by_span = {p.span_id: p for p in prompts}
    seen = load_clustered_spans()
    items = items_of(prompts, extractions)
    grouped = {i for p in cached for i in p.item_ids}
    new = [i for i in items if i not in seen and i not in grouped]
    by_id = {p.id: p for p in cached}
    if new and cached:
        agent = Agent(model, output_type=_Placements, instructions=ASSIGN_INSTRUCTIONS, name='fleet_miner_assign')
        groups = [{'id': p.id, 'rule': p.pattern, 'size': len(p.prompts)} for p in cached]
        output = await run_cached(
            agent,
            f'Existing groups:\n{json.dumps(groups, indent=2)}\n\n'
            f'New potential rules:\n{json.dumps([items[i].item() for i in new], indent=2)}',
            output_type=_Placements,
        )
        allowed = set(new)
        for assignment in output.to_existing:
            if (pattern := by_id.get(assignment.group_id)) is None:
                continue
            placed = [i for i in assignment.member_ids if i in allowed and i not in grouped]
            grouped.update(placed)
            by_id[pattern.id] = _pattern(
                pattern.id, pattern.pattern, pattern.confidence, pattern.item_ids + placed, by_span, pattern.existing_id
            )
    save_clustered_spans(seen | set(items))

    unplaced = sorted((i for i in items if i not in grouped), key=lambda i: (by_span[items[i].span_id].timestamp, i))
    unplaced = unplaced[-max_unplaced:]
    fresh: list[Pattern] = []
    if len({items[i].span_id for i in unplaced}) >= 2:
        agent = Agent(model, output_type=_Groups, instructions=UNPLACED_INSTRUCTIONS, name='fleet_miner_unplaced')
        output = await run_cached(
            agent,
            f'Potential rules in no group:\n{json.dumps([items[i].item() for i in unplaced], indent=2)}',
            output_type=_Groups,
        )
        allowed = set(unplaced)
        for group in output.groups:
            members = [i for i in dict.fromkeys(group.member_ids) if i in allowed and i not in grouped]
            pattern = _pattern(group.slug, group.pattern, group.confidence, members, by_span)
            if len(pattern.prompts) < 2:
                continue
            grouped.update(members)
            fresh.append(pattern)
        assign_stable_ids(
            fresh,
            existing,
            prior_spans=load_prior_spans(),
            single_rule_spans=single_rule_spans(extractions),
            taken=set(by_id),
        )
    return _by_size([*by_id.values(), *fresh])


@dataclass(frozen=True)
class Gates:
    """What a validated cluster needs to become a suggestion. Counts are over the prompts validation verified."""

    min_users: int = 2
    min_prompts: int = 3
    min_sessions: int = 2
    min_days: int = 2
    """Evidence from this many distinct days, or else `min_sessions_one_day` sessions: one afternoon of overlapping
    work in one shared thread is not a recurring need."""
    min_sessions_one_day: int = 3
    min_confidence: float = 0.8
    """braindump's "high" confidence."""
    min_coherence: float = 0.7
    max_pending: int = 5
    """Pending skills and instructions at most; the rest that pass are kept as `emerging`."""
    correction_bonus: float = 0.05
    """Added to the score per verified correction (up to 3): the agent had to be told, live."""

    def counting_failure(self, p: Pattern, *, verified: bool = False) -> str | None:
        """Why these prompts are too few, or None. Applied before validation (to skip the call) and after it."""
        users, prompts, sessions, days = len(p.users), len(p.prompts), len(p.sessions), len(p.days)
        v = 'verified ' if verified else ''
        if users < self.min_users:
            return f'only {users} {v}developer{"" if users == 1 else "s"}'
        if prompts < self.min_prompts:
            return f'only {prompts} {v}prompt{"" if prompts == 1 else "s"}'
        if sessions < self.min_sessions:
            return f'only 1 session among the {v}prompts'
        if days < self.min_days and sessions < self.min_sessions_one_day:
            return f'all {sessions} sessions of the {v}prompts on one day'
        return None


@dataclass
class Candidate:
    """A cluster after validation: verified evidence, the judgement, and whether it passed the gates."""

    pattern: Pattern
    """Rebuilt from the supporting (verified) prompts only."""
    validation: _Validation | None
    """None when the cluster was too small to be worth validating."""
    corrections: int = 0
    failure: str | None = None
    """Why it does not pass, for `status_reason` ("didn't pass: ...")."""

    @property
    def score(self) -> float:
        confidence = self.validation.confidence if self.validation else 0.0
        spread = SPREAD.get(len(self.pattern.users), 1.0)
        return round(confidence * spread, 3)

    def rank(self, gates: Gates) -> tuple[float, int, datetime]:
        bonus = gates.correction_bonus * min(self.corrections, 3)
        return (round(self.score + bonus, 3), len(self.pattern.sessions), self.pattern.latest)


def _is_correction(prompt: UserPrompt, extractions: dict[str, PromptExtraction]) -> bool:
    e = extractions.get(prompt.span_id)
    return (e is not None and e.kind == 'correction') or prompt.route in STEERING_ROUTES


async def validate_patterns(
    patterns: list[Pattern],
    extractions: dict[str, PromptExtraction],
    *,
    model: str,
    gates: Gates,
) -> list[Candidate]:
    """braindump's cluster analysis, plus evidence verification and the "does it change behaviour" judgement.

    One cached call per cluster that could pass on its raw counts; a cluster whose input is unchanged costs nothing.
    """
    agent = Agent(model, output_type=_Validation, instructions=VALIDATE_INSTRUCTIONS, name='fleet_miner_validate')
    rules = {
        f'{span_id}{RULE_KEY}{n}': r for span_id, e in extractions.items() for n, r in enumerate(e.potential_rules)
    }

    async def one(pattern: Pattern) -> Candidate:
        if failure := gates.counting_failure(pattern):
            return Candidate(pattern, None, failure=f"didn't pass: {failure}")
        by_span = {p.span_id: p for p in pattern.prompts}
        developers = {u: n for n, u in enumerate(dict.fromkeys(_user(p) for p in pattern.prompts), 1)}
        sessions = {s: n for n, s in enumerate(dict.fromkeys(p.session_id or p.trace_id for p in pattern.prompts), 1)}
        members: list[dict[str, object]] = []
        for item_id in pattern.item_ids:
            p, rule = by_span[span_of(item_id)], rules.get(item_id)
            if rule is None:
                continue
            members.append(
                {
                    'id': item_id,
                    'developer': developers[_user(p)],
                    'session': sessions[p.session_id or p.trace_id],
                    'date': p.timestamp.date().isoformat(),
                    'rule': rule.generalization,
                    'motivation': rule.motivation,
                    'kind': extractions[p.span_id].kind,
                    'typed_mid_run': p.route in STEERING_ROUTES if p.route is not None else None,
                    'source_prompt': p.text[:600],
                }
            )
        validation = await run_cached(
            agent,
            f'Cluster: {pattern.pattern}\n'
            f'{len(members)} potential rules from {len(developers)} developers in {len(sessions)} sessions.\n'
            f'{json.dumps(members, indent=2)}',
            output_type=_Validation,
        )
        supported = [i for i in pattern.item_ids if i in set(validation.supporting_ids)]
        verified = _pattern(
            pattern.id, validation.rule or pattern.pattern, pattern.confidence, supported, by_span, pattern.existing_id
        )
        corrections = sum(_is_correction(p, extractions) for p in verified.prompts)
        return Candidate(verified, validation, corrections, failure=_failure(validation, verified, gates))

    return list(await asyncio.gather(*(one(p) for p in patterns)))


def _failure(v: _Validation, verified: Pattern, gates: Gates) -> str | None:
    if v.rule is None:
        reason = v.rejection_reason or 'no single rule'
    elif not v.changes_behaviour:
        reason = f"doesn't change what an agent does ({v.value_reason})"
    elif v.cluster_coherence < gates.min_coherence:
        reason = f"the prompts don't share one pattern (coherence {v.cluster_coherence:.2f})"
    elif v.confidence < gates.min_confidence:
        reason = f'low confidence ({v.confidence:.2f})'
    elif counting := gates.counting_failure(verified, verified=True):
        reason = counting
    else:
        return None
    return f"didn't pass: {reason}"


def rank_and_cap(
    candidates: list[Candidate],
    gates: Gates,
    *,
    reviewed: set[str] = frozenset(),  # pyright: ignore[reportArgumentType]
) -> tuple[list[Candidate], list[Candidate]]:
    """Passing candidates, best first, split into the `max_pending` suggestions and the emerging rest.

    Ids already accepted or dismissed (`reviewed`) are left out: they are never re-proposed, so they take no slot.
    """
    passing = sorted(
        (c for c in candidates if c.failure is None and c.pattern.id not in reviewed),
        key=lambda c: c.rank(gates),
        reverse=True,
    )
    return passing[: gates.max_pending], passing[gates.max_pending :]


def unmatched_reason(spans: set[str], extractions: dict[str, PromptExtraction]) -> str:
    """Why an earlier pending suggestion that no cluster continues isn't one any more, from its prompts' extractions."""
    rejected = Counter(e.rejection.reason for s in spans if (e := extractions.get(s)) and e.rejection)
    actionable = sum(1 for s in spans if (e := extractions.get(s)) and e.is_actionable)
    if rejected and sum(rejected.values()) > actionable:
        kinds = ', '.join(f'{k} {n}' for k, n in rejected.most_common())
        return f"didn't pass: its prompts are not guidance for the agent ({kinds} of {len(spans)})"
    return "didn't pass: no longer recurs across 2+ developers"


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


def _scope_fields(
    pattern: Pattern, draft: _Draft, *, window_teams: set[str], window_repos: set[str]
) -> dict[str, object]:
    measured = measure_scope(
        [(p.team, p.repo_slug) for p in pattern.prompts], window_teams=window_teams, window_repos=window_repos
    )
    if measured is not None:
        scope, reason, applies_to = measured
        return {'scope': scope, 'scope_reason': reason, 'applies_to': applies_to}
    reason = draft.scope_reason or (
        'Nothing in it is specific to one codebase.' if draft.scope == 'organization' else ''
    )
    tagged = sum(1 for p in pattern.prompts if p.team or p.repo_slug)
    basis = (
        f'only {tagged} of {len(pattern.prompts)} prompts carry repo or team data' if tagged else 'no repo or team data'
    )
    return {'scope': draft.scope, 'scope_reason': f'LLM judgment ({basis}): {reason}'.strip()}


async def draft_proposals(
    candidates: list[Candidate],
    *,
    model: str,
    emerging: set[str] = frozenset(),
    max_evidence: int = 5,
    window_teams: set[str] = frozenset(),
    window_repos: set[str] = frozenset(),
) -> list[Proposal]:
    """Draft each passing candidate from its verified prompts; ids in `emerging` are kept `stale` and `emerging`."""
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

    async def one(candidate: Candidate) -> Proposal:
        pattern, validation = candidate.pattern, candidate.validation
        assert validation is not None and validation.rule is not None
        # The drafter never sees who asked: developers are numbered, not named.
        numbers = {user: n for n, user in enumerate(dict.fromkeys(_user(p) for p in pattern.prompts), 1)}
        examples = [{'developer': numbers[_user(p)], 'prompt': p.text[:2000]} for p in pattern.prompts[:12]]
        identifiers = personal_identifiers(pattern.prompts)
        target = median_prompt_chars(pattern)
        steps = '\n'.join(f'{n}. {s}' for n, s in enumerate(validation.steps, 1))
        draft = await run_cached(
            agent,
            f'Validated rule: {validation.rule}\nKind: {validation.kind}\n'
            + (f'Steps they spelled out:\n{steps}\n' if validation.kind == 'skill' and steps else '')
            + f'Asked by {len(pattern.users)} distinct developers across {len(pattern.sessions)} sessions'
            f'{f", {candidate.corrections} of them correcting the agent" if candidate.corrections else ""}.\n'
            f'Median prompt length: {target} characters. Target for `text`: about {target}, at most {2 * target}.\n'
            f'What they typed:\n{json.dumps(examples, indent=2)}',
            output_type=_Draft,
            deps=_DraftDeps(identifiers=identifiers, target_chars=target),
        )
        if leaked := leaked_identifiers(draft, identifiers):  # pragma: no cover - only if retries ran out
            print(f'warning: redacted {len(leaked)} personal identifier(s) from `{pattern.id}`')
            draft = _Draft.model_validate({k: _redact(v, leaked) if isinstance(v, str) else v for k, v in draft})
        is_emerging = pattern.id in emerging
        return Proposal(
            id=pattern.id,
            kind=validation.kind,
            pattern=validation.rule,
            distinct_users=len(pattern.users),
            sessions=len(pattern.sessions),
            verified_prompts=len(pattern.prompts),
            corrections=candidate.corrections,
            value_reason=validation.value_reason,
            evidence=_evidence(pattern, max_evidence),
            score=candidate.score,
            status='stale' if is_emerging else 'pending',
            status_reason='Emerging: passes every gate, but ranks below the pending suggestions.'
            if is_emerging
            else None,
            emerging=is_emerging,
            generated_by=f'fleet-miner {__version__} / {model}',
            **draft.model_dump(exclude={'scope', 'scope_reason'}),
            **_scope_fields(pattern, draft, window_teams=window_teams, window_repos=window_repos),
        )

    return list(await asyncio.gather(*(one(c) for c in candidates)))


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
    existing: list[Proposal],
    fresh: list[Proposal],
    *,
    stale_kinds: set[str] = frozenset(),
    stale_reasons: dict[str, str] | None = None,
) -> tuple[list[Proposal], dict[str, MergeAction]]:
    """Never re-propose an accepted or dismissed id; refresh a pending (or stale) one's evidence, draft and status.

    A pending proposal of a kind this run fully re-mined (`stale_kinds`) that it no longer suggests becomes `stale`,
    with `stale_reasons[id]` (or a generic reason) as its `status_reason`. Accepted and dismissed ones are untouched.
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
            reason = (stale_reasons or {}).get(id_, "didn't pass: no longer found often enough")
            by_id[id_] = proposal.model_copy(update={'status': 'stale', 'status_reason': reason, 'emerging': False})
            actions[id_] = 'stale'
        elif (
            id_ not in actions
            and proposal.status == 'stale'
            and proposal.kind in stale_kinds
            and (reason := (stale_reasons or {}).get(id_))
        ):
            by_id[id_] = proposal.model_copy(update={'status_reason': reason, 'emerging': False})
    return list(by_id.values()), actions
