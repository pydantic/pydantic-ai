"""Stage 1: is each typed prompt actionable guidance for the agent, and if so, which general rules does it imply?

braindump's extract stage (PR review comments -> potential rules), ported to prompts: a prompt either tells the agent
how to behave beyond the task at hand (a correction, a preference, a convention, a procedure it had to spell out), or
it is rejected as a question, a one-off task, an acknowledgment, or unclear. Cached per span id, so only prompts never
seen before cost anything.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, TypeAdapter

from pydantic_ai import Agent

from .llm_cache import USAGE
from .models import UserPrompt

if TYPE_CHECKING:
    from .policy import ToolCall

EXTRACT_INSTRUCTIONS = """\
You read the prompts one developer typed into their coding agent during one session, in order, and decide for each
whether it carries guidance about how the agent should work IN GENERAL, beyond the task at hand.

Each prompt comes with the developer's previous prompt and, when known, the agent's last shell commands and tool calls
before it (`agent_actions_before`), so you can tell when the developer is reacting to what the agent just did.
`typed_mid_run` is true when the developer typed it while the agent was still working, to steer it.

## Actionable

A prompt is actionable when it tells the agent how to behave, in a way that would apply to other tasks too:
- `correction`: the developer corrects or stops the agent ("no, don't X", "stop Y", "you should have Z", "why did you
  ask me, just do it"). The strongest signal: the agent did the wrong thing and had to be told.
- `preference`: a standing preference, even stated in passing inside a one-off task ("... and use British English").
- `convention`: a team or project convention (tools, branches, where things go, how PRs are handled).
- `procedure`: the developer had to spell out a multi-step way of working the agent should follow ("open the PR, then
  watch CI and fix failures until green, then address the review bot").

## Rejected (set `rejection`)

- `question`: asking about something ("how does X work?", "what do we do when Y?", "does our agent do Z?"). Questions
  about one piece of the codebase or a feature are never guidance, however often they recur.
- `task`: a one-off request whose content is the work itself ("fix this", "look at the screenshot", "resolve the
  merge conflicts", "review this PR", "add feature X"). Doing what was asked is what any agent does already.
- `acknowledgment`: "yes", "ok", "continue", "thanks", answers to the agent's questions.
- `unclear`: not enough context to tell.

A task that also states a general preference or correction in passing is actionable for that part only: extract the
preference, not the task.

## Potential rules (actionable prompts only)

For each piece of guidance, write ONE rule at the level of THE PATTERN: generalize the instance (strip names, paths,
issue numbers, the specific feature being worked on) but keep project or team conventions and named tools that make it
actionable.
- BAD: "Use British English in the FizzBuzz README" (the instance). BAD: "Write well" (too generic).
  GOOD: "Write all output in British English spelling."
- BAD: "Keep going on the auth refactor" (instance). GOOD: "Keep working until the task is done instead of stopping to
  ask for confirmation."
Do NOT generate rephrasings ("use X" and "prefer X"), complementary halves ("do X when Y" plus "don't do X when Z":
write one rule with the condition), inverses ("do X" and "avoid not-X"), or entity-specific copies of one general rule.
One good rule is better than three overlapping ones; most actionable prompts yield exactly one.

`motivation`: why the developer wants it, in a few words. `scope`: `global`, or the repository, tool or area it is
specific to. `steps`: only for a `procedure`, the steps the developer spelled out, in order, generalized like the rule.
"""


class PotentialRule(BaseModel):
    generalization: str = Field(description='The rule: one actionable imperative sentence of at most 25 words.')
    motivation: str
    scope: str | None = None
    steps: list[str] = Field(default=[], description='Only for a procedure: the spelled-out steps, in order.')


class Rejection(BaseModel):
    reason: Literal['question', 'task', 'acknowledgment', 'unclear']
    explanation: str = Field(description='A few words.')


GuidanceKind = Literal['correction', 'preference', 'convention', 'procedure']


class PromptExtraction(BaseModel):
    span_id: str
    is_actionable: bool
    kind: GuidanceKind | None = Field(default=None, description='Set when actionable.')
    rejection: Rejection | None = Field(default=None, description='Set when not actionable.')
    potential_rules: list[PotentialRule] = []


class _SessionExtractions(BaseModel):
    extractions: list[PromptExtraction]


_cache_adapter = TypeAdapter(dict[str, PromptExtraction])

STEERING_ROUTES = frozenset({'queued', 'run now', 'edited queued'})
"""clai2's `prompt submitted` routes for a prompt typed while the agent was working (see `live_prompt.accept`)."""

RULE_KEY = '#r'
"""Rule ids: `<span_id>#r<n>` for the n-th potential rule extracted from a prompt (see `models.span_of`)."""


class ExtractionCache:
    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, PromptExtraction] = (
            _cache_adapter.validate_json(path.read_bytes()) if path.exists() else {}
        )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(_cache_adapter.dump_json(self.items, indent=2))


def _session(prompt: UserPrompt) -> str:
    """The same key tool calls carry as `session_id` (see `policy.fetch_tool_calls`)."""
    return prompt.session_id or prompt.trace_id


def agent_actions_before(
    session_prompts: list[UserPrompt], calls: list[ToolCall], *, limit: int = 5
) -> dict[str, list[str]]:
    """Per prompt span: the agent's last shell commands and tool calls since the previous prompt (local data, free)."""
    by_time = sorted(calls, key=lambda c: c.timestamp)
    actions: dict[str, list[str]] = {}
    previous = None
    for p in session_prompts:
        start = previous.timestamp if previous else p.timestamp - timedelta(minutes=30)
        turn = [c for c in by_time if start <= c.timestamp < p.timestamp]
        actions[p.span_id] = [
            f'{c.tool}: {" ".join(c.command.split())[:160]}' if c.command else c.tool for c in turn[-limit:]
        ]
        previous = p
    return actions


async def extract_rules(
    prompts: list[UserPrompt],
    *,
    model: str,
    cache: ExtractionCache,
    calls: Sequence[ToolCall] = (),
    concurrency: int = 16,
) -> dict[str, PromptExtraction]:
    """One extraction per prompt span id, only calling the model for sessions with prompts never extracted."""
    agent = Agent(model, output_type=_SessionExtractions, instructions=EXTRACT_INSTRUCTIONS, name='fleet_miner_extract')
    by_session: dict[str, list[UserPrompt]] = defaultdict(list)
    for prompt in prompts:
        by_session[_session(prompt)].append(prompt)
    calls_by_session: dict[str, list[ToolCall]] = defaultdict(list)
    for call in calls:
        calls_by_session[call.session_id].append(call)
    semaphore = asyncio.Semaphore(concurrency)

    async def one(key: str, session_prompts: list[UserPrompt]) -> None:
        session_prompts = sorted(session_prompts, key=lambda p: p.timestamp)
        if all(p.span_id in cache.items for p in session_prompts):
            return
        actions = agent_actions_before(session_prompts, calls_by_session.get(key, []))
        payload = [
            {
                'span_id': p.span_id,
                'previous_prompt': prev.text[:300] if prev else None,
                'agent_actions_before': actions.get(p.span_id) or None,
                'typed_mid_run': p.route in STEERING_ROUTES if p.route is not None else None,
                'prompt': p.text[:2000],
            }
            for prev, p in zip([None, *session_prompts[:-1]], session_prompts)
            if p.span_id not in cache.items
        ]
        async with semaphore:
            result = await agent.run(
                'New prompts from one session, in order (extract one entry for every span_id):\n'
                + json.dumps(payload, indent=2)
            )
        USAGE.add(result)
        wanted = {p.span_id for p in session_prompts if p.span_id not in cache.items}
        for extraction in result.output.extractions:
            if extraction.span_id in wanted:
                cache.items[extraction.span_id] = _normalized(extraction)
        cache.save()  # per session, so an interrupted run keeps what it paid for

    await asyncio.gather(*(one(k, ps) for k, ps in by_session.items()))
    cache.save()
    return {p.span_id: cache.items[p.span_id] for p in prompts if p.span_id in cache.items}


def _normalized(e: PromptExtraction) -> PromptExtraction:
    """An actionable prompt without rules carries nothing to cluster; a rejected one never carries rules."""
    if e.is_actionable and e.potential_rules:
        return e.model_copy(update={'rejection': None})
    rejection = e.rejection or Rejection(reason='unclear', explanation='no general rule extracted')
    return e.model_copy(update={'is_actionable': False, 'kind': None, 'rejection': rejection, 'potential_rules': []})


def summary(extractions: dict[str, PromptExtraction]) -> str:
    """`41 actionable (correction 12, preference 9, ...), 122 rejected (task 80, question 25, ...)`."""
    kinds = Counter[str](e.kind for e in extractions.values() if e.kind)
    rejected = Counter[str](e.rejection.reason for e in extractions.values() if e.rejection)

    def fmt(c: Counter[str]) -> str:
        return ', '.join(f'{k} {n}' for k, n in c.most_common())

    return (
        f'{sum(kinds.values())} actionable ({fmt(kinds)}), {sum(rejected.values())} rejected ({fmt(rejected)}), '
        f'{sum(len(e.potential_rules) for e in extractions.values())} potential rules'
    )
