"""Stage 1: per session, turn each typed prompt into a reusable intent (cached by span id)."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

from pydantic_ai import Agent

from .llm_cache import USAGE
from .models import Facet, UserPrompt

FACET_INSTRUCTIONS = """\
You read the prompts one developer typed into their coding agent during one session, in order.
For each prompt, extract the reusable intent behind it: what would this request look like if any developer
on any repository typed it? Strip names, paths, issue numbers and task-specific detail.

Set `intent` to null for prompts that carry no reusable intent: answers to the agent's questions ("yes",
"option 2"), pure task statements with nothing general about how the work should be done, or chit-chat.
Pay special attention to prompts that tell the agent HOW to work or WHAT TO DO NEXT that the agent could
have done unprompted (e.g. "now open a PR and keep fixing CI until it's green", "run the tests before you
commit", "don't add comments"). Those are what we are looking for.

Separately, list in `preferences` every standing preference about how the agent should behave in general that the
prompt states, even in passing inside a one-off task: language or locale ("use British English"), tone, tooling
choices ("use uv, not pip"), habits ("always run the tests first"). Extract them even when the main intent is a
throwaway task or null; keep the main intent as it is. When the whole prompt is the preference, it is the `intent`
and `preferences` stays empty.
"""


class _SessionFacets(BaseModel):
    facets: list[Facet]


_cache_adapter = TypeAdapter(dict[str, Facet])


class FacetCache:
    def __init__(self, path: Path):
        self.path = path
        self.facets: dict[str, Facet] = _cache_adapter.validate_json(path.read_bytes()) if path.exists() else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(_cache_adapter.dump_json(self.facets, indent=2))


async def extract_facets(
    prompts: list[UserPrompt], *, model: str, cache: FacetCache, concurrency: int = 16
) -> dict[str, Facet]:
    """Return a facet per prompt span id, only calling the model for sessions with uncached prompts."""
    agent = Agent(model, output_type=_SessionFacets, instructions=FACET_INSTRUCTIONS, name='fleet_miner_facets')
    by_session: dict[str, list[UserPrompt]] = defaultdict(list)
    for prompt in prompts:
        by_session[prompt.session_id or prompt.trace_id].append(prompt)

    semaphore = asyncio.Semaphore(concurrency)

    async def one(session_prompts: list[UserPrompt]) -> None:
        # Incremental: only prompts never faceted are sent, each with the session's previous prompt cut to 300
        # characters as context, never the whole session again (that would grow quadratically with its length).
        session_prompts = sorted(session_prompts, key=lambda p: p.timestamp)
        payload = [
            {'span_id': p.span_id, 'previous_prompt': prev.text[:300] if prev else None, 'prompt': p.text[:2000]}
            for prev, p in zip([None, *session_prompts[:-1]], session_prompts)
            if p.span_id not in cache.facets
        ]
        if not payload:
            return
        async with semaphore:
            result = await agent.run(
                'New prompts from one session, in order; `previous_prompt` is only context '
                '(extract a facet for every span_id):\n' + json.dumps(payload, indent=2)
            )
        USAGE.add(result)
        todo = [p for p in session_prompts if p.span_id not in cache.facets]
        wanted = {p.span_id for p in todo}
        for facet in result.output.facets:
            if facet.span_id in wanted:
                cache.facets[facet.span_id] = facet
        cache.save()  # per session, so an interrupted run keeps what it paid for

    await asyncio.gather(*(one(ps) for ps in by_session.values()))
    cache.save()
    return {p.span_id: cache.facets[p.span_id] for p in prompts if p.span_id in cache.facets}
