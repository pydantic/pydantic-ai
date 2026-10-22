"""Relevance judgments from any Pydantic AI model.

Relevance is scored as the product of two independent judgments -- the snippet acts on the requested
*entity*, and it performs the requested *operation* -- so mentioning a concept is not enough; the code must
do the thing. On a decision model, such as one served over the System One API or TypeSafe's Jev, a bounded
`float` field is answered with the raw probability of yes, so there is no confidence-margin math to undo. Any
other model fills the same fields as structured output.

Every snippet is judged in its own run, so neighbouring snippets never lend each other evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import anyio
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness._judge import claim_request
from pydantic_ai_harness.smart_search._chunks import Chunk

REQUEST_TIMEOUT_SECONDS = 30
"""The request timeout of a judge given by name; a model instance keeps its own `settings`."""

JudgeModel = Model | KnownModelName | str
"""A judge model as `Agent` accepts one: an instance or a `provider:model` name."""


class Relevance(BaseModel):
    """Decide whether this source code is what a developer searching the codebase wants to find."""

    entity: float = Field(
        ge=0,
        le=1,
        description='Does this code act on the thing the search is about (the data, object, resource or concept it names)?',
    )
    operation: float = Field(
        ge=0,
        le=1,
        description=(
            'Does this code itself perform the action or behaviour the search describes, rather than only '
            'mentioning it, importing it, or calling something with a similar name?'
        ),
    )

    @property
    def score(self) -> float:
        """How relevant the snippet is, in `[0, 1]`: both judgments have to hold."""
        return self.entity * self.operation


def _state(chunk: Chunk) -> str:
    """What the judge sees: where the snippet lives, then the snippet itself."""
    where = f'{chunk.path}:{chunk.line}-{chunk.end_line}'
    header = f'{where} ({chunk.symbol})' if chunk.symbol else where
    return f'{header}\n\n{chunk.text}'


def build_agent(model: JudgeModel, query: str) -> Agent[None, Relevance]:
    """The one-shot judge agent for `query`, named after the capability."""
    agent = Agent(
        model,
        name='smart_search',
        output_type=Relevance,
        instructions=f'The text is a snippet of source code from a repository. A developer is searching the codebase for: {query}',
        # Agent settings override the model's own, so only bound a judge whose settings the user could not set.
        model_settings={'timeout': REQUEST_TIMEOUT_SECONDS} if isinstance(model, str) else None,
    )
    # The judge's prompt carries source snippets, so it must not inherit global instrumentation that could
    # ignore the host agent's content or opt-out settings.
    agent.instrument = False
    return agent


def has_request_room(ctx: RunContext[Any]) -> bool:
    """Whether the run's `request_limit` leaves room for one more judgment and still one for the model's next turn."""
    limits = ctx.usage_limits
    return limits is None or limits.request_limit is None or ctx.usage.requests + 1 < limits.request_limit


async def judge(
    ctx: RunContext[Any],
    model: JudgeModel,
    query: str,
    chunks: Sequence[Chunk],
    *,
    concurrency: int,
) -> list[float | None]:
    """Score each chunk in `[0, 1]`, order preserved, or `None` for a chunk the request budget left unjudged.

    Each judgment is one request on the run's usage, checked against its limits. It is claimed before the
    judge is awaited, so judgments running in parallel, and sibling searches, share the budget: once the
    run's `request_limit` leaves no room for another judgment besides the model's next turn, the remaining
    chunks (the lowest ranked, as they are judged in order) are left unjudged.

    Any failed judgment fails the whole search and cancels the rest: a partial scan reported as a complete
    one is worse than an error.
    """
    agent = build_agent(model, query)
    # The claim below already holds each judgment's request slot, checked by `has_request_room`, so the
    # nested run checks only the token limits; with no limits at all it would apply the default ones.
    limits = replace(ctx.usage_limits, request_limit=None) if ctx.usage_limits else UsageLimits(request_limit=None)
    gate = anyio.Semaphore(concurrency)
    scores: list[float | None] = [None] * len(chunks)
    errors: list[Exception] = []

    async def one(index: int, chunk: Chunk) -> None:
        async with gate:
            if not has_request_room(ctx):
                return
            try:
                claim_request(ctx)
                try:
                    result = await agent.run(_state(chunk), usage=ctx.usage, usage_limits=limits)
                finally:
                    ctx.usage.requests -= 1  # the nested run counted its own request
                scores[index] = result.output.score
            except Exception as error:
                # Raised below, outside the task group, so callers see the judge's own error rather
                # than an `ExceptionGroup`.
                errors.append(error)
                group.cancel_scope.cancel()

    async with anyio.create_task_group() as group:
        for index, chunk in enumerate(chunks):
            group.start_soon(one, index, chunk)
    if errors:
        raise errors[0]
    return scores
