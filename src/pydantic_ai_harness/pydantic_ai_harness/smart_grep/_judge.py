"""Relevance judgments from any Pydantic AI model.

Relevance is scored as the product of two independent judgments -- the snippet acts on the requested
*entity*, and it performs the requested *operation* -- so mentioning a concept is not enough; the code must
do the thing. On a decision model such as TypeSafe's Jev, a bounded `float` field is answered with the raw
probability of yes, so there is no confidence-margin math to undo. Any other model fills the same fields as
structured output.

Every snippet is judged in its own run, so neighbouring snippets never lend each other evidence.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from importlib.util import find_spec
from typing import cast

import anyio
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai_harness.smart_grep._chunks import Chunk

TYPESAFE_MODEL = 'typesafe:jev-latest'
"""The recommended judge, used by default when TypeSafe is installed and `TYPESAFE_API_KEY` is set."""

REQUEST_TIMEOUT_SECONDS = 30

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


def typesafe_available() -> bool:
    """Whether the recommended TypeSafe judge can run: its SDK is installed and `TYPESAFE_API_KEY` is set."""
    return bool(os.environ.get('TYPESAFE_API_KEY')) and find_spec('typesafe_sdk') is not None


def resolve_judge_model(model: JudgeModel | None, run_model: object) -> JudgeModel:
    """The configured judge; else TypeSafe's Jev when available; else the run's own model."""
    if model is not None:
        return model
    if typesafe_available():
        return TYPESAFE_MODEL
    if isinstance(run_model, Model):
        # `isinstance` narrows to `Model[Unknown]`; the bare `Model` restores its declared client default.
        return cast(Model, run_model)
    raise UserError(
        '`SmartFileSearch` could not pick a judge model: the run model is not a request-response `Model`. '
        'Pass `SmartFileSearch(model=...)`.'
    )


def _state(chunk: Chunk) -> str:
    """What the judge sees: where the snippet lives, then the snippet itself."""
    where = f'{chunk.path}:{chunk.line}-{chunk.end_line}'
    header = f'{where} ({chunk.symbol})' if chunk.symbol else where
    return f'{header}\n\n{chunk.text}'


def build_agent(model: JudgeModel, query: str) -> Agent[None, Relevance]:
    """The one-shot judge agent for `query`, named after the capability for tracing."""
    return Agent(
        model,
        name='smart_grep',
        output_type=Relevance,
        instructions=f'The text is a snippet of source code from a repository. A developer is searching the codebase for: {query}',
        model_settings={'timeout': REQUEST_TIMEOUT_SECONDS},
    )


async def judge(
    model: JudgeModel,
    query: str,
    chunks: Sequence[Chunk],
    *,
    concurrency: int,
) -> list[float]:
    """Score each chunk in `[0, 1]`, order preserved.

    Any failed judgment fails the whole search and cancels the rest: a partial scan reported as a complete
    one is worse than an error.
    """
    agent = build_agent(model, query)
    gate = anyio.Semaphore(concurrency)
    scores = [0.0] * len(chunks)
    errors: list[Exception] = []

    async def one(index: int, chunk: Chunk) -> None:
        async with gate:
            try:
                scores[index] = (await agent.run(_state(chunk))).output.score
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
