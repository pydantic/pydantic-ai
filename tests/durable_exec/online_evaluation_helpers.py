"""Helpers shared by the durable-execution `OnlineEvaluation` tests (DBOS, Prefect and Temporal)."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from pydantic_evals.online import SamplingContext


class Answer(BaseModel):
    text: str


evaluated: list[tuple[str, EvaluatorContext]] = []
"""The context each `RecordRun` evaluated, with its label. Tests clear it first."""


@dataclass
class RecordRun(Evaluator):
    label: str = 'record'

    def evaluate(self, ctx: EvaluatorContext) -> bool:
        evaluated.append((self.label, ctx))
        return True


def seen() -> list[tuple[str, str | None, object]]:
    """Each evaluation's label, run ID and output."""
    return [(label, ctx.name, ctx.output) for label, ctx in evaluated]


def failing_sample_rate(ctx: SamplingContext) -> bool:
    raise RuntimeError('sampling failed')
