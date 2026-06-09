"""Candidate NEW built-in evaluators for pydantic_evals.

These are the matchers a beginner reaches for that have no existing built-in
today. They are written as ordinary `Evaluator` subclasses -- exactly the shape
of the current built-ins (`Contains`, `MaxDuration`, ...) -- so they are real,
reusable, serializable primitives usable in plain pydantic_evals, NOT logic
hidden inside the ergonomic facade.

The facade (easy_evals.py) only *instantiates* these; it contains no evaluation
logic of its own.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext


def _text(output: object) -> str:
    return output if isinstance(output, str) else str(output)


@dataclass(repr=False)
class NotContains(Evaluator[object, object, object]):
    """Pass if none of the given substrings appear in the output."""

    value: str | Sequence[str]
    case_sensitive: bool = False
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        needles = [self.value] if isinstance(self.value, str) else list(self.value)
        text = _text(ctx.output)
        hay = text if self.case_sensitive else text.lower()
        found = [n for n in needles if (n if self.case_sensitive else n.lower()) in hay]
        return EvaluationReason(value=not found, reason=f'found banned: {found}' if found else None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or 'not_contains'


@dataclass(repr=False)
class OneOf(Evaluator[object, object, object]):
    """Pass if the output is (or contains) one of the allowed options. Great for classification."""

    options: Sequence[str]
    case_sensitive: bool = False
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        text = _text(ctx.output)
        hay = text if self.case_sensitive else text.lower()
        match = next((o for o in self.options if (o if self.case_sensitive else o.lower()) in hay), None)
        return EvaluationReason(value=match is not None, reason=None if match else f'expected one of {list(self.options)}')

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or 'one_of'


@dataclass(repr=False)
class Matches(Evaluator[object, object, object]):
    """Pass if the output matches a regular expression."""

    pattern: str
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        ok = re.search(self.pattern, _text(ctx.output)) is not None
        return EvaluationReason(value=ok, reason=None if ok else f'did not match /{self.pattern}/')

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or 'matches'


@dataclass(repr=False)
class MaxLength(Evaluator[object, object, object]):
    """Pass if the output is within a length budget (words or characters)."""

    words: int | None = None
    chars: int | None = None
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        text = _text(ctx.output)
        if self.words is not None and len(text.split()) > self.words:
            return EvaluationReason(value=False, reason=f'{len(text.split())} words (max {self.words})')
        if self.chars is not None and len(text) > self.chars:
            return EvaluationReason(value=False, reason=f'{len(text)} chars (max {self.chars})')
        return EvaluationReason(value=True)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or 'max_length'


@dataclass(repr=False)
class HasFields(Evaluator[object, object, object]):
    """Pass if the (structured) output has the given field values. Partial match.

    Works with Pydantic models, dataclasses, or dicts -- the natural matcher for
    agents with a structured `output_type`.
    """

    fields: dict[str, object]
    evaluation_name: str | None = field(default=None)

    def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        output = ctx.output
        mismatches: list[str] = []
        for key, expected in self.fields.items():
            actual = _field_value(output, key)
            if actual != expected:
                mismatches.append(f'{key}={actual!r} (expected {expected!r})')
        return EvaluationReason(value=not mismatches, reason='; '.join(mismatches) or None)

    def get_default_evaluation_name(self) -> str:
        return self.evaluation_name or 'has_fields'


@runtime_checkable
class _SupportsGet(Protocol):
    def get(self, key: str, /) -> object: ...


def _field_value(output: object, key: str) -> object:
    """Read a field from a mapping, Pydantic model, or dataclass uniformly."""
    if isinstance(output, _SupportsGet):
        return output.get(key)
    return getattr(output, key, None)
