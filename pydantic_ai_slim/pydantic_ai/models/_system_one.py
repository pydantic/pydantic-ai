"""Response validation shared by backends that speak the System One question/answer format."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Annotated, assert_never

import httpx2
from pydantic import Field, TypeAdapter

from ..exceptions import UnexpectedModelBehavior
from ..usage import RequestUsage
from .decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionQuestion,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    _probability_bounds,  # pyright: ignore[reportPrivateUsage]
    _score_fits,  # pyright: ignore[reportPrivateUsage]
)


@dataclass(kw_only=True)
class Usage:
    input_tokens: Annotated[int, Field(ge=0)] = 0
    output_tokens: Annotated[int, Field(ge=0)] = 0


@dataclass(kw_only=True)
class Response:
    model: str
    answers: dict[str, Annotated[DecisionAnswer, Field(discriminator='type')]]
    usage: Usage


response_adapter = TypeAdapter(Response)
questions_adapter = TypeAdapter(dict[str, Annotated[DecisionQuestion, Field(discriminator='type')]])


def validate_response(
    parsed: Response,
    questions: Mapping[str, DecisionQuestion],
    *,
    api_name: str,
    response_text: str,
    provider_response_id: str | None = None,
) -> DecisionResponse:
    if parsed.answers.keys() != questions.keys():
        raise UnexpectedModelBehavior(
            f'Invalid response from the {api_name}: answer names do not match the questions', response_text
        )
    for name, question in questions.items():
        answer = parsed.answers[name]
        if isinstance(question, NoulQuestion):
            valid = isinstance(answer, NoulAnswer) and isfinite(answer.noul) and 0 <= answer.noul <= 1
        elif isinstance(question, ChoiceQuestion):
            valid = (
                isinstance(answer, ChoiceAnswer)
                and answer.choice in question.criteria
                and answer.probabilities.keys() == question.criteria.keys()
            )
        elif isinstance(question, ScoreQuestion):
            levels = set(range(len(question.criteria)))
            valid = (
                isinstance(answer, ScoreAnswer)
                and answer.probabilities.keys() == levels
                and (not answer.legend or answer.legend.keys() == levels)
                and isfinite(answer.score)
                and 0 <= answer.score <= len(question.criteria) - 1
            )
        else:
            assert_never(question)
        if isinstance(answer, (ChoiceAnswer, ScoreAnswer)):
            probabilities = answer.probabilities.values()
            # Jev displays probabilities to two decimal places, so each may differ by half a unit.
            valid = (
                valid
                and isfinite(answer.confidence)
                and 0 <= answer.confidence <= 1
                and all(isfinite(p) and 0 <= p <= 1 for p in probabilities)
                and abs(sum(probabilities) - 1) <= 1e-6 + len(answer.probabilities) * 0.005
            )
            if valid and isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
                # A displayed score and its probabilities may each be rounded. Check whether any distribution
                # within their rounding intervals could produce that score, using at least Jev's two decimals.
                probabilities_by_level: list[float] = [
                    answer.probabilities[level] for level in range(len(question.criteria))
                ]
                lower, upper = _probability_bounds(probabilities_by_level)
                valid = _score_fits(lower, upper, answer.score)
        if not valid:
            raise UnexpectedModelBehavior(
                f'Invalid response from the {api_name}: answer {name!r} does not match its question: {answer!r}',
                response_text,
            )
    return DecisionResponse(
        answers=parsed.answers,
        model_name=parsed.model,
        usage=RequestUsage(input_tokens=parsed.usage.input_tokens, output_tokens=parsed.usage.output_tokens),
        provider_response_id=provider_response_id,
    )
