"""The question schema the model fills in, the payloads an answerer sees, and the `Answerer` protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_QUESTIONS = 10
"""Most questions one `ask_user_question` call may carry."""

MIN_OPTIONS = 2
MAX_OPTIONS = 6
MAX_HEADER_LENGTH = 25
MAX_LABEL_LENGTH = 50
MAX_DESCRIPTION_LENGTH = 200
MAX_QUESTION_LENGTH = 500


def _printable(value: str, *, newlines: bool) -> str:
    """Reject control characters: these strings are drawn on terminals, and an escape sequence is an attack."""
    if not all(char.isprintable() or (newlines and char == '\n') for char in value):
        raise ValueError('must not contain control characters')
    return value


_SCHEMA_CONFIG = ConfigDict(str_strip_whitespace=True, extra='forbid', frozen=True)
"""Strict on the way in (a field the schema lacks is a retry, not a silent drop) and immutable after.

The request is validated once and the response is checked against it later, so nothing between,
an event listener or the answerer, may change what was asked.
"""


class QuestionOption(BaseModel):
    """One choice the user can pick."""

    model_config = _SCHEMA_CONFIG

    label: str = Field(min_length=1, max_length=MAX_LABEL_LENGTH, description='Short option name, one to five words.')
    description: str | None = Field(
        default=None, max_length=MAX_DESCRIPTION_LENGTH, description='What choosing this option means.'
    )

    @field_validator('label')
    @classmethod
    def _label_is_one_line(cls, value: str) -> str:
        return _printable(value, newlines=False)

    @field_validator('description')
    @classmethod
    def _description_is_printable(cls, value: str | None) -> str | None:
        return None if value is None else _printable(value, newlines=True)


class Question(BaseModel):
    """One multiple-choice question."""

    model_config = _SCHEMA_CONFIG

    header: str = Field(
        min_length=1,
        max_length=MAX_HEADER_LENGTH,
        description='Short label naming the question, unique within the call; the answer is keyed by it.',
    )
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH, description='The full question text.')
    options: tuple[QuestionOption, ...] = Field(
        min_length=MIN_OPTIONS, max_length=MAX_OPTIONS, description='The choices to offer.'
    )
    multi_select: bool = Field(default=False, description='Whether the user may pick more than one option.')

    @field_validator('header')
    @classmethod
    def _header_is_one_line(cls, value: str) -> str:
        return _printable(value, newlines=False)

    @field_validator('question')
    @classmethod
    def _question_is_printable(cls, value: str) -> str:
        return _printable(value, newlines=True)

    @model_validator(mode='after')
    def _unique_labels(self) -> Question:
        _require_unique([option.label for option in self.options], what='option labels')
        return self


def _require_unique(values: list[str], *, what: str) -> None:
    folded = [value.casefold() for value in values]
    if len(folded) != len(set(folded)):
        raise ValueError(f'{what} must be unique')


def require_unique_headers(questions: list[Question]) -> list[Question]:
    """Reject a call whose questions share a header, since answers are keyed by it."""
    _require_unique([question.header for question in questions], what='question headers')
    return questions


@dataclass(frozen=True, kw_only=True)
class AskUserRequest:
    """One `ask_user_question` call, handed to the `Answerer` and carried by `AskUserRequestedEvent`."""

    questions: tuple[Question, ...]
    id: str = field(default_factory=lambda: uuid4().hex)
    """Distinguishes concurrent or repeated calls; a UI matches its reply to it."""


@dataclass(frozen=True, kw_only=True)
class AskUserAnswer:
    """The labels the user picked for one question."""

    header: str
    selected: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class AskUserResponse:
    """What the `Answerer` returns: one answer per question, or `cancelled` when the user declined."""

    answers: tuple[AskUserAnswer, ...] = ()
    cancelled: bool = False


class Answerer(Protocol):
    """Whatever puts the questions in front of a person: a terminal menu, a web form, a scripted test.

    A plain `async def` with this signature satisfies it. Return a cancelled response when the user
    declines; raise only for failures that should fail the run.
    """

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        """Put `request` to the user and return what they picked."""
        ...  # pragma: no cover


def check_response(request: AskUserRequest, response: AskUserResponse) -> None:
    """Raise `ValueError` when a response does not fit its request.

    A cancelled response carries no answers. A completed one answers every question with distinct
    labels that question offered, one of them unless `multi_select`.
    """
    if response.cancelled:
        if response.answers:
            raise ValueError('a cancelled response carries no answers')
        return
    questions = {question.header: question for question in request.questions}
    for answer in response.answers:
        question = questions.pop(answer.header, None)
        if question is None:
            raise ValueError(f'answer for unknown or repeated question {answer.header!r}')
        labels = {option.label for option in question.options}
        if unknown := [label for label in answer.selected if label not in labels]:
            raise ValueError(f'answer for {answer.header!r} picked options it does not offer: {unknown}')
        if not answer.selected:
            raise ValueError(f'answer for {answer.header!r} picked nothing')
        if len(set(answer.selected)) != len(answer.selected):
            raise ValueError(f'answer for {answer.header!r} picked the same option twice')
        if len(answer.selected) > 1 and not question.multi_select:
            raise ValueError(f'answer for {answer.header!r} picked several options but it is single-select')
    if questions:
        raise ValueError(f'unanswered questions: {sorted(questions)}')
