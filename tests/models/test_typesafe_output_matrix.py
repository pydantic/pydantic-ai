"""What `TypeSafeModel` does with a *composed* output type, as one table.

`test_typesafe.py` pins a field shape at a time. This file pins the shapes those combine into — an
optional, a union, a discriminated union, a mapping, a rubric of numbers, an output type beside an
output function — because that is the surface a user meets first and the one the
[docs](../../docs/models/typesafe.md) describe.

**Every refusal below is what the model does today, not what it ought to do.** Several rows are
candidates to be made to work: `X | None` as an output type, a pick-one of something other than
strings, a union of models as a field. When one of those changes, its row moves from `REFUSED` to
`ACCEPTED`; a row that disappears is a user-facing behaviour that went unnoticed. The refusals that
are not about a composed shape — `str`, `NativeOutput`, `PromptedOutput`, a field of plain text —
stay in `test_typesafe.py` beside the rest of the field shapes.

Nothing here reaches the network. Almost every refusal is raised while the request is still being
built, so the transport these tests hand the model raises if it is ever called: the refusal and its
cost (nothing) are pinned together. The rows that do work are answered by a scripted transport
rather than a cassette, because what they pin is *how many requests a shape costs* — a fact about
two exchanges that no single recording holds, and one a cassette matcher would not notice changing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Annotated, Any, Literal

import httpx2
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, UseEnumMemberDocstrings
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from ..conftest import try_import
from .test_typesafe import answers, mock_model

with try_import() as imports_successful:
    from pydantic_ai.models.typesafe import TypeSafeModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='typesafe-sdk not installed'),
    pytest.mark.anyio,
]


Area = Literal['billing', 'shipping', 'security']


class Ticket(BaseModel):
    """Triage the ticket."""

    urgent: bool = Field(description='Is this urgent?')


class Escalation(BaseModel):
    """Hand the ticket to a human specialist."""

    security: bool = Field(description='Does this involve a security risk?')


class Cat(BaseModel):
    """A cat."""

    kind: Literal['cat'] = 'cat'
    indoor: bool = Field(description='Does it live indoors?')


class Dog(BaseModel):
    """A dog."""

    kind: Literal['dog'] = 'dog'
    large: bool = Field(description='Is it a large breed?')


class Codes(IntEnum):
    """Which area, as codes."""

    billing = 10
    shipping = 20
    security = 30


def probe(name: str, annotation: Any, **field: Any) -> Any:
    """An output type whose one field carries `annotation`, so a row is the annotation and its message."""
    namespace: dict[str, Any] = {'__annotations__': {name: annotation}, '__doc__': 'Triage the ticket.'}
    if field:
        namespace[name] = Field(**field)
    return type('Probe', (BaseModel,), namespace)


# The remedy every "not supported" message ends with, spelled out once: a rewording is one failure, not twenty.
SUPPORTED_FIELDS = (
    'Use `bool`, a `Literal` or `Enum` of two or more strings, a `float` bounded with `ge=0` and `le=1`, a `list` of '
    'a `Literal` or `Enum`, a rubric of whole numbers from 0 with a description per level in its schema, or a model '
    'of these.'
)


def unsupported(field: str, because: str = '') -> str:
    return f'Output field {field!r} is not supported by this model{because}. {SUPPORTED_FIELDS}'


def says_nothing(route: str) -> str:
    return (
        f'Jev weighs each route by what it is for, and {route!r} says nothing about itself. '
        'Give the output type a docstring that says what filling it does.'
    )


NOT_OPTIONAL = ': only a `Literal` or `Enum` of strings can be optional, since `None` is one more option to pick'
NOT_A_LIST = ': a list must be of two or more string options'
NOT_A_RUBRIC = ': a rubric must be the whole numbers from 0 upwards, in order, and there must be at least two of them'
NOT_STRINGS = ': its options are not two or more strings'


@dataclass(frozen=True)
class Refused:
    """An output type Jev will not take, and the whole message the user gets for it."""

    id: str
    output_type: Any
    error: str


REFUSED = [
    # `None` is an option on a pick-one field and nothing else: as a route, or beside anything but a
    # pick-one, the output type is refused.
    Refused('model | None', Ticket | None, says_nothing('final_result_NoneType')),
    Refused('union | None', Ticket | Escalation | None, says_nothing('final_result_NoneType')),
    Refused('pick-one | None', Area | None, says_nothing('final_result_Literal')),
    Refused('field: model | None', probe('inner', Ticket | None), unsupported('inner', NOT_OPTIONAL)),
    Refused(
        'field: list | None',
        probe('areas', list[Area] | None, description='Which?'),
        unsupported('areas', NOT_OPTIONAL),
    ),
    # A route is weighed by what it says about itself, and a `Literal` has nowhere to write that down.
    Refused('union with a pick-one', [Ticket, Area], says_nothing('final_result_Literal')),
    # A union of structured types is a route set; the same union as a *field* is not a question.
    Refused('field: union of models', probe('animal', Cat | Dog, description='Which animal?'), unsupported('animal')),
    Refused(
        'field: discriminated union',
        probe('animal', Annotated[Cat | Dog, Field(discriminator='kind')], description='Which animal?'),
        unsupported('animal'),
    ),
    # A pick-one is a pick between strings; numbers are read as a rubric's levels instead.
    Refused(
        'field: pick-one of ints',
        probe('status', Literal[200, 404, 500], description='Which?'),
        unsupported('status', NOT_A_RUBRIC),
    ),
    Refused(
        'field: IntEnum of codes', probe('area', Codes, description='Which area?'), unsupported('area', NOT_A_RUBRIC)
    ),
    Refused(
        'field: pick-one of mixed types',
        probe('which', Literal['a', 1], description='Which?'),
        unsupported('which', NOT_STRINGS),
    ),
    Refused(
        'field: pick-one of booleans',
        probe('which', Literal[True, False], description='Which?'),
        unsupported('which', NOT_STRINGS),
    ),
    Refused(
        'field: pick-one of one option',
        probe('area', Literal['billing'], description='Which area?'),
        unsupported('area'),
    ),
    # A mapping keyed by a pick-one is not a question either, as an output type or as a field.
    Refused('mapping of options', dict[Area, bool], unsupported('response')),
    Refused(
        'field: mapping of options',
        probe('applies', dict[Area, bool], description='Which apply?'),
        unsupported('applies'),
    ),
    Refused('mapping of text', dict[str, str], unsupported('response')),
    # A list is one yes/no per option, so its items have to be the options.
    Refused('list of models', list[Ticket], unsupported('response', NOT_A_LIST)),
    # A number is only a question when it is a probability or a rubric level.
    Refused('field: bounded int', probe('clarity', int, ge=0, le=4, description='How clear?'), unsupported('clarity')),
    Refused('field: percentage', probe('risk', float, ge=0, le=100, description='How risky?'), unsupported('risk')),
]


def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
    raise AssertionError('a refused output type must not reach a request')


@pytest.mark.parametrize('behind_a_model', [False, True], ids=['jev alone', 'with a model behind it'])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in REFUSED])
async def test_a_refused_output_type_costs_no_request(allow_model_requests: None, case: Refused, behind_a_model: bool):
    """Each refusal, its message, and the fact that it is decided before anything goes out.

    A `FallbackModel` makes no difference, which `test_fallback_does_not_skip_a_user_error` pins for one output
    type and this pins for every row: the `UserError` is raised while the request is being prepared and
    `fallback_on=(ModelAPIError,)` does not catch it, so an agent Jev cannot serve fails the same way whether or
    not there is a language model behind it, rather than quietly running on the next model every time.
    """
    jev = mock_model(unreachable)
    model = FallbackModel(jev, TestModel()) if behind_a_model else jev

    with pytest.raises(UserError) as exc_info:
        await Agent(model, output_type=case.output_type).run('anything')

    assert str(exc_info.value) == case.error


class OptionalArea(BaseModel):
    """Triage the ticket."""

    area: Area | None = Field(description='Which area, if any?')


class Clarity(UseEnumMemberDocstrings, IntEnum):
    """How clearly is the problem stated?"""

    none = 0
    """Leaves a reader who did not already know none the wiser."""
    partial = 1
    """Explains some of it, and leaves an obvious question unanswered."""
    full = 2
    """A reader who did not already know could act on it."""


class Graded(BaseModel):
    """Grade the writing."""

    clarity: Clarity


def escalate() -> str:
    """Escalate to a human because nobody on this tier can resolve it."""
    return 'escalated'


def summarise(ctx: RunContext[None], area: Area) -> str:
    """Summarise the ticket for the named area."""
    return f'summary for {area}'


def answer(question: dict[str, Any], picks: str | None) -> dict[str, object]:
    """The plainest answer of the kind a question asks for: yes, `picks` or the first option, or the middle level."""
    if question['type'] == 'noul':
        return {'type': 'noul', 'noul': 0.9}
    if question['type'] == 'choice':
        options = list(question['criteria'])
        chosen = picks if picks in options else options[0]
        return {
            'type': 'choice',
            'choice': chosen,
            'confidence': 0.9,
            'probabilities': {option: 0.9 if option == chosen else 0.1 for option in options},
        }
    levels = range(len(question['criteria']))
    return {
        'type': 'score',
        'score': float(len(question['criteria']) // 2),
        'confidence': 0.9,
        'legend': {},
        'probabilities': {str(level): 1 / len(question['criteria']) for level in levels},
    }


def scripted(picks: str | None) -> tuple[TypeSafeModel, list[dict[str, Any]]]:
    """A model that answers whatever it is asked, taking `picks` where it is on offer, and what it was asked."""
    sent: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        return answers(**{name: answer(question, picks) for name, question in body['questions'].items()})

    return mock_model(respond), sent


@dataclass(frozen=True)
class Accepted:
    """An output type Jev fills, what it answers, and what the shape costs in requests."""

    id: str
    output_type: Any
    output: Any
    requests: int = 1
    picks: str | None = None
    """The route Jev takes, where there is one to take: the first on offer unless a row says otherwise."""


ACCEPTED = [
    Accepted('one output type', Ticket, Ticket(urgent=True)),
    Accepted('a pick-one', Area, 'billing'),
    Accepted('a list of options', list[Area], ['billing', 'shipping', 'security']),
    Accepted('an optional pick-one field', OptionalArea, OptionalArea(area='billing')),
    Accepted('a rubric field', Graded, Graded(clarity=Clarity.partial)),
    # A union is a route set: one request picks the member, a second asks only that member's fields.
    Accepted(
        'a union of output types',
        [Ticket, Escalation],
        Escalation(security=True),
        requests=2,
        picks='final_result_Escalation',
    ),
    # With one output type and one output function the route question rides along with the fields, so the
    # pick and the answer arrive together.
    Accepted('an output type beside an output function', [Ticket, escalate], Ticket(urgent=True)),
    Accepted('an output function picked as the route', [Ticket, escalate], 'escalated', picks='final_result_escalate'),
    Accepted('an output function Jev can fill', [summarise], 'summary for billing'),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in ACCEPTED])
async def test_an_accepted_output_type_and_what_it_costs(allow_model_requests: None, case: Accepted):
    """Each shape Jev fills, and the requests it takes to fill it.

    The count is the part a user pays for and cannot see from the output: a single output type is filled in the
    same request that asks for it, while picking between routes costs a second one, reported as
    `provider_details['requests']` only when there was more than one.
    """
    model, sent = scripted(case.picks)
    result = await Agent(model, output_type=case.output_type).run('anything')

    assert result.output == case.output
    assert len(sent) == case.requests
    assert (result.response.provider_details or {}).get('requests') == (case.requests if case.requests > 1 else None)


# The two rows below are defects, not considered refusals: an output type Jev cannot take should raise a
# `UserError` naming the field, as every row above does. They are pinned as they stand so that fixing either
# is a visible change; neither is fixed here.


async def test_a_tuple_of_options_raises_a_key_error(allow_model_requests: None):
    """A `tuple` renders as `prefixItems` with no `items`, and the list branch reads `items` unguarded."""
    Pair = probe('two', tuple[Area, Area], description='Which two?')

    with pytest.raises(KeyError, match='items'):
        await Agent(mock_model(unreachable), output_type=Pair).run('anything')


async def test_a_model_that_contains_itself_runs_out_of_stack(allow_model_requests: None):
    """Resolving `$ref`s follows a model into itself forever when the self-reference has no way out.

    A self-reference through a `list` or an optional is refused on that field first, before the walk repeats.
    """

    class Thread(BaseModel):
        """Triage the thread."""

        urgent: bool = Field(description='Is this urgent?')
        parent: Thread

    with pytest.raises(RecursionError):
        await Agent(mock_model(unreachable), output_type=Thread).run('anything')
