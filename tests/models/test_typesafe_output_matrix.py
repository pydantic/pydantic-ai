"""What `TypeSafeModel` does with a *composed* output type, as one table.

`test_typesafe.py` pins a field shape at a time. This file pins the shapes those combine into — an
optional, a union, a discriminated union, a mapping, a rubric of numbers, an output type beside an
output function — because that is the surface a user meets first and the one the
[docs](../../docs/models/typesafe.md) describe.

**Every refusal below is what the model does today, not what it ought to do.** Several rows are
candidates to be made to work: `X | None` as an output type, a union of models as a field. When one of those changes, its row moves from `REFUSED` to
`ACCEPTED`; a row that disappears is a user-facing behaviour that went unnoticed. The refusals that
are not about a composed shape — `str`, `NativeOutput`, `PromptedOutput`, a field of plain text —
stay in `test_typesafe.py` beside the rest of the field shapes.

Nothing here reaches the network. Every refusal below is raised while the request is still being
built, so the transport these tests hand the model raises if it is ever called: the refusal and its
cost (nothing) are pinned together. The rows that do work are answered by a scripted transport
rather than a cassette, because what they pin is *how many requests a shape costs* — a fact about
two exchanges that no single recording holds, and one a cassette matcher would not notice changing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Annotated, Any, Literal, Union

import httpx2
import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, UseEnumMemberDocstrings
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel

from .._inline_snapshot import snapshot
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


class Thread(BaseModel):
    """Triage the thread."""

    urgent: bool = Field(description='Is this urgent?')
    parent: Thread


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


class Statuses(UseEnumMemberDocstrings, IntEnum):
    """Which status the service returned."""

    ok = 200
    """The request worked."""
    missing = 404
    """Nothing is at that address."""


def probe(name: str, annotation: Any, **field: Any) -> Any:
    """An output type whose one field carries `annotation`, so a row is the annotation and its message."""
    namespace: dict[str, Any] = {'__annotations__': {name: annotation}, '__doc__': 'Triage the ticket.'}
    if field:
        namespace[name] = Field(**field)
    return type('Probe', (BaseModel,), namespace)


# The remedy every "not supported" message ends with, spelled out once: a rewording is one failure, not twenty.
SUPPORTED_FIELDS = (
    'Use `bool`, a `Literal` or `Enum` of two or more strings or whole numbers, a `float` bounded with `ge=0` and '
    '`le=1`, a `list` of a `Literal` or `Enum`, a rubric of whole numbers from 0 with a description per level in its '
    'schema, or a model of these.'
)


def unsupported(field: str, because: str = '') -> str:
    return f'Output field {field!r} is not supported by this model{because}. {SUPPORTED_FIELDS}'


def says_nothing(route: str, *, alone: bool = False) -> str:
    # With one output type left the agent's `instructions` can describe it, so the message offers that too.
    return (
        f'Jev weighs each route by what it is for, and {route!r} says nothing about itself. '
        'Give the output type a docstring that says what filling it does'
        + (', or the agent `instructions`.' if alone else '.')
    )


def contains_itself(field: str) -> str:
    return (
        f'Output field {field!r} is not supported by this model: a model that contains itself has no end '
        'to fill, and Jev asks a fixed set of questions. Give the field a type that does not contain itself.'
    )


NOT_OPTIONAL = ': only a pick-one of strings or whole numbers can be optional, since `None` is one more option to pick'
NOT_A_LIST = ': a list must be of two or more string options'


class OneArea(str, Enum):
    """The only area."""

    billing = 'billing'


class Areas(UseEnumMemberDocstrings, str, Enum):
    """Which area."""

    billing = 'billing'
    """Charges, refunds and invoices."""
    shipping = 'shipping'
    """Where an order is."""


EnumKeyed = probe('applies', dict[Areas, bool], description='Which apply?')
Stepped = probe('risk', float, ge=0, le=100, multiple_of=10, description='How risky?')
Capped = probe('applies', dict[Area, bool], max_length=2, description='Which apply?')
BoolLiteral = probe('which', Literal[True, False], description='Which?')
Percentage = probe('risk', float, ge=0, le=100, description='How risky?')
Applies = probe('applies', dict[Area, bool], description='Which apply?')


@dataclass(frozen=True)
class Refused:
    """An output type Jev will not take, and the whole message the user gets for it."""

    id: str
    output_type: Any
    error: str


REFUSED = [
    # `None` is a route, so what is left here is the route that cannot describe itself: a bare `Literal` has
    # no docstring, and `None` no longer counts towards the union that would have ruled out `instructions`.
    Refused('pick-one | None', Area | None, says_nothing('final_result_Literal', alone=True)),
    # As a *field*, `None` is still one more option on a pick-one and nothing else.
    Refused('field: model | None', probe('inner', Ticket | None), unsupported('inner', NOT_OPTIONAL)),
    Refused(
        'field: list | None',
        probe('areas', list[Area] | None, description='Which?'),
        unsupported('areas', NOT_OPTIONAL),
    ),
    # `None` beside `None` is a union with nothing to ask about, so there is no `X` for the extra option to
    # stand beside. It is refused as an unsupported field, not as an optional one.
    Refused(
        'field: None | None',
        probe(
            'nothing',
            Union[Annotated[None, Field(description='one')], Annotated[None, Field(description='the other')]],  # noqa: UP007
            description='Which?',
        ),
        unsupported('nothing'),
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
    Refused(
        'field: pick-one of one option',
        probe('area', Literal['billing'], description='Which area?'),
        unsupported('area'),
    ),
    Refused('mapping of text', dict[str, str], unsupported('response')),
    # A list is one yes/no per option, so its items have to be the options.
    Refused('list of models', list[Ticket], unsupported('response', NOT_A_LIST)),
    # A bound is the units a probability is asked in. A field that only accepts steps along it is a set of
    # levels instead, and every key of a mapping is answered, so a limit on how many there may be cannot hold.
    Refused('field: stepped number', Stepped, unsupported('risk')),
    Refused('field: mapping with a size limit', Capped, unsupported('applies')),
    Refused(
        'field: list with a size limit',
        probe('areas', list[Area], max_length=1, description='Which apply?'),
        unsupported('areas'),
    ),
    # `dict[str, Any]` says its values are unconstrained by writing `additionalProperties: true` rather than a
    # schema, which is not a thing to call `.get` on.
    Refused('field: mapping of anything', probe('blob', dict[str, Any], description='Anything?'), unsupported('blob')),
    # The values have to be a plain yes/no: anything narrower forbids an answer Jev is free to give.
    Refused(
        'field: mapping to a fixed value',
        probe('m', dict[Area, Literal[True]], description='Which?'),
        unsupported('m'),
    ),
    # And the keys have to be options: `dict[str, bool]` says nothing about what they are.
    Refused('field: mapping of free keys', probe('m', dict[str, bool], description='Which?'), unsupported('m')),
    # One option is not a set to fan out over, the same as a pick-one of one option. A single `Literal` key
    # reaches the schema as a `const`, which is not a set at all; a one-member `Enum` is a set of one.
    Refused(
        'field: mapping of one option',
        probe('m', dict[OneArea, bool], description='Which?'),
        unsupported('m', ': a mapping must be keyed by two or more options'),
    ),
    # A `tuple` is an array whose members are positional, which Pydantic renders as `prefixItems` and no
    # `items`, so there are no options to fan out over.
    Refused(
        'field: tuple of pick-ones',
        probe('two', tuple[Area, Area], description='Which two?'),
        unsupported('two', NOT_A_LIST),
    ),
    # A model that always contains itself is infinitely many questions; through a list or an optional it is
    # refused on that field first, by the rows above.
    Refused('field: model that contains itself', Thread, contains_itself('parent.parent')),
    # A number is only a question when it is a probability or a rubric level.
    Refused('field: bounded int', probe('clarity', int, ge=0, le=4, description='How clear?'), unsupported('clarity')),
]


def unreachable(request: httpx2.Request) -> httpx2.Response:  # pragma: no cover
    raise AssertionError('a refused output type must not reach a request')


# The refused rows that every other model takes. Being here is not a bug — Jev answers questions rather than
# writing values, and most of these are things only a model that writes can do. It is a list to decide against
# on purpose: `None` as a route was on it, and was worth closing. Anything joining it is worth the same look.
GAPS = [
    'field: None | None',
    'field: bounded int',
    'field: list with a size limit',
    'field: list | None',
    'field: mapping of anything',
    'field: mapping of free keys',
    'field: model | None',
    'field: pick-one of one option',
    'field: stepped number',
    'field: tuple of pick-ones',
    'field: union of models',
    'list of models',
    'mapping of text',
    'pick-one | None',
    'union with a pick-one',
]


async def test_which_refusals_are_gaps_with_the_rest_of_the_library(allow_model_requests: None):
    """Which refusals above are Jev's own limits, and which are output types every other model takes.

    The table says what Jev does. It cannot say whether a refusal is a deliberate limit or a hole, and the two
    look identical in it — which is why `None` as a route sat here looking like the first kind. Running each
    refused type against a model with no such limits separates them: a row that answers there is a gap with
    the rest of the library, and a row that does not is Jev's own.

    A row moving between the two lists is the point of this test. Adding a refusal the rest of the library
    accepts is a decision worth making on purpose rather than noticing later.
    """
    gaps: list[str] = []
    for case in REFUSED:
        try:
            await Agent(TestModel(), output_type=case.output_type).run('anything')
        except Exception:
            continue
        gaps.append(case.id)

    assert sorted(gaps) == GAPS


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


class DescribedNoneArea(BaseModel):
    """Triage the ticket."""

    area: Area | Annotated[None, Field(description='Nothing to route.')] = Field(description='Which area, if any?')


class Refunded(UseEnumMemberDocstrings, Enum):
    """Whether the money went back."""

    yes = True
    """Money was returned to the customer."""

    no = False
    """No refund was issued."""


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


# `True` and `False` are a yes/no's own two options, so either spelling of them is one.
TrueFalse = probe('which', Literal[True, False], description='Which?')


class Settled(BaseModel):
    """Review the transcript."""

    refunded: Refunded = Field(description='Was a refund issued?')


@dataclass(frozen=True)
class Accepted:
    """An output type Jev fills, what it answers, and what the shape costs in requests."""

    id: str
    output_type: Any
    output: Any
    requests: int = 1
    picks: str | None = None
    """The route Jev takes, where there is one to take: the first on offer unless a row says otherwise."""
    questions: dict[str, Any] | None = None
    """What the last request asked, where the shape of the question is the point of the row."""


class Status(BaseModel):
    """Report what the service returned."""

    status: Literal[200, 404, 500] = Field(description='Which status did the service return?')


class Checked(BaseModel):
    """Check the service."""

    check: Status


CodedArea = probe('area', Codes, description='Which area?')
Mixed = probe('which', Literal['a', 1], description='Which?')
Described = probe('status', Statuses, description='Which status?')
OptionalStatus = probe('status', Literal[200, 404] | None, description='Which status, if any?')
DefaultedStatus = probe('status', Literal[200, 404] | None, default=200, description='Which status, if any?')
DescribedNoneStatus = probe(
    'status',
    Literal[200, 404] | Annotated[None, Field(description='The service did not answer.')],
    description='Which status, if any?',
)


ACCEPTED = [
    Accepted('one output type', Ticket, Ticket(urgent=True)),
    Accepted('a pick-one', Area, 'billing'),
    Accepted('a list of options', list[Area], ['billing', 'shipping', 'security']),
    Accepted('an optional pick-one field', OptionalArea, OptionalArea(area='billing')),
    Accepted('a rubric field', Graded, Graded(clarity=Clarity.partial)),
    Accepted('field: yes/no from two options', TrueFalse, TrueFalse(which=True)),
    Accepted('field: yes/no with each answer described', Settled, Settled(refunded=Refunded.yes)),
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
    # `None` is a route like any other: one more option on the route question, described as "None of these.",
    # taken on the pick alone because there is nothing to fill.
    Accepted('model | None, declined', Ticket | None, None, picks='final_result_None'),
    Accepted('model | None, filled', Ticket | None, Ticket(urgent=True), picks='final_result_Ticket'),
    Accepted('union | None, declined', Ticket | Escalation | None, None, picks='final_result_None'),
    Accepted(
        'union | None, filled',
        Ticket | Escalation | None,
        Escalation(security=True),
        requests=2,
        picks='final_result_Escalation',
    ),
    Accepted('output function | None', [escalate, None], None, picks='final_result_None'),
    # `Literal[True, False]` spells out what a `bool` already is, so it asks the same yes/no.
    Accepted('field: pick-one of booleans', BoolLiteral, BoolLiteral(which=True)),
    # A bounded number asks for a probability; the bound is the units it comes back in.
    Accepted('field: percentage', Percentage, Percentage(risk=90.0)),
    # A mapping of options to yes/no fans out like a list of them, keeping every answer rather than the yeses.
    Accepted(
        'field: mapping of options', Applies, Applies(applies={'billing': True, 'shipping': True, 'security': True})
    ),
    Accepted('mapping of options', dict[Area, bool], {'billing': True, 'shipping': True, 'security': True}),
    # An `Enum` key reaches the schema as a `$ref` under `propertyNames`, which the walk resolves like any other.
    Accepted(
        'field: mapping keyed by an Enum', EnumKeyed, EnumKeyed(applies={Areas.billing: True, Areas.shipping: True})
    ),
    # Writing what `None` means puts a `description` beside its `{'type': 'null'}`, which does not stop it
    # being `None`: the route is still taken on the pick alone, and the field is still one more option.
    Accepted(
        'model | described None, declined',
        [Ticket, Annotated[None, Field(description='Nothing needs doing.')]],
        None,
        picks='final_result_None',
    ),
    Accepted('a described `None` option', DescribedNoneArea, DescribedNoneArea(area='billing')),
    # Whole numbers that are not a rubric -- not 0 upwards, or with nothing said about each -- are labels, so
    # they are a pick-one of their digits, and the answer is the number itself.
    Accepted(
        'field: pick-one of ints',
        Status,
        Status(status=200),
        questions=snapshot(
            {
                'status': {
                    'type': 'choice',
                    'criteria': {'200': None, '404': None, '500': None},
                    'instructions': {
                        'field': 'status',
                        'question': 'Which status did the service return?',
                        'goal': 'Report what the service returned.',
                    },
                }
            }
        ),
    ),
    Accepted(
        'field: IntEnum of codes',
        CodedArea,
        CodedArea(area=Codes.billing),
        questions=snapshot(
            {
                'area': {
                    'type': 'choice',
                    'criteria': {'10': None, '20': None, '30': None},
                    'instructions': {'field': 'area', 'question': 'Which area?', 'goal': 'Triage the ticket.'},
                }
            }
        ),
    ),
    Accepted(
        'field: pick-one of mixed types',
        Mixed,
        Mixed(which=1),
        picks='1',
        questions=snapshot(
            {
                'which': {
                    'type': 'choice',
                    'criteria': {'a': None, '1': None},
                    'instructions': {'field': 'which', 'question': 'Which?', 'goal': 'Triage the ticket.'},
                }
            }
        ),
    ),
    Accepted(
        'field: described IntEnum of codes',
        Described,
        Described(status=Statuses.ok),
        questions=snapshot(
            {
                'status': {
                    'type': 'choice',
                    'criteria': {'200': 'The request worked.', '404': 'Nothing is at that address.'},
                    'instructions': {'field': 'status', 'question': 'Which status?', 'goal': 'Triage the ticket.'},
                }
            }
        ),
    ),
    Accepted(
        'field: optional pick-one of ints',
        OptionalStatus,
        OptionalStatus(status=None),
        picks='none',
        questions=snapshot(
            {
                'status': {
                    'type': 'choice',
                    'criteria': {'200': None, '404': None, 'none': 'None of these.'},
                    'instructions': {
                        'field': 'status',
                        'question': 'Which status, if any?',
                        'goal': 'Triage the ticket.',
                    },
                }
            }
        ),
    ),
    # "None of these" on a field with a default leaves it to the default, a number as much as a string.
    Accepted(
        'field: optional pick-one of ints with a default', DefaultedStatus, DefaultedStatus(status=200), picks='none'
    ),
    # What the user wrote about `None` describes the extra option on a pick-one of numbers as on one of strings.
    Accepted(
        'field: optional pick-one of ints with a described None',
        DescribedNoneStatus,
        DescribedNoneStatus(status=None),
        picks='none',
        questions=snapshot(
            {
                'status': {
                    'type': 'choice',
                    'criteria': {'200': None, '404': None, 'none': 'The service did not answer.'},
                    'instructions': {
                        'field': 'status',
                        'question': 'Which status, if any?',
                        'goal': 'Triage the ticket.',
                    },
                }
            }
        ),
    ),
    Accepted(
        'field: nested pick-one of ints',
        Checked,
        Checked(check=Status(status=200)),
        questions=snapshot(
            {
                'check.status': {
                    'type': 'choice',
                    'criteria': {'200': None, '404': None, '500': None},
                    'instructions': {
                        'field': 'check.status',
                        'question': 'Which status did the service return?',
                        'goal': 'Check the service.',
                    },
                }
            }
        ),
    ),
    Accepted(
        'a union member with a pick-one of ints',
        [Ticket, Status],
        Status(status=200),
        requests=2,
        picks='final_result_Status',
        questions=snapshot(
            {
                'status': {
                    'type': 'choice',
                    'criteria': {'200': None, '404': None, '500': None},
                    'instructions': {
                        'field': 'status',
                        'question': 'Which status did the service return?',
                        'chosen': 'Status',
                        'goal': 'Report what the service returned.',
                    },
                }
            }
        ),
    ),
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
    if case.questions is not None:
        assert sent[-1]['questions'] == case.questions
    assert len(sent) == case.requests
    assert (result.response.provider_details or {}).get('requests') == (case.requests if case.requests > 1 else None)
