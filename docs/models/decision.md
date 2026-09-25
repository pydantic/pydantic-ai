---
description: "Use a decision model with Pydantic AI for classification, routing and yes-or-no judgements: typed answers with a confidence, cheaper and faster than an LLM."
---

# Decision models

A decision model answers typed questions about a text rather than writing text: is this true or not, which of these labels fits, where does this fall on a rubric. Each answer comes with a probability, or a distribution over the options, so you know how sure it is. An agent runs on one like on any other model, and uses it for the two things a decision can drive: filling a structured [output](../output.md), and picking which *route* to take — which output type, [output function](../output.md#output-functions) or [tool](../tools.md) the text calls for.

[`DecisionModel`][pydantic_ai.models.decision.DecisionModel] maps an agent run onto those questions. Each field of the `output_type` becomes one question, the prompt is the text, and the answers come back as the output: a `bool` is a yes or no, a `Literal` or `Enum` is one label out of several, a list of labels is a yes or no per label, a bounded `float` is the probability of yes itself, and a set of described levels is a score against a rubric. A Pydantic model with several fields extracts several values in one request. Output types, output functions and tools are [routes](#routes-which-thing-to-do): with more than one on offer, the model picks the one the text calls for, and fills that route's fields or arguments the same way. Change the model name and the same agent runs on a language model, so you can compare the two.

A decision model can also work together with a language model. When it picks a route it cannot continue down — a tool with an argument it cannot fill, such as a free-form `str`, or one of several output types with such a field — it escalates: behind a [`FallbackModel`](overview.md#fallback-model), a language model takes that whole step, with the same tools and output types to choose from. The same fallback can take the steps the decision model [was unsure about](#falling-back-on-low-confidence). The cheap model answers what it can, and the expensive one only runs when it is needed.

The decision model Pydantic AI supports out of the box is TypeSafe's Jev, through the [`TypeSafeModel`](typesafe.md) model class, and the examples on this page use it. This page covers what `DecisionModel` does for any backend; the [TypeSafe page](typesafe.md) covers setup, Jev's own limits and what it answers badly. To use another backend, [implement `decide`](#implementing-a-decision-model).

Reach for one when the answer is a classification — a verdict, a route, a label, a score against a rubric — and you want it cheaper and faster than a language model gives it, with a confidence you can act on. Keep a language model for anything that has to be written: a `str` field, a reply, a summary.

## Asking a question

The simplest decision model agent asks one question. The `output_type` is the kind of answer, the agent's `instructions` are the question, and the run's prompt is the text the question is about:

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-latest', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

A `Literal` output picks one label instead: `output_type=Literal['billing', 'bug', 'account']` with `instructions='Which team owns this ticket?'` answers with one of the three.

That split is the habit to learn first. The instructions say what to do, and the run's input is what it is done to. Keeping the two apart is good practice on any model, and a decision model depends on it, because it takes them as two separate things: the material to judge, and the questions to ask about it. The material is the *state*: the content and the facts that support it. The questions are the judgements to make about that material. So **the prompt is only what is being judged, and the question belongs on the agent** — in its `instructions` here, and on its output type [below](#asking-with-an-output-type).

A question written into the prompt is not lifted out: it is text to be judged, and it is judged. Almost nothing catches that for you. A bare `bool` or bounded `float` output with no instructions carries no question at all, so it is a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent — but a `bool` *field* of an output type is not refused, because its name is enough to ask about. So do not count on an error to catch a question in the wrong place.

## Asking with an output type

Most agents ask more than one thing. Give the agent an output type and each field is a question of its own, all of them sent in a single request and answered together. The type is also where the wording goes:

- The class docstring states what the type is for, as a goal: "Triage a support ticket." It is sent as the goal on every question about the type.
- A field's description is that field's question. Write it with `Field(description=...)`, or, as here, as a docstring under the field: setting `model_config = ConfigDict(use_attribute_docstrings=True)` on the model makes Pydantic read that docstring as the field's description.
- An `Enum` that mixes in [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings], with a docstring under each member, says what each option means. The options of a `Literal`, or of a plain `Enum`, are seen by their names alone.
- A `bool` field's question is usually all it needs. Where the line between yes and no is subtle, as it is for "urgent", [`BoolCriteria`][pydantic_ai.output.BoolCriteria] says what a yes and a no mean. It is `Annotated` metadata, so the field stays a plain `bool` to every type checker and at runtime.

```python
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict

from pydantic_ai import Agent, BoolCriteria, UseEnumMemberDocstrings


class Area(UseEnumMemberDocstrings, str, Enum):
    billing = 'billing'
    """Charges, invoices, plans and payment methods."""

    bug = 'bug'
    """Part of the product does not work as it should."""

    account = 'account'
    """Logging in, access, and account settings."""


class Ticket(BaseModel):
    """Triage a support ticket."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    area: Area
    """Which team owns this ticket?"""

    urgent: Annotated[
        bool,
        BoolCriteria(
            true='The customer is losing money or has a deadline today.',
            false='It can wait its turn in the queue.',
        ),
    ]
    """Should this ticket jump the queue?"""


agent = Agent('typesafe:jev-latest', output_type=Ticket)
result = agent.run_sync(
    'You have charged me twice and my account is now overdrawn. I need this reversed today.'
)
print(result.output)
#> area=<Area.billing: 'billing'> urgent=True
assert result.response.provider_details is not None
print(result.response.provider_details['confidence'])
#> {'area': 1.0, 'urgent': 0.86}
print(result.response.provider_details['probabilities'])
#> {'area': {'billing': 1.0, 'bug': 0.0, 'account': 0.0}}
```

Every answer comes with how sure the model is of it. `provider_details['confidence']` has one number per field, from 0 to 1, and `provider_details['probabilities']` the whole distribution over each pick-one's options: Jev is certain this is billing, and fairly sure it should jump the queue. [Confidence and thresholds](#confidence-and-thresholds) covers what the numbers mean and how to act on them, such as handing the unsure answers to a language model.

With an output type to describe, the agent's `instructions` are no longer the question. They are sent as framing shared by every question — the product, the domain, the voice to judge in — as in the [example below](#a-support-desk-end-to-end). [Where the wording comes from](#where-the-wording-comes-from) maps every input to the agent onto the state or the questions.

### Supported field types

Every field is asked as one of a few kinds of question: a yes or no (`bool`), a pick-one (`Literal`, `Enum` or [`Choices`](#options-known-only-at-run-time), optionally `| None`), a probability (a bounded `float`), a score against a rubric (an `IntEnum` of described levels), or a yes or no per option (a `list` or `dict` of options). A nested model of these is a question per field. [What each field type does](#what-each-field-type-does) has the full table.

A field of any other type is a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent, and the message names the field and lists what is supported. The ones to expect are a `str`, an unbounded `int` or `float`, a `datetime`, a `dict` of anything but options to yes/no, and a union of models as a field. That is about the fields of a type the model is asked to fill. An output type beside other [routes](#routes-which-thing-to-do), a [union member](#a-union-of-output-types) or a [tool](#tools-pick-then-fill) the model cannot fill is not an error — it is still offered as a route, and picking it [escalates](#escalating-to-a-language-model) the step to the model behind it.

A backend can also cap how many options a pick-one or how many levels a rubric may have, through [`max_choice_options`][pydantic_ai.models.decision.DecisionModel.max_choice_options] and [`max_score_levels`][pydantic_ai.models.decision.DecisionModel.max_score_levels]. A pick-one over its cap is refused the same way, before a request is sent, while whole numbers with more levels than a rubric's cap are not a rubric, and are [a pick-one instead](#what-each-field-type-does). Jev's caps are on the [TypeSafe page](typesafe.md#limits).

## A support desk, end to end

The pieces combine into an agent that does real work. This one runs a support desk: it triages problems for the team that owns them, refunds charges the customer did not owe, checks the status page when a service might be down, and leaves anything that has to be written to a language model:

```python {title="support_desk.py"}
from enum import Enum, IntEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict

from pydantic_ai import Agent, BoolCriteria, UseEnumMemberDocstrings
from pydantic_ai.models.fallback import FallbackModel


class Area(UseEnumMemberDocstrings, str, Enum):
    billing = 'billing'
    """Charges, invoices, plans and payment methods."""

    bug = 'bug'
    """Part of the product does not work as it should."""

    account = 'account'
    """Logging in, access, and account settings."""


class Impact(UseEnumMemberDocstrings, IntEnum):
    cosmetic = 0
    """An annoyance; the customer can do everything they need to."""

    degraded = 1
    """Something is slow or broken, and there is a way around it."""

    blocked = 2
    """The customer cannot do their work until it is fixed."""


class Triage(BaseModel):
    """Route a problem to the team that owns it."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    area: Area
    """Which team owns this ticket?"""

    urgent: Annotated[
        bool,
        BoolCriteria(
            true='The customer is losing money or has a deadline today.',
            false='It can wait its turn in the queue.',
        ),
    ]
    """Should this ticket jump the queue?"""

    app: Literal['web', 'ios', 'android'] | None
    """Which app is it about?"""

    impact: Impact
    """How badly does this get in their way?"""


class Refund(BaseModel):
    """Give back money for a charge the customer did not owe."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    reason: Literal['duplicate', 'unrecognised', 'after_cancelling']
    """Why was the charge not owed?"""


class Reply(BaseModel):
    """Answer the customer: a question, or a problem the status page already explains."""

    model_config = ConfigDict(use_attribute_docstrings=True)

    body: str
    """The reply to send, in two sentences at most."""


def check_status(service: Literal['payments', 'login', 'reports']) -> str:
    """Check the status page for an incident on a service.

    Args:
        service: Which service is the customer having trouble with?
    """
    return f'{service}: degraded since 09:12 UTC; a fix is rolling out.'


support = Agent(
    FallbackModel('typesafe:jev-latest', 'anthropic:claude-opus-5-5'),
    output_type=[Triage, Refund, Reply],
    instructions='Tickets to the support desk of a project-management app.',
    tools=[check_status],
)

result = support.run_sync('You charged me twice for the March invoice.')
print(repr(result.output))
#> Refund(reason='duplicate')

result = support.run_sync(
    'Exporting the timeline to PDF on my iPad cuts off every task after March. '
    'Client review is this afternoon.'
)
print(repr(result.output))
#> Triage(area=<Area.bug: 'bug'>, urgent=True, app='ios', impact=<Impact.blocked: 2>)

result = support.run_sync('Is login down? None of my team can sign in.')
print(repr(result.output))
"""
Reply(body="Yes, login has been having problems since 09:12 UTC and that's why your team can't sign in. A fix is rolling out now, so please try again shortly.")
"""
print(result.response.model_name)
#> claude-opus-5-5
```

`Area` is a pick-one whose options are described by their docstrings, `urgent` a yes/no whose subtle boundary [`BoolCriteria`][pydantic_ai.output.BoolCriteria] spells out, `app` a pick-one that can answer "none of these", and `Impact` a [rubric](#what-each-field-type-does): ordered levels, each described. `Reply` asks for a `str`, which no decision model can fill, and that is deliberate: it is the route that needs a language model.

### How it runs

Each ticket is a run, and Jev takes every step it can:

1. **The route question.** The first request asks Jev one pick-one question over the ticket: does it call for `Triage`, `Refund`, `Reply`, or `check_status`? Each option is described by its docstring. No fields are asked yet, since until a route is picked there is no telling which fields apply.
2. **The fill.** A second request asks only the picked route's questions, over the same ticket: `Refund`'s `reason`, or all four of `Triage`'s fields at once. The run ends with that output. The first two tickets go this way, and cost two Jev requests each and no language model call.
3. **A tool call.** A picked tool is filled the same way — here, `check_status`'s `service`. Your function runs, and the next step asks Jev the route question again, with the result in the history. `check_status` is not offered a second time in the same run.
4. **Escalation.** `Reply` has a field Jev cannot fill, so picking it — like picking a tool with such an argument — raises [`UnfillableRoute`][pydantic_ai.models.decision.UnfillableRoute], a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]. The `FallbackModel` hands the whole step to the language model, with the same tools and output types, and the language model decides afresh: it could triage, refund or check the status as well. On the third ticket, Jev checked the status and then picked `Reply`, so the language model wrote the reply with the status in view, and the response names it as the model that answered. That one step is the only language model call; the steps before it stayed on Jev.

The rest of this page takes each piece in turn: [where the wording comes from](#where-the-wording-comes-from), [each field type](#what-each-field-type-does), [routes](#routes-which-thing-to-do) through [tools](#tools-pick-then-fill) and [unions](#a-union-of-output-types), [escalation](#escalating-to-a-language-model), and [confidence](#confidence-and-thresholds) — where the same `FallbackModel` can also take the steps Jev was unsure about.

## Where the wording comes from

Every input to the agent ends up in one of two places: the state, which is judged, or the questions, which are asked.

**Judged**, as the state:

| Agent input | Where it ends up |
|---|---|
| the run's prompt | the whole state when there is no history, otherwise its `text` |
| the message history | the state's `history`, as user prompts, answers, thinking, tool calls and results, and retry prompts — see [judging a conversation](#judging-a-conversation) |
| a system prompt, including the agent's own `system_prompt=` | the state's `history`, as a `system` entry — [not part of the question](#judging-a-conversation) |

A question can point at a part of the state by its name, such as "Is the request in `text` already answered in `history`?", which TypeSafe [recommend](https://docs.typesafe.ai/model-jaggedness/jev-1.13#indirection) over leaving the model to work out which part is meant.

**Asked**, as the questions:

| Agent input | Where it ends up |
|---|---|
| the agent's `instructions` | the question itself, when the output is a bare `bool`, pick-one or `float` with no field to describe; otherwise, shared framing on every question |
| the output type's docstring | the goal, on every question about it, and its description when it is offered as a [route](#routes-which-thing-to-do) |
| a field's description — `Field(description=...)`, or the docstring under the field with `use_attribute_docstrings=True`; an `Enum` field's class docstring only when the field has neither | that field's question |
| a description on an option in the schema, such as an `Enum` member's docstring | that option's meaning |
| [`BoolCriteria`][pydantic_ai.output.BoolCriteria] on a `bool` field | what a yes and a no mean |
| a description on the `None` itself, `Annotated[None, Field(description=...)]` | what "none of these" means |
| a tool's description, and its arguments' descriptions | the tool's option on the route question, and the questions that [fill its arguments](#tools-pick-then-fill) |

Dependencies and anything else on the run context are not sent, unless a prompt, instructions function or history processor puts them into one of the rows above.

Prefer an output type with fields to a bare output with its question in `instructions`: each field carries its own question, so several questions can be asked in one request. Reach for `instructions` for framing that applies to every question — the voice to judge in, the domain, what the material is — and for the question itself only when there is one question and no field to describe.

Unless the schema describes an option, the model sees it by its name alone, so name `Literal` and `Enum` options for what they mean. A `Literal` has nowhere to write a meaning per option; where the difference between two options needs explaining, use an `Enum` that mixes in [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings] and put a docstring under each member, which is what puts a description on each option in the schema.

In short:

- **The output type's docstring states its purpose, as a goal:** "Triage a support ticket."
- **Every field carries its question**, as `Field(description=...)` or as a docstring under the field with `model_config = ConfigDict(use_attribute_docstrings=True)`.
- **A `bool` field needs a question, and that is usually enough.** Add [`BoolCriteria`][pydantic_ai.output.BoolCriteria] when the boundary between yes and no is subtle, or when a [literal reading](typesafe.md#what-jev-answers-badly) of the question would go wrong.
- **Option docstrings and `BoolCriteria` texts are statements** of what that outcome means, not questions: "The customer is losing money or has a deadline today." They must agree with the question: a `true` that describes a no confuses the model.
- **A `list` field is asked once per option**, so word its question as a yes or no about one option: "Does this ticket raise this topic?", not "Which topics does this ticket raise?".
- **A rubric level's description stands alone.** Its member name is not sent, so describe the situation the level covers, not a degree: "Something is broken, and there is a way around it", not "Moderate".
- **Give every outcome a route.** An agent with several output types needs [one for everything the text can call for](#give-every-outcome-a-route), including a way out to a language model.
- **One question per field, in one place.** An `Enum` class docstring is only sent when the field has no description, so give the field its question and leave the class docstring off.
- **`instructions` are framing shared by every question**, and the question itself only for a bare output with no field to describe.

## Ask one thing per field

This is the second habit to learn. Ask each field the kind of judgement a knowledgeable person makes in a second. A question that weighs several things at once does not fail — it returns a plausible answer with low confidence, and you find out later.

So instead of one field asking `'Is this a good pitch?'`, ask three and combine them in code:

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Pitch(BaseModel):
    """Assess a startup pitch."""

    large_market: bool = Field(description='Does this address a market worth more than $1B a year?')
    technically_feasible: bool = Field(description='Could a small team build this with current technology?')
    differentiated: bool = Field(description='Does this do something competitors do not already do?')

    @property
    def promising(self) -> bool:
        return sum([self.large_market, self.technically_feasible, self.differentiated]) >= 2


agent = Agent('typesafe:jev-latest', output_type=Pitch)
result = agent.run_sync('A dashboard that shows every SaaS subscription a company pays for.')
print(result.output)
#> large_market=True technically_feasible=True differentiated=False
print(result.output.promising)
#> True
```

Every field goes out in the same request, so a field you only need on some inputs costs no extra round trip. What it does cost depends on the backend: on Jev, an extra field [costs tokens rather than time](typesafe.md#limits).

## What each field type does

Each field type is asked as one kind of question, and comes back as one kind of answer:

| Field type | Question | Answer | Notes |
|---|---|---|---|
| `bool`, or `Literal[True, False]` | yes or no | `True` when the probability of yes is at least `decision_boolean_threshold` (0.5) | [`BoolCriteria`][pydantic_ai.output.BoolCriteria] says what a yes and a no mean |
| an `Enum` of `True` and `False`, with [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings] | yes or no | the member the answer picks | the same question as a `bool`, with a named answer |
| `Literal[...]` or `Enum` of strings or whole numbers | pick one | the chosen option | a `Literal`'s options are seen by name alone; an `Enum`'s members can be described |
| [`Choices(...)`][pydantic_ai.output.Choices] | pick one | the chosen option, or what it stands for | for options [known only at run time](#options-known-only-at-run-time), each with its meaning |
| any pick-one in a union with `None` | pick one, or "None of these." | the option, or the field's default, or `None` | not for a rubric |
| `float` with `ge=0` and an inclusive upper bound (`le=`) | the probability of yes | the probability, unrounded, in the field's own units | no confidence entry, since the probability is the answer |
| an `IntEnum` of `0, 1, 2, …` with a description per level, from `UseEnumMemberDocstrings` | score against a rubric | the nearest level | at most the backend's `max_score_levels`; the unrounded position is in `provider_details['scores']` |
| `list` of a `Literal` or `Enum` of strings | one yes or no per option | the options answered yes | word its question about one option; its confidence is the least sure option's |
| `dict` from a `Literal` or `Enum` to `bool` | one yes or no per option | every option, with its answer | the same question as the `list` |
| a nested model of these | its fields, asked as `outer.inner` | the model | the parent field's description is not sent |

The bound on a number field is the units it is asked in, not a second question: `ge=0, le=1` is the probability as the model gives it, and `ge=0, le=100` the same answer written as a percentage. A `dict` keyed by options and valued by `bool` asks what a `list` of those options asks — one yes or no each — and differs only in the answer, which keeps every option rather than just the ones answered yes.

Each of those yes/no questions carries the field's description as its question, with one option and its description beside it, so write the description as a yes or no about a single option: `topics: list[Topic]` described as "Does this ticket raise this topic?" reads right when it is asked about each topic in turn, where "Which topics does this ticket raise?" asks for a list the question is not in a position to give.

An optional pick-one field, `Area | None`, is the same question with one more option, "None of these.": an explicit option, so that "nothing fits" is an answer the model can give rather than something read off low confidence, which is what the field's confidence is for. Picking it is the absence of an answer, so a field with a default gets its default, and a field without one gets `None`. The same goes for a nested model with a default when nothing under it was answered: it gets its own default, not one built from the defaults of the fields inside it. A `default_factory` is not in the schema, which is all the answers are read against, so it counts as no default. Any pick-one, of strings or whole numbers, can be optional; a rubric cannot, since its levels are ordered and `None` is not one of them. To say what picking nothing means on this field rather than take the stock phrase, describe the `None` itself: `Area | Annotated[None, Field(description='Nothing here needs routing.')]` makes that description the option's meaning.

A yes/no is a `bool`, and its question is usually enough on its own. Where the line between yes and no is subtle, or where a [literal reading](typesafe.md#what-jev-answers-badly) of the question would go wrong, [`BoolCriteria`][pydantic_ai.output.BoolCriteria] says what each answer means, as it does for "urgent" in the [`Ticket` above](#asking-with-an-output-type). The two descriptions become the question's `criteria`, which the model weighs the text against, so they must agree with the question: each says what a yes or a no to *that* question looks like. TypeSafe [suggest](https://docs.typesafe.ai/primitives/noul) asking with and without criteria and keeping whichever answers better on your own examples.

Where the answer should be a named thing rather than `True` or `False` — because it is stored, or branched on by name — an `Enum` of `True` and `False` mixing in [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings] asks exactly the same question, and the field's value is the member the answer picks:

```python {title="say_what_yes_and_no_mean.py"}
from enum import Enum

from pydantic import BaseModel, Field

from pydantic_ai import Agent, UseEnumMemberDocstrings


class Refunded(UseEnumMemberDocstrings, Enum):
    yes = True
    """Money was returned to the customer."""

    no = False
    """No refund was issued."""


class Settled(BaseModel):
    """Review the transcript."""

    refunded: Refunded = Field(description='Was a refund issued?')


agent = Agent('typesafe:jev-latest', output_type=Settled)
result = agent.run_sync('We have sent the 40 pounds back to your card.')
print(result.output.refunded)
#> Refunded.yes
```

A `Literal[True, False]` has nowhere to write the two meanings at all, and asks exactly what a bare `bool` asks.

A rubric is a set of ordered levels rather than a set of alternatives: the whole numbers from 0 upwards, at least two of them and no more than the backend's [`max_score_levels`][pydantic_ai.models.decision.DecisionModel.max_score_levels], and every level needs a description in the schema saying what it means. The ordering is the numbers' own, so the order the levels are declared in does not matter. The model answers with a position along the rubric, which lands between levels, and the field gets the nearest one — a half rounds up. The unrounded position is in `provider_details['scores']`.

A level's description reaches the schema the [same way an option's meaning does](#where-the-wording-comes-from), which makes an `IntEnum` mixing in `UseEnumMemberDocstrings` the way to declare one. The descriptions are all the model sees of the levels: the member names, `opaque` and `actionable` below, are not sent. So each description has to stand alone, and should, as TypeSafe [put it](https://docs.typesafe.ai/primitives/score), "describe situations, not degrees": "A reader who did not already know could act on it" gives the model something to match the text against, where "Very clear" does not.

Any other whole numbers are labels rather than levels, and are a pick-one like strings: `Literal[200, 404, 500]`, an `IntEnum` of codes, or `Literal['a', 1]` mixing the two. That includes numbers from 0 upwards that miss being a rubric only because a level says nothing about itself — a bare `Literal[0, 1, 2]`, a plain `IntEnum` — or because there are more of them than the backend scores against. A pick-one weighs its options without their order, so describe every level of a rubric you mean as one. The model picks a number by its digits, and the field gets back the option itself, number and all. A number whose digits are already a string option is offered as `1 (number)`, so `Literal['1', 1]` is still two options.

```python {title="grade_with_a_rubric.py"}
from enum import IntEnum

from pydantic import BaseModel, Field

from pydantic_ai import Agent, UseEnumMemberDocstrings


class Clarity(UseEnumMemberDocstrings, IntEnum):
    opaque = 0
    """Leaves a reader who did not already know none the wiser."""

    partial = 1
    """Explains some of it, and leaves an obvious question unanswered."""

    actionable = 2
    """A reader who did not already know could act on it."""


class Review(BaseModel):
    """Grade a release note."""

    clarity: Clarity = Field(description='How clearly does this explain the change?')


agent = Agent('typesafe:jev-latest', output_type=Review)
result = agent.run_sync('Fixed a bug in the parser.')
print(result.output)
#> clarity=<Clarity.opaque: 0>
assert result.response.provider_details is not None
print(result.response.provider_details['scores'])
#> {'clarity': 0.12}
```

A nested model is its fields, asked as `outer.inner` and put back in place; the parent field's description is not sent, so put the context each question needs on the field that asks it. A dot in a field name is how a nested field is named, so a field whose own name contains one is refused.

### Options known only at run time

A `Literal` or an `Enum` names its options in the source. When a pick-one's options are only known once the run is under way — the teams configured in your help desk, the records a search returned — build them with [`Choices()`][pydantic_ai.output.Choices], from a mapping of each option to what it means. Each meaning reaches the model the way an `Enum` member's docstring does, and the set's `description` is the question:

```python {title="choices_at_run_time.py"}
from pydantic_ai import Agent, Choices

agent = Agent('typesafe:jev-latest')

teams = {
    'payments': 'Charges, refunds and invoices.',
    'identity': 'Signing in, single sign-on and two-factor authentication.',
    'mobile': 'The iOS and Android apps.',
}
result = agent.run_sync(
    'The app on my Pixel logs me out every time I lock the screen.',
    output_type=Choices(teams, description='Which team should take this ticket?'),
)
print(result.output)
#> mobile
assert result.response.provider_details is not None
print(result.response.provider_details['probabilities'])
#> {'response': {'payments': 0.0, 'mobile': 0.82, 'identity': 0.18}}
```

`Choices()` returns a type, so the same set also works as a field of an output type or as a tool argument, in `Annotated` on the type of the value it gives back: `Annotated[str, Teams]`. An option can also stand for a value other than its own key, or for an action that runs when it is picked, as in [choosing from a set built at run time](#choose-from-a-set-built-at-run-time). [Choices known only at run time](../output.md#choices) covers the helper in full.

## Routes: which thing to do

Fields are what the model fills. A *route* is a thing the text could call for: an output type, an [output function](../output.md#output-functions), or a [tool](#tools-pick-then-fill). When there is more than one on offer — tools attached, or a [union](#a-union-of-output-types) of output types and output functions — the model is asked one more question, the route question: which of these does this call for, framed by the agent's `instructions` like every other question. The options are every output type, every output function and every tool on offer, each described by its docstring.

Each option goes by the name you gave the route:

- a tool or an [output function](../output.md#output-functions) by the function's name;
- an output type by its class name, an `Enum`'s included, or by the `name` you gave it with [`ToolOutput`][pydantic_ai.output.ToolOutput];
- a `None` member of a union as `None`;
- an [on-demand capability](#on-demand-capabilities) by its `id`;
- a single output type with no class name of its own, such as a bare `bool` or `Literal`, as `output`.

Where a tool and an output type would share a name, the tool keeps it and the output type's option becomes, say, `Refund (output)`. A capability that shares a tool's name becomes `refund (capability)` the same way.

The route the model picks is the one that runs, and how much it costs to fill depends on which route it is. A single output type's fields, when the model can fill them, ride along in the *same* request as the route question, so its answers are already in hand when the pick comes back. Every other route is picked first and filled after: a chosen tool's arguments, or a chosen [union](#a-union-of-output-types) member's fields, go out in a second request carrying only that route's questions. A route with no arguments — an [output function](../output.md#output-functions) that takes nothing but the run context, or a tool with no parameters — is called on the pick alone, with no second request at all.

Fields are never merged across routes: a tool's arguments are only asked once the tool is picked, and a union member's fields once that member is. A route with even one field or argument the model cannot fill is still offered, and picking it [escalates](#escalating-to-a-language-model) the step:

| Route | How it is asked | If the model cannot fill it |
|---|---|---|
| a single output type or output function with arguments | its fields in the same request as the route question, or alone when nothing else is on offer | only the route is asked, and picking it escalates; with no route beside it that the model can take, a [`UserError`][pydantic_ai.exceptions.UserError] |
| a member of a union of output types or output functions | picked, then its fields in a second request | picking it escalates |
| a tool with arguments | picked, then its arguments in a second request | picking it escalates |
| an output function with no arguments, `None`, a tool with no arguments, or an [on-demand capability](#on-demand-capabilities) | picked, and taken with nothing to fill | — |

The pick is on the response, in `provider_details['route']`, under the same names: `choice` is the route the model picked, `probabilities` the probability it gave every route, and `offered` the routes the question offered. `provider_details` describes the response rather than each request: `confidence`, `probabilities` and `scores` are for the answers the output or the tool call was built from, which are the second request's when there was one, and are empty for a route called on the pick alone. A single output type's answers that rode along with the pick of something else built nothing, so they are not reported.

The second request is about the same text as the first, which on its own would leave nothing in it saying a route had been picked. So each of its questions names the chosen route, under the same name the route question offered it by, alongside the field's own question and whatever the route's docstring said about it. What you already said in the text is unchanged between the two requests; only the questions differ.

The response sums the input and output tokens from both requests, but [`RequestUsage.requests`][pydantic_ai.usage.RequestUsage.requests] is fixed at one request per model step and cannot carry the real count, so `provider_details['requests']` is `2` when the model picked and then filled.

The second request has already committed to the selected route. If that request fails or returns invalid answers, [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] names the route and stops the run; the default `FallbackModel` does not replay the original step and quietly choose another one.

The questions in one request are answered independently. A field cannot depend on another field's answer: two arguments of the same tool are decided separately, and neither sees the other. Where one judgement genuinely follows from another, they belong in different steps, not in two fields of the same call — which is what the [patterns below](#decision-models-inside-an-agent-run) build on: an output function is a route the model can choose, and choosing it *is* calling it.

Write the output type's docstring as the action it is — "Triage a support ticket", "Reply to the customer" — because that is what the other routes are weighed against. Asking whether the model *can* answer, rather than what the text calls for, is a question about the question rather than about the text, and [on Jev](typesafe.md#what-jev-answers-badly) it hands off nearly everything.

### Tools: pick, then fill

With tools attached, the first request carries the route question, with every output type first among the options and every tool after them. Each tool is described by its docstring, and the output type by its own docstring or, without one, by the agent's instructions; with tools attached one of the two is required, since it is what filling the output is weighed against. The pick decides which route the request takes:

| The model picks | What runs | Language model call |
|---|---|---|
| a single output type | the decision model fills the fields, in the same request | none |
| one member of a union of output types | the decision model fills that member's fields in a second request | none |
| a tool with no arguments | your function, then the decision model again with its result in view | none |
| an output function with no arguments | your function, and the run ends | only if the function makes one |
| a tool whose arguments the model can express | the decision model fills its arguments in a second request, then your function runs | none |
| a tool, output type or output function with any field or argument the model cannot fill | the model behind the decision model [takes the whole step](#escalating-to-a-language-model), tools and all | one |
| any route, below [`decision_route_threshold`](#handing-off-an-unsure-route) when you set one | the model behind the decision model takes the whole step | one |

The route taken is the likeliest one, however sure the model is of it. To hand the unsure picks to a language model instead, set [`decision_route_threshold`](#handing-off-an-unsure-route).

With no output type to fill and every other route already returned this turn, the one route left is taken without a route question at all: the model still fills its supported arguments in one request, and a route with no arguments costs no request. A single output type left on its own the same way is filled without a route question too, and one the model cannot fill escalates without a request.

A pick is a classification of the text, not a judgement that running the tool is safe: the framework emits the call and your function runs, exactly as on a language model's call, so a tool that sends mail or charges an account is one a decision model can set off, and approval and limits are the agent's job here as anywhere.

**A tool with no arguments: the decision model alone.** There is nothing to fill, so the call is made on the pick, and its result comes back as history for the next request. The model can work through a sequence of such tools. A tool whose result is already in the turn is not offered again, because a decision model has no notion of having made a call and would pick it again with the result in view; one that asked for a retry stays on offer. What was on offer is in `provider_details['route']['offered']`. Every request here is a decision model request:

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


escalated: list[str] = []


def escalate_to_human() -> str:
    """Hand the ticket to a person on the support team."""
    escalated.append('case #4821')
    return 'Escalated: case #4821 opened.'


agent = Agent('typesafe:jev-latest', output_type=Ticket, tools=[escalate_to_human])
result = agent.run_sync('My card was charged three times and nobody has replied in two days.')
print(result.output)
#> urgent=True
print(escalated)
#> ['case #4821']
```

Put a `UsageLimits(request_limit=...)` on a decision model agent with tools all the same, as on any agent that loops.

**An output function with no arguments: a hand-off that ends the run.** An output function that takes nothing, or only the run context, is picked the same way, and the run ends with what it returns — to a person, a queue, or another agent. The language model runs only inside the hand-off, so only the requests the decision model handed off pay for one:

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


support = Agent('anthropic:claude-opus-5-5', instructions='Reply to the customer.')


async def reply(ctx: RunContext) -> str:
    """Write the customer a reply."""
    # `ctx.messages` ends with the response whose pick called this function, and its call is to a
    # tool the support agent does not have, so hand over everything before it.
    result = await support.run(message_history=ctx.messages[:-1])
    return result.output


agent = Agent('typesafe:jev-latest', output_type=[Ticket, reply])


async def main():
    result = await agent.run('Could you tell me when my order ships?')
    print(result.output)
    #> It shipped this morning; the tracking link is on its way to you now.
```

**Supported arguments: the model picks, then fills.** Tool arguments use the same [mapping](#supported-field-types) as output fields. The argument name is the field, its `Args:` entry in the function docstring is the question, and the tool description is the goal. The first request picks the tool; the second carries only its argument questions over the same text and history, as `check_status` does in the [support desk](#a-support-desk-end-to-end).

An `Args:` entry describes the *argument*, not its options: a `Literal` argument's options go out named and nothing more, the same as [a `Literal` output field](#where-the-wording-comes-from). Where the difference between two of them needs explaining, make the argument an `Enum` that mixes in [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings] with a docstring under each member, or a [`Choices`][pydantic_ai.output.Choices] set built from a mapping of option to meaning — either puts a description on each option in the schema, which is what the model weighs them by. `Choices` built from a bare sequence of names describes nothing, and leaves the model weighing the names alone like a `Literal` does.

**Unsupported arguments: the model behind it.** A plain `str`, an unbounded number, or any other unsupported argument cannot be filled, so picking the tool [escalates](#escalating-to-a-language-model) the step.

### On-demand capabilities

An [on-demand capability](../capabilities/on-demand.md) is offered as a route of its own, under its `id` and described by its `description`, beside the tools and output types. Picking it loads it, on the pick alone: its instructions come back as the load's result, which the next request sees in the history, and its tools are on offer from then on. A capability the history shows loaded is not offered again, and loading one leaves the others on offer, so a turn that needs two can load both.

A language model loads a capability by calling `load_capability` with the capability's `id`, and reads which ids there are from a catalog appended to the agent's instructions. A decision model cannot fill a free-form `id`, and the catalog would otherwise be framing on every question, fields included, repeating every capability's description whatever was being asked. So a decision model leaves the catalog out, and offers each capability as the route it is instead. What a language model is sent does not change:

```python {title="decision_on_demand_capability.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelResponse, ToolCallPart


class Resolution(BaseModel):
    """Close the conversation once the customer's question has been answered."""

    resolved: bool = Field(description="Has the customer's question been answered?")


refunds = Capability(
    id='refunds',
    description='Use for refund eligibility, refund status, or processing a refund.',
    instructions='Confirm the order ID before issuing a refund.',
    defer_loading=True,
)


@refunds.tool_plain
def latest_refund() -> str:
    """Look up the status of the customer's latest refund."""
    return 'Refund of $40 issued on 2026-05-01.'


shipping = Capability(
    id='shipping',
    description='Use for where an order is, delivery dates, and tracking.',
    defer_loading=True,
)

agent = Agent(
    'typesafe:jev-latest',
    output_type=Resolution,
    instructions='You are a customer support assistant for an online store.',
    capabilities=[refunds, shipping],
)
result = agent.run_sync('Has the refund for my returned blender gone through?')
print(result.output)
#> resolved=True
print(
    [
        part.tool_name
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
)
#> ['load_capability', 'latest_refund', 'final_result']
```

The first request's route question offers `Resolution`, `refunds` and `shipping`. Once `refunds` has loaded, it offers `Resolution`, `shipping` and `latest_refund`, and once `latest_refund` has returned, `Resolution` and `shipping`. Every request here is a decision model request.

### A union of output types

An `output_type` of several structured types is a set of routes. The model picks which one the text calls for, then a second request asks only that type's fields — the same two steps a [selected tool's arguments](#tools-pick-then-fill) take, because it is the same question asked twice. The [support desk](#a-support-desk-end-to-end) is one: `Triage`, `Refund` and `Reply` are three routes.

Each member is described by **its own docstring**, which is what the model weighs the routes against. With one output type the agent's instructions can say what filling it is for; with several they cannot, because one instruction cannot describe two different routes, so a member without a docstring is a [`UserError`][pydantic_ai.exceptions.UserError].

The pick is reported in `provider_details['route']`, with the probability of every route, and `provider_details['requests']` is `2`.

#### Declining with `None`

`None` is a route like any other. Include it in the union and the model is offered one more option, "None of these.", for the text that calls for nothing at all:

```python {title="union_none.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


class Escalation(BaseModel):
    """Hand the ticket to a human specialist: security, privacy or legal problems."""

    security: bool = Field(description='Does this involve a security risk?')


agent = Agent('typesafe:jev-latest', output_type=[Ticket, Escalation, None])

result = agent.run_sync('Thanks, that fixed it. Nothing else needed.')
print(result.output)
#> None
```

`None` cannot carry a docstring, so the library describes it, the same way an [optional pick-one field](#what-each-field-type-does) gets its "None of these." option. There is nothing to fill either, so the route is taken on the pick alone: declining costs one request, never two.

To say what declining means on your agent rather than take the stock phrase, name the route yourself with [`ToolOutput`][pydantic_ai.output.ToolOutput]: `ToolOutput(type_=None, name='nothing', description='Nothing needs doing here.')` puts that description on the route instead.

#### Give every outcome a route

The route question is answered with one of the routes on offer, so the output types an agent offers should cover everything the text can call for. That is output type design for any model, not something particular to decision models: a language model offered only `Refund` and `Feedback` has no good answer to "What day is my order arriving?" either, and has to force it into one of the two.

It matters most after a tool returns. A tool whose result is already in the turn is [not offered again](#tools-pick-then-fill), so the next step picks one of the output types, whether or not any of them fits. Give that agent an order lookup tool, and "What day is my order arriving?" looks the order up and then comes back as a refund or as feedback — `Feedback`, most likely — with the answer the customer wanted nowhere in it.

The way out is a route that hands the step on: an output type like the [support desk](#a-support-desk-end-to-end)'s `Reply`, with a `str` field no decision model can fill. Picking it [escalates](#escalating-to-a-language-model) the step, so behind a `FallbackModel` a language model answers whatever the other routes do not cover, with the tool's result in view.

## Escalating to a language model

A route the model cannot fill is still offered, and picking it hands the step on: a tool with any unsupported argument, such as a plain `str` or an unbounded number, or an output type with such a field, like the support desk's `Reply`. The pick becomes an [`UnfillableRoute`][pydantic_ai.models.decision.UnfillableRoute] rather than a response. That is a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`](overview.md#fallback-model) with a language model behind the decision model hands that model the whole step, with the same tools and output types, and the language model decides the step again for itself. The rest of the requests never leave the decision model. Without a model behind it, the hand-off is the error, and it says which route the model picked and how sure it was. The request that proposed the call is not on the fallback response's usage.

The decision model answers the tickets it can and hands over the ones that need writing, so only those cost a language model call. How many that is depends on your tickets and on [`decision_route_threshold`](#handing-off-an-unsure-route) if you set it, and `provider_details['route']` on each response is how to see it.

An agent on which *every* route would hand off is refused before any request instead: an `output_type` the model cannot fill, or a union of them, with no route beside it that the model can take — a tool or output type it can fill, or one with nothing to fill. Every answer could only hand off, so the request asking for one would buy nothing — that is a coding error, and finding out at setup beats finding out from the bill. With such a route on offer, an output type the model cannot fill is a route like any other, whether it is alone or in a union.

The same `FallbackModel` can also take the steps the decision model answered but [was unsure about](#falling-back-on-low-confidence).

`UnfillableRoute` and [`UnsureRoute`](#handing-off-an-unsure-route) are both a [`DecisionHandOff`][pydantic_ai.models.decision.DecisionHandOff]: the decision model handing the step on, rather than failing. A `FallbackModel` falls back on every `ModelAPIError` by default, which includes an outage of the decision model's backend. To hand the language model only the steps the decision model hands off, and let a backend error fail the run instead of quietly costing a language model call on every step, pass `fallback_on=DecisionHandOff`:

```python {title="hand_offs_only.py"}
from pydantic_ai.models.decision import DecisionHandOff
from pydantic_ai.models.fallback import FallbackModel

model = FallbackModel('typesafe:jev-latest', 'anthropic:claude-opus-5-5', fallback_on=DecisionHandOff)
```

!!! warning "Watch the hand-off rate"
    A union that hands off on most requests costs a language model call **plus** a decision model call, and is
    slower than not using a decision model at all. Measure the rate on your own data before relying on the
    arrangement.

    Note where the number is. On a request the decision model answers, its pick and the probability of every
    route are in `provider_details['route']`. On a hand-off they are not:
    a [`DecisionHandOff`][pydantic_ai.models.decision.DecisionHandOff] is raised instead of a response, and
    [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] returns the *next* model's response, which carries
    none of the decision model's numbers. So counting hand-offs by their absence in `provider_details` is the
    measurement. If you would rather catch the exceptions, each hand-off carries the `route` the model picked and
    its `probability`, and `UnsureRoute` the `probabilities` of every route as well: run the models separately, or
    wrap the fallback, when you want both.

## Confidence and thresholds

Confidence in each answer is on the response, in `provider_details['confidence']`: 0 to 1, one number per field. It is a margin, not a probability that the answer is right. For a yes/no it is how far the probability of yes sits from the threshold that decided it, scaled to run from 0 at the threshold to 1 at certainty — at the default of 0.5 that is the distance from the coin flip, doubled, so a `False` answered from a probability of 0.01 reports 0.98 and one answered from 0.45 reports 0.10. The bar it measures from is the one [actually used](#what-true-has-to-mean), so a yes at 0.8 under a threshold of 0.75 reports 0.2 rather than the 0.6 it would report against a coin flip, and a [fallback on low confidence](#falling-back-on-low-confidence) keeps meaning what it meant. For a pick-one or a rubric it is the confidence the backend reports, from how its probabilities are spread; for a list of options it is the least sure option's. The kinds of question arrive at the number differently, so a bar tuned on a yes/no does not carry over to a pick-one — TypeSafe [say as much](https://docs.typesafe.ai/model-jaggedness/jev-1.13#common-sense-structural-invariants) of Jev — and neither does a bar tuned on one field carry over to another: [set one per field](#falling-back-on-low-confidence).

`provider_details['probabilities']` holds the whole distribution of each pick-one and rubric field — a pick-one's options keyed by the label the model picked them by, and a rubric's levels by their number as a string — and each option's probability for a list. `provider_details['scores']` holds each rubric field's unrounded position along its levels.

A `float` field has no entry in any of them. The probability *is* its answer, so nothing was lost to rounding and there is no second number to report — a `churn_risk` of 0.93 is the judgement, not a 93%-confident judgement — and `0.5` means the model is undecided, not that the answer is middling. Apply `abs(value - 0.5) * 2` yourself for the same reading the other fields give at the default threshold; against a `decision_boolean_threshold` of your own, the margin is the distance from *that* bar scaled to the room left on the side the answer fell — `(value - t) / (1 - t)` at or above it, `(t - value) / t` below.

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-latest', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
print(result.response.provider_details)
#> {'confidence': {'response': 0.84}, 'probabilities': {}, 'scores': {}}
```

Two thresholds turn a probability into what the agent does with it. They are [`DecisionModelSettings`][pydantic_ai.models.decision.DecisionModelSettings], set like any other [model settings](../agent.md#model-run-settings), and apply to every decision model:

- [`decision_boolean_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_boolean_threshold], default 0.5, is how likely a yes has to be before a `bool` field is `True`: [what `True` has to mean](#what-true-has-to-mean).
- [`decision_route_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_route_threshold], unset by default, is how likely the picked [route](#routes-which-thing-to-do) has to be before it is taken, rather than handed to the model behind the decision model: [handing off an unsure route](#handing-off-an-unsure-route).

Both are read before the request is sent, so a value outside 0 to 1 is a [`UserError`][pydantic_ai.exceptions.UserError] rather than a wasted request.

Every bar on this page — the confidence you decide to act on, [`decision_boolean_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_boolean_threshold] and [`decision_route_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_route_threshold] — belongs to what its answer is used for rather than to the system as a whole: acting automatically deserves a higher one than flagging something for review. Calibrate each against labelled examples of your own, and once you have tuned one, pin the model version it was tuned against, since a new version can shift the numbers under you. Not every backend's probabilities are calibrated the same way, so a bar tuned on one backend does not carry over to another.

### What `True` has to mean

A yes/no is answered with the probability of yes, and [`decision_boolean_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_boolean_threshold] decides where that rounds. The default of 0.5 is the coin flip: the answer is whichever side the model favours. That is the right default and the wrong setting for any field where the two mistakes do not cost the same.

Raise it where a false positive is the expensive one, so a `True` has to be earned:

```python {title="earn_a_true.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.decision import DecisionModelSettings


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    safe_to_run: bool = Field(description='Is this command safe to run without a human looking at it?')


agent = Agent(
    'typesafe:jev-latest',
    output_type=Handling,
    model_settings=DecisionModelSettings(decision_boolean_threshold=0.9),
)
result = agent.run_sync('pytest tests/test_agent.py')
print(result.output)
#> safe_to_run=True
```

Lower it where a false negative is, so a `True` only has to be plausible — a flag that sends a borderline case to a human is cheap, and one that misses a real case is not.

The threshold applies to every `bool` field and to each option of a fanned-out `list`. It does not apply to a `float` bounded with `ge=0` and `le=1`: a field whose bar you would want to vary per call is often better declared that way, and compared in your own code.

### Falling back on low confidence

[`FallbackModel`](overview.md#fallback-model) falls back on API errors by default, and its `fallback_on` also takes a handler that looks at the response. A decision model's confidence is on the response, so a language model can take over the requests the decision model was unsure about — the cheap model answers what it can, the expensive one only the rest. A `float` field has no confidence entry, for the reason above, so a handler like this one does not see its uncertainty and an output of nothing but `float`s never falls back:

```python {title="fall_back_when_unsure.py"}
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, ModelAPIError, ModelResponse
from pydantic_ai.models.fallback import FallbackModel


class Screening(BaseModel):
    """Screen a request to a coding agent."""

    harmful: bool = Field(description='Is this request harmful?')
    target: Literal['code', 'infrastructure', 'data'] = Field(
        description='What does the request act on?'
    )


# How sure each field has to be, set by what a wrong answer costs.
BARS = {'harmful': 0.8, 'target': 0.6}


def unsure(response: ModelResponse) -> bool:
    confidence = (response.provider_details or {}).get('confidence', {})
    return any(value < BARS[field] for field, value in confidence.items())


model = FallbackModel(
    'typesafe:jev-latest', 'anthropic:claude-opus-5-5', fallback_on=[ModelAPIError, unsure]
)
agent = Agent(model, output_type=Screening)

result = agent.run_sync('Rename the helper functions in utils.py to snake_case.')
print(result.output)
#> harmful=False target='code'
assert result.response.provider_details is not None
print(result.response.provider_details['confidence'])
#> {'harmful': 0.98, 'target': 1.0}

result = agent.run_sync("Drop the staging database and restore it from last night's backup.")
print(result.output)
#> harmful=False target='data'
print(result.response.model_name)
#> claude-opus-5-5
```

Each field has its own bar. A wrong `harmful` costs more than a wrong `target`, so it has to be surer, and the two are different kinds of question whose confidence is [not on the same scale](#confidence-and-thresholds) anyway. Jev was sure of both answers about the rename, so they stand. About the database it answered `data` with a confidence of 0.52 and `harmful` with 0.28, both under their bars, so the language model took the request. Tune each bar on labelled examples of that field.

The handler runs on every model in the chain, and a language model reports no `confidence`, so its answers pass through. A response handler on its own replaces the default exception fallback, which is why `ModelAPIError` is listed alongside it.

### Handing off an unsure route

The [route](#routes-which-thing-to-do) pick can be unsure too. The model has to pick one of the routes it is offered, so when none of them really fits the text — after a tool has returned, say, with only output types left to pick from — it still picks one, with a low probability, as a language model given the same choices would. The likeliest route is taken however unsure the model is of it, unless you set [`decision_route_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_route_threshold]: a pick below it raises [`UnsureRoute`][pydantic_ai.models.decision.UnsureRoute] before anything is filled. That is a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a `FallbackModel` hands the language model the whole step, with the same tools and output types:

```python {title="unsure_route.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.decision import DecisionModelSettings
from pydantic_ai.models.fallback import FallbackModel


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


class Escalation(BaseModel):
    """Hand the ticket to a human specialist: security, privacy or legal problems."""

    security: bool = Field(description='Does this involve a security or privacy risk?')


agent = Agent(
    FallbackModel('typesafe:jev-latest', 'anthropic:claude-opus-5-5'),
    output_type=[Ticket, Escalation],
    model_settings=DecisionModelSettings(decision_route_threshold=0.7),
)

result = agent.run_sync('The export button does nothing when I click it.')
print(result.output)
#> urgent=False
assert result.response.provider_details is not None
print(result.response.provider_details['route']['probabilities'])
#> {'Ticket': 1.0, 'Escalation': 0.0}

result = agent.run_sync('Can you recommend a good restaurant near your office?')
print(result.response.model_name)
#> claude-opus-5-5
```

Jev is sure the broken button is a ticket, so its pick stands. A restaurant recommendation is neither, and Jev gave `Ticket` only about 0.6, below the bar, so the language model took the step.

If you set `decision_route_threshold`, put a `FallbackModel` behind the decision model, or `agent.run` raises `UnsureRoute`. You can also catch it yourself, to send the text to a review queue, say: it carries the picked `route`, its `probability`, the `probabilities` of every route, and the `threshold` it fell below. The one route left when every other has returned this turn, and a single output type with nothing else on offer, are taken without a pick, so there is nothing to be unsure of and the threshold does not apply to them.

0.7 is a reasonable place to start. A higher threshold hands off more steps and gets more of the rest right, so tune it on your own data: [below](#tuning-a-threshold-on-your-own-data) is how, without paying for a run per setting. It is not a guard for a tool with side effects, such as a refund or an account change: [require approval](../deferred-tools.md#human-in-the-loop-tool-approval) for that tool instead.

This makes up for routes that do not cover the texts they will see. The better fix is to [give every outcome a route](#give-every-outcome-a-route): a route the decision model cannot fill, such as the support desk's `Reply`, lets the text that fits nothing else pick it and go to the language model on the pick.

Watch how often the fallback fires, not only how accurate the pair is. A chain that hands off nearly everything is accurate and costs full price, and the rate is the only number that shows it.

### Tuning a threshold on your own data

A decision model returns its probabilities with every answer, and Jev gives nearly the same ones for the same text every time. So you can run a labelled set of texts once, with no threshold set, keep the probabilities from each response, and try as many thresholds as you like afterwards, without another request. Each one trades how many steps are handed off against how many of the rest are right:

```python {title="tune_route_threshold.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


class Escalation(BaseModel):
    """Hand the ticket to a human specialist: security, privacy or legal problems."""

    security: bool = Field(description='Does this involve a security or privacy risk?')


agent = Agent('typesafe:jev-latest', output_type=[Ticket, Escalation])

# Each text, with the route a person on the team would take for it.
labelled = [
    ('The export button does nothing when I click it.', 'Ticket'),
    ('Someone else can see my invoices when they log in.', 'Escalation'),
    ('Our lawyer asked for a copy of your data processing agreement.', 'Escalation'),
    ('My colleague left the company last week. How do I remove her from our workspace?', 'Ticket'),
    ('Two-factor codes stopped arriving on my phone.', 'Ticket'),
    ('I shared a board by mistake. How do I make it private again?', 'Ticket'),
    ('The CSV export includes columns I had hidden.', 'Ticket'),
]

picks: list[tuple[str, float, str]] = []
for text, expected in labelled:
    result = agent.run_sync(text)
    assert result.response.provider_details is not None
    route = result.response.provider_details['route']
    picks.append((route['choice'], route['probabilities'][route['choice']], expected))

for threshold in (0.5, 0.8, 0.9):
    kept = [(choice, expected) for choice, probability, expected in picks if probability >= threshold]
    right = sum(choice == expected for choice, expected in kept)
    print(f'{threshold}: {len(picks) - len(kept)} handed off, {right} of {len(kept)} kept right')
    #> 0.5: 0 handed off, 5 of 7 kept right
    #> 0.8: 2 handed off, 5 of 5 kept right
    #> 0.9: 4 handed off, 3 of 3 kept right
```

Jev sent two ordinary product tickets that mention privacy to `Escalation`, at 0.6 and 0.67. A bar of 0.8 hands both of them on and keeps every pick it had right; 0.9 hands on two more that it had right as well, at 0.84 and 0.88. The numbers move by a few hundredths from one run to the next, so a bar is a range to choose from rather than a point.

The same works for [`decision_boolean_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_boolean_threshold]: declare the field as a `float` bounded with `ge=0` and `le=1` while you tune, which returns the probability of yes itself, and compare it with each bar. Tune on a few hundred texts rather than a handful, drawn from the traffic the agent will see, and tune again when you change the model version, the routes or their docstrings. Your production traces carry the same numbers on each decision model request, so the texts you label later can come from there.

## Judging a conversation

A run's message history goes to the model as `history`: user prompts, answers, thinking, tool calls and their results, and retry prompts, from whichever model produced them. With no new prompt, the conversation is the whole state, so a decision model agent given another agent's messages judges that run — and it is the run being judged, so there is nothing to put in the prompt:

```python
from pydantic_ai import Agent

assistant = Agent('openai:gpt-5.6-sol')
judge = Agent('typesafe:jev-latest', output_type=bool, instructions='Was the assistant polite?')

conversation = assistant.run_sync('hello')
result = judge.run_sync(message_history=conversation.all_messages())
print(result.output)
#> True
```

A new prompt on top of a history is judged as `text` beside it; the latest prompt with no history before it is the whole state, as plain text. Either way the conversation in the history goes to the backend — system prompts, tool arguments and tool results included, though a `CachePoint` is left out and a file is refused — so trim it to what the question is about: `message_history=conversation.all_messages()[-4:]`, a [history processor](../message-history.md#processing-message-history), or a compaction capability, which works on a decision model agent as on any other. A summary it writes goes along as a `summary` entry when it is a [`CompactionPart`][pydantic_ai.messages.CompactionPart], or as a `system` entry when it was written as a system prompt, which the [harness](https://github.com/pydantic/pydantic-ai-harness)'s compaction does.

A model's thinking goes along as a `thinking` entry, where it was in the response, so a question can be about the reasoning itself, such as whether the model considered getting around its tests. Thinking a provider returned only in encrypted form, as a [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] with a `signature` and no text, has nothing to judge and is left out.

Accuracy falls as the state grows with detail the question does not need, and a backend has a limit on how large the state can be; past it the request fails with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError], which a `FallbackModel` hands to the model behind it like any API error, so an over-long conversation quietly becomes a language model call. Compact earlier than a language model would need, since the decision model is being asked to *judge* the whole of it, not to continue from it. Jev's limits are on the [TypeSafe page](typesafe.md#limits).

!!! note "A system prompt is judged, not asked"
    A decision model is told what a conversation said and asked what the agent's `instructions` ask. A
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is part of what was said, so it joins the state as
    a `system` entry rather than becoming part of the question — including the agent's own `system_prompt=`.
    Nothing on the part says who wrote it, so treating any of them as an instruction would mean that judging
    another agent's run folded that agent's persona into the question. Give a decision model agent its question
    through `instructions=`.

## Streaming

A decision model answers in one piece, so there is nothing to stream, and nothing that stops working: `run_stream`, an `event_stream_handler`, and the AG-UI and Vercel AI adapters get the whole answer as a single event. There are no partial results and no earlier first token — it is compatibility, not streaming.

## Decision models inside an agent run

Everything above asks a decision model a question and uses the answer. The same question is worth as much *inside* a run as outside one: a decision that sits between the expensive steps — which model answers, whether a call should run, which tools are worth offering — is a classification, and a decision model fast enough for a real-time request path makes it cheap enough to ask every time rather than once at the top.

Each of these is a [capability](../capabilities/overview.md) hook, and none of them is specific to decision models: they take any model, and a language model will do the same job more slowly and more expensively. What a decision model changes is that the decision stops being something you ration.

### Classify, then act

The simplest shape is one run. An [output function](../output.md#output-functions) makes the decision a signature rather than a string to map afterwards — and because the function *runs* on the model's pick, it can do the work it routed to, so the router's result is the answer:

```python {title="route_to_a_model.py"}
from enum import Enum

from pydantic_ai import Agent, RunContext, UseEnumMemberDocstrings

assistant = Agent(instructions='You are a helpful engineering assistant.')


class Tier(UseEnumMemberDocstrings, str, Enum):
    fast = 'fast'
    """A lookup, an extraction, or a change confined to one place."""

    capable = 'capable'
    """Architecture, security, or a decision that is expensive to get wrong."""


async def route(ctx: RunContext, tier: Tier) -> str:
    """Answer the question on a model suited to it.

    Args:
        tier: Which model should answer this?
    """
    model = 'openai:gpt-5.6-sol' if tier is Tier.capable else 'openai:gpt-5.6-luna'
    return (await assistant.run(ctx.prompt, model=model)).output


router = Agent('typesafe:jev-latest', output_type=route)


async def main():
    result = await router.run('How do I centre a div?')
    print(result.output)
    #> Give the container `display: flex` and both `place-items: center`.
```

The decision model fills `tier` and the framework calls `route`, which runs the assistant and returns its answer, so one `router.run(...)` is the whole thing. The question itself is not an argument: it is already the text being judged, and [`ctx.prompt`][pydantic_ai.tools.RunContext.prompt] hands the same text to the function, so `tier` is the only question asked and the routing costs one request and no extra plumbing. A `str` parameter would not work here in any case — it is not [a type a decision model can fill](#supported-field-types), and an agent asking for one is refused before a request is sent.

The argument's `Enum` becomes the pick-one question: its `Args:` entry is the question, the function's summary line what the run is for, and each member's docstring what that option means — the [same split](#where-the-wording-comes-from) as an output field's, with the meanings on the options rather than folded into the question. The pick's confidence is in `provider_details['confidence']`, so an unsure route can go to the capable model rather than the cheap one, which is the conservative direction when a wrong route is expensive.

### Decide again on every step

A run is not one decision. [`SelectModel`][pydantic_ai.capabilities.SelectModel] is evaluated before each step, so the same question can be asked of the conversation as it stands rather than once up front — a conversation that starts simple and turns hard moves up when it turns:

```python {title="select_the_model_per_step.py"}
from enum import Enum

from pydantic_ai import Agent, ModelSelectionContext, UseEnumMemberDocstrings
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models import Model, infer_model

fast = infer_model('openai:gpt-5.6-luna')
capable = infer_model('openai:gpt-5.6-sol')


class Tier(UseEnumMemberDocstrings, str, Enum):
    fast = 'fast'
    """A lookup, or a change confined to one place."""

    capable = 'capable'
    """Architecture, security, or a decision that is expensive to get wrong."""


router = Agent(
    'typesafe:jev-latest',
    output_type=Tier,
    instructions='Which model should take the next step of this conversation?',
)


async def select_model(ctx: ModelSelectionContext) -> Model:
    picked = await router.run(message_history=ctx.messages)
    return capable if picked.output is Tier.capable else fast


agent = Agent(capabilities=[SelectModel(select_model)])


async def main():
    simple = await agent.run('What does this repo do?')
    print(simple.response.model_name)
    #> gpt-5.6-luna
    hard = await agent.run(
        'Now redesign its auth layer.', message_history=simple.all_messages()
    )
    print(hard.response.model_name)
    #> gpt-5.6-sol
    print(hard.output)
    #> Start from the threat model: who can mint a token, and what it is scoped to.
```

The selector returns a [`Model`][pydantic_ai.models.Model] here, but a model ID string is equally fine — anything `Agent(model=...)` takes. Returning an instance lets each candidate be built once, with whatever provider or [settings](overview.md#per-model-settings) it needs, instead of being inferred again every step.

The router is given the messages the selected model will be sent, which are the whole state it reads. They end with the request being routed — the user's question on a run's first step, tool results on a later one — so every step, the first included, is routed on what the model is about to answer. What the selected model will add to that request, its instructions and a fresh run's system prompt, isn't there yet.

Asking on every step is only affordable because the question is cheap; with a language model in the selector, the routing costs as much as the work it routes.

A router that reads the history has the same problem every agent does: the history grows. The state is the whole history, so a long run makes each routing question larger and slower, and eventually the input is dominated by turns that no longer bear on which model should take the next step. Pair this with [compaction](../capabilities/compaction.md) rather than letting it grow — the compacted history is what the router reads, which is usually what you wanted it to read anyway.

### Judge a tool call before it runs

A [hook](../hooks.md) on tool execution sees every call the model makes, with its arguments already validated, and can stop one before its body runs. That is a decision per call, which is the shape a decision model answers:

```python {title="judge_a_tool_call.py"}
import json

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext, SkipToolExecution, ToolDefinition
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    irreversible: bool = Field(
        description='Would running this destroy data or leak secrets?'
    )


judge = Agent('typesafe:jev-latest', output_type=Handling)


async def judge_tool_call(
    ctx: RunContext,
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
    args: dict[str, object],
) -> dict[str, object]:
    verdict = await judge.run(json.dumps({'tool': tool_def.name, 'args': args}))
    if verdict.output.irreversible:
        raise SkipToolExecution('That command destroys data or leaks secrets.')
    return args


agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[Hooks(before_tool_execute=judge_tool_call)],
)


@agent.tool_plain
def run_shell(command: str) -> str:
    return f'ran {command!r}'


async def main():
    result = await agent.run('Clear out the build directory.')
    print(result.output)
    """
    I did not run that: it destroys data or leaks secrets. Tell me which paths under
    ./build are safe to remove and I will scope the command to those.
    """
```

[`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution] stops the call and sends its message back as the tool's result, so the model learns what was refused and can try something else. Nothing is marked `requires_approval` and no tool opts in, so the hook sits on every *function tool* the agent can call, including ones added later.

!!! warning "Output functions and actions do not fire tool hooks"
    An [output function](../output.md#output-functions) is an internal tool, and tool-execution hooks are
    deliberately not run for it — the same way `prepare_tools` and toolset wrappers exclude output tools. So a
    guard written this way does not see an output function, nor the actions of a `Choices` set
    [built at run time](#choose-from-a-set-built-at-run-time) further down this page, which are called as output
    too. Put the side effect in a function tool if it needs to pass this guard, or validate it inside the output
    function or action itself.

The alternative is [deferred tools](../deferred-tools.md): mark a tool `requires_approval=True` and resolve the approval request with [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]. Use that when the decision has to leave the process — a person approving in another system, a queue, a run that is resumed later. Use the hook when the decision is made in-process, as it is here. Both see validated arguments; only the deferral can outlive the run.

The call goes to the judge as JSON, so the text it judges reads as structured data, with the tool and its arguments named, rather than as a Python repr. Those arguments go to the decision model's backend before the verdict comes back, so a call is disclosed to a third party even when it is then refused. Send the judge what it needs to decide — the tool name and the fields that bear on safety — rather than the whole argument dict, when those arguments can carry credentials or customer data.

This judges the call the model proposed, not the model's intent, so it is a check on what is about to happen rather than on what was said. Keep a human in the loop for the calls that matter most: a judgement this cheap is one you can afford to run on everything, which is exactly why it should not be the only thing standing between an agent and an irreversible action. For a guard the two mistakes rarely cost the same — a missed irreversible command costs more than a second look at a safe one — so set [what `True` has to mean](#what-true-has-to-mean) accordingly.

### Choose from a set built at run time

The examples above name their options in the source. When the options are only known once the run is under way — the actions available on the screen in front of an agent, the records a search returned — build a [`Choices`](#options-known-only-at-run-time) set at that point and pass it to the run as its output type. Each option is one candidate, described where it is built, and a [`Choice`][pydantic_ai.output.Choice] whose value is a callable makes it an action: **the one the model picks is the one that runs**:

```python {title="choose_a_candidate.py"}
from functools import partial

from pydantic_ai import Agent, Choice, Choices

agent = Agent('typesafe:jev-latest')


class Screen:
    """Whatever the agent is acting on."""

    def click(self, target: str) -> str:
        return f'clicked {target}'

    def observe(self) -> str:
        return 'a fresh look at the screen'


def candidates(screen: Screen, targets: dict[str, str]) -> type[str]:
    """One choice per available action, plus the two ways to decline."""
    actions = {
        target: Choice(description, value=partial(screen.click, target))
        for target, description in targets.items()
    }
    declines = {
        'reobserve': Choice('Look again before deciding.', value=screen.observe),
        'abstain': Choice(
            'Do nothing, because none of these is safe for what was observed.',
            value=lambda: 'did nothing',
        ),
    }
    if clashing := actions.keys() & declines.keys():
        raise ValueError(f'action IDs clash with the reserved ones: {sorted(clashing)}')
    return Choices(actions | declines, description='Which action should the agent take next?')


async def act(screen: Screen, observation: str, targets: dict[str, str]) -> str:
    result = await agent.run(observation, output_type=candidates(screen, targets))
    return result.output


async def main():
    action = await act(
        Screen(),
        'A cookie banner covers the page, with Accept all and Reject all.',
        {
            'accept_all': 'Accept every cookie.',
            'reject_all': 'Reject every optional cookie.',
        },
    )
    print(action)
    #> clicked reject_all
```

The key idea is that a candidate is **the action itself**, not a token standing for it. The model picks, the framework calls that option's value, and `result.output` is what the action returned — so there is no dispatch table to write and no second step where an ID is turned back into behaviour. An action is called with no arguments, so bind what it needs with `functools.partial` or a closure; it can be `async`, and raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] to ask for another pick. If you find yourself writing a function that returns its own name, the dispatch has just moved somewhere else; give the function the work instead.

Two things this gets right that are easy to lose. A decision model can only answer with an option it was given, so there is no step where a made-up action has to be validated away. And `reobserve` and `abstain` are options like any other, so declining is something the model can *choose* rather than something you infer from a low confidence — the difference between an agent that stops and one that acts on a coin flip.

A backend may cap how many options one question can carry, and the two reserved ones count: on Jev, which picks from [at most 255](typesafe.md#limits), a set built at run time needs a ceiling of 253 candidates and a plan for what to do above it — rank and offer the best few, or narrow by some cheaper filter first. An observation that produces hundreds of equally plausible actions is usually a sign the candidates are too fine-grained, not that the limit is too low.

`provider_details['probabilities']` carries the probability of every candidate, which is what to watch: a decision loop that abstains on most steps, or spreads its probability evenly, is telling you the candidates are not distinguishable by their descriptions.

### The same shape elsewhere

Any hook that takes a decision rather than a generation fits this way. [`PrepareTools`][pydantic_ai.capabilities.PrepareTools] can ask which of a large toolset this request calls for before the tools go on the wire; a [history processor](../capabilities/process-history.md) can ask which parts of a long conversation still matter before it is compacted. Both are classifications over text, both run on every step, and both are questions you would not ask a language model on every step.

Two things to hold on to. A classifier in the loop is a component like any other, so it needs the same measurement as the classifier you would deploy on its own — a router that is right 80% of the time sends one request in five to the wrong model, and nothing in the run will tell you. And text written to steer the answer can move a decision model — [Jev's list](typesafe.md#what-jev-answers-badly) includes it — so a guard built this way belongs alongside deterministic checks, not instead of them.

## What decision models cannot do

A decision model does not write text or read files, and it only fills tool arguments that map to the [typed questions](#supported-field-types) above. Its model profile records the first of those as [`supports_text_output=False`][pydantic_ai.profiles.ModelProfile.supports_text_output], and an agent that needs text output or files is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent:

- The `output_type` must be made of the field types above, or be [handed off](#escalating-to-a-language-model) when it is picked from beside other routes: no `str`, no [`NativeOutput`][pydantic_ai.output.NativeOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput]. An [output function](../output.md#output-functions)'s arguments are fields like any other, so they are subject to the same list, and one that takes nothing but the run context is a [hand-off](#tools-pick-then-fill) picked without filling anything. A [union](#a-union-of-output-types) of structured types is supported; a union of structured types as a *field* of an output type is not.
- No native tools. A function tool is offered to the model; supported arguments are [filled after it is picked](#tools-pick-then-fill), while any unsupported argument makes the pick an `UnfillableRoute` after the request rather than a refusal before it. With tools attached, the output type needs a docstring or the agent instructions to be weighed against them.
- No image, audio, video or document in the prompt or the history.
- A pick-one needs two or more options, each a string or a whole number: one option leaves nothing to pick, and `True` is not a label.
- No more options in one question than the backend's `max_choice_options`, where it sets one. A pick-one field counts its own options, and the route question counts every tool plus every output type, so on Jev 255 tools is already one too many once the output type is counted beside them.
- The model needs something to ask. A run with no user text and no history has nothing to judge, and an `output_type` with no fields to fill — a lone argumentless output function — leaves no question to ask unless there is more than one route to pick between.

A decision model does not revise an answer the way a language model does. Its previous answer and the validator's complaint both go back in the history, so they are part of what it judges, but the question is unchanged and a confident answer does not move: an output validator that raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] usually gets the same answer again, and one that keeps rejecting runs the agent out of retries.

## Implementing a decision model

Any backend that answers the three kinds of question can be a decision model. Subclass [`DecisionModel`][pydantic_ai.models.decision.DecisionModel] and implement [`decide`][pydantic_ai.models.decision.DecisionModel.decide], which sends one [`DecisionRequest`][pydantic_ai.models.decision.DecisionRequest] and returns a [`DecisionResponse`][pydantic_ai.models.decision.DecisionResponse]; the base class does everything else on this page. A `DecisionModel` is a [`Model`][pydantic_ai.models.Model], so it also needs `model_name`, `system` and `base_url`, like any [custom model](overview.md#custom-models).

This one has no opinion at all: every option is equally likely. Replace the body of `decide` with a call to your backend:

```python {title="undecided_model.py"}
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.decision import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionAnswer,
    DecisionModel,
    DecisionModelSettings,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
)


class UndecidedModel(DecisionModel[None]):
    max_choice_options = 50
    max_score_levels = 5

    @property
    def model_name(self) -> str:
        return 'undecided'

    @property
    def system(self) -> str:
        return 'example'

    @property
    def base_url(self) -> str:
        return 'https://decisions.example.com'

    async def decide(
        self, request: DecisionRequest, model_settings: DecisionModelSettings
    ) -> DecisionResponse:
        answers: dict[str, DecisionAnswer] = {}
        for name, question in request.questions.items():
            if isinstance(question, NoulQuestion):
                answers[name] = NoulAnswer(noul=0.5)
            elif isinstance(question, ChoiceQuestion):
                options = list(question.criteria)
                answers[name] = ChoiceAnswer(
                    choice=options[0],
                    confidence=0.0,
                    probabilities={option: 1 / len(options) for option in options},
                )
            else:
                levels = range(len(question.criteria))
                answers[name] = ScoreAnswer(
                    score=(len(levels) - 1) / 2,
                    confidence=0.0,
                    probabilities={level: 1 / len(levels) for level in levels},
                )
        return DecisionResponse(answers=answers, model_name=self.model_name)


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')
    area: Literal['billing', 'bug'] = Field(description='Which team owns it?')


agent = Agent(UndecidedModel(), output_type=Ticket)
result = agent.run_sync('My invoice lists a plan I never signed up for.')
print(result.output)
#> urgent=True area='billing'
assert result.response.provider_details is not None
print(result.response.provider_details['confidence'])
#> {'urgent': 0.0, 'area': 0.0}
```

A probability of 0.5 lands exactly on the default threshold, so `urgent` comes back `True` with a confidence of 0: an undecided answer reads as one.

What `decide` receives and owes:

- `request.state` is the text to judge as a `str`, or a dict with the conversation under `history` and the latest prompt under `text`, as [above](#judging-a-conversation). Each question's `instructions` is likewise a `str` when there is one thing to say, or a dict of labelled parts (`field`, `question`, `goal`, `instructions`, `option` on each yes/no a `list` or `dict` of options fans out to, and `chosen` on the second request of a [pick, then fill](#tools-pick-then-fill)).
- Every question in `request.questions` needs an answer of the matching kind under the same name: a [`NoulAnswer`][pydantic_ai.models.decision.NoulAnswer] with the probability of yes, a [`ChoiceAnswer`][pydantic_ai.models.decision.ChoiceAnswer] with a probability for every option in `criteria`, or a [`ScoreAnswer`][pydantic_ai.models.decision.ScoreAnswer] with a position along the levels. A missing or mismatched answer is an [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].
- Probabilities are read as probabilities, and the [thresholds](#confidence-and-thresholds) and reported confidence assume they are calibrated. A backend whose scores are not should calibrate them before returning them, or document that its bars need tuning of their own.
- Report the tokens a request used on `DecisionResponse.usage`, and the model version that answered on `model_name`.
- Raise [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] or [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] when the backend fails, so a [`FallbackModel`](overview.md#fallback-model) can take over, and [`UserError`][pydantic_ai.exceptions.UserError] when a request cannot be sent as given. Forward `timeout`, `extra_headers` and `extra_body` from `model_settings` where the backend supports them.

Set [`max_choice_options`][pydantic_ai.models.decision.DecisionModel.max_choice_options] and [`max_score_levels`][pydantic_ai.models.decision.DecisionModel.max_score_levels] to the backend's limits, so an agent over them is refused before a request is sent rather than by the backend after it. Leave them `None` when the backend has none.

What a question costs is the backend's business, and it is not the same everywhere. `decide` is called once per request with every question at once, but a backend that evaluates each option of a pick-one separately pays per option, where one that answers a whole request together pays once. Keep that in mind when you choose the limits, and document it for your users, since the same output type can be cheap on one backend and expensive on another.
