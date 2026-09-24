# TypeSafe (Jev)

[Jev](https://typesafe.ai) is TypeSafe's model, and a [decision model](decision.md): it answers typed questions about a text, each with a probability or a distribution over the options, rather than writing text. In Pydantic AI, an agent running on a decision model can use it both to produce a structured [output](../output.md) and to call [tools](../tools.md).

[`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] is the Pydantic AI model class for Jev, and a subclass of [`DecisionModel`][pydantic_ai.models.decision.DecisionModel]. This page covers what is specific to Jev and TypeSafe: setup, Jev's limits, and what it answers badly.

!!! tip "Start with Decision models"
    **[Decision models](decision.md) is where to learn how an agent's output types, tools and message history map
    onto Jev's questions**, and what the answers mean, with a [worked example](decision.md#a-support-desk-end-to-end)
    that combines them and escalates to a language model where Jev cannot answer.

Give an agent an output type, and Jev answers every field of it in one request, each with its own confidence:

```python {title="triage_with_jev.py"}
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent, BoolCriteria, UseEnumMemberDocstrings


class Area(UseEnumMemberDocstrings, str, Enum):
    """The team that owns the ticket."""

    billing = 'billing'
    """Charges, invoices, plans and payment methods."""

    bug = 'bug'
    """Part of the product does not work as it should."""

    account = 'account'
    """Logging in, access, and account settings."""


class Ticket(BaseModel):
    """Triage a support ticket."""

    area: Area = Field(description='Which team owns this ticket?')
    urgent: Annotated[
        bool,
        BoolCriteria(
            true='The customer is losing money or has a deadline today.',
            false='It can wait its turn in the queue.',
        ),
    ]
    app: Literal['web', 'ios', 'android'] | None = Field(description='Which app is it about?')


agent = Agent('typesafe:jev-latest', output_type=Ticket)
result = agent.run_sync(
    'The timeline on my Android phone has been blank since the update this morning, '
    'and my standup is in ten minutes.'
)
print(result.output)
#> area=<Area.bug: 'bug'> urgent=True app='android'
assert result.response.provider_details is not None
print(result.response.provider_details['confidence'])
#> {'area': 1.0, 'urgent': 0.46, 'app': 0.85}
```

The prompt is only the ticket; the questions are on the output type, in its field descriptions, its `Enum` members' docstrings and the [`BoolCriteria`][pydantic_ai.output.BoolCriteria] saying what counts as urgent. Jev is sure of the team and the app, and much less sure the ticket is urgent, which is the answer to send to a person or to [a language model behind it](decision.md#falling-back-on-low-confidence). Every answer is a value of the type, so there is no text to parse and no answer outside the options.

## Install

To use `TypeSafeModel`, install `pydantic-ai-slim` (or `pydantic-ai`) with the `typesafe` optional group:

```bash
pip/uv-add "pydantic-ai-slim[typesafe]"
```

## Configuration

To use Jev through the [TypeSafe](https://typesafe.ai) API, get an API key from your TypeSafe account and set it as an environment variable:

```bash
export TYPESAFE_API_KEY='your-api-key'
```

You can then use `TypeSafeModel` by name, as `typesafe:jev-latest` in the example above, or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

### Model names

`jev-latest` and `jev-preview` are aliases that move when TypeSafe ship a release; `jev-preview` runs ahead when there is a preview build. A versioned id is accepted too, whether or not it is listed:

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-1.13.0', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

[`ModelResponse.model_name`][pydantic_ai.messages.ModelResponse.model_name] reports the versioned id that answered, so a run logged against `jev-latest` still records which model produced it. Because `jev-latest` moves, a new release can shift the numbers under a [threshold](decision.md#confidence-and-thresholds) you have tuned; once you have tuned one, pin the version it was tuned against (`typesafe:jev-1.13.0`) and move deliberately.

## Costs and limits {#limits}

Jev answers the questions in one request in parallel, so asking several costs little more than asking one: a field you only need on some inputs costs tokens rather than time.

What one request can carry is limited, and `TypeSafeModel` keeps to the first two before a request is sent:

- **255 options in one pick-one question.** A pick-one field counts its own options, and the [route question](decision.md#routes-which-thing-to-do) counts every tool plus every output type, so 255 tools is already one too many once the output type is counted beside them. A 256th option is a 400 from the API, so a question over it is refused with a [`UserError`][pydantic_ai.exceptions.UserError] instead.
- **10 levels in one rubric.** An 11th is a 400 from the API, so eleven or more whole numbers from 0 are not a rubric, and are [asked as a pick-one](decision.md#what-each-field-type-does) instead.
- **64k tokens** for the state and questions together on `jev-1.13`, with 32k for the state plus the longest question. Past that the request fails with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] (`max_tokens_exceeded`), which a `FallbackModel` hands to the model behind Jev like any API error, so an over-long conversation quietly becomes a language model call. [Compact](decision.md#judging-a-conversation) earlier than a language model would need.

## What Jev answers badly

Everything below returns an answer rather than an error, which is what makes it worth knowing. TypeSafe publish these per model version, on their [jaggedness page for `jev-1.13`](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and revise them as models change. TypeSafe's own guide also calls asking [one thing per field](decision.md#ask-one-thing-per-field) "probably the most important concept".

- **Arithmetic, counting and dates.** Jev is not a calculator, does not count reliably, and reads dates as text rather than as ordered quantities. Compute these in Python and ask Jev about the result.
- **Several judgements in one question.** See [ask one thing per field](decision.md#ask-one-thing-per-field).
- **Indirection.** A question about a property of a property, or one needing several hops, costs accuracy.
- **Context it does not need.** Accuracy falls as the state grows with detail unrelated to the question, so filter before you send rather than after, and compact a long conversation before judging it.
- **A tool call that repeats.** With a tool's call and result in the history, the text usually still calls for it, so Jev picks it again. A tool is therefore not offered again once its result is in the turn, and comes back on offer at the next prompt; unsupported arguments are proposed to the model behind Jev, which decides. Put a `UsageLimits(request_limit=...)` on a Jev agent with tools all the same, as on any agent that loops.
- **Deciding what it cannot see.** A tool that needs an argument the text does not state — a refund amount, a date — is one Jev will propose and a language model may decline to call; the two judge the same option differently, and language models disagree with each other on such picks about as often. Compare Jev with the model behind it on your own tickets before trusting either's hand-off rate.
- **Adversarial text.** Jev treats the state as data, not as hostile: text written to steer the answer — an injected instruction, a misleading framing, an argument for its own classification — can move it. TypeSafe say they expect to improve this. A guard built on Jev belongs alongside deterministic checks, not instead of them, and is worth testing against your own adversarial inputs.
- **Option order.** The order of a `Literal`'s options or an `Enum`'s members is part of what Jev sees, and reordering them can move the answer. If a classification matters, test it with the options in more than one order.
- **A question about the question.** Asked whether it *can* answer, rather than what the text calls for, Jev hands off nearly everything, which is why the output type is offered as an action, [described by its docstring](decision.md#routes-which-thing-to-do).

!!! note "Measure on your own data"
    Measure accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.

## Asking Jev directly

An `output_type` is the question in almost every case, and it is what makes the same agent run on a language model later. One thing it cannot carry is a state that is a record rather than prose.

The TypeSafe SDK client is on the model for that, configured with the same API key, base URL and HTTP client:

```python {title="ask_jev_directly.py"}
from typesafe_sdk import Choice, JSONValue, Noul, NoulAnswer, NoulCriteria

from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')


async def judge_order(order: dict[str, JSONValue]) -> float:
    response = await model.client.system_one(
        {'order': order, 'policy': 'Refunds are allowed within 30 days.'},
        {
            'refundable': Noul(
                instructions='The order can still be refunded under the policy.',
                criteria=NoulCriteria(
                    true='The order is inside the refund window.',
                    false='The order is outside it, or was refunded already.',
                ),
            ),
            'risk': Choice(
                instructions='How risky is refunding anyway?',
                criteria={'low': None, 'high': 'The customer has prior chargebacks.'},
            ),
        },
        model=model.model_name,
    )
    refundable = response.answers['refundable']
    assert isinstance(refundable, NoulAnswer)
    return refundable.noul
```

This is the one example on this page that is not run by the documentation tests: the call never reaches a
[`Model`][pydantic_ai.models.Model], so there is nothing for the test suite to stand in for, and running it needs
a TypeSafe API key.

Pass `model=` yourself. The client does not know which model the `TypeSafeModel` around it was built with, so without it the SDK falls back to its own default, which `TYPESAFE_DEFAULT_MODEL` can change underneath you.

Nothing else in Pydantic AI sees a call made this way: no agent run, no message history, no usage on a run's total, no fallback to another model, and the span the rest of an agent's work appears under is not opened. It is the escape hatch, not the main road. Reach for it when the question genuinely will not fit an output type, and go back to an `output_type` as soon as it will. Spelling out what counts as a yes and what counts as a no fits one: that is what [`BoolCriteria`](decision.md#what-each-field-type-does) is for.

Passing a record as a mapping rather than as text is a convenience, not an accuracy setting. Jev reads a rendered sentence at least as well as the object it came from, so there is no need to restructure a prompt to get at this.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='your-api-key'))
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

You can also customize the [`TypeSafeProvider`][pydantic_ai.providers.typesafe.TypeSafeProvider] with a custom `http_client`:

```python
from httpx2 import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

custom_http_client = AsyncClient(timeout=30)
model = TypeSafeModel(
    'jev-latest',
    provider=TypeSafeProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

## SDK retries {#sdk-retries}

The TypeSafe SDK retries connection errors, timeouts and retryable HTTP statuses twice by default, with backoff. To change that, build the client yourself and hand it to the provider:

```python
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

client = AsyncTypeSafeClient(api_key='your-api-key', retry=RetryPolicy(max_retries=0))
model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(typesafe_client=client))
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```

See [Provider SDK retries](../retries.md#provider-sdk-retries) for how this interacts with Pydantic AI's own retries.

## Model settings

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request. [`TypeSafeModelSettings`][pydantic_ai.models.typesafe.TypeSafeModelSettings] adds the two [thresholds](decision.md#confidence-and-thresholds) every decision model has, `decision_boolean_threshold` and `decision_tool_call_threshold`. The tool-call default of 0.6 was chosen against Jev: there, its tool picks agree with a frontier model's as often as two frontier models agree with each other.

The former `typesafe_tool_call_threshold` and `typesafe_boolean_threshold` names remain as deprecated aliases.

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(
    model,
    output_type=bool,
    instructions='Is this request harmful?',
    model_settings={'timeout': 5},
)
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
```
