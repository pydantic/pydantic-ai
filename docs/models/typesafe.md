# TypeSafe (Jev)

[Jev](https://typesafe.ai) is TypeSafe's model, and a [decision model](decision.md): it answers typed questions about a text, each with a probability or a distribution over the options, rather than writing text. In Pydantic AI, an agent running on a decision model can use it both to produce a structured [output](../output.md) and to call [tools](../tools.md).

[`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] is the Pydantic AI model class for Jev, and a subclass of [`DecisionModel`][pydantic_ai.models.decision.DecisionModel]. This page covers what is specific to Jev and TypeSafe: setup, Jev's limits, and what it answers badly.

!!! tip "Start with Decision models"
    **[Decision models](decision.md) is where to learn how an agent's output types, tools and message history map
    onto Jev's questions**, and what the answers mean, with a [worked example](decision.md#a-support-desk-end-to-end)
    that combines them and escalates to a language model where Jev cannot answer.

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

You can then use `TypeSafeModel` by name, as `typesafe:jev-latest`, or initialise the model directly with just the model name:

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

## Example

Give an agent an output type, and Jev answers every field of it in one request, each with its own confidence:

```python {title="triage_with_jev.py"}
from enum import Enum
from typing import Annotated, Literal

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

    app: Literal['web', 'ios', 'android'] | None
    """Which app is it about?"""


agent = Agent('typesafe:jev-latest', output_type=Ticket)
result = agent.run_sync(
    'The timeline on my Android phone has been blank since the update this morning, '
    'and my standup is in ten minutes.'
)
print(result.output)
#> area=<Area.bug: 'bug'> urgent=True app='android'
assert result.response.provider_details is not None
print(result.response.provider_details['confidence'])
#> {'area': 1.0, 'urgent': 0.24, 'app': 0.81}
```

The prompt is only the ticket; the questions are on the output type: its docstring is the goal, the docstring under each field is that field's question, and the `Enum` members' docstrings and the [`BoolCriteria`][pydantic_ai.output.BoolCriteria] say what each answer means. Jev is sure of the team and the app, and much less sure the ticket is urgent, which is the answer to send to a person or to [a language model behind it](decision.md#falling-back-on-low-confidence). Every answer is a value of the type, so there is no text to parse and no answer outside the options.

## Costs and limits {#limits}

Jev answers the questions in one request in parallel, so asking several costs little more than asking one: a field you only need on some inputs costs tokens rather than time.

What one request can carry is limited, and `TypeSafeModel` keeps to the first two before a request is sent:

- **255 options in one pick-one question.** A pick-one field counts its own options, and the [route question](decision.md#routes-which-thing-to-do) counts every tool plus every output type, so 255 tools is already one too many once the output type is counted beside them. A 256th option is a 400 from the API, so a question over it is refused with a [`UserError`][pydantic_ai.exceptions.UserError] instead.
- **10 levels in one rubric.** An 11th is a 400 from the API, so eleven or more whole numbers from 0 are not a rubric, and are [asked as a pick-one](decision.md#what-each-field-type-does) instead.
- **32k tokens** for the state plus the longest question on `jev-1.13`, and 64k for the state and every question together. The state is counted once per request, so the 32k is the limit a long conversation reaches; the 64k only binds when the questions themselves are very large. Past it the request fails with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] (`max_tokens_exceeded`), which a `FallbackModel` hands to the model behind Jev like any API error. See [keeping a conversation under the limit](#compaction).

### Keeping a conversation under the limit {#compaction}

The whole message history is the state, so each turn adds to every request after it, and tool results, which go along whole, fill the budget fastest. Once a conversation is over the limit, every later turn fails on Jev, so behind a `FallbackModel` every one of them goes to the language model until the history is compacted.

`TypeSafeModel`'s [`context_window`][pydantic_ai.profiles.ModelProfile.context_window] is the 32k limit, so [`ctx.context_window_used`][pydantic_ai.tools.RunContext.context_window_used] measures against the limit that binds, and the [compact when the context window fills](../message-history.md#compact-when-the-context-window-fills) processor works on a Jev agent unchanged. A `FallbackModel` reports the smallest window among its models, which is Jev's:

```python {title="jev_compaction.py" requires="compact_when_window_fills.py"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory, ReinjectSystemPrompt
from pydantic_ai.models.fallback import FallbackModel

from compact_when_window_fills import compact_when_window_fills

model = FallbackModel('typesafe:jev-latest', 'openai:gpt-5.6-sol')
print(model.context_window)
#> 32000

agent = Agent(
    model,
    output_type=bool,
    instructions='Does the customer want a refund?',
    capabilities=[ProcessHistory(compact_when_window_fills), ReinjectSystemPrompt()],
)
```

To keep what the dropped turns said, summarize them instead, with a history processor like [summarize old messages](../message-history.md#summarize-old-messages) or the [harness](https://pydantic.dev/docs/ai/harness/compaction/)'s `SummarizingCompaction`. Either way the summary has to be written by a language model: Jev does not write text, so pass `SummarizingCompaction` a language model as `model=` rather than letting it default to the agent's. Its trigger estimates tokens at about four characters each, which undercounts the JSON Jev is sent by about a quarter, so give it a `max_tokens=` well under 32k, such as `20_000`, rather than a fraction of the window.

## What Jev answers badly

Everything below returns an answer rather than an error, which is what makes it worth knowing.

### Published by TypeSafe

TypeSafe publish these per model version, on their [jaggedness page for `jev-1.13`](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and revise them as models change. The page also warns against hiding several judgements in one question, which TypeSafe's guide calls "probably the most important concept": see [ask one thing per field](decision.md#ask-one-thing-per-field).

- **Literal reading.** Jev "answers the question you wrote, not the one you meant": scoping words, negations and implied conditions are read at face value. State the exact condition in the question, and put boundary cases in a `bool` field's [criteria](decision.md#what-each-field-type-does) or in the descriptions of a pick-one's options.
- **Arithmetic, counting and dates.** Jev is not a calculator, does not count reliably, and reads dates as text rather than as ordered quantities. Compute these in Python and ask Jev about the result.
- **Indirection.** A question about a property of a property, or one needing several hops, costs accuracy. Ask as directly as you can, and [name the part of the state](decision.md#where-the-wording-comes-from) the question is about.
- **Context it does not need.** Accuracy falls as the state grows with detail unrelated to the question, so filter before you send rather than after, and compact a long conversation before judging it.
- **Adversarial text.** Jev treats the state as data, not as hostile: text written to steer the answer — an injected instruction, a misleading framing, an argument for its own classification — can move it. TypeSafe say they expect to improve this. A guard built on Jev belongs alongside deterministic checks, not instead of them, and is worth testing against your own adversarial inputs.
- **Contradictory question and criteria.** Where a field's question and its criteria ask for different things, such as a [`BoolCriteria`][pydantic_ai.output.BoolCriteria] whose `true` describes a no, Jev answers worse. Write the criteria as an extension of the question.
- **Structural invariants.** A question and its negation need not add up to one, and the same question asked as a yes/no and as a pick-one gives numbers that do not compare. Ask each decision one way, enforce identities in code, and [tune each bar](decision.md#confidence-and-thresholds) on the kind of question it gates.
- **Generation.** Jev does not write text, so a route with a `str` field [escalates to a language model](decision.md#escalating-to-a-language-model).

### Observed in Pydantic AI

These come from running Jev behind Pydantic AI, not from TypeSafe.

- **A tool call that repeats.** With a tool's call and result in the history, the text usually still calls for it, so Jev picks it again. A tool is therefore not offered again once its result is in the turn, and comes back on offer at the next prompt; unsupported arguments are proposed to the model behind Jev, which decides. Put a `UsageLimits(request_limit=...)` on a Jev agent with tools all the same, as on any agent that loops.
- **Deciding what it cannot see.** A tool that needs an argument the text does not state — a refund amount, a date — is one Jev will propose and a language model may decline to call; the two judge the same option differently, and language models disagree with each other on such picks about as often. Compare Jev with the model behind it on your own tickets before trusting either's hand-off rate.
- **A question about the question.** Asked whether it *can* answer, rather than what the text calls for, Jev hands off nearly everything, which is why the output type is offered as an action, [described by its docstring](decision.md#routes-which-thing-to-do).

!!! note "Measure on your own data"
    Measure accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.

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

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request. [`TypeSafeModelSettings`][pydantic_ai.models.typesafe.TypeSafeModelSettings] adds the two [thresholds](decision.md#confidence-and-thresholds) every decision model has, `decision_boolean_threshold` and `decision_tool_call_threshold`.

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
