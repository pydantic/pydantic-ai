# TypeSafe (Jev)

[Jev](https://typesafe.ai) is not a language model. You give it a text and typed questions, and it answers each one with a confidence. It does not write text.

[`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] is a [decision model](decision.md): an agent whose job is to decide something runs on Jev like on any other model. Each field of the `output_type` becomes one question, the prompt is the text, and the answers come back as the output. **[Decision models](decision.md) covers how an agent's output type, tools and message history become Jev's questions**, and what the answers mean; this page covers setting Jev up and what is particular to it.

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

You can then use `TypeSafeModel` by name, with the `output_type` Jev should fill:

```python
from enum import Enum

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Verdict(str, Enum):
    """Run it, reject it, or ask a human: reversible work runs, destructive or secret-leaking work is rejected."""

    run = 'run'
    reject = 'reject'
    ask = 'ask'


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    verdict: Verdict
    irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


agent = Agent('typesafe:jev-latest', output_type=Handling)
result = agent.run_sync('rm -rf ./build')
print(result.output)
#> verdict=<Verdict.ask: 'ask'> irreversible=True
```

Or initialise the model directly with just the model name:

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
- **10 levels in one rubric.** An 11th is a 400 from the API, so eleven or more whole numbers from 0 are not a rubric, and are [asked as a pick-one](decision.md#what-each-mapping-does) instead.
- **64k tokens** for the state and questions together on `jev-1.13`, with 32k for the state plus the longest question. Past that the request fails with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] (`max_tokens_exceeded`), which a `FallbackModel` hands to the model behind Jev like any API error, so an over-long conversation quietly becomes a language model call. [Compact](decision.md#judging-a-conversation) earlier than a language model would need.

## Calibration notes

The defaults in Pydantic AI, and the claims on this page about what Jev answers well, come from a small internal set of support tickets: one domain, labelled by the maintainers, and too small to separate models with confidence. They say the mappings work, not how Jev will do on your task.

- **The tool-call threshold.** At the default [`decision_tool_call_threshold`][pydantic_ai.models.decision.DecisionModelSettings.decision_tool_call_threshold] of 0.6, Jev's tool picks agree with a frontier model as often as two frontier models agree with each other; higher takes fewer tools, and is right more often when it does.
- **"None of these".** The explicit option an [optional pick-one field](decision.md#what-each-mapping-does) adds was as accurate as an `other` member written into the options, and more accurate than reading `None` off low confidence.
- **How the route question is asked.** Asked whether it *can* answer, rather than what the text calls for, Jev hands off nearly everything, which is why the output type is offered as an action, [described by its docstring](decision.md#tools-pick-then-fill).
- **Lists and nested models** round-trip faithfully, but their accuracy against labels is not measured, so check them on your own data before relying on either.

!!! note "Measure on your own data"
    Measure accuracy, the hand-off rate and any threshold on labelled examples of your own before relying on them.

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

Nothing else in Pydantic AI sees a call made this way: no agent run, no message history, no usage on a run's total, no fallback to another model, and the span the rest of an agent's work appears under is not opened. It is the escape hatch, not the main road. Reach for it when the question genuinely will not fit an output type, and go back to an `output_type` as soon as it will. Spelling out what counts as a yes and what counts as a no fits one: that is what [`BoolCriteria`](decision.md#what-each-mapping-does) is for.

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

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request. [`TypeSafeModelSettings`][pydantic_ai.models.typesafe.TypeSafeModelSettings] adds the two thresholds every decision model has:

- `decision_tool_call_threshold` sets how sure Jev has to be before it [takes a tool](decision.md#tools-pick-then-fill).
- `decision_boolean_threshold` sets [what `True` has to mean](decision.md#what-true-has-to-mean) for a `bool` field.

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
