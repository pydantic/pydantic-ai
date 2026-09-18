# TypeSafe (Jev)

[Jev](https://typesafe.ai) is not a language model. You give it a text and typed questions, and it answers each one with a confidence. It does not write text.

[`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] lets an agent whose job is to decide something run on Jev like on any other model. Each field of the `output_type` becomes one question, the prompt is the text, and the answers come back as the output, so a Pydantic model with several fields extracts several values in one request. Change the model name and the same agent runs on a language model, so you can compare the two.

## Install

To use `TypeSafeModel`, install `pydantic-ai-slim` (or `pydantic-ai`) with the `typesafe` optional group:

```bash
pip/uv-add "pydantic-ai-slim[typesafe]"
```

## Configuration

To use Jev through the [TypeSafe](https://typesafe.ai) API, get an API key from your TypeSafe account.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export TYPESAFE_API_KEY='your-api-key'
```

You can then use `TypeSafeModel` by name, with the `output_type` Jev should fill:

```python
from enum import Enum

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Verdict(str, Enum):
    """How to handle this command."""

    run = 'run'
    """Reads, builds, tests or edits inside the project. Reversible."""
    reject = 'reject'
    """Destroys data, rewrites shared history, or sends secrets over the network."""
    ask = 'ask'
    """Legitimate but consequential enough that a human should confirm."""


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    verdict: Verdict
    irreversible: bool = Field(description='Would running this destroy data or leak secrets?')


agent = Agent('typesafe:jev-latest', output_type=Handling)
result = agent.run_sync('rm -rf ./build')
print(result.output)
#> verdict=<Verdict.ask: 'ask'> irreversible=False
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(model, output_type=bool)
...
```

## Model names

`jev-latest` and `jev-preview` are aliases that move when TypeSafe ship a release; `jev-preview` runs ahead when there is a preview build. A versioned id is accepted too, whether or not it is listed:

```python {test="skip" lint="skip"}
Agent('typesafe:jev-1.13.0', output_type=Ticket)
```

[`ModelResponse.model_name`][pydantic_ai.messages.ModelResponse.model_name] always reports the versioned id that answered, so a run logged against `jev-latest` still records which model produced it.

## What Jev can answer

Every field of the output type is one question, and all of them go out in a single request:

| Field type | Question | Answer |
|---|---|---|
| `bool` | yes or no | `True` when Jev's probability is at least 0.5 |
| `Literal[...]` or `Enum` of strings | pick one | the chosen option |
| `float` with `ge=0` and `le=1` | yes or no | Jev's probability |
| `IntEnum` of 0, 1, 2, … with a docstring each | score against a rubric | the score rounded to the nearest level |
| `list` of a `Literal` or `Enum` | one yes or no per option | the options Jev said yes to |
| a nested model of these | its fields, asked as `outer.inner` | the model |
| `Literal[...]` or `Enum`, or `None` | pick one, or none of these | the option, or `None` |

The field description is the question text; an `Enum` field without one uses the enum's class docstring. The output type's docstring and the agent's instructions are context, so put the framing there and the per-field wording in the descriptions — see [where the question goes](#where-the-question-goes). A docstring under an `Enum` member, as in the example above, describes that option. A `Literal` has nowhere to put descriptions, so Jev only sees its option names.

A bare `bool`, `Literal` or `float` as the `output_type` is a single question with no field to describe, so the agent's instructions are the question, as in the example below.

A `list` of options is TypeSafe's fan-out: one yes/no per option, all in the same request, and the answer is the options Jev said yes to. An optional pick-one field, `Area | None`, is the same question with one more option, "None of these.", and the answer is `None` when Jev picks it; on labelled tickets that reads as well as an `other` member the user wrote, and better than turning low confidence into `None`, which is what the field's confidence is for. A nested model is its fields, asked as `outer.inner` and put back in place.

Confidence in each answer is on the response, so you can act on how sure the model was, for example by asking a human below a threshold. It runs 0 to 1, where 0 is undecided, and means the same thing for every field, so one threshold reads the same way across an output type.

For a pick-one or a rubric field it is Jev's own number, computed from how its probabilities are spread. Jev reports none for a yes/no, because there the probability *is* the answer before we round it: what is lost in the rounding is how sure the answer is, so that field's confidence is how far the probability sits from the coin flip, doubled onto the same scale. A `False` answered from a probability of 0.01 is a confident no and reports 0.98; one answered from 0.45 reports 0.10.

Pick the threshold from what the answer is used for rather than once for the whole system: acting automatically deserves a higher bar than flagging something for review, and the right numbers depend on your data, so calibrate against labelled examples of your own. Note that `jev-latest` moves when TypeSafe ship a release, which can shift the numbers under you — once you have tuned a threshold, pin the version it was tuned against (`typesafe:jev-1.13.0`) and move deliberately.

A `float` field has no entry at all. It keeps the probability as its answer, so nothing was lost to rounding and there is no second number to report — a `churn_risk` of 0.93 is the judgement, not a 93%-confident judgement, and repeating it under `confidence` would invite a threshold that filters out the low-risk customers rather than the uncertain ones. Apply `abs(value - 0.5) * 2` yourself for the same reading the other fields give.

A `float` is still a yes/no question underneath, so `0.5` means Jev is undecided, not that the answer is middling. When you want a magnitude, use a rubric.

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-latest', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
print(result.response.provider_details)
#> {'confidence': {'response': 0.95}, 'probabilities': {}, 'scores': {}}
```

## Falling back on low confidence

[`FallbackModel`](overview.md#fallback-model) falls back on API errors by default, and its `fallback_on` also takes a handler that looks at the response. Jev's confidence is on the response, so a language model can take over exactly the requests Jev was unsure about — the cheap model answers what it can, the expensive one only the rest:

```python
from pydantic_ai import Agent, ModelAPIError, ModelResponse
from pydantic_ai.models.fallback import FallbackModel


def unsure(response: ModelResponse) -> bool:
    confidence = (response.provider_details or {}).get('confidence', {})
    return any(value < 0.8 for value in confidence.values())


model = FallbackModel('typesafe:jev-latest', 'openai:gpt-5.6-sol', fallback_on=[ModelAPIError, unsure])
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
...
```

The handler runs on every model in the chain, and a language model reports no `confidence`, so its answers pass through. A response handler on its own replaces the default exception fallback, which is why `ModelAPIError` is listed alongside it.

Watch how often the fallback fires, not only how accurate the pair is. A chain that hands off nearly everything is accurate and costs full price, and the rate is the only number that shows it.

## Tools: Jev picks, and calls what it can

Jev cannot write a tool's arguments, but it can tell which tool a text calls for. With tools attached, every request carries one more question — which of these does the text call for — with the output type first among the options and every tool after it, each described by its docstring, and Jev answers it like any other question. What happens next depends on the pick:

- **The output type.** The fields are filled as usual.
- **A tool that takes no arguments** — a function tool, or an output function that takes nothing or only the run context. There is nothing to write, so Jev calls it itself. A function tool runs and its result comes back as history for the next request, so Jev can work through a sequence of them — each offered once per run, since the same call could only return the same result; an output function ends the run, which makes it a hand-off Jev chooses, to a person, a queue or another agent.
- **A tool with arguments.** Jev cannot fill them, so the request ends in a [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed]. That is a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`](overview.md#fallback-model) with a language model behind Jev hands that model the whole step, tools and all, and only the requests Jev handed off cost a language model call. Without a model behind Jev, the proposal is the error, and it says which tool Jev wanted and how sure it was.

A tool is only taken at or above `typesafe_tool_call_threshold` (0.8 by default). Below that the pick is a lean: the output is filled, and the pick and its probabilities are reported in `provider_details['tool']`, so the hand-off rate can be watched alongside accuracy. When there is no output type to fill, only output functions, the pick is the answer whatever its probability.

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.fallback import FallbackModel


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


support = Agent('openai:gpt-5.6-sol', instructions='Reply to the customer.')


async def escalate(ctx: RunContext[None]) -> str:
    """Hand the ticket to the support assistant."""
    result = await support.run(message_history=ctx.messages)
    return result.output


def refund(amount: float) -> str:
    """Return a payment to the customer."""
    return f'Refunded {amount}'


model = FallbackModel('typesafe:jev-latest', 'openai:gpt-5.6-sol')
agent = Agent(model, output_type=[Ticket, escalate], tools=[refund])
...
```

Here Jev triages what it can, hands a ticket to `support` when that is what it calls for, and leaves a refund, which needs an amount, to the language model behind it.

Write the output type's docstring as the action it is — "Triage a support ticket", "Reply to the customer" — because that is what Jev weighs the tools against. Asked whether it *can* answer rather than what the text calls for, Jev hands off nearly everything. Tune the threshold on labelled examples of your own: higher hands off less, and is right more often when it does.

## Ask one thing per field

TypeSafe call this "probably the most important concept" in their guide, and it is the one habit that does not carry over from a language model. Ask each field the kind of judgement a knowledgeable person makes in a second. A question that weighs several things at once does not fail — it returns a plausible number with low confidence, and you find out later.

So instead of one field asking `'Is this a good pitch?'`, ask three and combine them in code:

```python
from pydantic import BaseModel, Field


class Pitch(BaseModel):
    """Assess a startup pitch."""

    large_market: bool = Field(description='Does this address a market worth more than $1B a year?')
    technically_feasible: bool = Field(description='Could a small team build this with current technology?')
    differentiated: bool = Field(description='Does this do something competitors do not already do?')

    @property
    def promising(self) -> bool:
        return sum([self.large_market, self.technically_feasible, self.differentiated]) >= 2
```

Extra fields are close to free: every field goes out in the same request, and Jev answers them in parallel branches over one shared copy of the text, so a field you only need on some inputs costs tokens rather than time.

## Scoring against a rubric

Jev's third primitive scores a text against an ordered rubric. An `IntEnum` whose members are `0`, `1`, `2`, … is that rubric, and the docstring under each member says what that score means, so the levels are written where they are declared:

```python
from enum import IntEnum

from pydantic import BaseModel

from pydantic_ai import Agent


class Clarity(IntEnum):
    """How clearly does the text explain itself?"""

    unclear = 0
    """Leaves a reader who did not already know none the wiser."""
    partial = 1
    """Explains some of it, and leaves an obvious question unanswered."""
    clear = 2
    """A reader who did not already know could act on it."""


class Review(BaseModel):
    """Grade a piece of writing."""

    clarity: Clarity


agent = Agent('typesafe:jev-latest', output_type=Review)
result = agent.run_sync('Jevantic gives Python programs typed, probabilistic decisions from Jev.')
print(result.output)
#> clarity=<Clarity.unclear: 0>
print(result.response.provider_details['scores'])
#> {'clarity': 0.16}
```

The answer is the expected score rounded to the nearest level, which is what TypeSafe's own docs do when code needs one outcome, so it is always one of yours. `provider_details['scores']` keeps the unrounded position along the rubric — `0.16` here, not `0` — which is what to rank or threshold on; TypeSafe warn it is weakly calibrated, so don't read the gap between two levels as a precise magnitude. `probabilities` holds the whole distribution, keyed by level.

Every level needs a docstring: a rubric whose levels are unexplained is not a rubric, so one without them is a [`UserError`][pydantic_ai.exceptions.UserError]. The levels must also start at `0` and run upwards without gaps, which is the shape Jev scores against.

## Where the question goes

Jev takes two separate things: the material to judge, and the questions to ask about it. TypeSafe's own guidance is that the state holds "the content and supporting facts" and the questions hold "the judgments the model should make about that material", so **the prompt is only what is being judged, and the question belongs on the output type**.

That is the opposite habit to the one a language model teaches, where the question and the material go into one prompt together and the model sorts them out. Jev will not. `Agent('typesafe:jev-latest', output_type=bool)` with the question written into the prompt asks Jev nothing, and is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent — Jev rejects a yes/no question that carries neither instructions nor criteria.

Put the question on the field, and the prompt carries the ticket alone:

```python
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')
    area: Literal['billing', 'bug', 'account', 'other'] = Field(description='Which team owns it?')


agent = Agent('typesafe:jev-latest', output_type=Ticket)
result = agent.run_sync('My card was charged twice.')
print(result.output)
#> urgent=True area='billing'
```

For a single question an agent's `instructions` do the same job, and Jev answers the two spellings alike. Prefer the output type anyway: each field carries its own question, so several questions can be asked in one request, which is the thing Jev is fast at. Reach for `instructions` for framing that applies to every question — the voice to judge in, the domain, what the material is — and for the question itself only when there is one question and no field to describe.

## Judging a conversation

The latest user prompt is the text Jev judges. Everything before it in the message history goes along beside it as `history`: user prompts, answers, tool calls and their results, from whichever model produced them. That makes Jev a cheap judge of another agent's run:

```python
from pydantic_ai import Agent

assistant = Agent('openai:gpt-5.6-sol')
conversation = assistant.run_sync('hello')

judge = Agent('typesafe:jev-latest', output_type=bool, instructions='Was the assistant polite?')
result = judge.run_sync('Judge the conversation above.', message_history=conversation.all_messages())
print(result.output)
#> True
```

The whole history goes, so trim it to what the question is about — `message_history=conversation.all_messages()[-4:]`, a [history processor](../message-history.md#processing-message-history), or a compaction capability such as the [harness](https://github.com/pydantic/pydantic-ai-harness)'s `SlidingWindow`, `ClearToolResults` and `SummarizingCompaction`, which work on a Jev agent as on any other; a summary they write arrives as a `system` entry in the history, which is what it is. Accuracy falls as the state grows with detail the question does not need, and `jev-1.13` takes 64k tokens for the state and questions together, with 32k for the state plus the longest question. Compact earlier than a language model would need, since Jev is being asked to *judge* the whole of it, not to continue from it.

!!! note "A system prompt is judged, not asked"
    Jev is told what a conversation said and asked what the agent's `instructions` ask. A
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is part of what was said, so it joins the state as
    a `system` entry rather than becoming part of the question — including the agent's own `system_prompt=`.
    Nothing on the part says who wrote it, so treating any of them as an instruction meant that judging another
    agent's run folded that agent's persona into Jev's question. Give a Jev agent its question through
    `instructions=`.

## Streaming

Jev answers in one piece, so there is nothing to stream, and nothing that stops working: `run_stream`, an `event_stream_handler`, and the AG-UI and Vercel AI adapters get the whole answer as a single event.

## What Jev answers badly

Everything below returns an answer rather than an error, which is what makes it worth knowing. TypeSafe publish these per model version, on their [jaggedness page for `jev-1.13`](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and revise them as models change.

- **Arithmetic, counting and dates.** Jev is not a calculator, does not count reliably, and reads dates as text rather than as ordered quantities. Compute these in Python and ask Jev about the result.
- **Several judgements in one question.** See [above](#ask-one-thing-per-field).
- **Indirection.** A question about a property of a property, or one needing several hops, costs accuracy.
- **Context it does not need.** Accuracy falls as the state grows with detail unrelated to the question, so filter before you send rather than after, and compact a long conversation before judging it.
- **A tool call that repeats.** With a tool's call and result in the history, the text usually still calls for it, so Jev picks it again. A tool with no arguments is therefore offered once per run; a tool with arguments is proposed to the model behind Jev, which decides. Put a `UsageLimits(request_limit=...)` on a Jev agent with tools all the same, as on any agent that loops.
- **Deciding what it cannot see.** A tool that needs an argument the text does not state — a refund amount, a date — is one Jev will propose and a language model may decline to call; the two judge the same option differently, and language models disagree with each other on such picks about as often. Compare Jev with the model behind it on your own tickets before trusting either's hand-off rate.
- **Adversarial text.** Jev treats the state as data, not as hostile: text written to steer the answer — an injected instruction, a misleading framing, an argument for its own classification — can move it. TypeSafe say they expect to improve this. A guard built on Jev belongs alongside deterministic checks, not instead of them, and is worth testing against your own adversarial inputs.
- **Option order.** The order of a `Literal`'s options or an `Enum`'s members is part of what Jev sees, and reordering them can move the answer. If a classification matters, test it with the options in more than one order.

## What Jev cannot do

Jev does not write text, write a tool's arguments or read files. An agent that needs any of those is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent:

- The `output_type` must be one structured type made of the field types above, beside any output functions that take no arguments: no `str`, no second type with fields, no [`NativeOutput`][pydantic_ai.output.NativeOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput].
- No native tools. A function tool with arguments is not called by Jev either, but [proposed](#tools-jev-picks-and-calls-what-it-can) for a model behind it.
- No image, audio, video or document in the prompt or the history.

Jev does not revise an answer either. An output validator that raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] gets the same answer again, so a validator that keeps rejecting runs the agent out of retries.

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel
from pydantic_ai.providers.typesafe import TypeSafeProvider

model = TypeSafeModel('jev-latest', provider=TypeSafeProvider(api_key='your-api-key'))
agent = Agent(model, output_type=bool)
...
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
agent = Agent(model, output_type=bool)
...
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
agent = Agent(model, output_type=bool)
...
```

See [Provider SDK retries](../retries.md#provider-sdk-retries) for how this interacts with Pydantic AI's own retries.

## Model settings

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request, and `typesafe_tool_call_threshold` sets how sure Jev has to be before it [takes a tool](#tools-jev-picks-and-calls-what-it-can):

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(model, output_type=bool, model_settings={'timeout': 5})
...
```
