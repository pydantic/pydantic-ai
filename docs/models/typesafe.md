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

## What Jev can answer

Every field of the output type is one question, and all of them go out in a single request:

| Field type | Question | Answer |
|---|---|---|
| `bool` | yes or no | `True` when Jev's probability is at least 0.5 |
| `Literal[...]` or `Enum` of strings | pick one | the chosen option |
| `float` with `ge=0` and `le=1` | yes or no | Jev's probability |
| `IntEnum` of 0, 1, 2, … with a docstring each | score against a rubric | the level Jev thought most likely |

The field description is the question text; an `Enum` field without one uses the enum's class docstring. The output type's docstring and the agent's instructions are context, so put the framing there and the per-field wording in the descriptions — see [where the question goes](#where-the-question-goes). A docstring under an `Enum` member, as in the example above, describes that option. A `Literal` has nowhere to put descriptions, so Jev only sees its option names.

A bare `bool`, `Literal` or `float` as the `output_type` is a single question with no field to describe, so the agent's instructions are the question, as in the example below.

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


The whole history goes, so trim it to what the question is about — `message_history=conversation.all_messages()[-4:]`, or a [history processor](../message-history.md#processing-message-history). Accuracy falls as the state grows with detail the question does not need, and `jev-1.13` takes 64k tokens for the state and questions together, with 32k for the state plus the longest question.

!!! warning "A judged agent's system prompt becomes part of the question"
    A [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] anywhere in the history is treated as an instruction to Jev, because that is what makes `Agent(system_prompt=...)` work across turns. When the history is another agent's, its system prompt is folded into what Jev is *asked* rather than into what Jev judges — so `'You are a pirate'` ends up in front of `'Was the assistant polite?'`. Give the judge its question through `instructions=`, which is never read from the history, and strip system prompts from a conversation you did not write:

    ```python {test="skip" lint="skip"}
    history = [
        message
        for message in conversation.all_messages()
        if not any(part.part_kind == 'system-prompt' for part in getattr(message, 'parts', []))
    ]
    ```

## What Jev answers badly

Everything below returns an answer rather than an error, which is what makes it worth knowing. TypeSafe publish these per model version, on their [jaggedness page for `jev-1.13`](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and revise them as models change.

- **Arithmetic, counting and dates.** Jev is not a calculator, does not count reliably, and reads dates as text rather than as ordered quantities. Compute these in Python and ask Jev about the result.
- **Several judgements in one question.** See [above](#ask-one-thing-per-field).
- **Indirection.** A question about a property of a property, or one needing several hops, costs accuracy.
- **Context it does not need.** Accuracy falls as the state grows with detail unrelated to the question, so filter before you send rather than after.
- **Adversarial text.** Jev treats the state as data, not as hostile: text written to steer the answer — an injected instruction, a misleading framing, an argument for its own classification — can move it. TypeSafe say they expect to improve this. A guard built on Jev belongs alongside deterministic checks, not instead of them, and is worth testing against your own adversarial inputs.
- **Option order.** The order of a `Literal`'s options or an `Enum`'s members is part of what Jev sees, and reordering them can move the answer. If a classification matters, test it with the options in more than one order.

## What Jev cannot do

Jev does not write text, call tools, read files or stream. An agent that needs any of those is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent:

- The `output_type` must be one structured type made of the field types above: no `str`, no union of output types, no [`NativeOutput`][pydantic_ai.output.NativeOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput].
- No function tools, toolsets or native tools.
- No image, audio, video or document in the prompt or the history.
- No streaming: `run_stream`, `event_stream_handler` and the AG-UI and Vercel AI adapters do not work with it.

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

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request:

```python
from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')
agent = Agent(model, output_type=bool, model_settings={'timeout': 5})
...
```
