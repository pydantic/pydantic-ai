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
agent = Agent(model, output_type=bool)
...
```

## Where the question goes

Jev takes two separate things: the material to judge, and the questions to ask about it. TypeSafe's own guidance is that the state holds "the content and supporting facts" and the questions hold "the judgments the model should make about that material", so **the prompt is only what is being judged, and the question belongs on the output type**.

That is the opposite habit to the one a language model teaches, where the question and the material go into one prompt together and the model sorts them out. Jev will not: a question written into the prompt is text to be judged, and Jev judges it. Only the case that can be seen is refused before a request is sent — a bare `bool` output with no field description and no instructions carries no question at all, and is a [`UserError`][pydantic_ai.exceptions.UserError] — so do not count on an error to catch a question in the wrong place.

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
result = agent.run_sync(
    'You have charged me twice and my account is now overdrawn. I need this reversed today.'
)
print(result.output)
#> urgent=True area='billing'
```

For a single question an agent's `instructions` do the same job, and Jev answers the two spellings alike. Prefer the output type anyway: each field carries its own question, so several questions can be asked in one request, which is the thing Jev is fast at. Reach for `instructions` for framing that applies to every question — the voice to judge in, the domain, what the material is — and for the question itself only when there is one question and no field to describe.

## Model names

`jev-latest` and `jev-preview` are aliases that move when TypeSafe ship a release; `jev-preview` runs ahead when there is a preview build. A versioned id is accepted too, whether or not it is listed:

```python {test="skip" lint="skip"}
Agent('typesafe:jev-1.13.0', output_type=Ticket)
```

[`ModelResponse.model_name`][pydantic_ai.messages.ModelResponse.model_name] always reports the versioned id that answered, so a run logged against `jev-latest` still records which model produced it.

## What Jev can answer

Each field of the output type is a question, and all of them go out in a single request. A field of a nested model is a question of its own, and a list of options fans out to one yes/no per option:

| Field type | Question | Answer |
|---|---|---|
| `bool` | yes or no | `True` when Jev's probability is at least 0.5 |
| `Literal[...]` or `Enum` of strings | pick one | the chosen option |
| `float` with `ge=0` and `le=1` | yes or no | Jev's probability |
| `list` of a `Literal` or `Enum` | one yes or no per option | the options Jev said yes to |
| a nested model of these | its fields, asked as `outer.inner` | the model |
| `Literal[...]` or `Enum`, or `None` | pick one, or none of these | the option, or `None` |

The field description is the question text; an `Enum` field without one uses the enum's class docstring. The output type's docstring and the agent's instructions are context, so put the framing there and the per-field wording in the descriptions — see [where the question goes](#where-the-question-goes). Unless the schema describes an option, Jev sees it by its name alone, so name `Literal` and `Enum` options for what they mean.

A bare `bool`, `Literal` or `float` as the `output_type` is a single question with no field to describe, so the agent's instructions are the question, as in the example below.

A `list` of options is TypeSafe's fan-out: one yes/no per option, all in the same request, and the answer is the options Jev said yes to. An optional pick-one field, `Area | None`, is the same question with one more option, "None of these.", and the answer is `None` when Jev picks it: an explicit option, rather than low confidence read as `None`, which is what the field's confidence is for. A nested model is its fields, asked as `outer.inner` and put back in place; the parent field's description is not sent, so put the context each question needs on the field that asks it. The round trip of lists and nested models is tested; their accuracy against labels is not measured, so check them on your own data before relying on either.

Confidence in each answer is on the response, in `provider_details['confidence']`: 0 to 1, one number per field, so one threshold reads the same way across an output type. It is a margin, not a probability that the answer is right. For a yes/no it is how far Jev's probability sits from the coin flip, doubled — a `False` answered from a probability of 0.01 reports 0.98, one answered from 0.45 reports 0.10. For a pick-one it is Jev's own number, from how its probabilities are spread; for a list of options it is the least sure option's. `provider_details['probabilities']` holds the whole distribution of each pick-one field, and each option's probability for a list.

A `float` field has no entry. The probability *is* its answer, so nothing was lost to rounding and there is no second number to report — a `churn_risk` of 0.93 is the judgement, not a 93%-confident judgement — and `0.5` means Jev is undecided, not that the answer is middling. Apply `abs(value - 0.5) * 2` yourself for the same reading the other fields give.

Pick the threshold from what the answer is used for rather than once for the whole system — acting automatically deserves a higher bar than flagging something for review — and calibrate it against labelled examples of your own. `jev-latest` moves when TypeSafe ship a release, which can shift the numbers under you; once you have tuned a threshold, pin the version it was tuned against (`typesafe:jev-1.13.0`) and move deliberately.

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-latest', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
print(result.response.provider_details)
#> {'confidence': {'response': 0.84}, 'probabilities': {}, 'scores': {}}
```

## Falling back on low confidence

[`FallbackModel`](overview.md#fallback-model) falls back on API errors by default, and its `fallback_on` also takes a handler that looks at the response. Jev's confidence is on the response, so a language model can take over the requests Jev was unsure about — the cheap model answers what it can, the expensive one only the rest. A `float` field has no confidence entry, for the reason below, so a handler like this one does not see its uncertainty and an output of nothing but `float`s never falls back:

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

Jev first tells which tool a text calls for. When that tool has arguments Jev can express as its typed questions, it asks only those arguments in a second request and returns the filled call. With tools attached, the first request carries one more question — which of these does the text call for — with the output type first among the options and every tool after it. Each tool is described by its docstring, and the output type by its own docstring or, without one, by the agent's instructions; with tools attached one of the two is required, since it is what filling the output is weighed against. Jev answers the question like any other, and the pick decides which path the request takes:

| Jev picks | What runs | Language model call |
|---|---|---|
| the output type | Jev fills the fields, in the same request | none |
| a tool with no arguments | your function, then Jev again with its result in view | none |
| an output function with no arguments | your function, and the run ends | only if the function makes one |
| a tool whose arguments Jev can express | Jev fills its arguments in a second request, then your function runs | none |
| a tool with any unsupported argument | the model behind Jev takes the whole step, tools and all | one |
| any tool, below the threshold | Jev fills the fields, or with only output functions takes the likeliest of them; the lean is reported in `provider_details['tool']` | none |
| the one route left | that route, without a choice request; Jev still asks for supported arguments | none |

A function tool is only taken at or above `typesafe_tool_call_threshold`, while there is still an output type to fill or an output function left to hand to. The default of 0.6 was chosen on a small internal set of support tickets and is a starting point, not a validated threshold: higher takes fewer tools, and is right more often when it does, so set it from labelled examples of your own. With no output type to fill, a pick below the threshold goes to the likeliest output function instead, and once every other route has returned, the one left is taken without a choice request. A pick is a classification of the text, not a judgement that running the tool is safe: the framework emits the call and your function runs, exactly as on a language model's call, so a tool that sends mail or charges an account is one Jev can set off, and approval and limits are the agent's job here as anywhere.

**A tool with no arguments: Jev alone.** There is nothing to write, so the call is made on Jev's pick, and its result comes back as history for the next request. Jev can work through a sequence of such tools; a tool whose result is already in the turn is not offered again, because Jev has no notion of having made a call and picks it again with the result in view, while one that asked for a retry stays on offer. What was on offer is in `provider_details['tool']['offered']`. Every request here is a Jev request:

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


def escalate_to_human() -> str:
    """Hand the ticket to a person on the support team."""
    return 'Escalated: case #4821 opened.'


agent = Agent('typesafe:jev-latest', output_type=Ticket, tools=[escalate_to_human])
...
```

**An output function with no arguments: a hand-off that ends the run.** An output function that takes nothing, or only the run context, is picked the same way, and the run ends with what it returns — to a person, a queue, or another agent. The language model runs only inside the hand-off, so only the requests Jev handed off pay for one:

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


support = Agent('openai:gpt-5.6-sol', instructions='Reply to the customer.')


async def reply(ctx: RunContext[None]) -> str:
    """Write the customer a reply."""
    result = await support.run(message_history=ctx.messages)
    return result.output


agent = Agent('typesafe:jev-latest', output_type=[Ticket, reply])
...
```

**Supported arguments: Jev chooses, then fills.** Tool arguments use the same mapping as output fields: `bool`, two or more string options, a bounded probability, a list of options, an optional pick-one, and a nested model of those. The argument name is the field, its description from the function docstring is the question, and the tool description is the goal. The first request chooses the tool; the second carries only its argument questions over the same text and history:

```python
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


def take_action(direction: Literal['left', 'right']) -> str:
    """Take the requested action.

    Args:
        direction: Which direction should be taken?
    """
    return f'Turned {direction}.'


agent = Agent('typesafe:jev-latest', output_type=Ticket, tools=[take_action])
...
```

The response sums the input and output tokens from both calls. [`RequestUsage.requests`][pydantic_ai.usage.RequestUsage.requests] is fixed at one request per model step and cannot carry the real count; `provider_details['requests']` is therefore `2` when Jev chose and filled a tool. See [#8498](https://github.com/pydantic/pydantic-ai/issues/8498).

The second request has already committed to the selected tool. If that request fails or returns invalid answers, [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] names the tool and stops the run; the default `FallbackModel` does not replay the original step and quietly choose another route.

**Unsupported arguments: the model behind Jev.** A plain `str`, an unbounded number, or any other unsupported argument leaves the selected call as a [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed]. That is a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`](overview.md#fallback-model) with a language model behind Jev hands that model the whole step, tools and all; the rest of the requests never leave Jev. Without a model behind Jev, the proposal is the error, and it says which tool Jev wanted and how sure it was. The Jev request that proposed the call is not on the fallback response's usage.

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


def escalate_to_human() -> str:
    """Hand the ticket to a person on the support team."""
    return 'Escalated: case #4821 opened.'


def refund(amount: float) -> str:
    """Return a payment to the customer."""
    return f'Refunded {amount}'


model = FallbackModel('typesafe:jev-latest', 'openai:gpt-5.6-sol')
agent = Agent(model, output_type=Ticket, tools=[escalate_to_human, refund])
...
```

Here Jev triages what it can, opens a case itself when the ticket calls for one and triages again with the case number in view, and leaves a refund's unbounded amount to the language model behind it. Most requests never leave Jev; how many depends on your tickets and the threshold, and `provider_details['tool']` on each response is how to see it.

Write the output type's docstring as the action it is — "Triage a support ticket", "Reply to the customer" — because that is what Jev weighs the tools against; asked whether it *can* answer rather than what the text calls for, it hands off nearly everything. Tune the threshold on labelled examples of your own.

!!! note "Measure on your own data"
    The defaults on this page, and the claims about what Jev answers well, come from a small internal set of
    support tickets: one domain, labelled by the maintainers, and too small to separate models with confidence.
    They say the mappings work, not how Jev will do on your task. Measure accuracy, the hand-off rate and any
    threshold on labelled examples of your own before relying on them.

## Jev inside an agent run

Everything above asks Jev a question and uses the answer. The same question is worth as much *inside* a run as
outside one: a decision that sits between the expensive steps — which model answers, whether a call should run,
which tools are worth offering — is a classification, and a classification at 180 ms is cheap enough to make every
time rather than once at the top.

Each of these is an existing [capability](../capabilities/overview.md) hook. None of them needs new API, and none
of them is specific to Jev: they take any model, and a language model will do the same job more slowly and more
expensively. What Jev changes is that the decision stops being something you ration.

### Classify, then act

The simplest shape is two runs: one to decide, one to do. An [output function](../output.md#output-functions) makes
the decision a signature rather than a string to map afterwards, so the routing table is the function and the
options are its argument:

```python {title="route_to_a_model.py"}
from typing import Literal

from pydantic_ai import Agent

MODELS = {'fast': 'openai:gpt-5.6-luna', 'capable': 'openai:gpt-5.6-sol'}


def route(tier: Literal['fast', 'capable']) -> str:
    """Route the request to a model.

    Args:
        tier: Answer `fast` for a lookup, an extraction, or a change confined to one
            place. Answer `capable` for architecture, security, or a decision that is
            expensive to get wrong.
    """
    return MODELS[tier]


router = Agent('typesafe:jev-latest', output_type=route)
assistant = Agent(instructions='You are a helpful engineering assistant.')


async def answer(question: str) -> str:
    picked = await router.run(question)
    return (await assistant.run(question, model=picked.output)).output
```

The argument's `Literal` becomes the pick-one question and its `Args:` entry becomes the wording, so the whole
routing decision costs one Jev request. The pick's confidence is in `provider_details['confidence']`, so an unsure
route can go to the capable model rather than the cheap one, which is the conservative direction when a wrong route
is expensive.

### Decide again on every step

A run is not one decision. [`SelectModel`][pydantic_ai.capabilities.SelectModel] is evaluated before each step, so
the same question can be asked of the conversation as it stands rather than of the first prompt alone — a run that
starts simple and turns hard moves up when it turns:

```python {title="select_the_model_per_step.py"}
from typing import Literal

from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel

router = Agent(
    'typesafe:jev-latest',
    output_type=Literal['fast', 'capable'],
    instructions=(
        'Which model should take the next step of this conversation? Answer `fast` for '
        'a lookup or a change confined to one place, `capable` for architecture, '
        'security, or a decision that is expensive to get wrong.'
    ),
)


async def select_model(ctx: ModelSelectionContext[None]) -> str:
    picked = await router.run(message_history=ctx.messages)
    return 'openai:gpt-5.6-sol' if picked.output == 'capable' else 'openai:gpt-5.6-luna'


agent = Agent(capabilities=[SelectModel(select_model)])
```

The router is given the history rather than a prompt, which is the whole state Jev reads. Asking on every step is
only affordable because the question is cheap; with a language model in the selector, the routing costs as much as
the work it routes.

### Judge a tool call before it runs

A tool marked `requires_approval=True` stops before its body runs and arrives as an approval request, which a
[deferred tool handler](../deferred-tools.md) resolves. That handler is a decision per call, which is the shape Jev
answers:

```python {title="judge_a_tool_call.py"}
from pydantic import BaseModel, Field

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    ToolDenied,
)
from pydantic_ai.capabilities import HandleDeferredToolCalls


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    irreversible: bool = Field(
        description='Would running this destroy data or leak secrets?'
    )


judge = Agent('typesafe:jev-latest', output_type=Handling)


async def judge_calls(
    ctx: RunContext, requests: DeferredToolRequests
) -> DeferredToolResults:
    approvals: dict[str, bool | ToolDenied] = {}
    for call in requests.approvals:
        verdict = await judge.run(str(call.args))
        approvals[call.tool_call_id] = (
            ToolDenied('That command destroys data or leaks secrets.')
            if verdict.output.irreversible
            else True
        )
    return requests.build_results(approvals=approvals)


agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[HandleDeferredToolCalls(handler=judge_calls)],
)


@agent.tool_plain(requires_approval=True)
def run_shell(command: str) -> str:
    return f'ran {command!r}'
```

This judges the call the model proposed, not the model's intent, so it is a check on what is about to happen rather
than on what was said. Denying a call sends the message back to the model, which can try something else. Keep a
human in the loop for the calls that matter most: a judgement at 180 ms is cheap enough to run on everything, which
is exactly why it should not be the only thing standing between an agent and an irreversible action.

`typesafe_boolean_threshold` matters here more than anywhere else on this page. The default rounds at the coin
flip, and for a guard the two mistakes rarely cost the same: see [what `True` has to mean](#what-true-has-to-mean).

### The same shape elsewhere

Any hook that takes a decision rather than a generation fits this way.
[`PrepareTools`][pydantic_ai.capabilities.PrepareTools] can ask which of a large toolset this request calls for
before the tools go on the wire; a [history processor](../capabilities/process-history.md) can ask which parts of a
long conversation still matter before it is compacted. Both are classifications over text, both run on every step,
and both are questions you would not ask a language model on every step.

Two things to hold on to. A classifier in the loop is a component like any other, so it needs the same
[measurement](#measure-on-your-own-data) as the classifier you would deploy on its own — a router that is right 80%
of the time sends one request in five to the wrong model, and nothing in the run will tell you. And Jev reads the
state as data rather than as instructions, so text written to steer it can move it: a guard built this way belongs
alongside deterministic checks, not instead of them.

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

## Judging a conversation

A run's message history goes to Jev as `history`: user prompts, answers, tool calls and their results, from whichever model produced them. With no new prompt, the conversation is the whole state, so a Jev agent given another agent's messages judges that run — and it is the run being judged, so there is nothing to put in the prompt:

```python
from pydantic_ai import Agent

assistant = Agent('openai:gpt-5.6-sol')
judge = Agent('typesafe:jev-latest', output_type=bool, instructions='Was the assistant polite?')

conversation = assistant.run_sync('hello')
result = judge.run_sync(message_history=conversation.all_messages())
print(result.output)
#> True
```

A new prompt on top of a history is judged as `text` beside it. Either way the conversation in the history goes to TypeSafe's API — system prompts, tool arguments and tool results included, though a model's private thinking and a `CachePoint` are left out and a file is refused — so trim it to what the question is about: `message_history=conversation.all_messages()[-4:]`, a [history processor](../message-history.md#processing-message-history), or a compaction capability, which works on a Jev agent as on any other. A summary it writes goes along as a `summary` entry when it is a [`CompactionPart`][pydantic_ai.messages.CompactionPart], or as a `system` entry when it was written as a system prompt, which the [harness](https://github.com/pydantic/pydantic-ai-harness)'s compaction does. Accuracy falls as the state grows with detail the question does not need, and `jev-1.13` takes 64k tokens for the state and questions together, with 32k for the state plus the longest question; past that the request fails with a [`ModelHTTPError`][pydantic_ai.exceptions.ModelHTTPError] (`max_tokens_exceeded`), which a `FallbackModel` hands to the model behind Jev like any API error, so an over-long conversation quietly becomes a language model call. Compact earlier than a language model would need, since Jev is being asked to *judge* the whole of it, not to continue from it.

!!! note "A system prompt is judged, not asked"
    Jev is told what a conversation said and asked what the agent's `instructions` ask. A
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is part of what was said, so it joins the state as
    a `system` entry rather than becoming part of the question — including the agent's own `system_prompt=`.
    Nothing on the part says who wrote it, so treating any of them as an instruction meant that judging another
    agent's run folded that agent's persona into Jev's question. Give a Jev agent its question through
    `instructions=`.

## Streaming

Jev answers in one piece, so there is nothing to stream, and nothing that stops working: `run_stream`, an `event_stream_handler`, and the AG-UI and Vercel AI adapters get the whole answer as a single event. There are no partial results and no earlier first token — it is compatibility, not streaming.

## What Jev answers badly

Everything below returns an answer rather than an error, which is what makes it worth knowing. TypeSafe publish these per model version, on their [jaggedness page for `jev-1.13`](https://docs.typesafe.ai/model-jaggedness/jev-1.13), and revise them as models change.

- **Arithmetic, counting and dates.** Jev is not a calculator, does not count reliably, and reads dates as text rather than as ordered quantities. Compute these in Python and ask Jev about the result.
- **Several judgements in one question.** See [above](#ask-one-thing-per-field).
- **Indirection.** A question about a property of a property, or one needing several hops, costs accuracy.
- **Context it does not need.** Accuracy falls as the state grows with detail unrelated to the question, so filter before you send rather than after, and compact a long conversation before judging it.
- **A tool call that repeats.** With a tool's call and result in the history, the text usually still calls for it, so Jev picks it again. A tool is therefore not offered again once its result is in the turn, and comes back on offer at the next prompt; unsupported arguments are proposed to the model behind Jev, which decides. Put a `UsageLimits(request_limit=...)` on a Jev agent with tools all the same, as on any agent that loops.
- **Deciding what it cannot see.** A tool that needs an argument the text does not state — a refund amount, a date — is one Jev will propose and a language model may decline to call; the two judge the same option differently, and language models disagree with each other on such picks about as often. Compare Jev with the model behind it on your own tickets before trusting either's hand-off rate.
- **Adversarial text.** Jev treats the state as data, not as hostile: text written to steer the answer — an injected instruction, a misleading framing, an argument for its own classification — can move it. TypeSafe say they expect to improve this. A guard built on Jev belongs alongside deterministic checks, not instead of them, and is worth testing against your own adversarial inputs.
- **Option order.** The order of a `Literal`'s options or an `Enum`'s members is part of what Jev sees, and reordering them can move the answer. If a classification matters, test it with the options in more than one order.

## What Jev cannot do

Jev does not write text or read files, and it only fills tool arguments that map to the typed questions above. An agent that needs text output or files is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent:

- The `output_type` must be one structured type made of the field types above, beside any output functions that take no arguments: no `str`, no second type with fields, no [`NativeOutput`][pydantic_ai.output.NativeOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput].
- No native tools. A function tool is offered to Jev; supported arguments are [filled after it is picked](#tools-jev-picks-and-calls-what-it-can), while any unsupported argument makes the pick a `ToolCallProposed` after the request rather than a refusal before it. With tools attached, the output type needs a docstring or the agent instructions to be weighed against them.
- No image, audio, video or document in the prompt or the history.
- At most 255 options in one question. A pick-one field counts its own options, and the tool question counts every tool plus the output type, so 255 tools is already one too many.

Jev does not revise an answer the way a language model does. Its previous answer and the validator's complaint both go back in the history, so they are part of what it judges, but the question is unchanged and a confident answer does not move: an output validator that raises [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] usually gets the same answer again, and one that keeps rejecting runs the agent out of retries.

## Asking Jev directly

An `output_type` is the question in almost every case, and it is what makes the same agent run on a language model later. Two things it cannot carry: a state that is a record rather than prose, and a question whose wording has nowhere to live, such as spelling out what counts as `true` and what counts as `false` for a yes/no.

The TypeSafe SDK client is on the model for those, configured with the same API key, base URL and HTTP client:

```python {title="ask_jev_directly.py"}
from typesafe_sdk import Choice, Noul, NoulCriteria

from pydantic_ai.models.typesafe import TypeSafeModel

model = TypeSafeModel('jev-latest')


async def judge_order(order: dict[str, object]) -> float:
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
    return response.answers['refundable'].noul
```

Pass `model=` yourself. The client does not know which model the `TypeSafeModel` around it was built with, so without it the SDK falls back to its own default, which `TYPESAFE_DEFAULT_MODEL` can change underneath you.

Nothing else in Pydantic AI sees a call made this way: no agent run, no message history, no usage on a run's total, no fallback to another model, and the span the rest of an agent's work appears under is not opened. It is the escape hatch, not the main road. Reach for it when the question genuinely will not fit an output type, and go back to an `output_type` as soon as it will.

Passing a record as a mapping rather than as text is a convenience, not an accuracy setting. Jev reads a rendered sentence at least as well as the object it came from, so there is no need to restructure a prompt to get at this.

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
