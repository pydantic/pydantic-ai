# TypeSafe (Jev)

[Jev](https://typesafe.ai) is not a language model. You give it a text and typed questions, and it answers each one with a confidence. It does not write text.

[`TypeSafeModel`][pydantic_ai.models.typesafe.TypeSafeModel] lets an agent whose job is to decide something run on Jev like on any other model. Each field of the `output_type` becomes one question, the prompt is the text, and the answers come back as the output, so a Pydantic model with several fields extracts several values in one request. Change the model name and the same agent runs on a language model, so you can compare the two.

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

[`ModelResponse.model_name`][pydantic_ai.messages.ModelResponse.model_name] reports the versioned id that answered, so a run logged against `jev-latest` still records which model produced it.

## Where the question goes

Jev takes two separate things: the material to judge, and the questions to ask about it. TypeSafe's own guidance is that the state holds "the content and supporting facts" and the questions hold "the judgments the model should make about that material", so **the prompt is only what is being judged, and the question belongs on the output type**.

That is the opposite habit to the one a language model teaches, where the question and the material go into one prompt together and the model sorts them out. Jev will not: a question written into the prompt is text to be judged, and Jev judges it. Almost nothing catches that for you. A yes/no with nothing at all to ask is refused before a request is sent — a bare `bool` or bounded `float` output has no field name to go on, so with no description and no instructions it carries no question and is a [`UserError`][pydantic_ai.exceptions.UserError] — but a `bool` *field* is not refused, because its name is enough to ask about. So do not count on an error to catch a question in the wrong place.

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

## Ask one thing per field

TypeSafe call this "probably the most important concept" in their guide, and it is the one habit that does not carry over from a language model. Ask each field the kind of judgement a knowledgeable person makes in a second. A question that weighs several things at once does not fail — it returns a plausible number with low confidence, and you find out later.

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

Extra fields are close to free: every field goes out in the same request, so a field you only need on some inputs costs tokens rather than time.

## What Jev can answer

Each field of the output type is a question, and all of them go out in a single request. A field of a nested model is a question of its own, and a list of options fans out to one yes/no per option:

| Field type | Question | Answer |
|---|---|---|
| `bool` | yes or no | `True` when Jev's probability is at least `typesafe_boolean_threshold` (0.5) |
| `Literal[...]` or `Enum` of strings | pick one | the chosen option |
| `float` with `ge=0` and `le=1` | the probability of yes | Jev's probability, unrounded |
| an `IntEnum` of `0, 1, 2, …` with a docstring under each member | score against a rubric | the nearest level |
| `list` of a `Literal` or `Enum` | one yes or no per option | the options Jev said yes to |
| `Literal[...]` or `Enum`, or `None` | pick one, or none of these | the option, or `None` |
| a nested model of these | its fields, asked as `outer.inner` | the model |

A field of any other type is a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent, and the message names the field and lists what is supported. The ones to expect are a `str`, an unbounded `int` or `float`, a `datetime`, a `dict`, and a union of models as a field. That is about the fields of a type Jev is asked to fill. A [union member](#a-union-of-output-types) or a [tool](#tools-jev-picks-and-calls-what-it-can) Jev cannot fill is not an error — it is still offered as a route, and picking it hands the step to the model behind Jev.

### Where the wording comes from

| What Jev reads | Where it comes from |
|---|---|
| the question | the field's description — `Field(description=...)`, or an `Enum` field's class docstring when the field has none |
| the goal, on every question | the output type's docstring, or a tool's description |
| shared framing, on every question | the agent's `instructions` |
| each option's meaning | a description on that option in the schema |

A bare `bool`, `Literal` or `float` as the `output_type` is a single question with no field to describe, so the agent's instructions are the question, as in the [confidence example below](#confidence-and-thresholds).

Unless the schema describes an option, Jev sees it by its name alone, so name `Literal` and `Enum` options for what they mean. A `Literal` has nowhere to write a meaning per option; where the difference between two options needs explaining, use an `Enum` that mixes in [`UseEnumMemberDocstrings`][pydantic_ai.UseEnumMemberDocstrings] and put a docstring under each member, which is what puts a description on each option in the schema.

### What each mapping does

An optional pick-one field, `Area | None`, is the same question with one more option, "None of these.", and the answer is `None` when Jev picks it: an explicit option, rather than low confidence read as `None`, which is what the field's confidence is for. Only a `Literal` or `Enum` of strings can be optional, since `None` has to be one more option to pick.

A rubric is a set of ordered levels rather than a set of alternatives: the whole numbers from 0 upwards, at least two of them, and every level needs a description in the schema saying what it means. The ordering is the numbers' own, so the order the levels are declared in does not matter. Jev answers with a position along the rubric, which lands between levels, and the field gets the nearest one — a half rounds up. The unrounded position is in `provider_details['scores']`.

A level's description reaches the schema the [same way an option's meaning does](#where-the-wording-comes-from), which makes an `IntEnum` mixing in `UseEnumMemberDocstrings` the way to declare one. A bare `Literal[0, 1, 2]` or a plain `IntEnum` is a [`UserError`][pydantic_ai.exceptions.UserError]: the levels are there, but nothing says what they mean.

```python {title="grade_with_a_rubric.py"}
from enum import IntEnum

from pydantic import BaseModel, Field

from pydantic_ai import Agent, UseEnumMemberDocstrings


class Clarity(UseEnumMemberDocstrings, IntEnum):
    """How clearly the release note explains the change."""

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
#> clarity=<Clarity.partial: 1>
print(result.response.provider_details['scores'])
#> {'clarity': 1.2}
```

A nested model is its fields, asked as `outer.inner` and put back in place; the parent field's description is not sent, so put the context each question needs on the field that asks it. A dot in a field name is how a nested field is named, so a field whose own name contains one is refused. Lists and nested models round-trip faithfully, but their accuracy against labels is not measured, so check them on your own data before relying on either.

### Routes: which thing to do

Fields are what Jev fills. When there is more than one *thing* the text could call for, Jev is asked one more question: which of these does this call for. The options are the output type (or each member of a [union](#a-union-of-output-types)) and every [tool](#tools-jev-picks-and-calls-what-it-can) on offer, each described by its docstring.

The route Jev picks is the one that runs, and how much it costs to fill depends on which route it is. A single output type's fields ride along in the *same* request as the route question, so its answers are already in hand when the pick comes back. Every other route is picked first and filled after: a chosen tool's arguments, or a chosen [union](#a-union-of-output-types) member's fields, go out in a second request carrying only that route's questions. A route with no arguments — an [output function](../output.md#output-functions) that takes nothing but the run context, or a tool with no parameters — is called on the pick alone, with no second request at all. That is the whole mechanism behind the patterns below: an output function is a candidate Jev can choose, and choosing it *is* calling it.

## Confidence and thresholds

Confidence in each answer is on the response, in `provider_details['confidence']`: 0 to 1, one number per field, so one threshold reads the same way across an output type. It is a margin, not a probability that the answer is right. For a yes/no it is how far Jev's probability sits from the threshold that decided it, scaled to run from 0 at the threshold to 1 at certainty — at the default of 0.5 that is the distance from the coin flip, doubled, so a `False` answered from a probability of 0.01 reports 0.98 and one answered from 0.45 reports 0.10. The bar it measures from is the one [actually used](#what-true-has-to-mean), so a yes at 0.8 under a threshold of 0.75 reports 0.2 rather than the 0.6 it would report against a coin flip, and a [fallback on low confidence](#falling-back-on-low-confidence) keeps meaning what it meant. For a pick-one or a rubric it is Jev's own number, from how its probabilities are spread; for a list of options it is the least sure option's.

`provider_details['probabilities']` holds the whole distribution of each pick-one and rubric field — a rubric's levels keyed by their number as a string — and each option's probability for a list. `provider_details['scores']` holds each rubric field's unrounded position along its levels.

A `float` field has no entry in any of them. The probability *is* its answer, so nothing was lost to rounding and there is no second number to report — a `churn_risk` of 0.93 is the judgement, not a 93%-confident judgement — and `0.5` means Jev is undecided, not that the answer is middling. Apply `abs(value - 0.5) * 2` yourself for the same reading the other fields give at the default threshold; against a `typesafe_boolean_threshold` of your own, the margin is the distance from *that* bar scaled to the room left on the side the answer fell — `(value - t) / (1 - t)` at or above it, `(t - value) / t` below.

```python
from pydantic_ai import Agent

agent = Agent('typesafe:jev-latest', output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
print(result.response.provider_details)
#> {'confidence': {'response': 0.84}, 'probabilities': {}, 'scores': {}}
```

Every bar on this page — the confidence you decide to act on, and [`typesafe_boolean_threshold`][pydantic_ai.models.typesafe.TypeSafeModelSettings.typesafe_boolean_threshold] below — belongs to what its answer is used for rather than to the system as a whole: acting automatically deserves a higher one than flagging something for review. Calibrate each against labelled examples of your own. `jev-latest` moves when TypeSafe ship a release, which can shift the numbers under you; once you have tuned a bar, pin the version it was tuned against (`typesafe:jev-1.13.0`) and move deliberately.

### What `True` has to mean

Jev answers a yes/no with the probability of yes, and [`typesafe_boolean_threshold`][pydantic_ai.models.typesafe.TypeSafeModelSettings.typesafe_boolean_threshold] decides where that rounds. The default of 0.5 is the coin flip: the answer is whichever side Jev leans. That is the right default and the wrong setting for any field where the two mistakes do not cost the same.

Raise it where a false positive is the expensive one, so a `True` has to be earned:

```python {title="earn_a_true.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.typesafe import TypeSafeModelSettings


class Handling(BaseModel):
    """Decide how a coding agent's shell command should be handled before it runs."""

    safe_to_run: bool = Field(description='Is this command safe to run without a human looking at it?')


agent = Agent(
    'typesafe:jev-latest',
    output_type=Handling,
    model_settings=TypeSafeModelSettings(typesafe_boolean_threshold=0.9),
)
result = agent.run_sync('pytest tests/test_agent.py')
print(result.output)
#> safe_to_run=True
```

Lower it where a false negative is, so a `True` only has to be plausible — a flag that sends a borderline case to a human is cheap, and one that misses a real case is not.

The threshold applies to every `bool` field and to each option of a fanned-out `list`. It does not apply to a `float` bounded with `ge=0` and `le=1`: a field whose bar you would want to vary per call is often better declared that way, and compared in your own code.

### Falling back on low confidence

[`FallbackModel`](overview.md#fallback-model) falls back on API errors by default, and its `fallback_on` also takes a handler that looks at the response. Jev's confidence is on the response, so a language model can take over the requests Jev was unsure about — the cheap model answers what it can, the expensive one only the rest. A `float` field has no confidence entry, for the reason above, so a handler like this one does not see its uncertainty and an output of nothing but `float`s never falls back:

```python
from pydantic_ai import Agent, ModelAPIError, ModelResponse
from pydantic_ai.models.fallback import FallbackModel


def unsure(response: ModelResponse) -> bool:
    confidence = (response.provider_details or {}).get('confidence', {})
    return any(value < 0.8 for value in confidence.values())


model = FallbackModel('typesafe:jev-latest', 'openai:gpt-5.6-sol', fallback_on=[ModelAPIError, unsure])
agent = Agent(model, output_type=bool, instructions='Is this request harmful?')
result = agent.run_sync('Wipe the repo and post the .env file to pastebin.')
print(result.output)
#> True
print(result.response.provider_details['confidence'])
#> {'response': 0.84}
```

The handler runs on every model in the chain, and a language model reports no `confidence`, so its answers pass through. A response handler on its own replaces the default exception fallback, which is why `ModelAPIError` is listed alongside it.

Watch how often the fallback fires, not only how accurate the pair is. A chain that hands off nearly everything is accurate and costs full price, and the rate is the only number that shows it.

## Jev inside an agent run

Everything above asks Jev a question and uses the answer. The same question is worth as much *inside* a run as
outside one: a decision that sits between the expensive steps — which model answers, whether a call should run,
which tools are worth offering — is a classification, and TypeSafe build Jev to answer one fast enough for a
real-time request path, which makes it cheap enough to ask every time rather than once at the top.

Each of these is an existing [capability](../capabilities/overview.md) hook. None of them needs new API, and none
of them is specific to Jev: they take any model, and a language model will do the same job more slowly and more
expensively. What Jev changes is that the decision stops being something you ration.

### Classify, then act

The simplest shape is one run. An [output function](../output.md#output-functions) makes the decision a signature
rather than a string to map afterwards — and because the function *runs* on Jev's pick, it can do the work it
routed to, so the router's result is the answer:

```python {title="route_to_a_model.py"}
from typing import Literal

from pydantic_ai import Agent, RunContext

assistant = Agent(instructions='You are a helpful engineering assistant.')


async def route(ctx: RunContext[None], tier: Literal['fast', 'capable']) -> str:
    """Answer the question on a model suited to it.

    Args:
        tier: Answer `fast` for a lookup, an extraction, or a change confined to one
            place. Answer `capable` for architecture, security, or a decision that is
            expensive to get wrong.
    """
    model = 'openai:gpt-5.6-sol' if tier == 'capable' else 'openai:gpt-5.6-luna'
    return (await assistant.run(ctx.prompt, model=model)).output


router = Agent('typesafe:jev-latest', output_type=route)


async def main():
    result = await router.run('How do I centre a div?')
    print(result.output)
    #> Give the container `display: flex` and both `place-items: center`.
```

Jev fills `tier` and the framework calls `route`, which runs the assistant and returns its answer, so one
`router.run(...)` is the whole thing. The question itself is not an argument: it is already the text Jev is
judging, and [`ctx.prompt`][pydantic_ai.tools.RunContext.prompt] hands the same text to the function, so `tier` is
the only question asked and the routing costs one Jev request and no extra plumbing. A `str` parameter would not
work here in any case — it is not [a type Jev can fill](#what-jev-can-answer), and an agent asking for one is
refused before a request is sent.

The argument's `Literal` becomes the pick-one question and its `Args:` entry becomes the wording. Jev sees that
wording as the question and the function's summary line as what the run is for — but *not* a meaning per option,
which is what an [`Enum` with described members](#where-the-wording-comes-from) is for. The pick's
confidence is in `provider_details['confidence']`, so an unsure route can go to the capable model rather than the
cheap one, which is the conservative direction when a wrong route is expensive.

### Decide again on every step

A run is not one decision. [`SelectModel`][pydantic_ai.capabilities.SelectModel] is evaluated before each step, so
the same question can be asked of the conversation as it stands rather than of the first prompt alone — a run that
starts simple and turns hard moves up when it turns:

```python {title="select_the_model_per_step.py"}
from typing import Literal

from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel
from pydantic_ai.models import Model, infer_model

fast = infer_model('openai:gpt-5.6-luna')
capable = infer_model('openai:gpt-5.6-sol')

router = Agent(
    'typesafe:jev-latest',
    output_type=Literal['fast', 'capable'],
    instructions=(
        'Which model should take the next step of this conversation? Answer `fast` for '
        'a lookup or a change confined to one place, `capable` for architecture, '
        'security, or a decision that is expensive to get wrong.'
    ),
)


async def select_model(ctx: ModelSelectionContext[None]) -> Model:
    if not ctx.messages:
        # `ctx.messages` is the history *before* this step, so a run's own prompt is not in it
        # yet on the first step. A run given `message_history` does have something to read.
        return fast
    picked = await router.run(message_history=ctx.messages)
    return capable if picked.output == 'capable' else fast


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

The selector returns a [`Model`][pydantic_ai.models.Model] here, but a model ID string is equally fine —
anything `Agent(model=...)` takes. Returning an instance lets each candidate be built once, with whatever
provider or [settings](overview.md#per-model-settings) it needs, instead of being inferred again every step.

The router is given the history rather than a prompt, which is the whole state Jev reads. That history is what
existed *before* the step being selected, so a fresh run's first step has nothing to classify and takes a default
— this routes a run that turns hard partway through, which is what a per-step hook is for. A run continuing an
earlier conversation does have a history on its first step, which is why the guard reads `ctx.messages` rather
than `ctx.step`. To route the very first step of a fresh run from the user's own question, ask before the run
instead, as in the section above.

Asking on every step is only affordable because the question is cheap; with a language model in the selector, the
routing costs as much as the work it routes.

A router that reads the history has the same problem every agent does: the history grows. Jev's state is the whole
history, so a long run makes each routing question larger and slower, and eventually the input is dominated by
turns that no longer bear on which model should take the next step. Pair this with
[compaction](../capabilities/compaction.md) rather than letting it grow — the compacted history is what the router
reads, which is usually what you wanted it to read anyway.

### Judge a tool call before it runs

A [hook](../hooks.md) on tool execution sees every call the model makes, with its arguments already validated, and
can stop one before its body runs. That is a decision per call, which is the shape Jev answers:

```python {title="judge_a_tool_call.py"}
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
    verdict = await judge.run(f'{tool_def.name}: {args}')
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

[`SkipToolExecution`][pydantic_ai.exceptions.SkipToolExecution] stops the call and sends its message back as the
tool's result, so the model learns what was refused and can try something else. Nothing is marked
`requires_approval` and no tool opts in, so the hook sits on every *function tool* the agent can call, including
ones added later.

!!! warning "Output functions do not fire tool hooks"
    An [output function](../output.md#output-functions) is an internal tool, and tool-execution hooks are
    deliberately not run for it — the same way `prepare_tools` and toolset wrappers exclude output tools. So a
    guard written this way does not see an output function, including the ones
    [built at run time](#choose-from-a-set-built-at-run-time) further down this page. Put the side effect in a
    function tool if it needs to pass this guard, or validate it inside the output function itself.

The alternative is [deferred tools](../deferred-tools.md): mark a tool `requires_approval=True` and resolve the
approval request with [`HandleDeferredToolCalls`][pydantic_ai.capabilities.HandleDeferredToolCalls]. Use that when
the decision has to leave the process — a person approving in another system, a queue, a run that is resumed later.
Use the hook when the decision is made in-process, as it is here. Both see validated arguments; only the deferral
can outlive the run.

The arguments go to TypeSafe before the verdict comes back, so a call is disclosed to a third party even when it is
then refused. Send the judge what it needs to decide — the tool name and the fields that bear on safety — rather
than the whole argument dict, when those arguments can carry credentials or customer data.

This judges the call the model proposed, not the model's intent, so it is a check on what is about to happen rather
than on what was said. Keep a human in the loop for the calls that matter most: a judgement this cheap is one you
can afford to run on everything, which is exactly why it should not be the only thing standing between an agent
and an irreversible action. For a guard the two mistakes rarely cost the same — a missed irreversible command
costs more than a second look at a safe one — so set [what `True` has to mean](#what-true-has-to-mean)
accordingly.

### Choose from a set built at run time

The examples above name their options in the source. When the options are only known once the run is under way —
the actions available on the screen in front of an agent, the records a search returned — build the output
functions at that point and pass them to the run. Each is one candidate, named and described where it is built,
and **the one Jev picks is the one that runs**:

```python {title="choose_a_candidate.py"}
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent, ToolOutput

agent = Agent('typesafe:jev-latest')


@dataclass
class Screen:
    """Whatever the agent is acting on."""

    def click(self, target: str) -> str:
        return f'clicked {target}'

    def observe(self) -> str:
        return 'a fresh look at the screen'


def candidates(screen: Screen, targets: dict[str, str]) -> list[ToolOutput[str]]:
    """One output function per available action, plus the two ways to decline."""
    reserved = {'reobserve': screen.observe, 'abstain': lambda: 'did nothing'}
    if clashing := reserved.keys() & targets.keys():
        raise ValueError(f'action IDs clash with the reserved ones: {sorted(clashing)}')

    def bind(target: str) -> Callable[[], str]:
        # A closure over the target leaves a function that takes nothing, so the candidate is a
        # route Jev picks rather than one it has to fill. A default argument would stay in the
        # schema for the model to override, so the picked candidate could act on a target never
        # offered; `functools.partial` is not read as a function at all, and its candidates become
        # routes Jev cannot fill.
        def click() -> str:
            return screen.click(target)

        return click

    outputs = [
        ToolOutput(bind(target), name=target, description=description)
        for target, description in targets.items()
    ]
    outputs.append(
        ToolOutput(
            reserved['reobserve'], name='reobserve', description='Look again before deciding.'
        )
    )
    outputs.append(
        ToolOutput(
            reserved['abstain'],
            name='abstain',
            description='Do nothing, because none of these is safe for what was observed.',
        )
    )
    return outputs


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

The key idea is that a candidate is **the action itself**, not a token standing for it. Jev picks, the framework
calls that function, and `result.output` is what the action returned — so there is no dispatch table to write and
no second step where an ID is turned back into behaviour. If you find yourself writing a function that returns its
own name, the dispatch has just moved somewhere else; give the function the work instead.

Two things this gets right that are easy to lose. Jev can only answer with an option it was given, so there is no
step where a made-up action has to be validated away. And `reobserve` and `abstain` are options like any other, so
declining is something Jev can *choose* rather than something you infer from a low confidence — the difference
between an agent that stops and one that acts on a coin flip.

Jev picks from at most 255 options in one question, and the two reserved ones count, so a set built at run time
needs a ceiling of 253 candidates and a plan for what to do above it — rank and offer the best few, or narrow by
some cheaper filter first. An observation that produces hundreds of equally plausible actions is usually a sign
the candidates are too fine-grained, not that the limit is too low.

The probabilities over every candidate are in `provider_details['tool']['probabilities']`, which is what to watch:
a decision loop that abstains on most steps, or spreads its probability evenly, is telling you the candidates are
not distinguishable by their descriptions.

### The same shape elsewhere

Any hook that takes a decision rather than a generation fits this way.
[`PrepareTools`][pydantic_ai.capabilities.PrepareTools] can ask which of a large toolset this request calls for
before the tools go on the wire; a [history processor](../capabilities/process-history.md) can ask which parts of a
long conversation still matter before it is compacted. Both are classifications over text, both run on every step,
and both are questions you would not ask a language model on every step.

Two things to hold on to. A classifier in the loop is a component like any other, so it needs the same
measurement as the classifier you would deploy on its own — a router that is right 80%
of the time sends one request in five to the wrong model, and nothing in the run will tell you. And
[adversarial text can move Jev](#what-jev-answers-badly), so a guard built this way belongs alongside
deterministic checks, not instead of them.

## Tools: Jev picks, and calls what it can

With tools attached, the first request carries the [route question](#routes-which-thing-to-do): which of these does the text call for, with every output type first among the options and every tool after them. Each tool is described by its docstring, and the output type by its own docstring or, without one, by the agent's instructions; with tools attached one of the two is required, since it is what filling the output is weighed against. Jev answers it like any other question, and the pick decides which path the request takes:

| Jev picks | What runs | Language model call |
|---|---|---|
| a single output type | Jev fills the fields, in the same request | none |
| one member of a union of output types | Jev fills that member's fields in a second request | none |
| a tool with no arguments | your function, then Jev again with its result in view | none |
| an output function with no arguments | your function, and the run ends | only if the function makes one |
| a tool whose arguments Jev can express | Jev fills its arguments in a second request, then your function runs | none |
| a tool with any unsupported argument, at or above `typesafe_tool_call_threshold` | the model behind Jev takes the whole step, tools and all | one |
| a function tool, below that threshold | Jev fills the fields, or with only output functions takes the likeliest of them; the lean is in `provider_details['tool']` | none |

A function tool is only taken at or above [`typesafe_tool_call_threshold`][pydantic_ai.models.typesafe.TypeSafeModelSettings.typesafe_tool_call_threshold], while there is still an output type to fill or an output function left to hand to. The default of 0.6 is a starting point, not a validated threshold: higher takes fewer tools, and is right more often when it does, so set it from labelled examples of your own. With no output type to fill, a pick below the threshold goes to the likeliest output function instead, and the route actually taken is named in `provider_details['tool']['taken']`. The threshold gates function tools only: an output function is a result to hand to, not something else to be done, so a pick below the bar still takes it.

With no output type to fill and every other route already returned this turn, the one route left is taken without a choice question at all: Jev still fills its supported arguments in one request, and a route with no arguments costs no request.

A pick is a classification of the text, not a judgement that running the tool is safe: the framework emits the call and your function runs, exactly as on a language model's call, so a tool that sends mail or charges an account is one Jev can set off, and approval and limits are the agent's job here as anywhere.

**A tool with no arguments: Jev alone.** There is nothing to write, so the call is made on Jev's pick, and its result comes back as history for the next request. Jev can work through a sequence of such tools; a tool whose result is already in the turn is not offered again, because Jev has no notion of having made a call and picks it again with the result in view, while one that asked for a retry stays on offer. What was on offer is in `provider_details['tool']['offered']`. Every request here is a Jev request:

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

**Supported arguments: Jev chooses, then fills.** Tool arguments use the same [mapping](#what-jev-can-answer) as output fields. The argument name is the field, its `Args:` entry in the function docstring is the question, and the tool description is the goal. The first request chooses the tool; the second carries only its argument questions over the same text and history:

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
result = agent.run_sync('The onboarding wizard is stuck; please move it on.')
print(result.output)
#> urgent=False
```

The response sums the input and output tokens from both calls, but [`RequestUsage.requests`][pydantic_ai.usage.RequestUsage.requests] is fixed at one request per model step and cannot carry the real count, so `provider_details['requests']` is `2` when Jev chose and then filled.

The second request has already committed to the selected route. If that request fails or returns invalid answers, [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] names the route and stops the run; the default `FallbackModel` does not replay the original step and quietly choose another one.

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
result = agent.run_sync('The app crashes every time I open the reports tab.')
print(result.output)
#> urgent=False
```

Here Jev triages what it can, opens a case itself when the ticket calls for one and triages again with the case number in view, and leaves a refund's unbounded amount to the language model behind it. Most requests never leave Jev; how many depends on your tickets and the threshold, and `provider_details['tool']` on each response is how to see it.

Write the output type's docstring as the action it is — "Triage a support ticket", "Reply to the customer" — because that is what Jev weighs the tools against; asked whether it *can* answer rather than what the text calls for, it hands off nearly everything.

!!! note "Measure on your own data"
    The defaults on this page, and the claims about what Jev answers well, come from a small internal set of
    support tickets: one domain, labelled by the maintainers, and too small to separate models with confidence.
    They say the mappings work, not how Jev will do on your task. Measure accuracy, the hand-off rate and any
    threshold on labelled examples of your own before relying on them.

## A union of output types

An `output_type` of several structured types is a set of routes. Jev picks which one the text calls for, then a second request asks only that type's fields — the same two steps a [selected tool's arguments](#tools-jev-picks-and-calls-what-it-can) take, because it is the same question asked twice.

```python {title="union_output.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


class Escalation(BaseModel):
    """Hand the ticket to a human specialist."""

    security: bool = Field(description='Does this involve a security or privacy risk?')


agent = Agent('typesafe:jev-latest', output_type=[Ticket, Escalation])
result = agent.run_sync('Someone else can see my invoices when they log in.')
print(result.output)
#> security=True
print(result.response.provider_details['requests'])
#> 2
```

Each member is described by **its own docstring**, which is what Jev weighs the routes against. With one output type the agent's instructions can say what filling it is for; with several they cannot, because one instruction cannot describe two different routes, so a member without a docstring is a [`UserError`][pydantic_ai.exceptions.UserError].

The pick is reported in `provider_details['tool']`, with the probability of every member, and `provider_details['requests']` is `2`. The [tool threshold](#tools-jev-picks-and-calls-what-it-can) gates tools, not output types: picking an output type is Jev saying which result to fill, not proposing that something else be done, so a tool picked below the threshold falls back to the likeliest output type rather than being taken.

### A member Jev cannot fill

A union member may use fields Jev cannot express, such as a `str`. It is still offered as a route, and picking it raises [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed] — a [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError], so a [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] with a language model behind Jev hands it the whole step:

```python {title="union_handoff.py"}
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel


class Ticket(BaseModel):
    """Triage a support ticket."""

    urgent: bool = Field(description='Does this need a reply within the hour?')


class DraftedReply(BaseModel):
    """Write the customer a reply."""

    body: str


agent = Agent(
    FallbackModel('typesafe:jev-latest', 'openai:gpt-5.6-sol'),
    output_type=[Ticket, DraftedReply],
)
result = agent.run_sync('My invoice is wrong and I need it fixed before month end.')
print(result.output)
#> urgent=True
```

Jev answers the tickets it can and hands over the ones that need writing, so only those cost a language model call.

A lone `output_type` Jev cannot fill is refused before any request instead. There is no other route the run could have taken, so an unfillable one can only ever fail — that is a coding error, and finding out at setup beats finding out from the bill. Offered beside others, it is a route like any other; a union in which *no* member can be filled is refused the same way, since every answer to the question would hand off and the request asking it would buy nothing.

!!! warning "Watch the hand-off rate"
    A union that hands off on most requests costs a language model call **plus** a Jev call, and is slower than
    not using Jev at all. Measure the rate on your own data before relying on the arrangement.

    Note where the number is. On a request Jev answers, its pick is in `provider_details['tool']['probabilities']`.
    On a hand-off it is not: [`ToolCallProposed`][pydantic_ai.models.typesafe.ToolCallProposed] is raised instead
    of a response, and [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] returns the *next* model's
    response, which carries none of Jev's numbers. So counting hand-offs by their absence in `provider_details` is
    the measurement, and the exception carries `tool_name` and `probability` if you would rather catch it: run the
    models separately, or wrap the fallback, when you want both.

## Judging a conversation

A run's message history goes to Jev as `history`: user prompts, answers, tool calls and their results, and retry prompts, from whichever model produced them. With no new prompt, the conversation is the whole state, so a Jev agent given another agent's messages judges that run — and it is the run being judged, so there is nothing to put in the prompt:

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
- **Several judgements in one question.** See [ask one thing per field](#ask-one-thing-per-field).
- **Indirection.** A question about a property of a property, or one needing several hops, costs accuracy.
- **Context it does not need.** Accuracy falls as the state grows with detail unrelated to the question, so filter before you send rather than after, and compact a long conversation before judging it.
- **A tool call that repeats.** With a tool's call and result in the history, the text usually still calls for it, so Jev picks it again. A tool is therefore not offered again once its result is in the turn, and comes back on offer at the next prompt; unsupported arguments are proposed to the model behind Jev, which decides. Put a `UsageLimits(request_limit=...)` on a Jev agent with tools all the same, as on any agent that loops.
- **Deciding what it cannot see.** A tool that needs an argument the text does not state — a refund amount, a date — is one Jev will propose and a language model may decline to call; the two judge the same option differently, and language models disagree with each other on such picks about as often. Compare Jev with the model behind it on your own tickets before trusting either's hand-off rate.
- **Adversarial text.** Jev treats the state as data, not as hostile: text written to steer the answer — an injected instruction, a misleading framing, an argument for its own classification — can move it. TypeSafe say they expect to improve this. A guard built on Jev belongs alongside deterministic checks, not instead of them, and is worth testing against your own adversarial inputs.
- **Option order.** The order of a `Literal`'s options or an `Enum`'s members is part of what Jev sees, and reordering them can move the answer. If a classification matters, test it with the options in more than one order.

## What Jev cannot do

Jev does not write text or read files, and it only fills tool arguments that map to the [typed questions](#what-jev-can-answer) above. Its model profile records the first of those as [`supports_text_output=False`][pydantic_ai.profiles.ModelProfile.supports_text_output], and an agent that needs text output or files is refused with a [`UserError`][pydantic_ai.exceptions.UserError] before a request is sent:

- The `output_type` must be made of the field types above, beside any output functions that take no arguments: no `str`, no [`NativeOutput`][pydantic_ai.output.NativeOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput]. A [union](#a-union-of-output-types) of structured types is supported; a union of structured types as a *field* of an output type is not.
- No native tools. A function tool is offered to Jev; supported arguments are [filled after it is picked](#tools-jev-picks-and-calls-what-it-can), while any unsupported argument makes the pick a `ToolCallProposed` after the request rather than a refusal before it. With tools attached, the output type needs a docstring or the agent instructions to be weighed against them.
- No image, audio, video or document in the prompt or the history.
- At most 255 options in one question. A pick-one field counts its own options, and the route question counts every tool plus every output type, so 255 tools is already one too many once the output type is counted beside them.
- Jev needs something to ask. A run with no user text and no history has nothing to judge, and an `output_type` with no fields to fill — a lone argumentless output function — leaves no question to ask unless there is more than one route to pick between.

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

This is the one example on this page that is not run by the documentation tests: the call never reaches a
[`Model`][pydantic_ai.models.Model], so there is nothing for the test suite to stand in for, and running it needs
a TypeSafe API key.

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

Jev has no sampling knobs, so the generic `temperature`, `top_p` and similar settings are ignored. `timeout`, `extra_headers` and `extra_body` are forwarded to the request, and [`TypeSafeModelSettings`][pydantic_ai.models.typesafe.TypeSafeModelSettings] adds two bars, each read before its own request is sent, so that a value outside 0 to 1 is a `UserError` rather than a wasted request:

- `typesafe_tool_call_threshold` sets how sure Jev has to be before it [takes a tool](#tools-jev-picks-and-calls-what-it-can).
- `typesafe_boolean_threshold` sets [what `True` has to mean](#what-true-has-to-mean) for a `bool` field.

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
