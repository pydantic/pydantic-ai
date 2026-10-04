---
title: Agent Control
description: Change a Pydantic AI agent's instructions, model, model settings, and tool descriptions from Logfire, without a redeploy.
---

# Agent Control

Agent Control is the Logfire feature for changing how a running agent behaves -- its instructions, the
model it runs on, that model's settings, and the tool descriptions the model reads -- from Logfire
instead of from code. Changes are versioned, labelled, and rolled out like anything else you publish
there, and they take effect without a redeploy.

`AgentControl` is the Pydantic AI [capability](index.md) that connects your agent to it. You add it,
and the agent you already wrote becomes editable from Logfire.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/logfire/)

Install the `logfire` extra:

```bash
pip/uv-add "pydantic-ai-harness[logfire]"
```

## The problem it solves

An agent's behavior is spread across knobs that all live in code: the instructions, the model and its
sampling settings, and the tool descriptions that make up half of what the model actually sees.
Tuning any one of them takes a redeploy. Tuning them *together* -- a new prompt that only works with a
smarter model, a reworded tool description the prompt now refers to -- takes coordinated redeploys,
and leaves nothing single to roll back when the combination misbehaves.

Agent Control makes the whole configuration one versioned unit that lands, rolls out, and rolls back
together.

## Usage

Write the agent exactly as you would anyway, give it an explicit `name`, and add the capability:

```python
import logfire
from pydantic_ai import Agent

from pydantic_ai_harness.logfire import AgentControl

logfire.configure()
logfire.instrument_pydantic_ai()


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f'The weather in {city} is sunny.'


agent = Agent(
    'anthropic:claude-fable-5-1',
    name='checkout_assistant',
    instructions='You are a concise checkout assistant.',
    tools=[get_weather],
    capabilities=[AgentControl(label='production')],
)

result = agent.run_sync('Refund my last order.')
print(result.output)
```

The agent's `name` is how Logfire matches it to the agent you can already see in your traces, and
the first run reports the agent to your project so there is nothing to describe by hand.

Agent Control needs a `LOGFIRE_API_KEY` with the `project:read_variables` scope -- a different
credential from the write token that sends spans. Read-only is all it is: nothing here writes to your
project's variables. `LOGFIRE_API_KEY` is read by the Logfire instance `logfire.configure()` set up,
which is the one `AgentControl` resolves on unless you pass `logfire_instance=`: if your application
configures a `logfire.Logfire` instance of its own, pass that. When the instance has no way to read
variables, `AgentControl` warns once per process and agent, naming what is missing, and the agent runs
as written. Instrumentation is not optional, though: it is how the agent tells Logfire it
exists at all (see [Registration and the baseline](#registration-and-the-baseline)), and without
spans neither the version that produced a given run nor whether the agent is picking its config up
makes it back to Logfire.

That's the whole setup on the code side. Creating the config itself is done in Logfire.

[`examples/agent_control.py`](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/examples/agent_control.py)
is a complete agent wired up this way: six separately addressable prompt blocks written in five
different ways, four tools in two toolsets, and a printout of what resolved on every run. It runs with no Logfire configured
at all, which is the behavior worth seeing first.

Pinning `label='production'` is the recommended default, for the same
[prompt-cache reasons](managed-prompt.md#prompt-cache-trade-off) as a managed prompt.

## What you can change from Logfire

| | What Logfire can do |
| --- | --- |
| **Instructions** | Rewrite or remove any block of the prompt, and add new ones |
| **Model** | Run the agent on a different model |
| **Model settings** | Change settings individually -- `temperature`, `max_tokens`, `thinking`, ... |
| **Tools** | Rename a tool and reword its description and its parameters' descriptions, for the model's eyes only |

Anything you don't change in Logfire keeps doing what the code says, and removing a change there puts
that piece back the way the code has it.

The settings are the canonical ones every framework has a knob for, under the names
[`ModelSettings`](https://ai.pydantic.dev/api/settings/) gives them. Provider-specific settings
(`openai_reasoning_effort`, `extra_headers`) stay in code, where they always were.

What a config may hold is the same contract in every Agent Control SDK, and it lives in one place:
the `logfire.agent_control` package that ships with the `logfire` extra. `AgentControl` reads it
from there, so what the Logfire UI offers, what a TypeScript service reads, and what this agent
applies cannot drift apart. Import `AgentConfig` from that package when you want to publish a value
from code rather than from the UI.

A tool's implementation and the shape of its arguments stay code-owned: Logfire edits what the model
is *told* about a tool, never what it *does*. A renamed tool still routes to the same function, and
`ctx.tool_name` is still the original name, so nothing in code has to know about the rename. Logfire
shows which toolset each tool came from, and a change can be narrowed to that toolset's tool of a
given name -- so two services sharing one config, each with its own `search`, can be told apart.

Your call site still wins. A `run(model=...)` or `run(model_settings=...)` argument overrides the
published value for that one run, and the published value overrides what the agent was constructed
with. Nothing in between changes it: another capability contributing a model or settings of its own
does not outrank the config the Logfire UI is showing you.

## Which parts of the prompt are editable

Everything else is one value, but instructions are assembled from many places -- your agent's text,
`@agent.instructions` functions injecting today's date or the signed-in user, each toolset and MCP
server, each capability, a skill catalog. So Logfire edits them **one block at a time**, and a block
it can identify is a block it can offer to change.

Most blocks already have an identity. Your agent's own text is `agent`; a toolset's is
`toolset:<id>`; a capability's is `capability:<id>`. Two cases are worth knowing about, and Logfire
marks both in the editor so you can see which of your blocks they are:

**A block computed on every request is shown but not editable.** That's any instruction *function* --
`@agent.instructions` and its toolset and capability equivalents -- even one that returns fixed text.
Rewriting it would freeze whatever it happened to return once, and removing it would delete the
computation.

Only the *fact* of such a block goes to Logfire, never its text. An instruction function reads the
run -- a tenant, a signed-in user, a retrieved document -- and what Logfire is told is the block's id
and that it is recomputed, which is all the editor needs to show it and to not offer to change it.

So if one function today produces a prompt that is mostly fixed with a dynamic piece in it, split it
in two: the fixed part as text, the varying part as its own function. The fixed half becomes editable
and the varying half stays where it belongs.

```python {test="skip"}
agent = Agent(
    'anthropic:claude-fable-5-1',
    name='checkout_assistant',
    instructions='You are a concise checkout assistant.',  # editable in Logfire
    capabilities=[AgentControl()],
)


@agent.instructions(name='today')  # shown, never edited, never sent
def today(_ctx: RunContext[None]) -> str:
    return f'Today is {date.today()}.'
```

**A block with nothing to identify it isn't offered at all.** That's text from a toolset or capability
with no `id` of its own, an unnamed callable in `Agent(instructions=...)`, or anything passed to
`run(instructions=...)`. Give the toolset or capability an `id` and its text becomes editable.

To make one block of a larger prompt separately editable, give that part a name:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.messages import InstructionPart

from pydantic_ai_harness.logfire import AgentControl

agent = Agent(
    'anthropic:claude-fable-5-1',
    name='checkout_assistant',
    instructions=[
        'You are a concise checkout assistant.',
        InstructionPart(content='Always confirm the order total before refunding.', name='refunds'),
    ],
    capabilities=[AgentControl()],
)
```

Logfire now offers those as two blocks, `agent` and `agent:refunds`, so the refund rule can be
reworded without touching the rest of the prompt. The same works wherever instructions are written:
`@agent.instructions(name=...)`, `Capability(instructions=...)`, a toolset's `get_instructions()`. A
name is relative to whatever owns it, and Pydantic AI qualifies it with the source, so
`name='limits'` on a toolset becomes `toolset:weather:limits` and no two can collide.

!!! warning "Take a prompt over, don't paste it back in"
    Logfire can also *add* instruction blocks, which is the one thing that would let the same text
    reach the model twice. Copying your agent's prompt out of a trace and publishing it as an added
    block, while that text is still in the code, sends every line of it twice -- and freezes anything
    per-request (`Today is 2026-07-29.`) at whatever it said when you copied. Edit the existing
    blocks instead.

## When something doesn't apply

If Logfire is unreachable, has nothing published, or has something this SDK is too old to understand,
the agent runs exactly as written -- a published config can never crash a run. The one exception is an
agent with no model of its own: `Agent(None, ...)` has nothing to fall back to, so it raises rather
than guessing. Keep a model in code if you want a Logfire outage to be a non-event.

Nothing this SDK can't act on costs more than the piece that contains it. A setting it doesn't
recognize drops that setting; a malformed tool entry drops that tool; a block it can't apply drops
that block. The rest of the config still applies, and each drop warns once per process naming what it
skipped, rather than once per run.

Only an explicit `"instructions": null` removes a block. An entry with an `id` and no `instructions`
key -- `{"id": "agent", "text": "..."}`, say, with the key misspelled -- is skipped with a warning and
the block keeps its code-defined text, rather than taking the prompt with it.

A published change can also be perfectly valid and still reach nothing *here*: an instruction block
this deployment never assembles (or only computes per request), a tool no toolset advertises, a
parameter a patched tool does not have, a rename another tool already answers to, a setting the
contract has no field for, a whole section it has none for, a `timeout` no request could be given.
`on_unmatched` decides what that costs:

```python {test="skip"}
AgentControl(label='production', on_unmatched='error')
```

- `'warn'` (the default) emits a `UserWarning` once per process, so a change Logfire shows and the
  agent isn't making is visible without stopping anything.
- `'error'` raises `UserError` naming everything the request could not apply, for a deployment that
  would rather stop than run with part of its published config unapplied.
- `'ignore'` applies nothing and says nothing.

Every section is judged before any of it is reported, so `'error'` fails on the settings key *and* the
tool override *and* the instruction block, rather than on whichever the agent reached first.

Warning is the default because tool availability is dynamic: one config is applied across services
that need not all install the same toolsets, and a toolset can advertise different tools from one
request to the next, so a change that reaches nothing now is not necessarily wrong.

Changes are picked up **once per run**. Publishing mid-run takes effect on the next run, which is what
lets every span of a run agree on the version that produced it.

## Registration and the baseline

**Nothing in your process ever writes to a variable.** Your agent tells Logfire what its code says,
and creating or updating a config from that happens in Logfire.

**Registering the agent.** `AgentControl` emits one `agent_control_config_hint` span carrying the name
of the variable the config belongs in and the code baseline below -- everything a config would be
created from. Until one is created, the agent keeps running exactly as the code says.

It reports **whether or not a config resolved**, and `agent_control.resolution_reason` on the span
says which: a `'code_default'` baseline is one Logfire can offer to create a config from, and a
`'resolved'` one is how the editor knows whether the baseline it stored still matches the code it is
showing changes against. An agent that reported only while unconfigured would go quiet the moment you
configured it, and the baseline you diff against would describe the deployment it was created from.

The hint is emitted **once per process per agent**, on the first model request that has a baseline to
describe. So an agent that never reaches a model reports nothing, and a restart after a code change
reports the new baseline.

**Which deployment reported it.** A variable is derived from the agent's name alone, so two services
that each define a `checkout_assistant`, and the same service's dev and prod, all land on one
`agent__checkout_assistant` -- a variable holds one value per project, and the dev/prod split is its
labels. The span therefore carries `agent_control.service_name`, `agent_control.environment`, and
`agent_control.service_version`, taken from what you already told `logfire.configure()`. Each is left
off when Logfire does not know it, rather than sent empty. `service_version` is whatever Logfire
resolved for the running code, which is the current commit when your process runs in a git checkout.

`agent_control.baseline_sha256` digests the baseline itself, so two reports of the same code agree
and a code change is visible even across the once-per-process guard. It is always taken over the
whole baseline, so it is unchanged by a reduction (below) and still there when the baseline itself
could not be carried.

Reporting rather than writing is what makes three things true: the credential your deployment holds
stays read-only, a snapshot of your code can never overwrite a value a teammate saved in Logfire, and
an agent that only ever runs **inside a durable workflow** (Temporal, DBOS, ...) registers like any
other one -- a write from a workflow would not be replay-safe, and a span is.

**The baseline.** Logfire's editor shows published values as changes *to something*, and that
something is a snapshot of what your agent does in code: its prompt block by block, its model, its
settings, its tool definitions. It is documentation -- never resolved, never applied to a run -- so a
missing or stale snapshot cannot change what your agent does.

It is a snapshot of one request, which matters when things vary: a prompt or toolset that changes
with `deps`, the run's input, or the step within a run is sampled rather than described. An
instruction block computed per request contributes that it exists and never what it rendered to.
Tool definitions are different: the baseline records the names, descriptions, and parameter
descriptions assembled for that request. If a dynamic toolset derives those fields from `deps` or
user input, that text reaches the hint span and may become the variable's example. Keep request data
and secrets out of tool-definition metadata. Settings the contract has no field for are omitted,
which keeps `extra_headers` and `extra_body` out of the baseline.

A very large baseline is reduced rather than cut: if it does not fit on a span attribute, its tool
definitions are left out, and if it still does not fit it is left out entirely. Either way the hint
says which of the two happened, so a config created from it is never quietly a config for a different
agent than the one you wrote.

## Which agent in Logfire it controls

The agent's `name`, which you have to **set explicitly** -- `AgentControl` refuses an agent that has
none. Pydantic AI would otherwise infer one from the Python variable the agent was assigned to, and
an agent's config is not something a local rename should be able to move. The refusal happens where
the capability is wired up, not mid-run:

```python {test="skip"}
Agent('anthropic:claude-fable-5-1', capabilities=[AgentControl()])
# UserError: `AgentControl` without an explicit `name` reads the agent's `name`, and this agent has none.
```

The name is normalized the way Logfire normalizes an agent's name in your traces, and that is lossy:
`checkout-assistant`, `Checkout Assistant`, and `checkout_assistant` are one agent to Logfire,
including across services reporting to the same project, so two agents differing only in punctuation
share one config. Pass an explicit `name` to `AgentControl` to decouple it from the agent's name
entirely -- to keep two such agents apart, or to deliberately point several at one config.

`targeting_key`, `attributes`, and `render_template` work exactly as they do for
[`ManagedPrompt`](managed-prompt.md#targeting). `AgentControl.resolved` exposes the config the
current run picked up (`value`, `label`, `version`, `reason`), and is `None` outside a run.

## API reference

::: pydantic_ai_harness.logfire.AgentControl

The config a variable holds -- `AgentConfig` and the models it is built from -- is documented with the
contract itself, in [`logfire.agent_control`](https://github.com/pydantic/logfire/tree/main/logfire/agent_control).
