# Pydantic AI vs smolagents

smolagents takes an unusual position and takes it seriously: instead of asking the model for
structured tool calls, it asks the model to write Python, then runs that Python. A `CodeAgent` loops —
model writes code, sandbox runs it, output goes back — until the code calls `final_answer`. It's a
small library with few dependencies, it's clear about its limits, and it fits when the task is
computational.

Its default sandbox is a restricted interpreter, not a container, and it says so. Running
`import os` gets you *"Import of os is not allowed. Authorized imports are: collections, datetime,
itertools, math, queue, random, re, stat, statistics, time, unicodedata"*, and `open(...)` is refused
outright. For real isolation you escalate to one of the remote executors — Docker, E2B, Modal, Blaxel,
or your own.

Pydantic AI does structured tool calls by default, and can do the write-code approach too through
`CodeMode` in [pydantic-ai-harness](https://github.com/pydantic/pydantic-ai-harness), which runs the
model's Python inside the [Monty](https://github.com/pydantic/monty) sandbox. The difference that
matters more day to day is that one library is synchronous and the other isn't.

## Synchronous, and what that costs

smolagents' loop is blocking. `CodeAgent.run()` has no async counterpart, so a run owns the thread it's
on. Two consequences follow.

The first is concurrency. If the model wants three independent things done — three lookups, three API
calls — they happen one after another, because the generated code runs in a single interpreter on one
thread. In Pydantic AI the tools are `async def` and the model can ask for several at once:

```python {title="parallel_tool_calls.py"}
"""Parallel tool calls in one turn.

The model asks for three slow calls in one response. The async loop starts all
three before any of them finishes, so they overlap instead of queueing up.
"""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

events: list[str] = []


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart('slow', {'name': 'a'}),
                ToolCallPart('slow', {'name': 'b'}),
                ToolCallPart('slow', {'name': 'c'}),
            ]
        )
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(model))


@agent.tool
async def slow(ctx, name: str) -> str:
    events.append(f'start:{name}')
    await asyncio.sleep(0.01)
    events.append(f'end:{name}')
    return f'{name}:done'


async def main():
    await agent.run('run the three jobs')
    print('first three events:', events[:3])
    #> first three events: ['start:a', 'start:b', 'start:c']
    print('all started before any finished:', events[:3] == ['start:a', 'start:b', 'start:c'])
    #> all started before any finished: True
```


All three calls entered before any of them came back. Run them one at a time and the log reads
`start:a`, `end:a`, `start:b` instead. That difference is the whole of it: three slow lookups cost you
one slow lookup of wall time.

The second is stopping. smolagents does have a stop: `agent.interrupt()` sets a flag the loop checks
between steps. It works, but because the run is blocking you need another thread to call it, and what
you get afterwards is an error, not a resumable conversation. In Pydantic AI a
`CancellationToken` or a tool calling `ctx.cancel()` ends the run in `RunCancelled` carrying the
history, and you resume by passing that history to the next run.

## The other differences

**Trusted state.** smolagents builds tool schemas from docstrings and type hints, and there's nowhere
to put something the model shouldn't see. Pydantic AI's `deps_type` is a separate typed argument that
tools read and the model never does — so a database handle or a customer ID stays out of the
conversation entirely.

**Testing.** Both are testable offline, and smolagents deserves credit here: its `Model` base class is
a real place to plug a scripted stub, and we used one to drive a full `CodeAgent` run with no network.
Pydantic AI ships `TestModel` and `FunctionModel` instead of asking you to write one, and
`ALLOW_MODEL_REQUESTS = False` turns any stray real call into an error.

**Crash recovery.** smolagents has none in core. Pydantic AI's runs can be wrapped by Temporal, DBOS,
Prefect, Restate, Kitaru, or Airflow without changing the agent.

## Side by side

| | smolagents 1.26.0 | Pydantic AI 2.42 |
|---|---|---|
| How the model acts | Writes Python that a sandbox runs | Structured tool calls; `CodeMode` in the harness if you want code |
| Async | Synchronous; a run owns the thread | Async throughout, with `run_sync` when you want blocking |
| Parallel tool calls | Sequential in one interpreter | Genuinely concurrent |
| Sandbox by default | Restricted interpreter, 11 stdlib modules, no `open` | Tools are your functions; `CodeMode` runs model code in Monty |
| Stronger isolation | Docker, E2B, Modal, Blaxel, or remote executors | Sandbox providers in the harness |
| Stopping a run | `interrupt()` sets a flag checked between steps; needs another thread | `CancellationToken`, `ctx.cancel()`, `RunCancelled` with resumable history |
| Trusted state | Nothing separate from the prompt | `deps_type`, read by tools, invisible to the model |
| Crash recovery | None in core | Six engines wrap the agent object |
| Testing offline | Subclass `Model` yourself | `TestModel` calls your tools with no scripting; `FunctionModel` scripts them |
| Tracing | OpenInference spans under its own attribute names; zero `gen_ai.*` | The GenAI semantic conventions, 36 `gen_ai.*` attributes |
| Budgets | Step caps; no money limit | `cost_limit` in USD across 41 providers, checked before the next request |
| Evals | None in core | `pydantic-evals` in your test suite |

## Choose smolagents when

- The task is computational and writing Python is genuinely the best way for the model to express it.
- You want very few dependencies and a codebase you can read in an afternoon.
- You're in the Hugging Face ecosystem already.
- A restricted interpreter is the right level of isolation for what you're doing.

## Choose Pydantic AI when

- Your tools do I/O and you want them to overlap.
- You need a stop button that leaves you a conversation you can resume.
- Credentials and identity must sit where the model can't reach them.
- You want crash recovery, spend limits, and evals without assembling them.

## FAQ

**Can Pydantic AI do the write-code-instead-of-tool-calls thing?**
Yes, through `CodeMode` in the harness. The model writes one Python program that calls your tools as
functions — with loops and `asyncio.gather` — inside the Monty sandbox, instead of one round trip per
call.

**Is smolagents' sandbox safe?**
For accidents, largely yes, and the defaults are sensible. For a model that might be adversarially
prompted, their own documentation points you at Docker or a remote executor, which is the right
answer.

**What does smolagents do better?**
Being small. If the write-code approach suits your problem, it's less machinery than anything else,
and that's a real virtue.

---

*Checked against smolagents 1.26.0 and Pydantic AI 2.42 on 2026-09-10. The sandbox messages are the actual
errors from running `import os` and `open(...)` through its local executor; the absence of an async run and
the behaviour of `interrupt()` come from reading the installed package. The Pydantic AI example is executed by
this repository's test suite, and the timing shown is from that run. The `gen_ai.*` counts are distinct
semantic-convention attribute names found in each installed package's source; ours were also captured from a
live run through a plain OpenTelemetry exporter. We recheck this page's version pins and behaviour claims each
time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
