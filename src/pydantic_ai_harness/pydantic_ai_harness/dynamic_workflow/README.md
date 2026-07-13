# Dynamic Workflow

Let one agent coordinate a whole team of sub-agents by writing a small Python script.

> [!NOTE]
> Import this capability from its submodule -- there is no top-level `pydantic_ai_harness` re-export:
>
> ```python
> from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow
> ```
>
> The API may change between releases. Where practical, breaking changes ship with a deprecation warning.
> The extensions planned in [What is coming](#what-is-coming), structured sub-agent inputs and durable
> workflows, touch the sub-agent call contract itself.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/dynamic_workflow/)

## The idea

Say you have a few specialist agents. One reviews code. One summarizes findings. One writes the
final note. Each one is easy to call on its own. The hard part is the choreography between them. You
want to review three files at once, keep only the reports that found something, summarize those, and
hand the summary to the writer.

The usual way to do this is one tool call per step. The agent calls the reviewer and waits. It reads
the result, calls the reviewer again, and waits again. And so on. Every intermediate result travels
back into the agent's context, and every step that depends on the previous one is a separate model
turn.

`DynamicWorkflow` takes a different route. You hand it a catalog of named sub-agents, and it gives
the model a single tool, `run_workflow`. Inside that tool the model writes ordinary Python. Each of
your sub-agents is an `async` function it can call, loop over, and combine. The script runs to
completion in one tool call, and only its final value comes back to the model.

The choreography moves out of the conversation and into code.

Claude Code ships this same idea as a feature, also called dynamic workflows. Jarred Sumner used it
to port Bun from Zig to Rust: around 750,000 lines of Rust, 99.8% of the existing test suite passing,
and eleven days from first commit to merge. One workflow mapped the right Rust lifetime for every
struct field. The next wrote every file as a behavior-identical port, with hundreds of agents in
parallel and two reviewers on each file. A fix loop then drove the build and tests until both ran
clean. Claude Code runs that at the scale of a whole session. This capability brings the same idea
into your own Pydantic AI agents, inside a single `run_workflow` tool call. You can
[read the Bun story here](https://bun.com/blog/bun-in-rust).

> **Tip**
>
> If you have met [Code Mode](../../code_mode/README.md), this will feel familiar. It is the same
> sandbox and the same idea: write a script instead of many tool calls. The difference is what the
> script gets to call. In Code Mode it calls the agent's own tools. Here it calls whole sub-agents.

## How this relates to SubAgents

The harness has two delegation capabilities. They trade in the same currency, named and isolated
sub-agent runs, but they work at different altitudes:

- [`SubAgents`](../subagents/README.md) exposes one `delegate_task(agent_name, task)` tool. Each
  delegation is its own tool call and its own model turn. The parent calls, waits, reads the result
  into context, then decides the next step. It is simple to reason about. It is the right fit when
  delegations are occasional, or when each result needs the parent's judgment before the next one.
- `DynamicWorkflow` moves the choreography into a script. Fan-out, chaining, voting, and retry loops
  all run inside one tool call, and intermediate results never enter the parent's context. It is the
  right fit when the coordination between sub-agents is the actual work.

Start with `SubAgents` if you are not sure. A `delegate_task` orchestrator converts to a workflow
catalog without changing the sub-agents themselves.

## Install

The script runs inside the [Monty](https://github.com/pydantic/monty) sandbox, so install the extra:

```bash
uv add "pydantic-ai-harness[dynamic-workflow]"
```

## Your first workflow

Let's build the smallest thing that works. Two sub-agents, one orchestrator.

```python
from pydantic_ai import Agent
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

reviewer = Agent('openai:gpt-5', name='reviewer', description='Reviews code for bugs.')
summarizer = Agent('openai:gpt-5', name='summarizer', description='Summarizes findings.')

orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[DynamicWorkflow(agents=[reviewer, summarizer])],
)
```

That is the whole setup. Here is what each piece does:

1. `reviewer` and `summarizer` are plain agents. There is nothing special about them. They are the
   same `Agent` you already know.
2. Their `name` becomes the function name the model calls in the script, so pick names that are
   valid Python identifiers.
3. Their `description` tells the model what each one is for. The model reads these to decide how to
   wire them together, so write them the way you would document a function.
4. `DynamicWorkflow(agents=[...])` bundles them into one capability and hands the orchestrator a
   single `run_workflow` tool.

## What the model does with it

When the orchestrator decides to use the tool, it does not call your sub-agents one at a time. It
writes a script. For a "review these two files and summarize" task, the script it writes looks like
this:

```python
import asyncio

reports = await asyncio.gather(
    reviewer(task="Review auth.py for bugs:\n<file contents>"),
    reviewer(task="Review parser.py for bugs:\n<file contents>"),
)
await summarizer(task="Summarize these review findings:\n" + "\n\n".join(reports))
```

Here are the parts that matter most, because they are the core of how you use this capability:

- Each sub-agent is an `async` function. You call it with `await`.
- You pass the work as a single keyword argument, `task`. Always by keyword. Write
  `reviewer(task="...")`, not `reviewer("...")`.
- `asyncio.gather(...)` runs the two reviews at the same time instead of one after the other.
- The last line's value becomes the result the model sees. The intermediate `reports` list never
  leaves the sandbox.

> **Info: what "call a sub-agent" actually means**
>
> Each call is a full `Agent.run`. It has its own model loop, its own message history, its own
> tools, and its own typed output. It is not a lightweight function. It is a real agent doing real
> work. Two things follow from that, and both matter when you write or debug workflows:
>
> - **Calls are isolated.** A sub-agent remembers nothing from an earlier call. Put everything it
>   needs into `task`.
> - **Calls cost tokens and take time.** That is why this capability gives you budgets, which we
>   get to below.

Because the coordination is ordinary Python, it scales past one-shot fan-out. Take a rule like
"re-dispatch only the files that failed review, with the reviewer's issues attached, for up to two
more rounds." That is a `for` loop over `asyncio.gather`, with the retry task text rebuilt from each
failed review. Without this, the model would run that control flow turn by turn, one round-trip per
step, with every intermediate draft flowing through its context.

## Sub-agents can return structured data

A sub-agent returns whatever its `output_type` produces. By default that is a string. But give a
sub-agent a Pydantic model, and the script receives a `dict`:

```python
from pydantic import BaseModel

class Score(BaseModel):
    value: int
    reason: str

critic = Agent('openai:gpt-5', name='critic', description='Scores an answer 0-10.', output_type=Score)
```

Inside the script, the model reads the fields by subscript, the way you read a JSON object:

```python
result = await critic(task="Score this answer: ...")
result["value"]   # not result.value
```

The catalog the model sees renders each output type as a `TypedDict`, so it knows the fields and
reads them by subscript on its own.

## A complete, runnable example

Now let's put it together into something you can actually run. This orchestrator runs a small
tournament. It drafts three candidate answers in parallel, scores each one, picks the winner, and
refines it. All of that happens in a single tool call.

You need an Anthropic key and the `anthropic` package:

```bash
export ANTHROPIC_API_KEY=sk-...
uv run --with 'pydantic-ai-harness[dynamic-workflow]' --with anthropic --with logfire python wf.py
```

```python
# wf.py
import asyncio

import logfire
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

# With Logfire configured, the trace shows the orchestrator turn, the run_workflow call (including
# the exact script the model wrote), and every sub-agent run nested underneath it.
logfire.configure(send_to_logfire='if-token-present', service_name='dynamic-workflow')
logfire.instrument_pydantic_ai()

MODEL = 'anthropic:claude-sonnet-4-6'  # or 'anthropic:claude-opus-4-8'

class Score(BaseModel):
    value: int  # 0-10
    reason: str

drafter = Agent(
    MODEL,
    name='drafter',
    description='Writes one candidate answer to a task.',
    instructions='Write one concise candidate answer to the task.',
)
critic = Agent(
    MODEL,
    name='critic',
    description='Scores a candidate answer 0-10, returns {value, reason}.',
    output_type=Score,
    instructions='Score the candidate 0-10 with a one-line reason.',
)
editor = Agent(
    MODEL,
    name='editor',
    description='Improves an answer given a critique.',
    instructions='Improve the given answer using the critique. Return only the answer.',
)

orchestrator = Agent(
    MODEL,
    instructions=(
        'Use run_workflow to: draft 3 candidate answers in parallel, score each with the critic, '
        'pick the highest-scoring one, then have the editor refine it using that critique. '
        'Return the refined answer.'
    ),
    capabilities=[DynamicWorkflow(agents=[drafter, critic, editor])],
)

async def main() -> None:
    result = await orchestrator.run(
        'Explain, for a new hire, why our service uses idempotency keys on payment requests.',
        usage_limits=UsageLimits(request_limit=20),
    )
    logfire.info('done', answer=result.output, requests=result.usage.requests)

asyncio.run(main())
```

Given just those three sub-agents and the instructions, the model writes and runs a script along
these lines:

```python
import asyncio

# 1. Draft three candidates at the same time.
drafts = await asyncio.gather(
    drafter(task="explain idempotency keys on payments"),
    drafter(task="explain idempotency keys on payments"),
    drafter(task="explain idempotency keys on payments"),
)
# 2. Score each one. Structured output arrives as {"value": int, "reason": str}.
scores = await asyncio.gather(*[critic(task="Score this answer:\n" + d) for d in drafts])
# 3. Pick the winner and refine it, all in plain Python, no extra model turns.
best = max(range(len(drafts)), key=lambda i: scores[i]["value"])
await editor(task="Answer:\n" + drafts[best] + "\n\nCritique:\n" + scores[best]["reason"])
```

Read that script and notice what did not happen. The three drafts, the three scores, and the
selection logic never traveled back through the orchestrator's context. The model issued one tool
call and got back one answer. The comparison, the `max(...)`, and the string assembly are ordinary
Python running in the sandbox.

> **Tip**
>
> The [Logfire](https://pydantic.dev/logfire) trace is the best way to see what a workflow did.
> Each sub-agent run appears nested under the `run_workflow` span, and the span carries the exact
> `code` argument the model wrote, so you can read the script it actually ran.

## How results come back

The value of the script's last expression becomes the tool result. The model does not `print()` it.

For the common cases, that is all you need to know. If you want the exact rules, including what
happens when the script also prints for debugging, here they are:

> **Info: the precise return shape**
>
> | Sub-agent `output_type` | Value inside the script |
> | --- | --- |
> | `str` (the default) | the string |
> | a Pydantic model | a `dict`, read as `r['field']` |
> | list or scalar | the list or scalar |
>
> And here is how the final tool result is shaped:
>
> | The script... | The model receives |
> | --- | --- |
> | ends in a value, no print | that value directly (or `{}` if it is `None`) |
> | prints and ends in a value | `{"output": "<printed text>", "result": <value>}` |
> | prints and ends in `None` | `{"output": "<printed text>"}` |
>
> `print()` is for debug logging. It stringifies, so it is the wrong tool for returning structured
> data. Let the last expression carry the real result.

## Choosing sub-agent models

By default, each sub-agent uses the model it was constructed with. Set `inherit_model=True` when the
host passes a per-run model override to the parent agent, for example from a `/model` command, and
every sub-agent dispatch should follow that resolved parent model. Leave it `False` when a sub-agent
is deliberately pinned to a different model.

## Keeping it safe: budgets

A sub-agent is non-deterministic and costs tokens, and it can fan out into more sub-agents. So a
workflow needs two kinds of ceiling. It needs a cap on *how many* sub-agent runs happen, and a cap
on *how much* they spend. `DynamicWorkflow` gives you both, plus a guard against runaway sandbox
scripts.

### `max_agent_calls`: an exact count

```python
DynamicWorkflow(agents=[...], max_agent_calls=50)  # 50 is the default
```

This is a hard, host-enforced ceiling on the number of sub-agent runs in one parent run. It is one
budget shared across every `run_workflow` call in that run, not a per-script allowance. It holds
exactly, even when the script fans out with `asyncio.gather`. When the budget runs out, the workflow
stops calling sub-agents and returns a terminal result that tells the model to conclude with what it
has. That result includes the sub-agent results that did complete, so nothing you already paid for
is wasted.

> **Note**
>
> `max_agent_calls` is the only knob that bounds the number of runs exactly. Reach for it when you
> need a guarantee. The token-based limits below are budgets, not guarantees.

### `sub_agent_usage_limits` and `forward_usage`: bounding cost

`sub_agent_usage_limits` is a `UsageLimits` applied to each sub-agent run. How tight a ceiling it
gives depends on `forward_usage`, which controls whether the whole tree shares one usage counter:

| `forward_usage` | Counter | What the limit means |
| --- | --- | --- |
| `True` (default) | the parent's `usage` is shared across the tree | a tree-wide cap, checked against the shared counter. Under concurrent fan-out it is best-effort: several sub-agents can pass the check before any of them adds to the count. |
| `False` | each sub-agent run counts on its own | per-run limits. A per-run `total_tokens_limit` of `T` with `max_agent_calls` of `N` bounds the tree to roughly `N * T` tokens. |

> **Warning**
>
> The `usage_limits` you pass to the parent `run()` is not forwarded into sub-agents. Core does not
> expose that limit value to the capability, so it is re-checked only at the parent's own request
> boundaries. If you want to bound sub-agents, set `sub_agent_usage_limits`. If you want an exact
> ceiling on the number of runs, use `max_agent_calls`.

### `resource_limits`: guarding the script itself

These limits guard the orchestration script's own memory and allocations, not the sub-agents it
calls. The default backstop is 256 MB and 50 million allocations, with no time limit.

```python
DynamicWorkflow(agents=[...], resource_limits={'max_duration_secs': 30})
```

There is no default duration cap. To see why, it helps to know what the timer actually measures.

> **Info: what `max_duration_secs` measures**
>
> Monty checks this limit once per bytecode step. So it measures the time your script spends
> running sandbox code. It does not measure wall-clock time.
>
> This is the part that matters for sub-agents. While your script waits on a sub-agent, it is
> suspended on the host. It is not running. That time does not count. The same is true whether you
> await one sub-agent at a time or fan several out with `asyncio.gather`. A normal workflow spends
> most of its time waiting, so the cap will not fire on it, no matter how long the sub-agents take.
>
> So what is the cap for? One thing: a pure-CPU runaway. Picture a `while True:` loop that never
> awaits. It burns a whole core and blocks the event loop, and none of the sub-agent budgets
> (`max_agent_calls`, `sub_agent_usage_limits`) can stop it, because it never calls a sub-agent. If
> you want that guard, set `max_duration_secs` yourself.
>
> Two more knobs. Pass `'unlimited'` to remove every limit. Pass a partial dict like
> `{'max_memory': ...}` and it merges onto the backstop, so you override only the caps you name and
> the rest keep their defaults.

### Workflows do not nest

A sub-agent cannot start its own workflow. If one tries, the nested `run_workflow` call returns a
terminal error instead of running.

> **Tip**
>
> Here is the practical rule: do not give the sub-agents in your catalog the `DynamicWorkflow`
> capability. They are the leaves of the orchestration, not orchestrators themselves.

## Renaming a sub-agent: `WorkflowAgent`

By default, a sub-agent shows up in the script under its own `name` and `description`. Sometimes you
want a different name or a different description for one particular workflow, without editing the
agent itself. Wrap it in a `WorkflowAgent`:

```python
from pydantic_ai_harness.dynamic_workflow import WorkflowAgent

DynamicWorkflow(
    agents=[
        WorkflowAgent(
            reviewer,
            name='check',
            description='Checks one code change and returns actionable review findings.',
        ),
    ],
)
```

Now the model calls `check(task=...)` instead of `reviewer(task=...)`. Passing a bare agent is just
shorthand for wrapping it in a `WorkflowAgent` with no overrides.

## Adding sub-agents while a run is going: `reveal()`

The catalog is fixed when a run starts, which keeps it in the prompt-cache prefix across turns. But
sometimes you learn during a run that a new sub-agent should be available, say once a fixer agent has
been provisioned. Keep a reference to the `DynamicWorkflow` instance and call `reveal()`:

```python
workflow = DynamicWorkflow(agents=[reviewer])
orchestrator = Agent('openai:gpt-5', deps_type=MyDeps, capabilities=[workflow])

# later, from the host or from another tool:
workflow.reveal(fixer)
```

The revealed sub-agent becomes callable on the next step. The model learns about it through a short
announcement message that carries the new function's signature. The `run_workflow` description itself
stays frozen at the agents present when the run started, so even a runtime reveal never moves the
prompt-cache prefix.

> **Note**
>
> `reveal()` is append-only. Once a sub-agent appears it stays for the rest of the run, and there
> is no way to remove or hide it again. Plan the catalog as something that only grows.
>
> It validates right away. A missing name, an invalid identifier, a reserved keyword, or a name
> collision raises `UserError` at the call site. And if you share one `DynamicWorkflow` instance
> across concurrent runs, `reveal()` reaches all in-flight runs and joins the baseline for runs that
> start afterward.

## Loading it only when needed: `defer_loading`

`DynamicWorkflow` carries a fair amount of instruction text, and most turns do not need it. You can
keep it collapsed to a one-line entry until the model actually loads it. That pays close to zero
tokens on turns that never orchestrate:

```python
DynamicWorkflow(
    agents=[reviewer, summarizer],
    id='workflow',
    defer_loading=True,
)
```

`defer_loading=True` needs a stable `id`. See
[on-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities)
for the full picture.

## What runs in the sandbox

The script runs in Monty, a subset of Python. The subset is what makes the sandbox safe, so it is
worth knowing where the edges are:

- No class definitions, and no third-party libraries.
- Useful standard-library modules: `asyncio`, `math`, `json`, `re`, `typing`. Import what you use.
  Other modules are unavailable or stubbed.
- No wall-clock or timing primitives. There is no `asyncio.sleep`, no `datetime.now()`, and no
  `time` module.
- `asyncio.gather(...)` runs sub-agents concurrently, but it does not support
  `return_exceptions=True`.

Before a script runs, it is statically type-checked against the sub-agent signatures. A misspelled
function, a positional `task`, or a wrong-typed argument costs one retry, but no sub-agent budget and
no sandbox execution.

> **Warning: errors abort the whole script**
>
> A sub-agent that raises cannot be caught inside the script. One failure aborts the whole script,
> and the model retries it. So write scripts where sub-agents do not depend on catching each other's
> errors. If a script does fail after some sub-agents already finished, the retry prompt lists those
> completed results, so the model can reuse them as plain values instead of paying for the same
> calls again.

## What is coming

A suspended Monty program is a small serializable value you can dump, reload, and fork. That points
at two patterns that do not ship yet. The first is forking one expensive shared prefix into N
best-of-N branches. The second is durable workflows that resume from a persisted snapshot after a
crash or a redeploy. Two smaller extensions are also planned: structured sub-agent inputs (a
`parameters` schema per `WorkflowAgent`, instead of only `task: str`) and first-class progress
streaming. Until then, set `event_stream_handler` on each sub-agent `Agent`, or use Logfire, to
watch sub-agent runs inside the one tool call.

## API

```python
DynamicWorkflow(                  # all parameters are keyword-only
    agents=[...],                 # Sequence[AbstractAgent | WorkflowAgent], required
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    inherit_model=False,          # True -> sub-agents run with the parent run's resolved model
    sub_agent_usage_limits=None,  # UsageLimits per sub-agent run; None -> pydantic-ai default
    resource_limits=None,         # None -> backstop (256 MB, 50M allocs, no time cap);
                                  # 'unlimited' -> off; a dict is merged onto the backstop
    id=None,                      # required when defer_loading=True
    description=None,             # one-line catalog entry shown while deferred
    defer_loading=False,
)

workflow.reveal(agent)            # AbstractAgent | WorkflowAgent; validates before appending

WorkflowAgent(
    agent,                        # Agent, required, positional
    name=None,                    # sandbox function name; falls back to agent.name
    description=None,             # function docstring; falls back to agent.description
)
```

## Further reading

- [Code Mode](../../code_mode/README.md), the same sandbox, calling the agent's own tools instead of
  sub-agents.
- [SubAgents](../subagents/README.md), one-delegation-per-tool-call sub-agents, without the
  scripted choreography.
- [Tool use via code](https://www.anthropic.com/engineering/code-execution-with-mcp) (Anthropic),
  the mechanism this applies to sub-agents.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
  (Anthropic), the orchestration patterns a script can express.
- [Rewriting Bun in Rust](https://bun.com/blog/bun-in-rust) (Bun), the same pattern at scale: Jarred
  Sumner's port of Bun from Zig to Rust with Claude Code dynamic workflows.
- [Capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/) and
  [on-demand capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/#on-demand-capabilities).
- [Monty](https://github.com/pydantic/monty), the sandbox.
