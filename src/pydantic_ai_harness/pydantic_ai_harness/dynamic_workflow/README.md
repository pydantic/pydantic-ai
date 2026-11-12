# Dynamic Workflow

Let one agent coordinate a whole team of sub-agents by writing a small Python script.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/dynamic_workflow/)

![The orchestrator agent has one tool, run_workflow. Its model writes a Python script that runs in the Monty sandbox, where each sub-agent is an async function: the script runs reviewer twice in parallel with asyncio.gather, then passes both reports to summarizer. Each call is a separate sub-agent run with its own model loop. Only the script's final value, the summary, goes back to the orchestrator's next model request.](https://pydantic.dev/docs/ai/img/harness/dynamic-workflow.svg)

## The idea

Say you have a few specialist agents: one reviews code, one summarizes findings, one writes the
final note. Each is easy to call on its own; the hard part is the choreography between them. You
want to review three files at once, keep only the reports that found something, summarize those, and
hand the summary to the writer.

The usual way to do this is one tool call per step. The agent calls the reviewer and waits, reads
the result, calls the reviewer again, and so on. Every intermediate result travels back into the
agent's context, and every step that depends on the previous one is a separate model turn.

`DynamicWorkflow` instead hands the model a catalog of named sub-agents and a single tool,
`run_workflow`. Inside that tool the model writes ordinary Python: each sub-agent is an `async`
function it can call, loop over, and combine. The script runs to completion in one tool call, and
only its final value comes back to the model.

[Code Mode](../code_mode/README.md) uses the same sandbox and the same idea: write a script instead
of many tool calls. In Code Mode the script calls the agent's own tools; here it calls whole
sub-agents.

## How this relates to SubAgents

The harness has two delegation capabilities. They trade in the same currency, named and isolated
sub-agent runs, but they work at different altitudes:

- [`SubAgents`](../subagents/README.md) exposes one `delegate_task(agent_name, task)` tool. Each
  delegation is its own tool call and its own model turn. It is the right fit when delegations are
  occasional, or when each result needs the parent's judgment before the next one.
- `DynamicWorkflow` moves the choreography into a script. Fan-out, chaining, voting, and retry loops
  all run inside one tool call, and only what the script returns or prints (plus bounded previews of
  completed results on failure or budget exhaustion) reaches the parent. It is the right fit when the
  coordination between sub-agents is the actual work.

Start with `SubAgents` if you are not sure. A `delegate_task` orchestrator converts to a workflow
catalog without changing the sub-agents themselves.

## Install

The script runs inside the [Monty](https://github.com/pydantic/monty) sandbox, so install the extra:

uv:

```bash
uv add "pydantic-ai-harness[dynamic-workflow]"
```

pip:

```bash
pip install "pydantic-ai-harness[dynamic-workflow]"
```

## Your first workflow

Two sub-agents, one orchestrator:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DynamicWorkflow

reviewer = Agent('openai:gpt-5', name='reviewer', description='Reviews code for bugs.')
summarizer = Agent('openai:gpt-5', name='summarizer', description='Summarizes findings.')

orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[DynamicWorkflow(agents=[reviewer, summarizer])],
)
```

`reviewer` and `summarizer` are plain agents. Their `name` becomes the function name the model
calls in the script, so pick names that are valid Python identifiers. Their `description` tells the
model what each one is for; write it the way you would document a function.
`DynamicWorkflow(agents=[...])` bundles them into one capability and hands the orchestrator a single
`run_workflow` tool.

## What the model does with it

When the orchestrator decides to use the tool, it does not call your sub-agents one at a time. It
writes a script. For a "review these two files and summarize" task:

```python
import asyncio

reports = await asyncio.gather(
    reviewer(task="Review auth.py for bugs:\n<file contents>"),
    reviewer(task="Review parser.py for bugs:\n<file contents>"),
)
await summarizer(task="Summarize these review findings:\n" + "\n\n".join(reports))
```

The parts that matter:

- Each sub-agent is an `async` function. You call it with `await`.
- You pass the work as a single keyword argument, `task`, always by keyword:
  `reviewer(task="...")`, not `reviewer("...")`.
- `asyncio.gather(...)` runs the two reviews concurrently instead of one after the other.
- The last line's value becomes the result the model sees. The intermediate `reports` list never
  leaves the sandbox.

> **Info: what "call a sub-agent" actually means**
>
> Each call is a full `Agent.run`, with its own model loop, message history, tools, and typed
> output. Two things follow:
>
> - **Calls are isolated.** A sub-agent remembers nothing from an earlier call. Put everything it
>   needs into `task`.
> - **Calls cost tokens and take time.** That is why this capability gives you budgets, below.

Because the coordination is ordinary Python, it scales past one-shot fan-out. Take a rule like
"re-dispatch only the files that failed review, with the reviewer's issues attached, for up to two
more rounds." That is a `for` loop over `asyncio.gather`, with the retry task text rebuilt from each
failed review. Without this, the model would run that control flow turn by turn, one round-trip per
step, with every intermediate draft flowing through its context.

## Sub-agents can return structured data

A sub-agent returns whatever its `output_type` produces. The default is a string, but give a
sub-agent a Pydantic model and the script receives a `dict`:

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

An `output_type` can include
[`DeferredToolRequests`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) so a top-level
caller can approve tools and resume the run, or
[`PendingModelResponse`](https://pydantic.dev/docs/ai/core-concepts/batch/#ending-a-run-while-it-waits) so
it can check on a model request the provider is still processing later. A workflow script cannot do
either, so the catalog leaves both out of the sub-agent's return type, and a sub-agent run that ends
with one raises a `RuntimeError` in the script, like any other sub-agent failure. A model request
it left pending is cancelled, best effort: a
[batch](https://pydantic.dev/docs/ai/core-concepts/batch/) is only cancelled once no other request in
it is still waiting.

## A complete, runnable example

This orchestrator drafts three candidate answers in parallel, scores each one, picks the winner,
and refines it, all in a single tool call.

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

from pydantic_ai_harness import DynamicWorkflow

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

The three drafts, the three scores, and the selection logic never pass through the orchestrator's
context: the model issues one tool call and gets back one answer.

> **Tip**
>
> The [Logfire](https://pydantic.dev/logfire) trace is the best way to see what a workflow did.
> Each sub-agent run appears nested under the `run_workflow` span, and the span carries the exact
> `code` argument the model wrote, so you can read the script it actually ran. To watch sub-agent
> runs inside the one tool call, set `event_stream_handler` on each sub-agent `Agent`.

## How results come back

The value of the script's last expression becomes the tool result. The model does not `print()` it.

> **Info: the precise return shape**
>
> | Sub-agent `output_type` | Value inside the script |
> | --- | --- |
> | `str` (the default) | the string |
> | a Pydantic model | a `dict`, read as `r['field']` |
> | list or scalar | the list or scalar |
>
> The final tool result is shaped as follows:
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

A sub-agent is non-deterministic, costs tokens, and can fan out into more sub-agents.
`DynamicWorkflow` gives you a hard count ceiling, token budgets, and a guard against runaway sandbox
scripts.

### `max_agent_calls`: an exact count

```python
DynamicWorkflow(agents=[...], max_agent_calls=50)  # 50 is the default
```

A hard, host-enforced ceiling on the number of sub-agent runs in one parent run. It is one budget
shared across every `run_workflow` call in that run, not a per-script allowance, and it holds
exactly even when the script fans out with `asyncio.gather`. When the budget runs out, the workflow
stops calling sub-agents and returns a terminal result that tells the model to conclude with what it
has. That result includes bounded previews of up to the 20 most recent completed sub-agent results.

> **Note**
>
> `max_agent_calls` is the only knob that bounds the number of runs exactly. The token-based limits
> below are budgets, not exact ceilings.

### `sub_agent_usage_limits` and `forward_usage`: bounding cost

`sub_agent_usage_limits` is a `UsageLimits` applied to each sub-agent run. How tight a ceiling it
gives depends on `forward_usage`, which controls whether the whole tree shares one usage counter:

| `forward_usage` | Counter | What the limit means |
| --- | --- | --- |
| `True` (default) | the parent's `usage` is shared across the tree | a tree-wide cap, checked against the shared counter. Under concurrent fan-out it is best-effort: several sub-agents can pass the check before any of them adds to the count. |
| `False` | each sub-agent run counts on its own | per-run limits. A per-run `total_tokens_limit` of `T` with `max_agent_calls` of `N` bounds the tree to roughly `N * T` tokens. |

Without `sub_agent_usage_limits`, sub-agents sharing the counter (`forward_usage=True`) run under
the `usage_limits` you pass to the parent `run()`, so the parent's budget bounds the whole tree. With
`forward_usage=False` they fall back to Pydantic AI's default `request_limit` of 50 per run. For an
exact ceiling on the number of runs, use `max_agent_calls`.

### `resource_limits`: guarding the script itself

These limits guard the orchestration script's own memory, not the sub-agents it calls. The default
backstop is 256 MB with no time limit. Printed output is collected separately with Monty's 10 MiB
default cap.

```python
DynamicWorkflow(agents=[...], resource_limits={'max_duration_seconds': 30})
```

> **Info: what `max_duration_seconds` measures**
>
> Monty checks this limit once per bytecode step, so it measures the time your script spends
> running sandbox code, not wall-clock time. While the script waits on a sub-agent (one at a time
> or fanned out with `asyncio.gather`) it is suspended on the host and that time does not count, so
> the cap will not fire on a normal workflow no matter how long the sub-agents take.
>
> What it catches is a pure-CPU runaway: a `while True:` loop that never awaits. That loop burns a
> core in the sandbox worker (and blocks the workflow thread under Temporal), and none of the
> sub-agent budgets (`max_agent_calls`, `sub_agent_usage_limits`) can stop it because it never calls
> a sub-agent. There is no default duration cap; set `max_duration_seconds` yourself for that guard.
>
> Pass `'unlimited'` to remove every sandbox resource limit; the separate 10 MiB print cap remains.
> A partial dict like `{'max_memory': ...}` merges onto the backstop, so you override only the caps
> you name. `max_duration_secs` is the deprecated name of `max_duration_seconds` and still works,
> with a warning.

### Workflows do not nest

A sub-agent cannot start its own workflow. A sub-agent that has the `DynamicWorkflow` capability is
not offered `run_workflow` while it runs inside a workflow.

> **Tip**
>
> Do not give the sub-agents in your catalog the `DynamicWorkflow` capability. They are the leaves
> of the orchestration, not orchestrators.

## Renaming a sub-agent: `WorkflowAgent`

By default, a sub-agent shows up in the script under its own `name` and `description`. To give it a
different name or description for one workflow without editing the agent itself, wrap it in a
`WorkflowAgent`:

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

The catalog is fixed when a run starts, which keeps it in the prompt-cache prefix across turns. To
make a new sub-agent available during a run (say once a fixer agent has been provisioned), keep a
reference to the `DynamicWorkflow` instance and call `reveal()`:

```python
workflow = DynamicWorkflow(agents=[reviewer])
orchestrator = Agent('openai:gpt-5', deps_type=MyDeps, capabilities=[workflow])

# later, from the host or from another tool:
workflow.reveal(fixer)
```

The revealed sub-agent becomes callable on the next step. The model learns about it through a short
system-level announcement message that carries the new function's signature. The `run_workflow`
description itself stays frozen at the agents present when the run started, so even a runtime reveal
never moves the prompt-cache prefix. If compaction or a history processor removes the announcement,
the capability announces the sub-agent again on the next step.

> **Note**
>
> `reveal()` is append-only: once a sub-agent appears it stays for the rest of the run.
>
> It validates right away. A missing name, an invalid identifier, a reserved keyword, or a name
> collision raises `UserError` at the call site. And if you share one `DynamicWorkflow` instance
> across concurrent runs, `reveal()` reaches all in-flight runs and joins the baseline for runs that
> start afterward.

## Loading it only when needed: `defer_loading`

`DynamicWorkflow` carries a fair amount of instruction text, and most turns do not need it. Keep it
collapsed to a one-line entry until the model loads it:

```python
DynamicWorkflow(
    agents=[reviewer, summarizer],
    id='workflow',
    defer_loading=True,
)
```

`defer_loading=True` needs a stable `id`. See
[on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/)
for the full picture.

## What runs in the sandbox

The script runs in Monty, a subset of Python:

- No third-party libraries.
- Importable standard-library modules include `sys`, `typing`, `asyncio`, `math`, `json`, `re`,
  `unicodedata`, `datetime`, `time`, and `random`. Import what you use. There is no filesystem
  or environment access.
- No clock or randomness: `datetime.datetime.now()`, `datetime.date.today()`, `time.time()`, and
  unseeded `random` fail. `time.sleep` and `asyncio.sleep` really wait; with `max_duration_seconds` set,
  a script may sleep for at most that long in total.
- `asyncio.gather(...)` runs sub-agents concurrently with positional awaitables but no keyword
  arguments, including `return_exceptions=True`. Other task creation and wait APIs are unavailable.

Before a script runs, it is statically type-checked against the sub-agent signatures. An ordinary,
statically provable mistake such as a misspelled function, a positional `task`, or a wrong-typed
argument costs one retry but no sub-agent budget or sandbox execution. Values typed as `Any` can
reach runtime validation; they are still rejected before a sub-agent runs.

> **Warning: an uncaught error aborts the whole script**
>
> A sub-agent failure surfaces as `RuntimeError`. The script can catch it with
> `try`/`except RuntimeError`; an uncaught failure aborts the whole script and the model retries it.
> If a script does fail after some sub-agents already finished, the retry prompt lists bounded
> previews of up to the 20 most recent completed results. The model can reuse an untruncated preview
> as a plain value instead of paying for the same call again.

## Running inside a Temporal workflow

Attach `TemporalDurability` to the orchestrating agent alongside `DynamicWorkflow`, and to each
sub-agent whose model requests should run as activities, then register every agent with its own
`AgentPlugin`. The script runs in workflow code, like [Code Mode's `run_code`](../code_mode/README.md#temporal-durability),
and is re-executed during replay against the recorded sub-agent results.
Unlike Code Mode, a `max_duration_seconds` you set still applies inside the workflow, so a script that
runs close to it can stop at a different point on replay and cause a `NondeterminismError`.

## Running under other durable engines

`DBOSDurability` numbers steps in the order they start, so the steps of concurrent sub-agents would
interleave by timing and recovery could replay one sub-agent's results into another call. When a
sub-agent carries it, each call to that sub-agent is instead recorded as a single DBOS step, whose ID
is drawn in the order the script makes the calls. Sub-agents still run concurrently. Inside that step,
DBOS runs the sub-agent's own steps, including tools decorated with `@DBOS.step`, as plain function
calls, without their own checkpoints or DBOS role checks. Recovery returns each finished call's
recorded output and runs an unfinished call again from the start, and a replayed call's usage is not
added to the run's usage again.

`AWSLambdaDurability` numbers a context's steps in the order they start too. When sub-agents carry it,
the calls a script makes before it next waits run as the branches of one Lambda `parallel` operation,
named after the tool, in the order the script made them. Sub-agents still run concurrently. Each
branch has a child context of its own, whose step IDs derive from the branch's position, so a
sub-agent's model requests and tool calls keep their own checkpoints. A branch's output is checkpointed
with the `serdes` in the sub-agent's `step_config`, like its steps. Recovery returns each finished
branch's recorded output, and an unfinished branch replays its own checkpoints and runs on from there.

Under other engines, sub-agents run concurrently.

## API

```python
DynamicWorkflow(                  # all parameters are keyword-only
    agents=[...],                 # Sequence[AbstractAgent | WorkflowAgent], required
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    inherit_model=False,          # True -> sub-agents run with the parent run's resolved model
    sub_agent_usage_limits=None,  # UsageLimits per sub-agent run; None -> parent run's limits when forwarding usage
    resource_limits=None,         # None -> backstop (256 MB, no time cap);
                                  # 'unlimited' -> sandbox limits off; a dict merges onto the backstop
    id=None,                      # required when defer_loading=True
    description=None,             # one-line catalog entry shown while deferred
    defer_loading=False,
)

workflow.reveal(agent)            # AbstractAgent | WorkflowAgent; validates before appending

WorkflowAgent(
    agent,                        # AbstractAgent, required, positional
    name=None,                    # sandbox function name; falls back to agent.name
    description=None,             # function docstring; None falls back to agent.description, '' hides it
)
```

## Further reading

- [Code Mode](../code_mode/README.md), the same sandbox, calling the agent's own tools instead of
  sub-agents.
- [SubAgents](../subagents/README.md), one-delegation-per-tool-call sub-agents, without the
  scripted choreography.
- [Tool use via code](https://www.anthropic.com/engineering/code-execution-with-mcp) (Anthropic),
  the mechanism this applies to sub-agents.
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
  (Anthropic), the orchestration patterns a script can express.
- [Rewriting Bun in Rust](https://bun.com/blog/bun-in-rust) (Bun), the same pattern at scale: Jarred
  Sumner's port of Bun from Zig to Rust with Claude Code dynamic workflows.
- [Capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) and
  [on-demand capabilities](https://pydantic.dev/docs/ai/capabilities/on-demand/).
- [Monty](https://github.com/pydantic/monty), the sandbox.
