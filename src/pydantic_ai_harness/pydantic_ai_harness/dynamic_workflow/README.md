# Dynamic Workflow

Coordinate sub-agents in parallel, chain their results, or retry them in one tool call. Use
`DynamicWorkflow` when this coordination can run as Python rather than needing the orchestrator's
judgment after each step. On success, intermediate results stay out of the orchestrator's context
unless the script returns or prints them.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/dynamic_workflow/)

```mermaid
sequenceDiagram
    accTitle: Dynamic Workflow: coordinate sub-agents with one script
    accDescr: The orchestrator model calls run_workflow with one Python script, and the DynamicWorkflow capability runs it in the Monty sandbox. The capability starts two reviewer sub-agent runs in parallel, one for auth.py and one for parser.py, and their two reports stay in the sandbox. It passes both reports to a summarizer run and returns only the summary to the orchestrator model.
    participant model as orchestrator model
    participant cap as DynamicWorkflow()<br/>Monty sandbox
    participant agents as sub-agent runs
    model->>cap: run_workflow,<br/>one Python script
    cap->>agents: reviewer, auth.py
    cap->>agents: reviewer, parser.py,<br/>in parallel
    agents-->>cap: two reports
    Note over cap: both reports stay here
    cap->>agents: summarizer,<br/>both reports
    agents-->>cap: summary
    cap-->>model: summary only
```

## The idea

`DynamicWorkflow` gives the orchestrator one tool, `run_workflow`. The model writes a Python script
that runs in the [Monty](https://github.com/pydantic/monty) sandbox, with each named sub-agent
available as an async function. Like [Code Mode](../code_mode/README.md), it replaces repeated tool
calls with code; unlike Code Mode, it calls whole agents rather than individual tools.

## How this relates to Subagents

Use [`SubAgents`](../sub_agents/README.md) for occasional delegations or when the parent needs to
judge each result before continuing. Use `DynamicWorkflow` for scripted fan-out, chaining, voting,
or retries. You can switch between them without changing the underlying agents. Start with
`SubAgents` if you are unsure.

The controls they share cover the same ground at different levels. `DynamicWorkflow` budgets the
whole catalog because one script fans out over it; `SubAgents` budgets each delegate because the
parent picks one per call:

| Control | `DynamicWorkflow` | `SubAgents` |
|---|---|---|
| Retries of the parent's tool | `max_retries`, like other script tools such as `CodeMode` | `tool_retries` |
| Number of sub-agent runs | `max_agent_calls`, across the catalog | `SubAgent.max_calls`, per delegate |
| Usage limits of one sub-agent run | `sub_agent_usage_limits`, for every run | `SubAgent.usage_limits`, per delegate |
| Shared usage accounting | `forward_usage` | `forward_usage` |
| Sub-agent model | `inherit_model`, for every run | the `models` menu and `SubAgent.models` |

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

Each agent's `name` becomes its sandbox function name and must be a valid Python identifier. Its
`description` tells the model when to call it.

## What the model does with it

The model can review two files concurrently, then summarize both reports:

```python
import asyncio

reports = await asyncio.gather(
    reviewer(task='Review auth.py for bugs:\n<file contents>'),
    reviewer(task='Review parser.py for bugs:\n<file contents>'),
)
await summarizer(task='Summarize these review findings:\n' + '\n\n'.join(reports))
```

Call sub-agents with `await` and a keyword-only `task`. `asyncio.gather` runs calls concurrently
(except in a DBOS workflow where a sub-agent
[carries `DBOSDurability`](#running-under-other-durable-engines)); the last expression supplies the
tool result.

Each call is an isolated `Agent.run` with its own model loop, history, tools, and output.
Include all required context in `task`: calls do not remember earlier runs. Each run costs tokens;
use the [budgets below](#keeping-it-safe-budgets) to limit usage.

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
result = await critic(task='Score this answer: ...')
result['value']  # not result.value
```

The catalog the model sees renders each output type as a `TypedDict`, so it knows the fields and
reads them by subscript on its own.

Sub-agent runs cannot pause for [tool
approval](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) or a [pending model
response](https://pydantic.dev/docs/ai/core-concepts/batch/#ending-a-run-while-it-waits). The
catalog omits `DeferredToolRequests` and `PendingModelResponse`; a sub-agent run that ends with
either raises `RuntimeError` in the script. Pending requests are cancelled best effort; a batch is
cancelled only when no other request in it is waiting.

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
    drafter(task='explain idempotency keys on payments'),
    drafter(task='explain idempotency keys on payments'),
    drafter(task='explain idempotency keys on payments'),
)
# 2. Score each one. Structured output arrives as {"value": int, "reason": str}.
scores = await asyncio.gather(*[critic(task='Score this answer:\n' + d) for d in drafts])
# 3. Pick the winner and refine it, all in plain Python, no extra model turns.
best = max(range(len(drafts)), key=lambda i: scores[i]['value'])
await editor(task='Answer:\n' + drafts[best] + '\n\nCritique:\n' + scores[best]['reason'])
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

`max_duration_seconds` counts sandbox execution time, excluding waits for sub-agents. There is no
default duration cap. A CPU-bound loop that never awaits is not stopped by sub-agent budgets and
blocks the workflow thread under Temporal. Set `max_duration_seconds` to stop such loops. A partial
limits dict overrides only the named caps. Pass `'unlimited'` to remove sandbox resource limits; the
separate 10 MiB print cap remains. `max_duration_secs` is the deprecated name of
`max_duration_seconds` and still works, with a warning.

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
orchestrator = Agent('openai:gpt-5', capabilities=[workflow])

# later, from the host or from another tool:
workflow.reveal(fixer)
```

The new agent becomes callable on the next step, announced by an `AgentRevealPart` (a capability
part whose `agent_name` field names the sub-agent) containing its signature. The tool description
stays fixed to preserve the prompt-cache prefix. If compaction or a history processor removes the
announcement, it is sent again.

`reveal()` is append-only. Missing names, invalid identifiers, reserved keywords, and collisions
raise `UserError` immediately.

When one `DynamicWorkflow` instance is shared across concurrent runs, `reveal()` reaches every
in-flight run and joins the baseline for later runs.

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
- `asyncio.gather(...)` takes positional awaitables but no keyword arguments, including
  `return_exceptions=True`. It runs sub-agents and `asyncio.sleep` calls concurrently, except in a
  DBOS workflow where a sub-agent [carries `DBOSDurability`](#running-under-other-durable-engines),
  where sub-agent calls and sleeps run one at a time. Other task creation and wait APIs are
  unavailable.

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

`DBOSDurability` numbers a workflow's steps in the order they start, so the steps of concurrent
sub-agents would interleave by timing and recovery could replay one sub-agent's results into another
call. When any sub-agent in the catalog carries it, every sub-agent call the script makes, and every
`asyncio.sleep` it gathers, runs one at a time inside a DBOS workflow, including calls to sub-agents
without `DBOSDurability`. An `asyncio.gather` over sub-agents then takes the sum of their run times
rather than the longest. Each call runs as ordinary workflow code, as it would outside
`DynamicWorkflow`. A sub-agent that carries `DBOSDurability` keeps checkpoints for its model
requests, MCP tool calls, and tools decorated with `@DBOS.step`, and DBOS checks the roles those
`@DBOS.step` tools require. Recovery replays a call's finished steps and runs on from there. Other
function tools run as workflow code and run again on recovery, so put side effects in `@DBOS.step`
tools. A sub-agent without `DBOSDurability` keeps no checkpoints for its model requests, so recovery
runs those again. Recording each call as one DBOS step would let the calls overlap, but DBOS runs a
step called from inside another step as a plain function call, without its own checkpoint or role
check.

`AWSLambdaDurability` numbers a context's steps in the order they start too. When sub-agents carry it,
the calls a script makes before it next waits run as the branches of one Lambda `parallel` operation,
named after the tool, in the order the script made them. Sub-agents still run concurrently. Each
branch has a child context of its own, whose step IDs derive from the branch's position, so a
sub-agent's model requests and tool calls keep their own checkpoints. A branch's output is checkpointed
with the `serdes` in the sub-agent's `step_config`, like its steps. Recovery returns each finished
branch's recorded output, and an unfinished branch replays its own checkpoints and runs on from there.

Under other engines, sub-agents run concurrently.

### Scheduling sub-agent calls

`call_scheduler` decides when the script's sub-agent calls start. It is called with the run
context once per `run_workflow` call, and returns a `WorkflowCallScheduler`. Each call reaches its
`run` as a `WorkflowCall` carrying the sub-agent's `agent_name`, and starts when `run` awaits
`call.start()`. Its `flush` is awaited each time the script waits on its calls, once every call
made before the wait has reached `run`. A durable engine integration can use this to register
every call of a batch, in the order the script made them, before any of them starts:

```python
import asyncio

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow, WorkflowCall


class StartTogether:
    def __init__(self) -> None:
        self._held: list[asyncio.Future[None]] = []

    async def run(self, call: WorkflowCall) -> object:
        held = asyncio.get_running_loop().create_future()
        self._held.append(held)  # register `call.agent_name` with the engine here
        await held
        return await call.start()

    async def flush(self) -> None:
        for held in self._held:
            held.set_result(None)
        self._held.clear()


def start_together(ctx: RunContext[None]) -> StartTogether:
    return StartTogether()


reviewer = Agent('openai:gpt-5', name='reviewer', description='Reviews code for bugs.')
workflow = DynamicWorkflow(agents=[reviewer], call_scheduler=start_together)
```

A call only completes after `run` awaits its `call.start()`, so a scheduler that holds calls back
has to release them in `flush`. Under `AWSLambdaDurability`, the calls that have started when
`flush` returns run as one `parallel` operation, so a released call has to await `call.start()`
without suspending first: one that waits on something else first, such as a concurrency limit,
misses the operation and the script waits on it indefinitely. An exception that `run` raises
itself reaches the script as a `RuntimeError` naming only its type. Under `DBOSDurability` the
calls already start one at a time in dispatch order, and the `call_scheduler` factory is not called.

## API

```python
DynamicWorkflow(  # all parameters are keyword-only
    agents=[...],  # Sequence[AbstractAgent | WorkflowAgent], required
    tool_name='run_workflow',
    max_agent_calls=50,
    max_retries=3,
    forward_usage=True,
    inherit_model=False,  # True -> sub-agents run with the parent run's resolved model
    sub_agent_usage_limits=None,  # UsageLimits per sub-agent run; None -> parent run's limits when forwarding usage
    resource_limits=None,  # None -> 256 MB, no time cap; a dict overrides only named caps; 'unlimited' -> limits off
    call_scheduler=None,  # (RunContext) -> WorkflowCallScheduler; None starts each call when made
    id=None,  # required when defer_loading=True
    description=None,  # one-line catalog entry shown while deferred
    defer_loading=False,
)

workflow.reveal(agent)  # AbstractAgent | WorkflowAgent; validates before appending

WorkflowAgent(
    agent,  # AbstractAgent, required, positional
    name=None,  # sandbox function name; falls back to agent.name
    description=None,  # function docstring; None falls back to agent.description, '' hides it
)
```

## Further reading

- [Code Mode](../code_mode/README.md), the same sandbox, calling the agent's own tools instead of
  sub-agents.
- [SubAgents](../sub_agents/README.md), one-delegation-per-tool-call sub-agents, without the
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
