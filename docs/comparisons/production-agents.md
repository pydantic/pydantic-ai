# What a production agent needs

Any framework can run an agent. The gap between a demo and something you'd put in front of customers
is made of unglamorous things: keeping secrets away from the model, stopping a run without losing it,
capping what it can spend, and knowing what it did afterwards.

This is our list of what that takes. Use it as a checklist against anything you're evaluating,
including us. Every item below has a script you can copy. This repository's test suite runs all of
them on every commit, so what's printed is what the code prints today, not what it printed when
someone wrote the page.

## 1. The model can't reach your secrets

A database password shouldn't be in a conversation the model can read. In Pydantic AI, trusted state
goes in a separate typed argument: [`deps_type`][pydantic_ai.Agent] plus
[`RunContext`][pydantic_ai.tools.RunContext]. Tools read it; the model gets tool definitions and
nothing else.

That is a typed dependency API, not a secrecy boundary other frameworks lack. OpenAI's `TContext` is
local and is not sent to the LLM. Google ADK state is programmatic unless you interpolate it into
instructions. The difference is that `deps_type` is in the agent's type, so your type checker knows
the shape and a tool cannot forget to take it.

```python {title="deps_boundary.py"}
"""Trusted state lives in deps. Tools read it; the model never sees it."""
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext, capture_run_messages


@dataclass
class Warehouse:
    db_password: str


agent = Agent('openai:gpt-5.6-luna', deps_type=Warehouse)


@agent.tool
def check_warehouse(ctx: RunContext[Warehouse], probe: str) -> str:
    ok = ctx.deps.db_password == 'hunter2-keep-secret'
    return f'warehouse:{probe}:{"ok" if ok else "auth-failed"}'


with capture_run_messages() as msgs:
    result = agent.run_sync(
        'Is the warehouse database ready?',
        deps=Warehouse(db_password='hunter2-keep-secret'),
    )
payload = str(msgs)
leaked = 'hunter2' in payload
print(f'request payload contained the db password: {leaked}')
#> request payload contained the db password: False
print(f'tool executed with deps ({result.output!r})')
#> tool executed with deps ('Warehouse is up.')
assert not leaked
```


## 2. Tools that aren't there until they're needed

An agent with sixty tools is a worse agent. Capabilities can wait until the model asks for them.
Inspect what the model is offered with a [`Hooks`][pydantic_ai.capabilities.Hooks]
`before_model_request` hook: on the first request only `load_capability` is visible, then after the
model loads `refunds`, `check_refund` is too.

How that withholding is represented on the wire depends on the provider. OpenAI-style requests omit
the deferred definition; Anthropic still sends it, marked `defer_loading=True`. Either way the model
cannot call the tool until the capability is loaded.

```python {title="deferred_capability.py"}
"""Deferred capability: the model cannot call what it hasn't loaded."""
from pydantic_ai import Agent, ModelRequestContext, RunContext, capture_run_messages
from pydantic_ai.capabilities import Capability, Hooks
from pydantic_ai.messages import ToolReturnPart

refunds = Capability(
    id='refunds',
    description='Use when the customer asks about a refund.',
    instructions='Always confirm the order ID before answering.',
    defer_loading=True,
)


@refunds.tool_plain
def check_refund(order_id: str) -> str:
    """Look up whether an order was refunded."""
    return f'Order {order_id}: refunded.'


declared: list[list[str]] = []
hooks = Hooks()


@hooks.on.before_model_request
async def record_tools(
    ctx: RunContext, request_context: ModelRequestContext
) -> ModelRequestContext:
    params = request_context.model_request_parameters
    declared.append(sorted(t.name for t in params.declared_function_tools))
    return request_context


agent = Agent('openai:gpt-5.6-luna', capabilities=[refunds, hooks])
with capture_run_messages() as msgs:
    agent.run_sync('Was order A-4471 refunded?')
refund_ran = any(
    isinstance(p, ToolReturnPart) and p.tool_name == 'check_refund' for m in msgs for p in m.parts
)
print('tools declared on the first request:', declared[0])
#> tools declared on the first request: ['load_capability']
print('check_refund executed after it was loaded:', refund_ran)
#> check_refund executed after it was loaded: True
assert declared[0] == ['load_capability']
assert refund_ran
```


That's least privilege without the ceremony, and it keeps the tool list short enough for the model to
choose well.

## 3. Stopping a run gives you something back

Users close tabs. Quotas run out. A tool discovers the job shouldn't continue.

A tool can stop its own run:

```python {title="cancel_from_tool.py"}
"""A tool may stop the run. ctx.cancel() requests cancellation; the run ends
in a catchable RunCancelled carrying everything completed before it stopped.
"""
import asyncio

from pydantic_ai import Agent, RunCancelled, RunContext

agent = Agent('openai:gpt-5.6-luna')


@agent.tool
async def look_up_shipment(ctx: RunContext, order_id: str) -> str:
    ctx.cancel()  # cooperative: returns normally, lands at the next await
    await asyncio.sleep(0)
    return 'never used'


try:
    agent.run_sync('Look up shipment for order C-110.')
    print('BUG: run completed')
except RunCancelled as exc:
    history = exc.all_messages()
    print(f'run ended with RunCancelled; completed work preserved ({len(history)} message(s))')
    #> run ended with RunCancelled; completed work preserved (2 message(s))
    assert len(history) >= 1
```


And one token can stop several runs at once, from another thread, which is what a stop button in a UI
actually needs. Hang the work in the tool, not in a fake model:

```python {title="cancel_token_thread.py"}
"""A stop button, from another thread. CancellationToken interrupts a blocked
run_sync(); the run ends in RunCancelled instead of hanging forever.
"""
import asyncio
import threading
import time

from pydantic_ai import Agent, CancellationToken, RunCancelled, RunContext

agent = Agent('openai:gpt-5.6-luna')


@agent.tool
async def wait_on_warehouse(ctx: RunContext, order_id: str) -> str:
    await asyncio.sleep(3600)
    return 'never'


def main() -> None:
    token = CancellationToken()

    def stop_handler():
        time.sleep(0.1)
        token.cancel()  # thread-safe: delivered onto the run's loop

    stop = threading.Thread(target=stop_handler)
    stop.start()
    try:
        agent.run_sync('Wait on the warehouse for order H-900.', cancellation_token=token)
        print('BUG: run completed')
    except RunCancelled:
        stop.join()
        print('blocked run_sync interrupted from another thread -> RunCancelled')
        #> blocked run_sync interrupted from another thread -> RunCancelled
```


Either way the run ends by raising [`RunCancelled`][pydantic_ai.exceptions.RunCancelled], and that
exception carries the conversation, so resuming is just passing it to the next run. Cancellation from
outside (`asyncio.timeout()`, a task group shutting down) still behaves like normal Python
cancellation.

[`CancellationToken`][pydantic_ai.CancellationToken] is same-process state. It works on
[`Agent.run`][pydantic_ai.Agent.run], [`run_sync`][pydantic_ai.Agent.run_sync], and
[`run_stream`][pydantic_ai.Agent.run_stream] in this process. It cannot be passed through Temporal,
DBOS, or Prefect durable entry points; cancel that engine's workflow instead.

Most of the others can stop a run; what differs is what you're holding afterwards. The OpenAI SDK
cancels a streamed run. The Claude SDK sends an `interrupt()` control request, in streaming mode only.
LangGraph's `abort()` lives on its experimental v3 stream and closes the graph iterator. smolagents
sets a flag that's checked between steps. AG2 cancels through its durable task envelope. CrewAI has no
stop method at all, and Google ADK exposes no cancellation API anywhere on `Runner` or `LlmAgent`.

## 4. Budgets that stop things before they happen

A budget you find out about afterwards is a bill, not a budget. [`UsageLimits`][pydantic_ai.usage.UsageLimits]
is checked before the next request goes out and before a batch of tool calls executes, so nothing
runs when the run is already over its limit.

```python {title="usage_limits_atomic.py"}
"""A usage limit stops a run BEFORE a side-effect batch executes.

The model asks for two tool calls in one response; the limit allows one.
The whole batch is rejected, so neither tool runs: budget checks precede
execution, not polite suggestions after it.
"""
from pydantic_ai import Agent, RunContext, UsageLimitExceeded, UsageLimits

side_effects: list[tuple[str, int]] = []
agent = Agent('openai:gpt-5.6-luna')


@agent.tool
def credit_wallet(ctx: RunContext, amount: int) -> str:
    side_effects.append(('credited', amount))
    return 'ok'


try:
    agent.run_sync(
        'Credit the wallet twice for order C-110.',
        usage_limits=UsageLimits(tool_calls_limit=1),
    )
    print('BUG: exceeded the limit')
except UsageLimitExceeded as exc:
    print('stopped by:', type(exc).__name__)
    #> stopped by: UsageLimitExceeded
    print(f'tool executions that happened: {len(side_effects)}')
    #> tool executions that happened: 0
    assert not side_effects, 'a side effect ran despite the budget'
```


Spend is a related limit, in money instead of tokens, and it is not the same check. A dollar budget
needs prices for every model you might call, so it only works if the pricing data is part of the
library: [`cost_limit`][pydantic_ai.usage.UsageLimits.cost_limit] is backed by
[genai-prices](https://github.com/pydantic/genai-prices), which we maintain. Unlike
`tool_calls_limit`, `cost_limit` is checked after each response, because a response's output cost
isn't known until it arrives. The first request can exceed the cap. Pair it with
[`request_limit`][pydantic_ai.usage.UsageLimits.request_limit] or your provider's own spend controls;
don't treat it as a hard billing guarantee. Unpriced models emit
[`CostNotFoundWarning`][pydantic_ai.exceptions.CostNotFoundWarning] rather than running unconstrained
in silence.

Two other frameworks accept a dollar figure (Agno and the Claude Agent SDK both take `max_budget_usd`)
and in both cases it is the same thing: a value passed through to the Claude CLI's own budget. It
works when your model is Claude, through that CLI. LangChain, LangGraph, the OpenAI Agents SDK, CrewAI,
smolagents and Google ADK have no money limit at all; they cap tokens or iterations, which is a proxy
that gets worse every time model pricing changes.

```python {title="cost_limit.py"}
"""Cost is tracked in USD. Pair cost_limit with request_limit."""
from decimal import Decimal

from pydantic_ai import Agent, UsageLimits

agent = Agent('openai:gpt-5.6-luna')
result = agent.run_sync(
    'Say hello to the customer.',
    usage_limits=UsageLimits(cost_limit=Decimal('1.00'), request_limit=5),
)
print('run completed under cost_limit and request_limit')
#> run completed under cost_limit and request_limit
print('output:', result.output)
#> output: Hello.
```


## 5. History that repairs itself

Interrupt a run at the wrong moment and the conversation is left with a tool call that has no result.
Send that to a provider and you get a 400. Pydantic AI notices and fills the gap before the request
goes out, so a resumed conversation is always well formed:

```python {title="history_repair.py"}
"""Interrupted history repairs itself before it reaches the model.

A run that dies mid-tool leaves a dangling tool call. The next run closes
that call out (outcome=<interrupted>) before the request goes out, so the
provider never rejects the history as malformed.
"""
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

agent = Agent('openai:gpt-5.6-luna')


@agent.tool_plain
def tag_ticket(n: int) -> str:
    return f'tagged {n}'


interrupted = [
    ModelRequest(parts=[UserPromptPart(content='Tag ticket 9 as urgent.')]),
    ModelResponse(parts=[ToolCallPart('tag_ticket', {'n': 9}, tool_call_id='t1')], state='interrupted'),
]

with capture_run_messages() as msgs:
    agent.run_sync('Tag ticket 9 as urgent.', message_history=interrupted)
repaired = [
    p for m in msgs for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 't1'
]
print('dangling tool call was repaired before the request went out:', bool(repaired))
#> dangling tool call was repaired before the request went out: True
assert repaired, 'the dangling tool call was not repaired'
```


This is the quiet one that saves you a bad afternoon. It only works because the history is typed data
the framework owns rather than a dictionary of whatever the last thing put there.

## 6. Agents you can ship as configuration

An agent can be a YAML file, and the file is checked when it loads, not when it runs. A typo in
a prompt template fails immediately and names the field:

```python {title="spec_validation.py"}
"""An agent spec fails at load time, not at runtime.

Templates are validated against typed deps when the spec is built from a
dict: a single typo error names the field.
"""
from pydantic import BaseModel

from pydantic_ai import Agent


class UserContext(BaseModel):
    user_name: str
    user_role: str


bad = {
    'name': 'support',
    'model': 'openai:gpt-5.6-luna',
    'instructions': 'You are {{non_existent_field}}. Be nice.',
    'tools': [],
    'capabilities': [],
}
try:
    Agent.from_spec(bad, deps_type=UserContext)
    print('BUG: invalid template accepted')
except Exception as exc:
    print('rejected at load:', type(exc).__name__)
    #> rejected at load: TemplateSchemaError

good = {
    'name': 'support',
    'model': 'openai:gpt-5.6-luna',
    'instructions': 'You are {{user_role}} {{user_name}}. Be nice.',
    'tools': [],
    'capabilities': [],
}
agent = Agent.from_spec(good, deps_type=UserContext)
print('the corrected spec loads:', agent.name)
#> the corrected spec loads: support
assert agent.name == 'support'
```


Worth being precise about the limit: that check runs when a spec is loaded from a dictionary via
[`Agent.from_spec`][pydantic_ai.Agent.from_spec]. [`Agent.from_file`][pydantic_ai.Agent.from_file]
builds an [`AgentSpec`][pydantic_ai.AgentSpec] first and does not re-run that template check.

## 7. You can watch it work

The run emits typed events as it happens (the model starting to speak, each tool call and its result,
the final answer) and you consume them with a normal `async for`:

```python {title="event_stream.py"}
"""The run is an event stream you can observe or transform."""
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5.6-luna')


@agent.tool
def twice(ctx: RunContext, n: int) -> int:
    return n * 2


async def main():
    kinds = []
    async with agent.run_stream_events('What is 21 times 2?') as run:
        async for event in run:
            kinds.append(type(event).__name__)
        final = run.result.output
    print('events seen while the run happened:', len(kinds))
    #> events seen while the run happened: 9
    print('the tool call arrived as an event:', 'FunctionToolCallEvent' in kinds)
    #> the tool call arrived as an event: True
    print('final output:', final)
    #> final output: 42
    assert 'FunctionToolCallEvent' in kinds
```


A capability can also wrap that stream to filter or rewrite it, which is how you build an auditor that
travels with the agent instead of a separate observability integration.

OpenTelemetry is off by default. When you enable it (with
[`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all] or
[`InstrumentationSettings`][pydantic_ai.models.instrumented.InstrumentationSettings]), spans go out as
the OpenTelemetry [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
That distinction is the whole point. Plenty of frameworks produce spans; what decides whether those
spans are useful is whether they use the attribute names the rest of the industry agreed on:

```python {title="otel_semconv.py"}
"""Agent spans that a GenAI dashboard can read, once instrumentation is on."""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

support = Agent('openai:gpt-5.6-luna', name='support')
support.instrument = InstrumentationSettings(tracer_provider=provider)


@support.tool_plain
def issue_store_credit(order_id: str) -> str:
    """Issue store credit for an order."""
    return f'credited {order_id}'


support.run_sync('Issue store credit for order W-882.')
spans = {s.name: s for s in exporter.get_finished_spans()}
print('agent span:', 'invoke_agent support' in spans)
#> agent span: True
print('tool span:', 'execute_tool issue_store_credit' in spans)
#> tool span: True
print('model span:', 'chat gpt-5.6-luna' in spans)
#> model span: True
print('the tool span names the tool:', spans['execute_tool issue_store_credit'].attributes['gen_ai.tool.name'])
#> the tool span names the tool: issue_store_credit
print('the model span reports usage:', 'gen_ai.usage.input_tokens' in spans['chat gpt-5.6-luna'].attributes)
#> the model span reports usage: True
```

Those spans went to a plain OpenTelemetry exporter, not to us. Point them at Logfire if you want the
first-party view, or at Datadog, Honeycomb or Grafana, and the agent shows up in the GenAI dashboards
those vendors already ship, because the attribute names match. Uninstrumented agents emit nothing, so
an existing dashboard does not start reading spans just because you imported Pydantic AI.

We counted distinct `gen_ai.*` attributes in each framework's source. Google ADK emits them too. The
rest go through third-party instrumentation using its own namespace (`llm.model_name`,
`openinference.span.kind`), so a standards-based GenAI dashboard stays empty:

| | Distinct `gen_ai.*` attributes |
|---|---|
| Google ADK 2.8.0 | 49 |
| Pydantic AI 2.42 | 36 |
| CrewAI 1.15.21 | 1 |
| LangChain 1.4.0, openai-agents 0.22.2, Agno 3.0.9, smolagents 1.26.0 | 0 |

## 8. Evals in your test suite

Evals shouldn't need a platform login. `pydantic-evals` takes cases and evaluators as ordinary Python,
uses the agent's own types, and runs in CI next to your unit tests:

```python {title="evals_ci.py"}
"""Regressions are typed and run in CI."""
from pydantic_ai import Agent, RunContext
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected

agent = Agent('openai:gpt-5.6-luna')


@agent.tool
def shout(ctx: RunContext, text: str) -> str:
    return text.upper()


dataset = Dataset(
    name='shout',
    cases=[Case(name='hello', inputs='Shout hello for the ticket.', expected_output='HELLO')],
    evaluators=[EqualsExpected(), Contains(value='HELLO', case_sensitive=True)],
)


def run_case(text: str) -> str:
    return str(agent.run_sync(text).output)


report = dataset.evaluate_sync(run_case, progress=False)
averages = report.averages()
print(f'assertions passed: {averages.assertions * 100:.0f}%')
#> assertions passed: 100%
assert averages.assertions == 1.0
```


## 9. Crash recovery without rewriting the agent

A run is an ordinary coroutine, so durability is a capability you add, not a shape you have to adopt.
The same agent definition can take Temporal, DBOS, or Prefect as a capability, and there are adapters
for Restate, Kitaru, and Airflow maintained in those projects.

Adding [`TemporalDurability()`][pydantic_ai.durable_exec.temporal.TemporalDurability] does not by
itself make [`agent.run()`][pydantic_ai.Agent.run] durable. You still need that engine's worker and
workflow (or the DBOS/Prefect equivalent). The capability is what you attach so the same agent object
is the thing the worker runs.

```python {title="durability_wrap.py"}
"""One agent definition; the durable engine is a capability you add."""

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.durable_exec.prefect import PrefectDurability
from pydantic_ai.durable_exec.temporal import TemporalDurability

INSTRUCTIONS = 'Be terse.'


def support_agent(durability) -> Agent:
    """The agent is defined once; only the capability list changes."""
    return Agent(
        'openai:gpt-5.6-luna',
        name='support',
        instructions=INSTRUCTIONS,
        capabilities=[durability],
    )


agents = [support_agent(d) for d in (TemporalDurability(), DBOSDurability(), PrefectDurability())]

print('same definition, three engines:', [a.name for a in agents])
#> same definition, three engines: ['support', 'support', 'support']
print('all of them still just run():', all(hasattr(a, 'run') for a in agents))
#> all of them still just run(): True
assert [a.name for a in agents] == ['support'] * 3
```


The point isn't that we have durability. It's that you choose the engine, and it's probably one your
company already runs and already knows how to operate.

## Where we're not the answer

Everything above is a reason to pick us. Here's the other side, so you don't find it out later.

- **No hosted platform.** No managed runtime, no control plane, no dashboard you get by signing up.
  Agno, Mastra, LangSmith, and the OpenAI platform all give you one, and if that's what you want, that's
  a real reason to pick them.
- **No TypeScript.** If your product lives in the browser, the Vercel AI SDK or Mastra will serve you
  better. We have UI adapters, including for the AI SDK's protocol, but the agent stays in Python.
- **A smaller integration catalogue.** LangChain's is far bigger. If you need a connector that exists
  only there, that matters more than anything on this page.
- **Turnkey coding agents.** Claude Code and Pi are finished products. Ours is a library of the parts
  they're made of, which is more work and more yours afterwards.
- **Memory.** Mastra and Agno ship more developed memory than we do. Ours is dependencies and history
  processors you wire up.
- **Some harness pieces are experimental.** Planning, subagents, compaction, and runtime authoring are
  moving.

## The comparisons

Framework by framework, with what each does better:
[LangChain and LangGraph](vs-langchain-langgraph.md) ·
[OpenAI Agents SDK](vs-openai-agents-sdk.md) ·
[Claude Agent SDK](vs-claude-agent-sdk.md) ·
[CrewAI](vs-crewai.md) ·
[smolagents](vs-smolagents.md) ·
[Google ADK](vs-google-adk.md) ·
[AG2](vs-ag2.md) ·
[Agno](vs-agno.md) ·
[Mastra](vs-mastra.md) ·
[Vercel AI SDK](vs-vercel-ai-sdk.md) ·
[Pi](vs-pi.md)

---

*Pydantic AI 2.42, checked 2026-09-10. Every example on this page is executed by this repository's test suite
on every commit, so the output shown is what it printed. Claims about other frameworks are checked on their
pages against a pinned version. We recheck this page's version pins and behaviour claims each time Pydantic AI
ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
