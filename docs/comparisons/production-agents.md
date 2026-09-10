# What a production agent needs

Any framework can run an agent. The gap between a demo and something you'd put in front of customers
is made of unglamorous things: keeping secrets away from the model, stopping a run without losing it,
capping what it can spend, and knowing what it did afterwards.

This is our list of what that takes. Use it as a checklist against anything you're evaluating,
including us. Every item below has a script under it that runs on your laptop in a few seconds with no
API key, and our test suite runs all of them on every commit — so what's printed is what the code
prints today, not what it printed when someone wrote the page.

## 1. The model can't reach your secrets

A database password shouldn't be in a conversation the model can read. In Pydantic AI, trusted state
goes in a separate typed argument. Tools read it; the model gets tool definitions and nothing else, and
can't name it or ask for it.

```python {title="deps_boundary.py"}
"""The deps boundary: the model never sees trusted state.

The password lives in deps. Only the tool may read it. The model only ever
receives tool definitions; its request payload contains no secret.
"""
import asyncio

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

DB = {'db_password': 'hunter2-keep-secret'}


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('check_db', {'key': 'readiness probe'})])
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(model), deps_type=dict)


@agent.tool
async def check_db(ctx, key: str) -> str:
    ok = ctx.deps['db_password'] == 'hunter2-keep-secret'
    return f'db:{key}:{"ok" if ok else "auth-failed"}'  # never echoes the secret


async def main():
    with capture_run_messages() as msgs:
        result = await agent.run('Is the db ready?', deps=DB)
    payload = str(msgs)
    leaked = 'hunter2' in payload
    assert not leaked, 'the secret crossed into model-visible messages!'
    print(f'request payload contained the db password: {leaked}')
    #> request payload contained the db password: False
    print(f'tool executed with deps ({result.output!r}); model saw only tool definitions')
    #> tool executed with deps ('done'); model saw only tool definitions


asyncio.run(main())
```


Most frameworks pass some kind of context through the run — LangChain's `context_schema`, the OpenAI
SDK's `TContext`, CrewAI's inputs, ADK's invocation context. They're useful, but they're part of the
same material the conversation is built from. A separate boundary is a different guarantee.

## 2. Tools that aren't there until they're needed

An agent with sixty tools is a worse agent. Capabilities can wait until the model asks for them: the
tool isn't in the request at all, the model calls `load_capability`, and then it is — along with the
instructions and settings that belong with it.

```python {title="deferred_capability.py"}
"""Deferred capability: the model cannot touch what it hasn't loaded.

Request 1: the deferred tool is absent from the request payload.
Request 2: the model asks to load the capability; request 3 sees the tool.
"""
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

CAP_INSTRUCTION = 'Always confirm the order ID before issuing a refund.'

refunds = Capability(
    id='refunds', description='Use for refunds.', instructions=CAP_INSTRUCTION, defer_loading=True
)


@refunds.tool_plain
def refund_status(order_id: str) -> str:
    """Look up refund status."""
    return f'Order {order_id}: refunded.'


seen = []  # (tool names, cap_instruction_in_messages)


async def model(messages, info):
    tools = sorted(t.name for t in info.function_tools)
    seen.append((tools, CAP_INSTRUCTION in str(messages)))
    n = len(seen)
    if n == 1:
        return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'X'})])  # blocked: not loaded
    if n == 2:
        return ModelResponse(parts=[ToolCallPart('load_capability', {'id': 'refunds'})])
    if n == 3:
        return ModelResponse(parts=[ToolCallPart('refund_status', {'order_id': 'Y'})])
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(model), capabilities=[refunds])


def main() -> None:
    agent.run_sync('go')
    print('model requests:', len(seen))
    #> model requests: 4
    print('tools offered on the first request:', seen[0][0])
    #> tools offered on the first request: ['load_capability']
    print('tools offered after load_capability:', seen[2][0])
    #> tools offered after load_capability: ['load_capability', 'refund_status']
    assert 'refund_status' not in seen[0][0], 'deferred tool was offered before loading'


main()
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

from pydantic_ai import Agent, RunCancelled
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel


async def model(messages, info):
    return ModelResponse(parts=[ToolCallPart('slow_job', {})])


agent = Agent(FunctionModel(model))


@agent.tool
async def slow_job(ctx) -> str:
    ctx.cancel()  # cooperative: returns normally, lands at the next await
    await asyncio.sleep(0)
    return 'never used'


def main():
    try:
        agent.run_sync('start the job')
        print('BUG: run completed')
    except RunCancelled as exc:
        history = exc.all_messages()
        print(f'run ended with RunCancelled; completed work preserved ({len(history)} message(s))')
        #> run ended with RunCancelled; completed work preserved (2 message(s))
        print('cancellation is a typed, catchable, resumable outcome')
        #> cancellation is a typed, catchable, resumable outcome
        assert len(history) >= 1
main()
```


And one token can stop several runs at once, from another thread — which is what a stop button in a UI
actually needs:

```python {title="cancel_token_thread.py"}
"""A stop button, from another thread. CancellationToken interrupts a blocked
run_sync(); the run ends in RunCancelled instead of hanging forever.
"""
import asyncio
import threading
import time

from pydantic_ai import Agent, CancellationToken, RunCancelled
from pydantic_ai.models.function import FunctionModel


async def model(messages, info):
    await asyncio.sleep(3600)  # model appears to hang


token = CancellationToken()
agent = Agent(FunctionModel(model))


def stop_handler():
    time.sleep(0.5)
    token.cancel()  # thread-safe: delivered onto the run's loop


def main() -> None:
    stop = threading.Thread(target=stop_handler)
    stop.start()
    try:
        agent.run_sync('go', cancellation_token=token)
        print('BUG: run completed')
    except RunCancelled:
        stop.join()
        print('blocked run_sync interrupted from another thread -> RunCancelled')
        #> blocked run_sync interrupted from another thread -> RunCancelled
        #> blocked run_sync interrupted from another thread -> RunCancelled


main()
```


Either way the run ends by raising `RunCancelled`, and that exception carries the conversation, so
resuming is just passing it to the next run. Cancellation from outside — an `asyncio.timeout()`, a task
group shutting down — still behaves like normal Python cancellation.

Most of the others can stop a run; what differs is what you're holding afterwards. The OpenAI SDK
cancels a streamed run. The Claude SDK sends an `interrupt()` control request, in streaming mode only.
LangGraph's `abort()` lives on its experimental v3 stream and closes the graph iterator. smolagents
sets a flag that's checked between steps. AG2 cancels through its durable task envelope. CrewAI has no
stop method at all, and Google ADK exposes no cancellation API anywhere on `Runner` or `LlmAgent`.

## 4. Budgets that stop things before they happen

A budget you find out about afterwards is a bill, not a budget. `UsageLimits` is checked before the
next request goes out and before a batch of tool calls executes, so nothing runs when the run is
already over its limit.

```python {title="usage_limits_atomic.py"}
"""A usage limit stops a run BEFORE a side-effect batch executes.

The model asks for two tool calls in one response; the limit allows one.
The whole batch is rejected, so neither tool runs: budget checks precede
execution, not polite suggestions after it.
"""
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

side_effects = []


async def model(messages, info):
    return ModelResponse(
        parts=[
            ToolCallPart('credit_customer', {'amount': 100}),
            ToolCallPart('credit_customer', {'amount': 100}),
        ]
    )


agent = Agent(FunctionModel(model))


@agent.tool
def credit_customer(ctx, amount: int) -> str:
    side_effects.append(('credited', amount))
    return 'ok'


def main() -> None:
    try:
        agent.run_sync('credit the customer twice', usage_limits=UsageLimits(tool_calls_limit=1))
        print('BUG: exceeded the limit')
    except UsageLimitExceeded as exc:
        print('stopped by:', type(exc).__name__)
        #> stopped by: UsageLimitExceeded
        print(f'tool executions that happened: {len(side_effects)}')
        #> tool executions that happened: 0
        assert not side_effects, 'a side effect ran despite the budget'


main()
```


Spend works the same way, in money instead of tokens. This is the one on this page that nothing else
can do. A dollar budget needs prices for every model you might call, so it only works if the pricing
data is part of the library: `cost_limit` is backed by [genai-prices](https://github.com/pydantic/genai-prices),
which we maintain, covering 41 providers and 1,646 models. The check runs *before* the next request
goes out, so the run stops instead of the number arriving on your bill.

Two other frameworks accept a dollar figure — Agno and the Claude Agent SDK both take `max_budget_usd`
— and in both cases it is the same thing: a value passed through to the Claude CLI's own budget. It
works when your model is Claude, through that CLI. LangChain, LangGraph, the OpenAI Agents SDK, CrewAI,
smolagents and Google ADK have no money limit at all; they cap tokens or iterations, which is a proxy
that gets worse every time model pricing changes.

```python {title="cost_limit.py"}
"""Cost is a unit, not a rumor.

Per-request USD comes from genai-prices (first-party). A dollar budget is
enforced like any other usage limit - and when a model can't be priced,
you get a warning, not a silently unconstrained run.
"""
import warnings

from pydantic_ai import Agent, UsageLimits
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel


async def model(messages, info):
    return ModelResponse(parts=[TextPart('ok')])


agent = Agent(FunctionModel(model))


def main() -> None:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        agent.run_sync('hi', usage_limits=UsageLimits(cost_limit=0.10))
    names = sorted({type(x.message).__name__ for x in w if 'Cost' in type(x.message).__name__})
    print('run completed; per-request USD cost is tracked (genai-prices)')
    #> run completed; per-request USD cost is tracked (genai-prices)
    print(f'unpriced model under a cost budget -> {names}')
    #> unpriced model under a cost budget -> ['CostNotFoundWarning']
    assert 'CostNotFoundWarning' in names


main()
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
import asyncio

from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel


async def model(messages, info):
    return ModelResponse(parts=[TextPart('ok')])


agent = Agent(FunctionModel(model), tools=[])


@agent.tool
def add(ctx, n: int) -> int:
    return n


interrupted = [
    ModelRequest(parts=[UserPromptPart(content='add 1')]),
    ModelResponse(parts=[ToolCallPart('add', {'n': 1}, tool_call_id='t1')], state='interrupted'),
]


async def main():
    with capture_run_messages() as msgs:
        await agent.run('add 1', message_history=interrupted)
    repaired = [
        p for m in msgs for p in m.parts if isinstance(p, ToolReturnPart) and p.tool_call_id == 't1'
    ]
    print('dangling tool call was repaired before the request went out:', bool(repaired))
    #> dangling tool call was repaired before the request went out: True
    #> dangling tool call was repaired before the request went out: True
    assert repaired, 'the dangling tool call was not repaired'


asyncio.run(main())
```


This is the quiet one that saves you a bad afternoon. It only works because the history is typed data
the framework owns rather than a dictionary of whatever the last thing put there.

## 6. Agents you can ship as configuration

An agent can be a YAML file, and the file is checked when it loads, not when it runs. A typo in
a prompt template fails immediately and names the field:

```python {title="spec_validation.py"}
"""An agent spec fails at load time, not at runtime.

Templates are validated against typed deps when the spec is built: a single
typo error names the field and the file. The same spec typechecks into a
running agent offline.
"""
from pydantic import BaseModel

from pydantic_ai import Agent


class UserContext(BaseModel):
    user_name: str
    user_role: str


bad = {
    'name': 'support',
    'model': 'test',  # offline stub backend
    'instructions': 'You are {{non_existent_field}}. Be nice.',
    'tools': [],
    'capabilities': [],
}
def main() -> None:
    try:
        Agent.from_spec(bad, deps_type=UserContext)
        print('BUG: invalid template accepted')
    except Exception as exc:
        print('rejected at load:', type(exc).__name__)
        #> rejected at load: TemplateSchemaError


    good = {
        'name': 'support',
        'model': 'test',
        'instructions': 'You are {{user_role}} {{user_name}}. Be nice.',
        'tools': [],
        'capabilities': [],
    }
    agent = Agent.from_spec(good, deps_type=UserContext)
    #> the corrected spec loads: support
    print('the corrected spec loads:', agent.name)
    #> the corrected spec loads: support
    assert agent.name == 'support'


main()
```


Worth being precise about the limit: that check runs when a spec is loaded from a dictionary or a file,
not when you build an `AgentSpec` object directly in Python.

## 7. You can watch it work

The run emits typed events as it happens — the model starting to speak, each tool call and its result,
the final answer — and you consume them with a normal `async for`:

```python {title="event_stream.py"}
"""The run is an event stream you can observe or transform.

Part deltas, tool calls, results, and the final result — all typed, all
streamed. No framework opinion on what you do with them.
"""
import asyncio

from pydantic_ai import Agent
from pydantic_ai.models.function import DeltaToolCall, FunctionModel


async def stream(messages, info):
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='twice', json_args='{"n": 21}', tool_call_id='c1')}
    else:
        yield '42'


agent = Agent(FunctionModel(stream_function=stream))


@agent.tool
def twice(ctx, n: int) -> int:
    return n * 2


async def main():
    kinds = []
    async with agent.run_stream_events('what is 21*2?') as run:
        async for event in run:
            kinds.append(type(event).__name__)
        final = run.result.output
    print('events seen while the run happened:', len(kinds))
    #> events seen while the run happened: 8
    print('the tool call arrived as an event:', 'FunctionToolCallEvent' in kinds)
    #> the tool call arrived as an event: True
    print('final output:', final)
    #> final output: 42
    assert 'FunctionToolCallEvent' in kinds


asyncio.run(main())
```


A capability can also wrap that stream to filter or rewrite it, which is how you build an auditor that
travels with the agent instead of a separate observability integration.

Everything also goes out as OpenTelemetry — and specifically, as the OpenTelemetry
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). That distinction is
the whole point. Plenty of frameworks produce spans; what decides whether those spans are useful is
whether they use the attribute names the rest of the industry agreed on, because that is what your
existing dashboards, alerts and vendor integrations read:

```python {title="otel_semconv.py" requires="event_stream.py"}
"""Agent spans that your existing tooling already understands."""
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))

support = Agent(TestModel(), name='support')
support.instrument = InstrumentationSettings(tracer_provider=provider)


@support.tool_plain
def refund(order_id: str) -> str:
    """Refund an order."""
    return 'refunded'


support.run_sync('refund A-1')
spans = {s.name: s for s in exporter.get_finished_spans()}
print('span names:', sorted(spans))
#> span names: ['chat test', 'execute_tool refund', 'invoke_agent support']
tool_span = spans['execute_tool refund']
print('the tool span names the tool:', tool_span.attributes['gen_ai.tool.name'])
#> the tool span names the tool: refund
print('the model span reports usage:', 'gen_ai.usage.input_tokens' in spans['chat test'].attributes)
#> the model span reports usage: True
```

Those spans went to a plain OpenTelemetry exporter, not to us. Point them at Logfire if you want the
first-party view, or at Datadog, Honeycomb or Grafana, and the agent shows up in the GenAI dashboards
those vendors already ship — because the attribute names match.

For comparison, we counted distinct `gen_ai.*` attributes in each framework's source. Google ADK is a
peer here and does this properly. The rest emit spans through third-party instrumentation that uses
its own namespace — `llm.model_name`, `openinference.span.kind` — so a standards-based GenAI dashboard
stays empty:

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
"""Regressions are typed and run in CI, offline.

Same types as the agent, same harness as CI: dataset -> evaluators -> report.
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Contains, EqualsExpected


async def model(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('shout', {'text': 'hello'})])
    return ModelResponse(parts=[TextPart('HELLO WORLD')])


agent = Agent(FunctionModel(model))


@agent.tool
def shout(ctx: RunContext[None], text: str) -> str:
    return text.upper()


dataset = Dataset(
    name='shout',
    cases=[Case(name='hello', inputs='hello', expected_output='HELLO WORLD')],
    evaluators=[EqualsExpected(), Contains(value='HELLO', case_sensitive=True)],
)


def run_case(text: str) -> str:
    return str(agent.run_sync(text).output)


def main() -> None:
    report = dataset.evaluate_sync(run_case, progress=False)
    averages = report.averages()
    print(f'assertions passed: {averages.assertions * 100:.0f}%')
    #> assertions passed: 100%
    #> assertions passed: 100%
    assert averages.assertions == 1.0


main()
```


## 9. Crash recovery without rewriting the agent

A run is an ordinary coroutine, so durability is a capability you add, not a shape you have to adopt.
The same agent definition runs under Temporal, DBOS, or Prefect, and there are adapters for Restate,
Kitaru, and Airflow maintained in those projects:

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
        'openai:gpt-5.2', name='support', instructions=INSTRUCTIONS, capabilities=[durability]
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

*Pydantic AI 2.42, checked 2026-09-10. Every example on this page is executed by this repository's test
suite on every commit, so the output shown is what it printed. Claims about other frameworks are
checked on their pages against a pinned version. We recheck this page's
version pins and behaviour claims each time Pydantic AI ships a minor release; if something here has
gone stale, [tell us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
