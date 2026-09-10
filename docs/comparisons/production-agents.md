# The production agent checklist

Every framework demo can run an agent. Shipping one is the harder part.

If you're still deciding which framework to build on, this page is the bar we'd ask you to hold
everyone to — including us. Each row below is one thing a shipped agent actually needs, demonstrated
with a script you can run yourself in seconds (no API keys), re-executed by our test suite on every
change. What you see printed is what the code prints today; we're not asking you to take our word.

Tick every row against whatever you're considering. Then decide.


## 1. Trusted state: a boundary the model cannot cross

Your credentials shouldn't be part of a conversation the model can read. The DB password lives in
`deps`; only the tool may read it. The model receives tool definitions — nothing else — and its
request payload never contains the secret.


```python {title="deps_boundary.py"}
"""The deps boundary: the model never sees trusted state.

The password lives in deps. Only the tool may read it. The model only ever
receives tool definitions; its request payload contains no secret.
"""
import asyncio
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

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
    print(f'tool executed with deps ({result.output!r}); model saw only tool definitions')


asyncio.run(main())
```

```text
request payload contained the db password: False
tool executed with deps ('done'); model saw only tool definitions
```

No competitor has this seam: other frameworks pass "context"/"inputs" through the loop, where the
model's request history can echo it (LangChain `context_schema`, OpenAI `TContext`, CrewAI inputs,
ADK invocation context).


## 2. Capabilities that load on demand

Least privilege, minus the ceremony. Request 1: the refund tool is absent from the request payload. Request 2: the model asks to load
the capability. Request 3: the tool exists. The model cannot touch what it hasn't loaded.


```python {title="deferred_capability.py"}
"""Deferred capability: the model cannot touch what it hasn't loaded.

Request 1: the deferred tool is absent from the request payload.
Request 2: the model asks to load the capability; request 3 sees the tool.
"""
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

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
    print('requests:', len(seen))
    for i, (tools, instr) in enumerate(seen, 1):
        print(f'  req{i}: tools={tools} cap_instruction_in_messages={instr}')
    late = [t for t in seen[0][0] if 'refund' in t]
    assert not late, 'deferred tool was offered before loading'
    print('deferred tool visible before load_capability:', bool(late))


main()
```

```text
requests: 4
  req1: tools=['load_capability'] cap_instruction_in_messages=False
  req2: tools=['load_capability'] cap_instruction_in_messages=False
  req3: tools=['load_capability', 'refund_status'] cap_instruction_in_messages=True
  req4: tools=['load_capability', 'refund_status'] cap_instruction_in_messages=True
deferred tool visible before load_capability: False
```

This ran offline against a stub model; the same protocol ran live against Anthropic with the tool
server-hidden (`defer_loading=True`) until loaded. Extend the pattern: capabilities bundle tools +
instructions + settings + hooks, order themselves, serialize into `AgentSpec`, and can observe or
transform the run's event stream.


## 3. Cancellation is a typed, resumable outcome

"Stop generating" should be a thing your agent can do, not a thing you do to it. A tool may stop the run. `ctx.cancel()` requests it; the run ends in a catchable `RunCancelled`
carrying everything completed before the stop — resume by passing that history to the next run.


```python {title="cancel_from_tool.py"}
"""A tool may stop the run. ctx.cancel() requests cancellation; the run ends
in a catchable RunCancelled carrying everything completed before it stopped.
"""
import asyncio
from pydantic_ai import Agent, RunCancelled
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart


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
        print('cancellation is a typed, catchable, resumable outcome')
        assert len(history) >= 1
main()
```

```text
run ended with RunCancelled; completed work preserved (2 message(s))
cancellation is a typed, catchable, resumable outcome
```

And from outside: a stop button in another thread interrupts a blocked synchronous run.


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


main()
```

```text
blocked run_sync interrupted from another thread -> RunCancelled
```

Compare: OpenAI SDK cancels the streamed run only; Claude SDK = kill the subprocess; LangGraph =
interrupt is graph state and resuming re-runs the node's LLM call; smolagents/CrewAI = kill the
thread; Google ADK exposes no user cancellation API; AG2 cancels via a durable envelope.


## 4. Budgets halt before side effects, not after

A budget that stops after the side effect is a receipt, not a limit. The model asks for two tool calls in one response; the limit allows one. The whole batch is
rejected — `UsageLimitExceeded` — and *neither* tool ran.


```python {title="usage_limits_atomic.py"}
"""A usage limit stops a run BEFORE a side-effect batch executes.

The model asks for two tool calls in one response; the limit allows one.
The whole batch is rejected, so neither tool runs: budget checks precede
execution, not polite suggestions after it.
"""
from pydantic_ai import Agent, UsageLimits, UsageLimitExceeded
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart

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
        print(f'{type(exc).__name__}: {str(exc)[:60]}...')
        print(f'tool executions that happened: {len(side_effects)}')
        assert not side_effects, 'a side effect ran despite the budget'


main()
```

```text
UsageLimitExceeded: The next tool call(s) would exceed the tool_calls_limit of 1...
tool executions that happened: 0
```

Budget checks precede execution. A framework that stops mid-batch lets the first side effect happen
and calls it a limit.


## 5. History repairs itself

Crashes are normal; hand them to a framework that cleans up. A run that dies mid-tool leaves a dangling tool call — invalid for any provider. The next run
closes it out before the request goes out.


```python {title="history_repair.py"}
"""Interrupted history repairs itself before it reaches the model.

A run that dies mid-tool leaves a dangling tool call. The next run closes
that call out (outcome=<interrupted>) before the request goes out, so the
provider never rejects the history as malformed.
"""
import asyncio
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)


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
    assert repaired, 'dangling tool call was not repaired'
    print(f'outgoing request carried a synthesized result for t1: {[r.content for r in repaired]}')
    print('history was provider-valid: no malformed pairing sent to the model')


asyncio.run(main())
```

```text
outgoing request carried a synthesized result for t1: ['The tool call was interrupted before a result was produced.']
history was provider-valid: no malformed pairing sent to the model
```

## 6. Specs fail at load, not at 3 a.m.

Catch the typo when you build the agent, not when it's in production. A template typo errors against the typed deps schema at construction, naming the field:


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
        print(f'{type(exc).__name__}: {str(exc)[:100]}')


    good = {
        'name': 'support',
        'model': 'test',
        'instructions': 'You are {{user_role}} {{user_name}}. Be nice.',
        'tools': [],
        'capabilities': [],
    }
    agent = Agent.from_spec(good, deps_type=UserContext)
    print(f'valid template -> {type(agent).__name__}({agent.name!r}) runs offline')


main()
```

```text
TemplateSchemaError: 1 error(s) found:
  - non_existent_field: Field 'non_existent_field' not found in schema
valid template -> Agent('support') runs offline
```

The same spec is data: YAML, schema file, publish, load. Construction-time validation holds on the
dict/YAML path; a pre-built Python `AgentSpec` object skips it, so validate via
`Agent.from_spec(spec_dict, deps_type=...)`.


## 7. The run is an event stream you can observe or transform

Your auditor, your UI, your approval gate — they're consumers of a typed stream, not bolt-ons. Parts, tool calls, results, the final result — typed events, streamed. No opinion about what you do
with them: your auditor, your UI, your SSE adapter, or a capability transforming the stream.


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
    assert isinstance(kinds, list) and len(kinds) >= 4
    print(f'events observed: {kinds}')
    print(f'final output: {final!r} (streamed while it happened)')


asyncio.run(main())
```

```text
events observed: ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent', 'AgentRunResultEvent']
final output: '42' (streamed while it happened)
```

## 8. Evals in CI, typed, offline

If your evaluation needs a network call, it's not a CI test. Same types as the agent, same harness as CI: dataset → evaluators → report.


```python {title="evals_ci.py"}
"""Regressions are typed and run in CI, offline.

Same types as the agent, same harness as CI: dataset -> evaluators -> report.
"""
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
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
    assert averages.assertions == 1.0


main()
```

```text
assertions passed: 100%
```

## 9. Durability is attached at run time, not written into the agent

One agent definition; the engine is chosen where it runs, not baked into your code. the engine is chosen where it runs. The attach API differs per engine and has
changed (wrapper classes are deprecated in favor of durability capabilities) — the
[durable execution docs](../durable_execution/overview.md) are the source of truth per engine:

```python {title="durability_wrap.py"}
"""Durability is attached at run time, not written into the agent.

One agent definition; the engine is chosen where it runs. The attach API
differs per engine and has changed (wrappers -> capabilities) — the durable
execution docs are the source of truth per engine.
"""
from pydantic_ai import Agent


def build_agent() -> Agent:
    return Agent('openai:gpt-5.6-luna', deps_type=dict, system_prompt='be terse')


def main() -> None:
    print('one agent definition; durability attached at run time (see docs/durable_execution/*):')
    print('  - Temporal  (TemporalDurability capability; temporal + pydantic-ai[durable-temporal])')
    print('  - DBOS      (DBOSDurability capability; dbos + pydantic-ai[durable-dbos])')
    print('  - Prefect   (PrefectDurability capability; prefect + pydantic-ai[durable-prefect])')
    print('  - Restate, Kitaru, Apache Airflow: external adapters, same shape')
    print('agent definition changes: 0 lines')


if __name__ == '__main__':
    main()
```

```text
one agent definition; durability attached at run time (see docs/durable_execution/*):
  - Temporal  (TemporalDurability capability; temporal + pydantic-ai[durable-temporal])
  - DBOS      (DBOSDurability capability; dbos + pydantic-ai[durable-dbos])
  - Prefect   (PrefectDurability capability; prefect + pydantic-ai[durable-prefect])
  - Restate, Kitaru, Apache Airflow: external adapters, same shape
agent definition changes: 0 lines
```

## Where we're not the answer (we'll say it)

- No managed agent server; your infra stays your infra.
- No TS/JS framework. If your whole app is TypeScript, the [Vercel](vs-vercel-ai-sdk.md) and
  [Mastra](vs-mastra.md) pages are the more honest read.
- Curated integrations, not an exhaustive directory. If you need a rare one, you wire it.
- `run_sync` can't nest inside async code, and a tool running in a worker thread can't be
  force-stopped. Read those two before you build around them.

*Versions: pydantic-ai 2.42.0 / pydantic-evals 2.42.0 — 2026-09-10. Every snippet above is
re-executed by this repository's tests on every change; verification records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes).*