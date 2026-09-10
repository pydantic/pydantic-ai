# Pydantic AI vs Mastra

**Mastra, at its best:** a batteries-included TS/Node framework — agents, tools, workflows,
observability, and evals under one roof, with a processor-style extension model.

**Pydantic AI, at its best:** one extension unit — a capability can carry tools, defer, and observe
or transform the run's event stream — on a typed, async-first Python loop.

*Verified against Mastra docs 2026-09 (TS framework; docs-grounded). Pydantic AI claims below are
self-contained scripts — offline, no API keys — re-executed by this repository's test suite.*

## Quick comparison

| What you get | Mastra | Pydantic AI |
|---|---|---|
| Stack | TS/Node; agents + workflows + tools in one package | Python 3.11+, packages split: core, evals, graph |
| Extension | Processors/guardrails/workflow concepts | Capabilities: one unit (tools + instructions + hooks), deferrable, serializable |
| Observability | Built-in (a real strength) | OTel + Logfire instrumentation; your capability can also see the stream (proven below) |
| Typed seams | Zod at boundaries | `deps_type` through construction, tools, specs, tests, evals |
| Cancellation | TS `AbortSignal` norm | Typed: `ctx.cancel()`, thread-safe token, catchable `RunCancelled` with resumable history |
| Tests/evals | Vitest + their evals | Offline `TestModel`/`FunctionModel` + typed datasets, CI-runnable |

## Prove it yourself

An auditor that sees every tool call and result is not a separate observability product — it's a
capability, the same unit that bundles tools:

```python {title="capability_observes_events.py"}
"""Your auditor is just a capability.

The same extension unit that bundles tools can also wrap the run's event
stream: count every tool call and result without touching the loop.
"""
import asyncio
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.function import DeltaToolCall, FunctionModel


@dataclass
class Auditor(AbstractCapability[Any]):
    seen: list[str]

    async def wrap_run_event_stream(
        self, ctx: RunContext[Any], *, stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterable[AgentStreamEvent]:
        async for event in stream:
            self.seen.append(type(event).__name__)
            yield event


async def stream(messages, info):
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='twice', json_args='{"n": 21}', tool_call_id='c1')}
    else:
        yield '42'


aud = Auditor(seen=[])
agent = Agent(FunctionModel(stream_function=stream), capabilities=[aud])


@agent.tool
def twice(ctx, n: int) -> int:
    return n * 2


async def main():
    async with agent.run_stream_events('what is 21*2?') as run:
        async for _ in run:
            pass
    print('auditor (a capability) observed:', aud.seen)
    assert 'FunctionToolCallEvent' in aud.seen
    assert 'FinalResultEvent' in aud.seen


asyncio.run(main())```

```text
auditor (a capability) observed: ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent']
```

```text
auditor (a capability) observed: ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent']
```

One extension model, and it reaches the event stream: your auditor, your UI adapter, your approval
gate are all just capabilities.

## Key differences

**Their best:** the all-in-one TS package with built-in observability and workflows is a genuinely
batteries-included offer for the JS ecosystem.

**Ours:** extension is one noun with prod reach — the capability that adds a tool can also wrap the
run's events, defer loading, and serialize into a spec. Observability is a capability, not a module
you bolt on the side.

## When to choose Mastra

Your stack is TypeScript and you want agents + workflows + observability from one package.

## When to choose Pydantic AI

You want the loop in Python where the extension unit reaches the seams — including the event stream
— and the production checklist (deps, budgets, cancellation, evals).

## Summary

Their observability is a module; our auditor is a capability. The same unit that observes
'['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent']' was built-in to the agent — no extra product.

*Mastra behavior docs-grounded 2026-09; records in the
[framework-comparison series](https://github.com/pydantic/pydantic-ai-notes). Pydantic AI verified
on 2.42.0, 2026-09-10.*