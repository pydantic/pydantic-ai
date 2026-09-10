# Pydantic AI vs Mastra

You're choosing a Python agent framework and have narrowed it to [Pydantic AI](../agent.md) and Mastra.
This page makes the call — and lets you check the evidence yourself: every snippet runs offline,
no API keys.

## Pydantic AI fits if you need

- a **Python** agent
- **one extension unit** that reaches the seams — including the event stream
- deps, budgets, cancellation, and evals on the same loop

## Why the answers differ

Their observability is a module; our auditor is a capability — the same unit that bundles tools can wrap the run's events. Same concern, one product ship versus one extension noun.

## See it work

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

asyncio.run(main())

```

```text
auditor (a capability) observed: ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent']
```

## The details

| What you get | Mastra | Pydantic AI |
|---|---|---|
|---|---|---|
| Stack | TS/Node; agents + workflows + tools in one package | Python 3.10+; packages split: core, evals, graph |
| Extension | Processors/guardrails/workflow concepts | Capabilities: one unit (tools + instructions + hooks), deferrable, serializable |
| Observability | Built-in (a real strength) | OTel + Logfire instrumentation; your capability can also see the stream (proven below) |
| Typed seams | Zod at boundaries | `deps_type` through construction, tools, specs, tests, evals |
| Cancellation | TS `AbortSignal` norm | Typed: `ctx.cancel()`, thread-safe token, catchable `RunCancelled` with resumable history |
| Tests/evals | Vitest + their evals | Offline `TestModel`/`FunctionModel` + typed datasets, CI-runnable |

## If this answer doesn't fit you

If your stack is TypeScript and agents + workflows + observability from one package is the offer you want, Mastra is built for that. For a Python loop where the extension unit reaches the event stream, this page shows what that looks like.

---

---

*Versions: Mastra (docs); Pydantic AI 2.42.0 — 2026-09. Snippets re-executed by this repository's tests.*
