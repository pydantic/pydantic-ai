# Pydantic AI vs Mastra

You're choosing a Python agent framework and you're down to
[Pydantic AI](../agent.md) and Mastra. This page is the tiebreaker — the answer first, then code
you can run in seconds.

## Pydantic AI fits if you need

- a **Python** agent
- **one extension unit** that reaches the seams — including the event stream
- deps, budgets, cancellation, and evals on the same loop

## Why the answers differ

Their observability is a module; our auditor is a capability — the same unit that bundles tools can wrap the run's events. Same concern, one product ship versus one extension noun.

## See it work

Say you want to observe what your agent did in production — without a separate observability product.

Mastra (TS) spreads this across surfaces: agent config, tools, processors, and an observability module (their docs).

Your side, runs offline:

```python {title="one_capability_two_jobs.py"}
"""One capability, two jobs: add a tool and watch the stream.

Your auditor is not a separate observability product; it is the same
extension unit that carries tools. This capability both provides the
tool the model calls and records every event it sees.
"""
import asyncio
from collections.abc import AsyncIterable

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.function import DeltaToolCall, FunctionModel


def refund_status(order_id: str) -> str:
    """Look up a refund status (provided by the capability)."""
    return f'Order {order_id}: refunded.'


class Auditor(Capability):
    def __init__(self):
        super().__init__(id='auditor', description='watch and act', tools=[refund_status])
        self.seen: list[str] = []

    async def wrap_run_event_stream(
        self, ctx: RunContext[object], *, stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterable[AgentStreamEvent]:
        async for event in stream:
            self.seen.append(type(event).__name__)
            yield event


async def stream(messages, info):
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='refund_status', json_args='{"order_id": "X"}', tool_call_id='c1')}
    else:
        yield 'done'


aud = Auditor()
agent = Agent(FunctionModel(stream_function=stream), capabilities=[aud])


async def main():
    async with agent.run_stream_events('check my refund') as run:
        async for _ in run:
            pass
    print(f'tool from the capability executed; auditor (same unit) observed: {aud.seen}')
    assert 'FunctionToolCallEvent' in aud.seen


asyncio.run(main())


```

```text
tool from the capability executed; auditor (same unit) observed: ['PartStartEvent', 'PartEndEvent', 'FunctionToolCallEvent', 'FunctionToolResultEvent', 'PartStartEvent', 'FinalResultEvent', 'PartEndEvent']```

**Notice:** The same unit added the tool *and* watched the stream. One noun, both jobs — your auditor is just a capability.

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

If yours is a TypeScript stack and you want one package that does agents, workflows, and observability, Mastra is built for exactly that, and we're not going to talk you out of it. Ours is the Python expression of the same instinct — one extension unit that reaches the event stream.

---

---

*Versions: Mastra (docs); Pydantic AI 2.42.0 — 2026-09. Snippets re-executed by this repository's tests.*
