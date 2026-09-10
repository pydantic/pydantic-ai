# Pydantic AI vs Mastra

Mastra is the batteries-included TypeScript framework for agents. One package gives you agents,
workflows with snapshots and time travel, a memory system with several kinds of recall, evaluation
scorers, a sandboxed way to run model-written TypeScript, a local dev server with a playground, and a
hosted Studio and Cloud if you want them.

If your stack is TypeScript it belongs on your shortlist. Pydantic AI is Python, so for a TypeScript
team the honest answer is usually Mastra or the Vercel AI SDK, and the rest of this page is for people
whose agent is going to be in Python either way.

Two things are still worth comparing across that line, because they're design choices and not
language ones.

## One thing to learn instead of several

Mastra separates concerns by giving each its own concept: tools, processors, guardrails, subagents,
workflow steps, scorers. Each is well shaped, and there are a lot of them.

Pydantic AI has one extension: a capability. A capability can add tools, add instructions, change model
settings, hook into the run's lifecycle, and watch or rewrite the stream of events — all as one unit,
and it can wait to load until the model asks for it. It exposes 63 hooks, with a matching `before_`,
`after_`, `wrap_` and error handler at each stage of the run: the model request, each node, tool
validation, tool execution, output validation, output processing, and the run itself. One thing to
learn, and it goes everywhere. Here's a single capability doing two of those jobs
at once, adding a tool and auditing what the run emits:

```python {title="one_capability_two_jobs.py"}
"""One capability, two jobs: it supplies the tool and audits the run.

The auditor is not a separate observability product bolted on; it is the same
extension unit that carries the tool, watching the events the run emits.
"""
from collections.abc import AsyncIterable

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.function import DeltaToolCall, FunctionModel


def refund_status(order_id: str) -> str:
    """Look up a refund status."""
    return f'Order {order_id}: refunded.'


class Refunds(Capability):
    def __init__(self):
        super().__init__(id='refunds', description='refund tools, audited', tools=[refund_status])
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


async def main():
    refunds = Refunds()
    agent = Agent(FunctionModel(stream_function=stream), capabilities=[refunds])
    async with agent.run_stream_events('check my refund') as run:
        async for _ in run:
            pass
    print('the capability ran its own tool:', 'FunctionToolCallEvent' in refunds.seen)
    #> the capability ran its own tool: True
    print('and saw the whole run:', len(refunds.seen), 'events')
    #> and saw the whole run: 7 events
```



The practical effect is that "add refunds to this agent" is one object to write, one object to test,
and one object to hand to another team — rather than a tool registered here, an instruction appended
there, and a listener wired up somewhere else.

## Observability you own

Mastra's tracing goes to their dev server, Studio, and Cloud, and that integration is part of what
makes it pleasant.

Pydantic AI emits OpenTelemetry. It goes to Logfire if you want the first-party experience, or to
Datadog, Honeycomb, Grafana, or whatever your company already runs, and the agent's traces sit next to
your database and HTTP spans instead of in a separate tool. That's less polished on day one and less
of a commitment on day two hundred.

The same pattern shows up in durability. Mastra's story is its workflow engine, with a variant built on
Inngest. Ours is a wrapper around whichever engine you already operate — Temporal, DBOS, Prefect,
Restate, Kitaru, or Airflow. Neither is better in the abstract; one has fewer moving parts, the other
has fewer opinions.

## Where Mastra is ahead

Its memory is more developed than ours. Working memory, observational memory, and semantic recall are
real features with real depth, and the equivalent in Pydantic AI is dependencies and history processors
you wire up yourself. If memory is the centre of your product, that gap is on our side.

Mastra also has a reconnectable streaming story under durability — a client can drop and rejoin a
running agent. Ours doesn't do that; under a durable engine you stream through an external sink you
provide.

## Side by side

| | Mastra 1.28 (`@mastra/core` 1.65) | Pydantic AI 2.42 |
|---|---|---|
| Language | TypeScript | Python |
| Extending an agent | Tools, processors, guardrails, subagents, scorers | One capability that can do all of those, and load on demand |
| Trusted state | Zod validates tool inputs | `deps_type`, read by tools, invisible to the model |
| Workflows | A workflow engine with snapshots and time travel | Ordinary async code, and `pydantic_graph` when you want a state machine |
| Durability | Their workflow engine, or the Inngest variant | Six engines wrap the agent; you pick |
| Memory | Working, observational, and semantic recall | Dependencies and history processors you wire up |
| Evals | Scorers, with their tooling and Vitest | `pydantic-evals` in your test suite, using the agent's own types |
| Tracing | Dev server, Studio, Cloud | OpenTelemetry GenAI semantic conventions (36 `gen_ai.*` attributes) — your existing dashboards read them |
| Deployment | Their Cloud is the paved road | Anywhere; it's a library |

## Choose Mastra when

- Your product is TypeScript. This is most of the decision.
- You want workflows, memory, evals, and a playground without assembling them.
- Their Cloud is somewhere you're happy to run things.
- Memory is central to what you're building.

## Choose Pydantic AI when

- Your agent belongs in Python, near your data or your existing services.
- You want traces in the observability stack you already pay for.
- You want durability from an engine your company already operates.
- Credentials and identity must sit where the model can't reach them.

## FAQ

**Can I use both?**
Yes — Mastra in front, a Python agent behind an HTTP endpoint. Pydantic AI's UI adapters mean the
front end doesn't need to know.

**Is Pydantic AI a drop-in replacement?**
No, it's a different language. What ports is the design: tools, prompts, schemas, and how you think
about evals.

**What does Mastra do better?**
Memory, the local development experience, and reconnectable streaming. All three are real, and the
first two are why people like it.

---

*Mastra versions checked on the npm registry on 2026-09-10 (`mastra` 1.28.0, `@mastra/core` 1.65.0). Unlike
the other pages in this series, the Mastra behaviour described here comes from their published documentation
rather than from code we ran — it's TypeScript and we didn't install it. Treat those claims as their
documentation's, and tell us if any have gone stale. The Pydantic AI example is executed by this repository's
test suite on every commit. We recheck this page's version pins and behaviour claims each time Pydantic AI
ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
