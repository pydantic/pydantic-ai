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

## One extension, not several

Mastra separates concerns by giving each its own concept: tools, processors, guardrails, subagents,
workflow steps, scorers. Each is well shaped, and there are a lot of them.

Pydantic AI has one extension: a capability. A capability can add tools, add instructions, change model
settings, hook into the run's lifecycle, and watch or rewrite the stream of events, all as one unit,
and it can wait to load until the model asks for it. "Add refunds to this agent" is one object to
write, one object to test, and one object to hand to another team.

## Observability you own

Mastra's tracing goes to their dev server, Studio, and Cloud, and that integration is part of what
makes it pleasant.

Pydantic AI emits OpenTelemetry when instrumentation is enabled
([`Agent.instrument_all()`][pydantic_ai.Agent.instrument_all], the
[`Instrumentation`][pydantic_ai.capabilities.Instrumentation] capability, or Logfire). It is off by
default. It goes to Logfire if you want the first-party experience, or to Datadog, Honeycomb, Grafana,
or whatever your company already runs, and the agent's traces sit next to your database and HTTP spans
instead of in a separate tool. That's less polished on day one and less of a commitment on day two
hundred.

The same pattern shows up in durability. Mastra's story is its workflow engine, with a variant built on
Inngest. Ours is a wrapper around whichever engine you already operate, Temporal, DBOS, Prefect,
Restate, Kitaru, or Airflow.

Mastra's memory is more developed than ours. Working memory, observational memory, and semantic recall
are real features with real depth, and the equivalent in Pydantic AI is dependencies and history
processors you wire up yourself. Mastra also has a reconnectable streaming story under durability; ours
doesn't, under a durable engine you stream through an external sink you provide.

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
| Tracing | Dev server, Studio, Cloud | OpenTelemetry GenAI semantic conventions when instrumentation is enabled |
| Deployment | Their Cloud is the paved road | Anywhere; it's a library |

## FAQ

**Can I use both?**
Yes, Mastra in front, a Python agent behind an HTTP endpoint. Pydantic AI's UI adapters mean the
front end doesn't need to know.

**Is Pydantic AI a drop-in replacement?**
No, it's a different language. What ports is the design: tools, prompts, schemas, and how you think
about evals.

---

*Mastra versions checked on the npm registry on 2026-09-10 (`mastra` 1.28.0, `@mastra/core` 1.65.0). Unlike
the other pages in this series, the Mastra behaviour described here comes from their published documentation
rather than from code we ran, it's TypeScript and we didn't install it. Treat those claims as their
documentation's, and tell us if any have gone stale. We recheck this page's version pins and behaviour claims
each time Pydantic AI ships a minor release; if something here has gone stale, [tell
us](https://github.com/pydantic/pydantic-ai/issues/new) and we'll correct it.*
