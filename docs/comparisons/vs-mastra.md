# Pydantic AI vs Mastra

Mastra is TypeScript all-in-one: agents, workflows, memory, evals, a playground, Studio, Cloud.
Pydantic AI is Python. One extension point (a capability) instead of tools, processors, guardrails,
and scorers as separate concepts. Tracing is OpenTelemetry you already run, off by default. Memory
is thinner than theirs.

## Side by side

| | Mastra | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Extending | Tools, processors, scorers, workflows | One capability |
| Memory | Working, observational, semantic | Deps and history processors |
| Durability | `createDurableAgent()` in core; Inngest via `@mastra/inngest` | The same agent, inside Temporal, DBOS, or Prefect |
| Tracing | `mastra dev`, Studio, Cloud | OpenTelemetry, when you turn it on |
| Deploy | Mastra Cloud, or `mastra start` | Anywhere |

## FAQ

**Can the UI stay in TypeScript?** Yes. A Python agent behind HTTP. UI adapters exist.

**Can I get a chat UI without Studio?** Yes. [`to_web()`][pydantic_ai.agent.Agent.to_web].
