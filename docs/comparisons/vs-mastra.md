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
| Durability | `DurableAgent` in core; Inngest via `@mastra/inngest` | The same agent, inside Temporal, DBOS, or Prefect |
| Tracing | `mastra dev`, Studio, Cloud | OpenTelemetry, when you turn it on |
| Deploy | Mastra Cloud, or `mastra start` | Anywhere |

## FAQ

**The product is TypeScript?** Stay in TypeScript (Mastra or the Vercel AI SDK). This page is for
Python.

**Both?** Mastra in the browser, a Python agent behind HTTP. UI adapters exist.

**Drop-in?** No. Different language.
