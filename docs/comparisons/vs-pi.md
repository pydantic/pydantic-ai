# Pydantic AI vs Pi

Pi is a TypeScript coding agent you can embed without forking: extensions, skills, and `pi install`
packages. Pydantic AI is a typed Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is a capability on that same object.

## Side by side

| | Pi | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | Telemetry contract, no exporter | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Extensibility | Extensions, skills, `pi install` | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Build your own harness | No | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Multi-agent | Packages, not core | [Sub-agents, hand-offs, or graph](../multi-agent-applications.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| License | MIT | MIT |

| | Pi | Pydantic AI |
|---|---|---|
| Structured output | No | [Structured output](../output.md#structured-output) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | No | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes | [Testing](../testing.md) |

## FAQ

**Can I build a coding agent like this in Python?** Yes. [Code Puppy](https://github.com/mpfaffenberger/code_puppy)
and [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) did.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) if you want ours.
