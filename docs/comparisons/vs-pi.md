# Pydantic AI vs Pi

## Side by side

| | Pi | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Multiple | [Multiple](../models/overview.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | Telemetry contract, no exporter | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) |
| Extensibility | Extensions, skills, `pi install` | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| Multi-agent | Packages, not core | [Sub-agents, hand-offs, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Build your own harness | No | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | MIT | MIT |

| | Pi | Pydantic AI |
|---|---|---|
| Structured output | No | [Structured output](../output.md#structured-output) |
| Memory | No | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | No | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | No | [Pydantic Evals](../evals.md) |

## FAQ

**Can I build a coding agent like this in Python?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability and it can read, edit and run
code. If you want to see a complete coding agent built on Pydantic AI,
[Code Puppy](https://github.com/mpfaffenberger/code_puppy) and
[Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) are both open source.
