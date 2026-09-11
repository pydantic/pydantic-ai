# Pydantic AI vs Pi

Pi is a TypeScript coding agent you can embed without forking: extensions, skills, and `pi install`
packages. Pydantic AI is a typed Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is a capability on that same object.

## Side by side

| | Pi | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| License | MIT | MIT |
| Model providers | Many | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | Session tree on disk | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | Logs | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Extensions, skills, `pi install` | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | No — you extend Pi | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | The product | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Multi-agent | Packages, not core | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image generation](../image-generation.md) |

| | Pi | Pydantic AI |
|---|---|---|
| Structured output | Terminating tool you write | [Type on the agent](../output.md) |
| Guardrails | No | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Packages, not core | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Sessions | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Test without API keys | Yes (`registerProvider()`) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

## FAQ

**Can I build a coding agent like this in Python?** Yes. [Code Puppy](https://github.com/mpfaffenberger/code_puppy)
and [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) did.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) if you want ours.

[Install Pydantic AI](../install.md).
