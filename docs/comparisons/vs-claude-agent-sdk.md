# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK is Claude Code as a library: it wraps the `claude` CLI (TypeScript) and you
configure that program in data. Pydantic AI is a native Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is the coding stack in your process, on any
model.

## Side by side

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Language | Python SDK wrapping the TypeScript `claude` CLI | Python |
| License | MIT | MIT |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | [Any](../models/overview.md), plus [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] |
| Durable execution | Sessions | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | CLI telemetry | [OpenTelemetry](../capabilities/instrumentation.md), any backend including [Pydantic Logfire](../logfire.md) |
| Extensibility | Hooks, `allowed_tools` | [Capabilities](../extensibility.md) and [toolsets](../toolsets.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [A2A](https://github.com/datalayer/fasta2a) (community) |
| Build your own harness | No — one harness, you drive it | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | The `claude` CLI | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Multi-agent | Sub-agents | [Sub-agents](../multi-agent-applications.md), [graph](../graph.md), or `async` |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image generation](../image-generation.md) |

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes (`json_schema`) | [Type on the agent](../output.md) |
| Guardrails | CLI permissions | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | CLI permissions | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes (Chrome) | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Sessions | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Test without API keys | No (launch the CLI) | [`TestModel`](../testing.md), [`FunctionModel`](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

## FAQ

**Is Pydantic AI a native Python SDK?** Yes. The Claude Agent SDK wraps a TypeScript CLI.

**Can I build my own coding agent harness?** Yes. [Code Puppy](https://github.com/mpfaffenberger/code_puppy)
and [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) did, on the same
[`Agent`][pydantic_ai.Agent]. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) if you want ours.

[Install Pydantic AI](../install.md).
