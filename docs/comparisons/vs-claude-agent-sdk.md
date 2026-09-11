# Pydantic AI vs Claude Agent SDK

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Language | Python SDK wrapping the TypeScript `claude` CLI | Python |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | [Multiple](../models/overview.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry from the CLI | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Extensibility | Hooks, `allowed_tools` | [Capabilities and toolsets](../extensibility.md) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| Multi-agent | Sub-agents | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Build your own harness | No (the loop lives in the CLI) | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| License | MIT | MIT |

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |
| Evals | No | [Pydantic Evals](../evals.md) |

## FAQ

**Can I build my own coding agent harness?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]. For a complete coding agent built that way,
[Code Puppy](https://github.com/mpfaffenberger/code_puppy) and
[Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) are both open source.
