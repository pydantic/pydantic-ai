# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK is Claude Code as a library: it wraps the `claude` CLI (TypeScript), which you
drive with `query()` for one run or a `ClaudeSDKClient` session for a conversation, configuring that
program in data. Pydantic AI is a native Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is the coding stack in your process, on any
model.

## Side by side

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Language | Python SDK wrapping the TypeScript `claude` CLI | Python |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | [Multiple](../models/overview.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | CLI telemetry | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Hooks, `allowed_tools` | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | No | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Multi-agent | Sub-agents | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | No | [Yes](../realtime/overview.md) |
| Image generation | No | [Yes](../image-generation.md) |
| License | MIT | MIT |

| | Claude Agent SDK | Pydantic AI |
|---|---|---|
| Structured output | Yes | [Yes](../output.md) |
| Guardrails | Yes | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | Yes | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Yes | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Yes](../capabilities/compaction.md) |
| Evals | No | [Yes](../evals.md) |
| Test without API keys | No | [Yes](../testing.md) |

## FAQ

**Is Pydantic AI a native Python SDK?** Yes. The Claude Agent SDK wraps a TypeScript CLI.

**Can I build my own coding agent harness?** Yes. [Code Puppy](https://github.com/mpfaffenberger/code_puppy)
and [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) did, on the same
[`Agent`][pydantic_ai.Agent]. [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) if you want ours.
