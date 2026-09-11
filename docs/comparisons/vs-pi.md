# Pydantic AI vs Pi

Pi is a TypeScript coding agent you can embed without forking: extensions, skills, and `pi install`
packages. Pydantic AI is a typed Python [`Agent`][pydantic_ai.Agent].
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) is a capability on that same object.

## Side by side

| | Pi | Pydantic AI |
|---|---|---|
| Language | TypeScript | Python |
| Model providers | Many | [Multiple](../models/overview.md) |
| Durable execution | Session tree on disk | Yes — [5+ integrations](../durable_execution/overview.md) |
| Observability | Telemetry contract, no exporter | [OpenTelemetry](../capabilities/instrumentation.md) |
| Extensibility | Extensions, skills, `pi install` | [Capabilities and toolsets](../extensibility.md) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), A2A |
| Build your own harness | No — you extend Pi | [Yes](https://pydantic.dev/docs/ai/harness/#build-your-own) |
| Coding harness | The product | Yes ([`Coder()`](https://pydantic.dev/docs/ai/harness/coder/)) |
| Graph library | No | Yes ([`pydantic-graph`](../graph.md)) |
| Multi-agent | Packages, not core | [Sub-agents](../multi-agent-applications.md), [hand-offs](../multi-agent-applications.md#programmatic-agent-hand-off), or [graph](../graph.md) |
| Realtime voice | No | [Yes](../realtime/overview.md) |
| Image generation | No | [Yes](../image-generation.md) |
| License | MIT | MIT |

| | Pi | Pydantic AI |
|---|---|---|
| Structured output | Terminating tool you write | [Yes](../output.md) |
| Guardrails | No | [Yes](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Packages, not core | [Yes](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser | No | [Yes](https://pydantic.dev/docs/ai/harness/#web-research) |
| Memory | Sessions | [Yes](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Yes](../capabilities/compaction.md) |
| Evals | No | [Yes](../evals.md) |
| Test without API keys | Yes (`faux` provider) | [Yes](../testing.md) |

Cells linked to pydantic.dev/docs/ai/harness ship in [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/),
a separate package.

## FAQ

**Can I build a coding agent like this in Python?** Yes. [Code Puppy](https://github.com/mpfaffenberger/code_puppy)
and [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) (Vstorm) did.
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) if you want ours.

[Install Pydantic AI](../install.md).
