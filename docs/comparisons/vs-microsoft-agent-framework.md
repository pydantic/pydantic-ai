# Pydantic AI vs Microsoft Agent Framework

Microsoft Agent Framework is the successor to AutoGen and Semantic Kernel, for Python and .NET: agents, graph workflows with checkpointing, a configurable harness agent, and hosting on Microsoft Foundry and Azure Functions. Pydantic AI starts from one typed [`Agent`][pydantic_ai.Agent] on [any model](../models/overview.md), with [structured output](../output.md#structured-output) validated by Pydantic, [`pydantic-graph`](../graph.md) for workflows, and [capabilities](../capabilities/overview.md) that compose into complete agents such as [`Coder`](https://pydantic.dev/docs/ai/harness/coder/).

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | Microsoft Agent Framework | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python (also .NET and Go) | Python |
| License | MIT | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Tools, middleware, context providers | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | `create_harness_agent`, or build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | With a [Microsoft Unified support agreement](https://github.com/microsoft/agent-framework/blob/main/SUPPORT.md), when used with Azure AI services | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Yes | [Eight integrations](../durable_execution/overview.md) |
| Interfaces | DevUI (sample app), AG-UI, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Image generation | Provider-hosted tools only | [Image Generation](../image-generation.md) |

## Features

| | Microsoft Agent Framework | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes (beta) | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Computer-use tools; you run the browser | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I use Pydantic AI with Microsoft Foundry?** Yes. Run agents on [Azure OpenAI and other Foundry model deployments](../models/azure.md), including [Claude on Foundry](../models/azure.md#claude-on-microsoft-foundry), and voice agents on [Azure AI Voice Live](../realtime/azure.md#azure-ai-voice-live).

**Can I build a coding agent on Pydantic AI?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions, or run a Python script directly with `uv run`;
nothing requires an Action.

*Last checked: 2026-10-30*

## Sources

- [Microsoft Agent Framework documentation](https://learn.microsoft.com/en-us/agent-framework/overview/)
- [Microsoft Agent Framework repository](https://github.com/microsoft/agent-framework)
