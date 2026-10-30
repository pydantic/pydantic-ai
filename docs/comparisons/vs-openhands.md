# Pydantic AI vs OpenHands Software Agent SDK

The OpenHands Software Agent SDK is a Python SDK for agents that work with code, and the engine behind the OpenHands CLI and Cloud: an agent loop with terminal, file-editing and browser tools, workspaces on your machine or in Docker, and an agent server to run them remotely. In Pydantic AI, a [coding agent](https://pydantic.dev/docs/ai/harness/coder/) is one [configuration](../capabilities/overview.md) of a general [`Agent`][pydantic_ai.Agent], so the same [sandboxes](https://pydantic.dev/docs/ai/harness/#execution-environments), [skills](https://pydantic.dev/docs/ai/harness/skills/) and [sub-agents](https://pydantic.dev/docs/ai/harness/subagents/) serve whatever else you build on it.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | OpenHands Software Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | Many, via LiteLLM | [Many](../models/overview.md) |
| Extensibility | Tools, hooks, plugins | [Capabilities and toolsets](../extensibility.md); [80+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Coding agent, composed from its tools | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | With [OpenHands Enterprise](https://docs.openhands.dev/enterprise) | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Conversation persistence only | [Eight integrations](../durable_execution/overview.md) |
| Interfaces | Agent server (REST, WebSocket, TypeScript client), CLI, ACP client | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | Coding benchmarks, in a separate repository | [Pydantic Evals](../evals.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | OpenHands Software Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I build my own coding agent harness?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions, or run a Python script directly with `uv run`;
nothing requires an Action.

*Last checked: 2026-10-30*

## Sources

- [OpenHands Software Agent SDK documentation](https://docs.openhands.dev/sdk)
- [OpenHands Software Agent SDK repository](https://github.com/OpenHands/software-agent-sdk)
