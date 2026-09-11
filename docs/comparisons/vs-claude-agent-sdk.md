# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK gives you the agent loop behind Claude Code: a Python package that drives the bundled `claude` CLI, with Anthropic's built-in tools, permissions and sub-agents already wired up. Pydantic AI gives you the loop itself: a typed [`Agent`][pydantic_ai.Agent] on [any model](../models/overview.md), and [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) as a complete coding agent built from [capabilities](../capabilities/overview.md) you can swap, extend or leave out.

## Framework

| | Claude Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python SDK wrapping the TypeScript `claude` CLI | Python |
| License | MIT | MIT |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | [Many](../models/overview.md) |
| Extensibility | Hooks, `allowed_tools` | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Claude Code's; hooks extend it, you cannot recompose it | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry from the CLI | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | Claude Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I build my own coding agent harness?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions, or run a Python script directly with `uv run`;
nothing requires an Action.
