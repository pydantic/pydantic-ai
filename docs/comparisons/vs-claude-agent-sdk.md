# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK gives you the agent loop behind Claude Code: a Python package that drives the bundled `claude` CLI, with Anthropic's built-in tools, permissions and sub-agents already wired up. Pydantic AI gives you the loop itself: a typed [`Agent`][pydantic_ai.Agent] on [any model](../models/overview.md), and [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) as a complete coding agent built from [capabilities](../capabilities/overview.md) you can swap, extend or leave out.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pax](../harness/pax.md) to run any of your agents in a coding terminal, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

**Already built on the Claude Agent SDK?** The [`migrating-claude-agent-sdk-to-pydantic-ai`](../framework-migration.md) skill ports an existing application to Pydantic AI one working path at a time, preserving behavior rather than translating API names.

## Framework

| | Claude Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python SDK wrapping the TypeScript `claude` CLI | Python |
| License | MIT | MIT |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | [Many](../models/overview.md) |
| Extensibility | Hooks, `allowed_tools` | [Capabilities and toolsets](../extensibility.md); [100+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Claude Code's; hooks extend it, you cannot recompose it | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own, and run any of them in [Pax](../harness/pax.md) |
| Observability | OpenTelemetry from the CLI | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | None documented | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | No | [Nine integrations](../durable_execution/overview.md) |
| Interfaces | CLI, GitHub Action | [Pax](../harness/pax.md) coding terminal (`pax --agent` runs yours), [legacy CLI](../cli.md) (`clai`), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Evals | No | [Pydantic Evals](../evals.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | Claude Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Your machine, SSH and 11 sandboxes](../workspace.md#supported-workspaces), with per-path [authority](../workspace-custom.md#path-authority) |
| Browser use | No | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research), and [computer use](../harness/computer-use.md) on a desktop in 11 sandboxes |

## FAQ

**Can I build my own coding agent harness?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I continue a Claude Code session in Pydantic AI?** Yes. `pax --resume-claude` imports a Claude Code session into [Pax](../harness/pax.md#pick-up-where-you-left-off-resume), tool calls and sub-agents included, and you continue it on any model.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions, or run a Python script directly with `uv run`;
nothing requires an Action.

**Can I control which content my agent treats as instructions?** Yes. What a capability tells the model mid-conversation is a typed part that records who wrote it (you, your user, or nobody you trust), and Pydantic AI speaks it in the voice that authority allows; what a capability enqueues can be lowered, never raised, as it moves through a run. Tool results and fetched pages stay data on every provider, and text that imitates the framework's own frames is flagged to the model. A workspace's files carry your authority unless you mark them untrusted, path by path. See [Where the content came from](../capabilities/custom.md#where-the-content-came-from) for the source and authority table, [Capability parts](../message-history.md#capability-parts) for how they are stored, and [Marking who wrote the files](../workspace-custom.md#workspace-authority).

For the longer argument, read [Would you deploy an agent you don't control?](https://pydantic.dev/articles/pydantic-ai-vs-claude-agent-sdk) on the Pydantic blog.

*Last checked: 2026-09-26*

## Sources

- [Claude Agent SDK overview](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-overview)
- [Claude Agent SDK for Python](https://github.com/anthropics/claude-agent-sdk-python)
