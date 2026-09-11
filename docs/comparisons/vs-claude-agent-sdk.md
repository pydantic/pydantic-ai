# Pydantic AI vs Claude Agent SDK

The Claude Agent SDK gives you the agent loop behind Claude Code: a Python package that drives the bundled `claude` CLI, with Anthropic's built-in tools, permissions and sub-agents already wired up. Pydantic AI vs Claude Agent SDK is a choice between that loop and one you own, in Python, on any model.

## Framework

| | Claude Agent SDK | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python SDK wrapping the TypeScript `claude` CLI | Python |
| License | MIT | MIT |
| Model providers | Claude (Anthropic, Bedrock, Vertex, Foundry) | [Many](../models/overview.md) |
| Extensibility | Hooks, `allowed_tools` | [Capabilities and toolsets](../extensibility.md) |
| Build a custom harness | No (the loop lives in the CLI) | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Interfaces | CLI | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | No | [Realtime](../realtime/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |
| Durable execution | No | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenTelemetry from the CLI | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Evals | No | [Pydantic Evals](../evals.md) |

## Features

| | Claude Agent SDK | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Multi-agent | Sub-agents | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
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

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents from a Markdown workflow file in GitHub Actions.
