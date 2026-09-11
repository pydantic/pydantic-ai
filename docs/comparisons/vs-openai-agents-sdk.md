# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is a small Python library built close to the grain of OpenAI's own API: agents, `handoff` primitives, guardrails, and first-class access to the hosted tools. Pydantic AI vs OpenAI Agents SDK matters when you want the same agent on more than one provider.

## Framework

| | OpenAI Agents SDK | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | OpenAI first; others via LiteLLM | [Many](../models/overview.md) |
| Extensibility | Tools, guardrails, handoffs | [Capabilities and toolsets](../extensibility.md) |
| Build a custom harness | Yes | [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
| Coding harness | Yes | [`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) |
| Interfaces | None built in | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Image generation | Via provider tools | [Image Generation](../image-generation.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Observability | OpenAI tracing, OTel via adapters | [OpenTelemetry](../capabilities/instrumentation.md), including [Pydantic Logfire](../logfire.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |

## Features

| | OpenAI Agents SDK | Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Structured output | Yes | [Structured output](../output.md#structured-output) |
| Multi-agent | Handoffs | [Delegation (tools or `SubAgents`), hand-off in your code, or graph](../multi-agent-applications.md) |
| Graph library | No | [`pydantic-graph`](../graph.md) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Bring your own `Computer` | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I build a coding agent on Pydantic AI?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions.
