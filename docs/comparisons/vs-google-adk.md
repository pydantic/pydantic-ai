# Pydantic AI vs Google ADK

Google's Agent Development Kit is an agent platform built around Gemini and Vertex AI: workflow graphs, evaluation, a dev UI, and deploy commands for Cloud Run, GKE, Docker and Agent Engine. Pydantic AI is a provider-agnostic library you run on your own infrastructure, with [`pydantic-graph`](../graph.md) for workflows, [Pydantic Evals](../evals.md), a [web chat UI](../web.md), and [Pydantic Logfire](https://pydantic.dev/logfire) to watch it all run.

## Framework

| | Google ADK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | Apache-2.0 | MIT |
| Model providers | Gemini first (`LiteLlm`, `AnthropicLlm` exist) | [Many](../models/overview.md) |
| Extensibility | Tools, plugins | [Capabilities and toolsets](../extensibility.md); [30+ in the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | TBD-FACTCHECK | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Interfaces | CLI, web, A2A | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Durable execution | Yes | [5+ integrations](../durable_execution/overview.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | Google ADK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Sub-agents | TBD-FACTCHECK | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | TBD-FACTCHECK | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | TBD-FACTCHECK | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Yes | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I build a coding agent on Pydantic AI?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai-harness/blob/main/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions, or run a Python script directly with `uv run`;
nothing requires an Action.
