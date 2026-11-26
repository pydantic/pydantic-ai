# Pydantic AI vs OpenAI Agents SDK

The OpenAI Agents SDK is a small Python library built close to OpenAI's own API: agents, `handoff` primitives, guardrails, and access to the hosted tools. Pydantic AI runs one typed [`Agent`][pydantic_ai.Agent] on [any provider](../models/overview.md), with [structured output](../output.md#structured-output) validated by Pydantic and a [Harness SDK](https://pydantic.dev/docs/ai/harness/) of capabilities from sandboxes to sub-agents.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pax](../harness/pax.md) to run any of your agents in a coding terminal, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

**Already built on the OpenAI Agents SDK?** The [`migrating-openai-agents-sdk-to-pydantic-ai`](../framework-migration.md) skill ports an existing application to Pydantic AI one working path at a time, preserving behavior rather than translating API names.

## Framework

| | OpenAI Agents SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python | Python |
| License | MIT | MIT |
| Model providers | OpenAI first; others via LiteLLM | [Many](../models/overview.md) |
| Extensibility | Tools, guardrails, handoffs | [Capabilities and toolsets](../extensibility.md); [100+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | `SandboxAgent`, or build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenAI tracing, OTel via adapters | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | None documented for the SDK; [API Enterprise](https://openai.com/api/pricing/) for the API | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Yes | [Nine integrations](../durable_execution/overview.md) |
| Interfaces | REPL demo loop | [Pax](../harness/pax.md) coding terminal (`pax --agent` runs yours), [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Image generation | Provider-hosted tools only | [Image Generation](../image-generation.md) |

## Features

| | OpenAI Agents SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | No | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Your machine, SSH and 11 sandboxes](../workspace.md#supported-workspaces), with per-path [trust](../workspace-custom.md#workspace-authority) |
| Browser use | Bring your own `Computer` | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research), and [computer use](../harness/computer-use.md) with a ready desktop in 11 sandboxes, on OpenAI's native computer tool where the model supports it |

## FAQ

**Can I build a coding agent on Pydantic AI?** Yes. Give your agent the
[`Coder()`](https://pydantic.dev/docs/ai/harness/coder/) capability, or assemble your own from the
same [`Agent`][pydantic_ai.Agent]; the harness repository has a
[complete coding agent](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/examples/coding_agent.py)
built from the pieces `Coder` puts together.

**Can I run my agents in CI?** Yes. [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) runs Pydantic AI agents
from a Markdown workflow file in GitHub Actions, or run a Python script directly with `uv run`;
nothing requires an Action.

**Can I control which content my agent treats as instructions?** Yes. Everything a capability adds to the conversation is a typed part that records who wrote it (you, your user, or nobody you trust), and Pydantic AI speaks it in the voice that authority allows; what a capability enqueues can be lowered, never raised, as it moves through a run. Tool results and fetched pages stay data on every provider, and text that imitates the framework's own frames is flagged to the model. A workspace's files carry your authority unless you mark them untrusted, path by path. See [Where the content came from](../capabilities/custom.md#where-the-content-came-from) for the source and authority table, [Capability parts](../message-history.md#capability-parts) for how they are stored, and [Marking who wrote the files](../workspace-custom.md#workspace-authority).

*Last checked: 2026-09-26*

## Sources

- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK voice documentation](https://openai.github.io/openai-agents-python/voice/quickstart/)
