# Pydantic AI vs Strands Agents

Strands Agents is AWS's open-source agent SDK for Python and TypeScript, for any model and cloud with Amazon Bedrock as the default: a model-driven agent loop, a ready-made harness, multi-agent patterns, voice agents, evals, and deployment guides for AWS. Pydantic AI starts from one typed [`Agent`][pydantic_ai.Agent] on [any model](../models/overview.md), with [structured output](../output.md#structured-output) validated by Pydantic, [`pydantic-graph`](../graph.md) for workflows, and [capabilities](../capabilities/overview.md) that compose into complete agents such as [`Coder`](https://pydantic.dev/docs/ai/harness/coder/).

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | Strands Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python (also TypeScript) | Python |
| License | Apache-2.0 | MIT |
| Model providers | Many | [Many](../models/overview.md) |
| Extensibility | Tools, hooks, plugins | [Capabilities and toolsets](../extensibility.md); [80+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Strands harness, or build your own | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | None documented for Strands; [AWS Support](https://aws.amazon.com/premiumsupport/) for the cloud | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Yes (Temporal) | [Eight integrations](../durable_execution/overview.md) |
| Interfaces | CLI, A2A; AG-UI via a partner package | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | Yes | [Realtime](../realtime/overview.md) |
| Evals | Yes | [Pydantic Evals](../evals.md) |
| Image generation | No | [Image Generation](../image-generation.md) |

## Features

| | Strands Agents | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Yes | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Yes (harness) | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Yes | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Yes | [Compaction](../capabilities/compaction.md) |
| Guardrails | Yes | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Yes | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | No; you drive Playwright | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Can I use Pydantic AI on AWS?** Yes. Run agents on [Amazon Bedrock](../models/bedrock.md) models, including [Bedrock Mantle](../models/bedrock.md#bedrock-mantle), with [Bedrock embeddings](../embeddings.md#bedrock), and make them durable on [AWS Lambda durable functions](https://pydantic.dev/docs/ai/harness/aws-lambda/), an integration [AWS announced](https://aws.amazon.com/about-aws/whats-new/2026/09/aws-lambda-durable-pydantic-ai/) and documents in its [Durable Execution SDK guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/integrations/pydantic-ai/). [LocalStack](https://pydantic.dev/docs/ai/harness/localstack/) gives an agent an emulated AWS environment to work in.

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

- [Strands Agents documentation](https://strandsagents.com/docs/user-guide/sdk/)
- [Strands Agents repository](https://github.com/strands-agents/harness-sdk)
