# Pydantic AI vs OpenHands Software Agent SDK

The OpenHands Software Agent SDK is a Python framework for agents that work with code: `openhands-sdk` supplies the agent loop, `openhands-tools` the tools, `openhands-workspace` the execution environments, and `openhands-agent-server` a REST/WebSocket server. Pydantic AI starts from a typed [`Agent`][pydantic_ai.Agent] for general applications, with [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) as a complete coding agent assembled from capabilities you can swap or extend.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | OpenHands Software Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python 3.12+; TypeScript client | Python |
| License | MIT | MIT |
| Model providers | Many, via LiteLLM | [Many](../models/overview.md) |
| Extensibility | Custom tools, MCP, hooks, skills and plugins | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | Composable coding agent with terminal, file editor and task tracker | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry; Laminar or any OTLP backend | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | With [OpenHands Enterprise](https://docs.openhands.dev/enterprise); SDK-only support not documented | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Conversation persistence and resume; journaled execution replay not documented | [Eight integrations](../durable_execution/overview.md) |
| Interfaces | Agent server (REST/WebSocket), TypeScript client, CLI; hosts ACP agents | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | None documented | [Realtime](../realtime/overview.md) |
| Evals | [Benchmark evaluation pipelines](https://github.com/OpenHands/benchmarks) | [Pydantic Evals](../evals.md) |
| Image generation | None documented | [Image Generation](../image-generation.md) |

## Features

| | OpenHands Software Agent SDK | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Resumable sub-agents through `TaskToolSet` | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Task tracker | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | AgentSkills (`SKILL.md`), repository context and keyword triggers | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Opt-in user and project memory across conversations | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Configurable context condensers, including LLM summarization | [Compaction](../capabilities/compaction.md) |
| Guardrails | Confirmation policies, security analyzers and hooks | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Docker, Apptainer and remote API/Cloud workspaces | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | `BrowserToolSet`, built on browser-use | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Does OpenHands support structured output?** Yes. A tool's `response_schema` accepts a Pydantic model or JSON Schema, validates the model's arguments, and exposes parsed results. Attach it to `FinishTool` for a structured final answer. Pydantic AI's [`output_type`](../output.md#structured-output) declares the agent's result type directly, with tool, native and prompted output modes.

**Is saving a conversation the same as durable execution?** OpenHands saves conversation state and events to disk and can restore them. Its agent server marks a previously running conversation as errored after a restart and records an error for an interrupted tool call. Pydantic AI's [durable integrations](../durable_execution/overview.md) journal model requests and wrapped tool calls so a restarted run can replay completed work and retry the step in flight.

**How do Agent Canvas and paid hosting fit in?** The `OpenHands/OpenHands` repository now contains Agent Canvas, a TypeScript UI for conversations and automations across OpenHands and other ACP agents. OpenHands Cloud is commercial hosted infrastructure; Enterprise is a commercial self-hosted/private-cloud platform with dedicated support. The SDK works independently. SDK-only enterprise support terms are not documented. OpenHands is ahead in shipping an agent server and a control UI for multiple agent backends; Pydantic AI offers [web chat and protocol adapters](../interfaces.md), rather than an equivalent management platform.

**What about benchmarks?** OpenHands has a longer published coding-agent benchmark record, including SWE-bench Verified results [reported in April 2025](https://openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model). Its self-run [OpenHands Index](https://index.openhands.dev/) evaluates models with its SDK across five benchmarks; the newest model release date in its [leaderboard data](https://index.openhands.dev/api/leaderboard) was June 9, 2026 when checked. These results depend on the model, SDK version and evaluation setup; they are not a head-to-head comparison with Pydantic AI.

Last checked: October 30, 2026.

## Sources

- [SDK architecture and packages](https://docs.openhands.dev/sdk/arch/overview), [repository and client/server boundaries](https://github.com/OpenHands/software-agent-sdk), [Python requirement](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-sdk/pyproject.toml), [MIT license](https://github.com/OpenHands/software-agent-sdk/blob/main/LICENSE), and [LLM providers](https://docs.openhands.dev/sdk/arch/llm).
- [Custom tools](https://docs.openhands.dev/sdk/guides/custom-tools), [MCP](https://docs.openhands.dev/sdk/guides/mcp), [hooks](https://docs.openhands.dev/sdk/guides/hooks), [plugins](https://docs.openhands.dev/sdk/guides/plugins), [structured output](https://docs.openhands.dev/sdk/guides/structured-output), [sub-agents](https://docs.openhands.dev/sdk/guides/task-tool-set), and [task tracker](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-tools/openhands/tools/task_tracker/definition.py).
- [Skills](https://docs.openhands.dev/sdk/guides/skill), [persistent memory](https://docs.openhands.dev/sdk/guides/persistent-memory), [condensers](https://docs.openhands.dev/sdk/guides/context-condenser), [security and confirmation](https://docs.openhands.dev/sdk/guides/security), and [browser use](https://docs.openhands.dev/sdk/guides/agent-browser-use).
- [Agent server](https://docs.openhands.dev/sdk/guides/agent-server/overview), [Docker](https://docs.openhands.dev/sdk/guides/agent-server/docker-sandbox), [Apptainer](https://docs.openhands.dev/sdk/guides/agent-server/apptainer-sandbox), [API sandboxes](https://docs.openhands.dev/sdk/guides/agent-server/api-sandbox), [Cloud workspaces](https://docs.openhands.dev/sdk/guides/agent-server/cloud-workspace), [persistence](https://docs.openhands.dev/sdk/guides/convo-persistence), and [server restart handling](https://github.com/OpenHands/software-agent-sdk/blob/main/openhands-agent-server/openhands/agent_server/event_service.py).
- [Tracing](https://docs.openhands.dev/sdk/guides/observability), [ACP agents](https://docs.openhands.dev/sdk/guides/agent-acp), [CLI](https://github.com/OpenHands/OpenHands-CLI), [Agent Canvas](https://github.com/OpenHands/OpenHands), [commercial offerings](https://docs.openhands.dev/enterprise/enterprise-vs-oss), and [enterprise support](https://docs.openhands.dev/enterprise#dedicated-support). The [SDK documentation index](https://docs.openhands.dev/llms.txt) and repository were also checked for realtime voice and image generation; neither has a documented SDK feature.
- [Benchmark infrastructure](https://github.com/OpenHands/benchmarks), [SWE-bench Verified report](https://openhands.dev/blog/sota-on-swe-bench-verified-with-inference-time-scaling-and-critic-model), [Index methodology](https://openhands.dev/blog/openhands-index), and [dated Index results](https://index.openhands.dev/api/leaderboard).
