# Pydantic AI vs Microsoft Agent Framework

Microsoft Agent Framework (`agent-framework`) is the successor to AutoGen and Semantic Kernel: agents, graph workflows, a configurable harness, and integrations with Microsoft Foundry and Azure Functions. Its Python SDK also supports non-Microsoft model providers. Pydantic AI starts from a typed [`Agent`][pydantic_ai.Agent], with [structured output](../output.md#structured-output), [`pydantic-graph`](../graph.md) for workflows and a [Harness SDK](https://pydantic.dev/docs/ai/harness/) of composable capabilities.

Pydantic AI is one part of a stack: the [Harness SDK](https://pydantic.dev/docs/ai/harness/) for capabilities and complete agents, [Pydantic Evals](../evals.md) for evaluation, [Pydantic Graph](../graph.md) for graph-based workflows, [Pydantic Logfire](https://pydantic.dev/logfire) for observability, and [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) itself for validation. The tables below cover the whole of it.

## Framework

| | Microsoft Agent Framework | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Language | Python 3.11–3.15 supported; some integrations have limits | Python |
| License | MIT | MIT |
| Model providers | OpenAI, Azure OpenAI, Foundry, Anthropic, Bedrock, Gemini, Ollama and others | [Many](../models/overview.md) |
| Extensibility | Function tools, MCP, middleware and context providers | [Capabilities and toolsets](../extensibility.md); [50+ with the Harness SDK](https://pydantic.dev/docs/ai/harness/) |
| Harnesses | `create_harness_agent`, with configurable providers and tools | Built-in [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) and [`Researcher`](https://pydantic.dev/docs/ai/harness/researcher/), or compose your own |
| Observability | OpenTelemetry | [OpenTelemetry](../logfire.md#using-opentelemetry), including [Pydantic Logfire](https://pydantic.dev/logfire) |
| Enterprise support | [Unified support agreement and Azure AI service usage required](https://github.com/microsoft/agent-framework/blob/main/SUPPORT.md) | [For the framework itself](../enterprise-support.md), no platform required |
| Durable execution | Durable Task extension; Azure Functions or self-hosted workers | [Eight integrations](../durable_execution/overview.md) |
| Interfaces | AG-UI, A2A client and hosting adapters, DevUI sample app | [CLI](../cli.md), [web chat](../web.md), [AG-UI](../ui/ag-ui.md), [Vercel AI](../ui/vercel-ai.md), [ACP](https://pydantic.dev/docs/ai/harness/acp/) (experimental), [GitHub Agentic Workflows](https://pydantic.dev/docs/ai/harness/gh-aw/) |
| Realtime voice | None documented for the Python SDK | [Realtime](../realtime/overview.md) |
| Evals | Local evaluators and Microsoft Foundry integration | [Pydantic Evals](../evals.md) |
| Image generation | Provider-hosted tools | [Image Generation](../image-generation.md) |

## Features

| | Microsoft Agent Framework | Pydantic AI and [Harness SDK](https://pydantic.dev/docs/ai/harness/) |
|---|---|---|
| Multi-agent | Graph workflows, orchestration patterns; background agents (experimental) | [Subagents](https://pydantic.dev/docs/ai/harness/subagents/), [delegation](../multi-agent-applications.md), or [`pydantic-graph`](../graph.md) |
| Planning | Todo tracking and plan/execute modes | [Planning](https://pydantic.dev/docs/ai/harness/planning/) |
| Skills | Yes; opt-in in the Python harness | [Skills](https://pydantic.dev/docs/ai/harness/skills/) |
| Memory | Session history, file memory and context providers | [Memory](https://pydantic.dev/docs/ai/harness/memory/) |
| Compaction | Truncation, summarization and tool-result strategies | [Compaction](../capabilities/compaction.md) |
| Guardrails | Tool approvals, middleware and Purview policy integration (preview) | [Guardrails](https://pydantic.dev/docs/ai/harness/guardrails/) |
| Code sandboxes | Hyperlight sandbox; Monty restricted interpreter (both beta) | [Execution environments](https://pydantic.dev/docs/ai/harness/#execution-environments) |
| Browser use | Foundry browser automation tool (preview) | [Web & research](https://pydantic.dev/docs/ai/harness/#web--research) |

## FAQ

**Do both support structured output and human approval?** Yes. Microsoft's Python SDK accepts Pydantic models or JSON Schema through `response_format`, and its tools can pause for approval. Pydantic AI has [validated structured output](../output.md#structured-output) and [deferred tool approval](../deferred-tools.md#human-in-the-loop-tool-approval).

**What survives a restart?** Microsoft has session state and [workflow checkpoints](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints); its Durable Extension adds recovery on Durable Task infrastructure. Pydantic AI separates [conversation and step persistence](../persistence.md) from [eight durable execution integrations](../durable_execution/overview.md).

**What if I already deploy on Azure?** Microsoft's [Foundry hosting integration](https://learn.microsoft.com/en-us/agent-framework/hosting/foundry-hosted-agent) and [Azure Functions Durable Extension](https://learn.microsoft.com/en-us/agent-framework/hosting/durable-extension) provide documented deployment paths. Pydantic AI supports [Azure OpenAI](../models/azure.md) and runs in your Python application; choose its durable engine separately.

Last checked: 2026-10-30. This page compares the Python SDK; capabilities and release status differ from .NET.

## Sources

- [Overview and successor to AutoGen and Semantic Kernel](https://learn.microsoft.com/en-us/agent-framework/overview/), [Python installation and supported versions](https://github.com/microsoft/agent-framework/blob/main/python/README.md), [MIT license](https://github.com/microsoft/agent-framework/blob/main/LICENSE) and [support policy](https://github.com/microsoft/agent-framework/blob/main/SUPPORT.md).
- [Python model providers](https://learn.microsoft.com/en-us/agent-framework/integrations/by-component/model-providers/?pivots=programming-language-python), [tools and MCP](https://learn.microsoft.com/en-us/agent-framework/agents/tools/), [structured output](https://learn.microsoft.com/en-us/agent-framework/agents/structured-outputs) and [tool approval](https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval).
- [Harness composition, planning, memory, skills and background agents](https://learn.microsoft.com/en-us/agent-framework/concepts/harness), [compaction](https://learn.microsoft.com/en-us/agent-framework/concepts/agents/conversations/compaction) and [Purview middleware](https://github.com/microsoft/agent-framework/blob/main/python/packages/purview/README.md).
- [Workflows](https://learn.microsoft.com/en-us/agent-framework/concepts/workflows/), [checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints), [Durable Extension](https://learn.microsoft.com/en-us/agent-framework/hosting/durable-extension) and [Foundry hosting](https://learn.microsoft.com/en-us/agent-framework/hosting/foundry-hosted-agent).
- [OpenTelemetry](https://learn.microsoft.com/en-us/agent-framework/agents/observability), [evaluation](https://learn.microsoft.com/en-us/agent-framework/agents/evaluation), [AG-UI](https://github.com/microsoft/agent-framework/blob/main/python/packages/ag-ui/README.md), [A2A client](https://github.com/microsoft/agent-framework/blob/main/python/packages/a2a/README.md), [A2A hosting](https://github.com/microsoft/agent-framework/blob/main/python/packages/hosting-a2a/README.md) and [DevUI](https://github.com/microsoft/agent-framework/blob/main/python/packages/devui/README.md).
- [Hyperlight](https://learn.microsoft.com/en-us/agent-framework/integrations/by-component/context-providers/hyperlight), [Monty](https://learn.microsoft.com/en-us/agent-framework/integrations/by-component/context-providers/monty), [Foundry browser tools](https://github.com/microsoft/agent-framework/blob/main/python/packages/foundry/README.md) and [image generation example](https://github.com/microsoft/agent-framework/blob/main/python/samples/02-agents/providers/openai/client_image_generation.py).
