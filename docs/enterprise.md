---
description: "Adopt Pydantic AI in an enterprise: agents run in your own process with no usage telemetry, on self-hosted models or Azure OpenAI, Amazon Bedrock and Google Vertex AI, with human approval, OpenTelemetry audit trails and commercial support with an SLA."
---

# Pydantic AI for Enterprise

Pydantic AI is open source, with [commercial enterprise support from Pydantic](enterprise-support.md). Agents run in your own process against the models you choose, from [Azure OpenAI](models/azure.md), [Amazon Bedrock](models/bedrock.md) and [Google Vertex AI](models/google-cloud.md) to models you host yourself, and the library sends no usage telemetry.

This page answers the questions security, platform and procurement teams ask before adopting it, each with the short answer and a link to the page that covers it in full. [Going to Production](production.md) covers the engineering side: crash recovery, cost limits, testing and deployment.

| Your team wants to know... | Short answer |
|---|---|
| [What leaves our infrastructure?](#data) | The model requests and tool calls you configure; no usage telemetry and no Pydantic-hosted runtime |
| [Can we run on-prem or air-gapped models?](#self-hosted-models) | Yes: [Ollama](models/ollama.md), [vLLM](models/compatible-apis.md#vllm) or any OpenAI-compatible server you run |
| [Can we use our cloud's models?](#cloud) | Azure OpenAI and Microsoft Foundry, Amazon Bedrock and Google Vertex AI, selected by the model string |
| [How do we keep a person in control?](#control) | Tool approval, guardrails, and a sandbox for model-written code |
| [Can we audit what an agent did?](#audit) | Every run, model request and tool call as an OpenTelemetry trace |
| [How do we standardize across teams?](#governance) | Agent specs, shared capabilities and [Pydantic AI Harness](harness/index.md) |
| [Who do we call when it breaks?](#support) | Pydantic AI engineers, under a commercial agreement with an SLA |

## What leaves our infrastructure? {#data}

[Pydantic AI](index.md), [Pydantic AI Harness](harness/index.md), [Pydantic Evals](evals.md) and [Pydantic Graph](graph.md) are Python packages that run in your own process: there is no Pydantic-hosted agent runtime. The library sends no usage telemetry; it connects to the model providers, [MCP servers](mcp/client.md) and tools you configure.

- **Durable execution** runs on an engine you operate, such as [Temporal, DBOS or Prefect](durable_execution/overview.md), or on that engine's own cloud.
- **Traces** go where you send them. [Pydantic Logfire](logfire.md) is an optional hosted service and [any OpenTelemetry backend](logfire.md#otel-without-logfire) works instead. [`include_content=False`](logfire.md#excluding-prompts-and-completions) keeps prompts, completions and tool arguments out of the spans.
- **The [Pydantic AI Gateway](gateway.md)** is an optional hosted service: one key for models across providers, with spending caps. Calling providers directly works without it.
- **Model prices** for cost estimates ship with each release; [`update_in_background()`][pydantic_ai.prices.update_in_background] downloads newer ones only when you call it.

## Can we run on-prem or air-gapped models? {#self-hosted-models}

Yes. Point an agent at a model server inside your network and prompts and outputs stay there:

- [Ollama](models/ollama.md) on your own servers, with the `ollama:` prefix;
- [vLLM](models/compatible-apis.md#vllm), with the `vllm:` prefix;
- any other OpenAI-compatible server through [`OpenAICompatibleProvider`][pydantic_ai.providers.openai_compatible.OpenAICompatibleProvider], as [Other endpoints](models/compatible-apis.md#other-endpoints) explains, or any other API through a [custom model](models/overview.md#custom-models).

For an air-gapped deployment, install the packages from your own package mirror and point agents at a model server inside the network.

## Can we use our cloud's models? {#cloud}

Pydantic AI runs on [Azure OpenAI and Microsoft Foundry](models/azure.md), [Amazon Bedrock](models/bedrock.md) and [Google Vertex AI](models/google-cloud.md), authenticating with that cloud's own credentials. Switching an agent between them, or to OpenAI, Anthropic, Gemini or a local model, means changing one string, without rewriting the agent. [Models and Providers](models/overview.md) lists every supported service, and a [fallback model](models/overview.md#fallback-model) moves to another provider when one is down.

## How do we keep a person in control? {#control}

- **Approval**: mark a tool [`requires_approval=True`](deferred-tools.md#human-in-the-loop-tool-approval) and the run ends with the pending call; it resumes, days later if need be, once a person approves or denies it. [Going to Production](production.md#approval) shows the full flow, and [`ToolCallJudge`](harness/tool-call-judge.md) (Harness) puts a second model in front of a call instead.
- **Guardrails**: [guardrails](harness/guardrails.md) (Harness) check the prompt, each tool call and result, and the final output, and can block, redact secrets and PII, or ask for approval. The [prompt injection defender](harness/prompt-injection-defender.md) screens tool results.
- **Least privilege**: give each caller only the tools it may use, and run model-written code in a [sandbox](harness/index.md#capabilities); [How do I limit what the agent can touch?](production.md#safety) covers both.

## Can we audit what an agent did? {#audit}

Pydantic AI [instruments](logfire.md) every run, model request and tool call as an OpenTelemetry trace, following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), with token usage and estimated cost. Send the traces to [Logfire](logfire.md) or to the [OpenTelemetry backend](logfire.md#using-opentelemetry) you already run, and keep them as long as your retention policy says. The [message history](message-history.md) of each run is plain data you can store next to the trace.

## How do we standardize across teams? {#governance}

- **[Agent specs](agent-spec.md)** keep an agent's model, instructions and capabilities in a YAML or JSON file, so its configuration is reviewed and versioned apart from application code.
- **[Capabilities](capabilities/overview.md)** bundle tools, instructions, hooks and model settings into one reusable unit: a platform team writes the company's policies once and every team's agents add them with one line.
- **[Pydantic AI Harness](harness/index.md)** provides ready-made capabilities, such as [guardrails](harness/guardrails.md), [spend limits](harness/spend.md) per tenant or month, and [managed prompts](harness/managed-prompt.md) rolled out from Logfire without a redeploy.
- **The [Gateway](gateway.md)** sets spending caps per project, user and API key before a request reaches a provider.

## Who do we call when it breaks? {#support}

[Enterprise support](enterprise-support.md) gives your team direct access to Pydantic AI engineers for production incidents, architecture reviews, upgrades and coordinated vulnerability response, with 24/7 priority support and an SLA defined in your agreement. See [pydantic.dev/enterprise](https://pydantic.dev/enterprise) or [contact us](https://pydantic.dev/contact). Pydantic AI follows a [version policy](version-policy.md) for its releases, and the open-source license, docs, issue tracker and [community help](help.md) remain open to everyone.
