---
description: "Adopt Pydantic AI in an enterprise: agents run in your own process with no usage telemetry, on self-hosted models or Azure OpenAI, Amazon Bedrock and Google Vertex AI, durable on AWS Lambda, with human approval and OpenTelemetry traces of every run. Pydantic Logfire and the Pydantic AI Gateway add observability, evals, central model access and spending caps, in Pydantic's cloud or self-hosted, and Pydantic offers commercial support with an SLA."
---

# Pydantic AI for Enterprise

Pydantic AI is an open-source Python agent framework from Pydantic, the company behind the Pydantic validation library, with [commercial enterprise support from Pydantic](enterprise-support.md). Agents run in your own process against the models you choose, from [Azure OpenAI](models/azure.md), [Amazon Bedrock](models/bedrock.md) and [Google Vertex AI](models/google-cloud.md) to models you host yourself, and the library sends no usage telemetry.

Around the open-source framework, Pydantic offers optional commercial services: [Pydantic Logfire](https://pydantic.dev/logfire) for tracing, evals and cost tracking, the [Pydantic AI Gateway](gateway.md), part of Logfire, for one controlled entry point to the supported model providers, and [enterprise support](enterprise-support.md). Pydantic AI works without any of them, and support for the framework does not require Logfire. Logfire and the Gateway run in Pydantic's cloud or [in your own Kubernetes cluster](https://pydantic.dev/docs/logfire/deploy/self-hosted-deployment/overview/). [Pydantic Enterprise](https://pydantic.dev/enterprise) describes the plans.

This page answers the questions security, platform and procurement teams ask before adopting it, each with the short answer and a link to the page that covers it in full. [Going to Production](production.md) covers the engineering side: crash recovery, cost limits, testing and deployment.

| Your team wants to know... | Short answer |
|---|---|
| [What leaves our infrastructure?](#data) | The model requests, tool calls and trace exports you configure; no usage telemetry and no Pydantic-hosted runtime |
| [Can we run on-prem or air-gapped models?](#self-hosted-models) | Yes: [Ollama](models/ollama.md), [vLLM](models/compatible-apis.md#vllm) or any OpenAI-compatible server you run |
| [Can we run on our cloud?](#cloud) | Yes: models and embeddings on AWS, Azure and Google Cloud, AWS Lambda durable functions, and realtime voice on Azure and Google Cloud |
| [How do we control model access and spend?](#model-access) | The [Pydantic AI Gateway](gateway.md): one key per team or service, spending caps, provider failover, every request traced |
| [How do we keep a person in control?](#control) | Tool approval, guardrails, and a sandbox for model-written code |
| [Can we audit what an agent did?](#audit) | With instrumentation on, every run, model request and tool call as an OpenTelemetry trace, in [Logfire](logfire.md) or your own backend |
| [How do we know quality holds in production?](#evals) | [Pydantic Evals](evals.md) on datasets before a release and on live traffic after it, with results in Logfire |
| [Which compliance certifications apply?](#compliance) | The library runs in your process; for Logfire and the Gateway, see the [Pydantic Trust Center](https://pydantic.dev/security) |
| [How do we standardize across teams?](#governance) | Agent specs, shared capabilities, [Pydantic AI Harness](harness/index.md) and Gateway policies |
| [Who do we call when it breaks?](#support) | Pydantic AI engineers, under a commercial agreement with an SLA |

## What leaves our infrastructure? {#data}

[Pydantic AI](index.md), [Pydantic AI Harness](harness/index.md), [Pydantic Evals](evals.md) and [Pydantic Graph](graph.md) are Python packages that run in your own process: there is no Pydantic-hosted agent runtime. The library sends no usage telemetry; it connects to the model providers, [MCP servers](mcp/client.md) and tools you configure.

- **Durable execution** runs on an engine you operate, such as [Temporal, DBOS or Prefect](durable_execution/overview.md), or on that engine's own cloud.
- **Traces** go where you send them. [Pydantic Logfire](logfire.md) is optional, and runs in Pydantic's cloud in a US or EU region, as a single-tenant [Enterprise Dedicated](https://pydantic.dev/enterprise) deployment in any GCP region with customer-managed encryption keys, or [self-hosted](https://pydantic.dev/docs/logfire/deploy/self-hosted-deployment/overview/) on your own Kubernetes cluster so telemetry stays in your infrastructure; [any OpenTelemetry backend](logfire.md#otel-without-logfire) works instead. Logfire's SDK scrubs likely-sensitive values such as tokens and passwords, but not prompt and response content; [`include_content=False`](logfire.md#excluding-prompts-and-completions) keeps prompts, completions and tool arguments out of the spans.
- **The [Pydantic AI Gateway](gateway.md)** is optional: model requests pass through it to the provider in the provider's own format. It runs on Logfire Cloud in the same US or EU region as your Logfire organization, or on self-hosted Logfire. Calling providers directly works without it.
- **Model prices** for cost estimates ship with each release; [`update_in_background()`][pydantic_ai.prices.update_in_background] downloads newer ones only when you call it.

## Can we run on-prem or air-gapped models? {#self-hosted-models}

Yes. Point an agent at a model server inside your network and model requests stay there; for a whole run to stay inside, its tools, [MCP servers](mcp/client.md) and trace exporter must be inside too:

- [Ollama](models/ollama.md) on your own servers, with the `ollama:` prefix;
- [vLLM](models/compatible-apis.md#vllm), with the `vllm:` prefix;
- any other OpenAI-compatible server through [`OpenAICompatibleProvider`][pydantic_ai.providers.openai_compatible.OpenAICompatibleProvider], as [Other endpoints](models/compatible-apis.md#other-endpoints) explains, or any other API through a [custom model](models/overview.md#custom-models).

For an air-gapped deployment, install the packages from your own package mirror and point agents at a model server inside the network. A few optional integrations download files on first use, such as `tiktoken` encodings and [Sentence Transformers](embeddings.md#sentence-transformers-local) models; mirror those too.

## Can we run on our cloud? {#cloud}

Yes. Pydantic AI supports models on AWS, Microsoft Azure and Google Cloud, authenticating with each cloud's own credentials, along with the embeddings, durable execution and realtime voice integrations below:

| Capability | AWS | Microsoft Azure | Google Cloud |
|---|---|---|---|
| Models | [Amazon Bedrock](models/bedrock.md), including [Bedrock Mantle](models/bedrock.md#bedrock-mantle) | [Azure OpenAI and Microsoft Foundry](models/azure.md), including [Claude on Foundry](models/azure.md#claude-on-microsoft-foundry) | [Gemini and Model Garden](models/google-cloud.md), including [Claude on Google Cloud](models/google-cloud.md#claude-on-google-cloud) |
| Embeddings | [Bedrock](embeddings.md#bedrock) | [Azure OpenAI](embeddings.md#openai-compatible) | [Google Cloud](embeddings.md#google-cloud) |
| Durable execution | [AWS Lambda durable functions](harness/aws-lambda.md) | No native integration; use any [portable durable execution engine](durable_execution/overview.md) on infrastructure you run or the engine's own cloud | No native integration; use any [portable durable execution engine](durable_execution/overview.md) on infrastructure you run or the engine's own cloud |
| Realtime voice | No native AWS voice integration; host a voice agent on AWS compute with [another supported provider](realtime/overview.md#provider-support) | [Azure OpenAI and Azure AI Voice Live](realtime/azure.md) | [Gemini Live](realtime/gemini.md) |

<!-- TODO(#11332): once Azure Durable Task durability merges, replace the Azure "Durable execution" cell with a link to its Harness page. -->

The [AWS Lambda durable functions integration](harness/aws-lambda.md) is co-maintained by the Pydantic and AWS teams, [announced by AWS](https://aws.amazon.com/about-aws/whats-new/2026/09/aws-lambda-durable-pydantic-ai/) and documented in the [AWS Durable Execution SDK guide](https://docs.aws.amazon.com/durable-execution/sdk-reference/integrations/pydantic-ai/).

Each cloud's enterprise features work too:

- **AWS**: Bedrock [guardrails](models/bedrock.md#customizing-bedrock-runtime-api), [prompt caching](models/bedrock.md#prompt-caching), [application inference profiles](models/bedrock.md#using-aws-application-inference-profiles) and [service tiers](models/bedrock.md#service-tier), plus [LocalStack](harness/localstack.md) for agents that work against an emulated AWS environment.
- **Azure**: the [Chat Completions](models/azure.md#configuration) and [Responses](models/azure.md#using-azure-with-the-responses-api) APIs, and Microsoft Entra ID authentication for [Claude on Foundry](models/anthropic.md#microsoft-foundry) and [realtime voice](realtime/azure.md#browser-webrtc-and-microsoft-entra-id).
- **Google Cloud**: [Application Default Credentials](models/google-cloud.md#application-default-credentials), [service accounts](models/google-cloud.md#service-account), [Model Armor](models/google.md#model-armor-google-cloud-only), [service tiers and provisioned throughput](models/google.md#service-tier-service_tier-google_cloud_service_tier), [Gemini image generation](image-generation.md#google), [Google Workspace tools](harness/google-workspace.md), and [hosting voice agents on Cloud Run](realtime/deployment.md#hosting).

Once a provider's credentials are configured, changing the model string moves an agent between clouds, or to OpenAI, Anthropic, Gemini or a local model, without rewriting it, for the features both providers support; deployment and durable execution are configured separately. [Models and Providers](models/overview.md) lists every supported service, and a [fallback model](models/overview.md#fallback-model) moves to another provider when one is down. Weighing your cloud's own agent framework? See [Pydantic AI vs Google ADK](comparisons/vs-google-adk.md).

<!-- TODO(#12021): add "Pydantic AI vs Microsoft Agent Framework" (comparisons/vs-microsoft-agent-framework.md) to the sentence above once it merges. -->
<!-- TODO(#12026): add "Pydantic AI vs Strands Agents" (comparisons/vs-strands.md) to the sentence above once it merges. -->

## How do we control model access and spend? {#model-access}

Without extra infrastructure, each provider reads its key from an environment variable or your secret manager, and [`SpendLimits`](harness/spend.md) (Harness) refuses new requests once a per-tenant, daily or monthly budget is spent; its page explains when spend can overshoot and how to share budgets across processes. When many teams call models, the [Pydantic AI Gateway](gateway.md) moves that control to one place:

- **One key for many providers**: an application holds one Gateway key for OpenAI, Anthropic, Google Vertex AI, Groq and Amazon Bedrock models, and selects a model with a `gateway/` [model string](gateway.md#usage). Use your own provider keys ([BYOK](https://pydantic.dev/docs/logfire/manage/ai-gateway/byok-providers/)) at no markup, or pay for inference through Pydantic at the markup on the [pricing page](https://pydantic.dev/pricing).
- **Spending caps**: daily, weekly and monthly [limits per project, user and API key](gateway.md#key-features), checked before a request is forwarded to the provider, so you can issue one key per service or environment.
- **Failover and load balancing**: [gateway endpoints](gateway.md#gateway-endpoints) route a model across providers by priority and weight, and fall back to the next provider on retryable errors.
- **A record of every request**: with [Gateway telemetry](https://pydantic.dev/docs/logfire/manage/ai-gateway/telemetry/) on, every request becomes a span in Logfire with the model, token counts, cost and timing, nested in your agent's trace, since Pydantic AI's `gateway/` provider sends the trace headers. Prompts and responses are recorded only when the project's **Include messages** setting is on.
- **Data protection**: [Gateway guardrails](https://pydantic.dev/docs/logfire/manage/ai-gateway/protect-data/) (in Beta) detect secrets and personal data in requests and can flag, redact or block them before they are forwarded.
- **Access**: the Gateway uses Logfire's organizations and roles: admins manage providers, endpoints and spending, and members create their own keys. On the Enterprise plan that includes SSO with your identity provider.

## How do we keep a person in control? {#control}

- **Approval**: mark a tool [`requires_approval=True`](deferred-tools.md#human-in-the-loop-tool-approval) and the run ends with the pending call; your application stores the conversation, and the run resumes, days later if need be, once a person approves or denies the call. [Going to Production](production.md#approval) shows the full flow, and [`ToolCallJudge`](harness/tool-call-judge.md) (Harness) replaces the person with an automated check by a second model.
- **Guardrails**: [guardrails](harness/guardrails.md) (Harness) check the prompt, each tool call and result, and the final output, and can block, redact secrets and PII, or ask for approval. The [prompt injection defender](harness/prompt-injection-defender.md) screens tool results. The Gateway's [guardrails](https://pydantic.dev/docs/logfire/manage/ai-gateway/protect-data/) apply a second check on the traffic path, for every application that uses the Gateway.
- **Least privilege**: give each caller only the tools it may use, and run model-written code in a [sandbox](harness/index.md#execution-environments); [How do I limit what the agent can touch?](production.md#safety) covers both.

## Can we audit what an agent did? {#audit}

With [instrumentation](logfire.md#using-logfire) turned on, Pydantic AI records every run, model request and tool call as an OpenTelemetry trace, following the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), with token usage and estimated cost. Send the traces to the [OpenTelemetry backend](logfire.md#using-opentelemetry) you already run, or to [Logfire](logfire.md), which shows agent runs next to the HTTP requests, database queries and background jobs around them, and lets you query every span with SQL, for example to total token usage and cost per agent or model. Retention is set in the backend you choose; Logfire's Enterprise plan has custom retention. Traces record what agents did; changes to keys, roles and policies are recorded by the systems that manage them. The [message history](message-history.md) of each run is plain data you can store next to the trace.

## How do we know quality holds in production? {#evals}

[Pydantic Evals](evals.md) runs an agent against a dataset of cases with scoring rules you write in code, so a prompt or model change can be checked before release, including in CI. [Online evaluation](evals/online-evaluation.md) scores live traffic, every call or a sample, in the background and records the scores as OpenTelemetry events. With [Logfire](evals/how-to/logfire-integration.md), experiment results and online scores sit next to the traces they came from, and you can compare runs to see what changed. Pydantic Evals is open source and works without Logfire.

## Which compliance certifications apply? {#compliance}

Pydantic AI, Harness, Evals and Graph run in your environment, under the security controls you apply to the rest of your application; the library holds no certifications of its own. For Logfire and the Gateway, the [Pydantic Trust Center](https://pydantic.dev/security) has the certifications and reports, and [Logfire for platform and security teams](https://pydantic.dev/docs/logfire/get-started/for-platform-and-security/) covers regions, scrubbing and self-hosting.

## How do we standardize across teams? {#governance}

- **[Agent specs](agent-spec.md)** keep an agent's model, instructions and capabilities in a YAML or JSON file, so its configuration is reviewed and versioned apart from application code.
- **[Capabilities](capabilities/overview.md)** bundle tools, instructions, hooks and model settings into one reusable unit: a platform team writes the company's policies once and every team's agents can reuse them.
- **[Pydantic AI Harness](harness/index.md)** provides ready-made capabilities, such as [guardrails](harness/guardrails.md), [spend limits](harness/spend.md) per tenant or month, and [managed prompts](harness/managed-prompt.md), which are versioned in Logfire and rolled out without a redeploy.
- **The [Gateway](gateway.md)** gives every application that routes through it the same model access, spending caps per project, user and API key, and failover rules, set once by an admin instead of in each application.
- **[Logfire](https://pydantic.dev/logfire)** gives every team's agents one place for traces, costs and evals; on the Enterprise plan with SSO, SCIM group provisioning and custom roles.

## Who do we call when it breaks? {#support}

[Enterprise support](enterprise-support.md) gives your team direct access to Pydantic AI engineers, with an SLA defined in your agreement; that page lists what it covers. It is its own agreement, separate from Logfire support: it does not require a Logfire or Gateway subscription, and a Logfire plan, at any tier, does not include it. [Logfire's Enterprise plans](https://pydantic.dev/enterprise) include their own support for Logfire and the Gateway. Pydantic AI follows a [version policy](version-policy.md), vulnerabilities are reported through its [security policy](https://github.com/pydantic/pydantic-ai/security/policy), and the docs, issue tracker and [community help](help.md) stay open to everyone. Open-source projects that run on it are listed on [Built with Pydantic AI](built-with.md).
