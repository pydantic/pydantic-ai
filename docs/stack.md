---
description: "What each piece of the Pydantic stack is, how you install it or sign up for it, and which pieces a Pydantic AI app needs: packages, extras and hosted products."
---

# The Pydantic Stack

Pydantic AI is a set of open source Python packages you install from PyPI and run in your own process; the only piece an agent needs is `pydantic-ai` (or `pydantic-ai-slim`). Two products are hosted, both optional: [Pydantic Logfire](https://pydantic.dev/logfire) for observability (with a free plan) and the [Pydantic AI Gateway](gateway.md) for model access, which you can also self-host.

We keep the pieces in separate packages on purpose: you install what you use. `pydantic-ai` is the batteries-included install and `pydantic-ai-slim` is the same framework with only the extras you name. [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) (`pip install pydantic-ai-harness`) adds the capabilities long-running agents need without growing the core, and [Pydantic Graph](graph.md) (`pip install pydantic-graph`) works without an agent at all. The framework packages (`pydantic-ai`, `pydantic-ai-slim`, `pydantic-graph` and `pydantic-evals`) are released together under one version number, while [Pydantic](https://pydantic.dev/docs/validation/latest/), the [Logfire SDK](https://github.com/pydantic/logfire) and [genai-prices](https://github.com/pydantic/genai-prices) (all three installed with `pydantic-ai`) and [Monty](https://github.com/pydantic/monty) (`pip install pydantic-monty`) each ship on their own schedule.

| Piece | What it is | How you get it | Required? | Docs |
|---|---|---|---|---|
| Pydantic AI | The Python AI SDK: a typed, extensible agent loop with every model a string swap away | `pip install pydantic-ai` | Core (this or `pydantic-ai-slim`) | [Installation](install.md) |
| `pydantic-ai-slim` and its extras | The same framework without optional dependencies; you name the model providers and integrations you use as extras, such as `openai`, `bedrock` or `temporal` | `pip install "pydantic-ai-slim[openai]"` | Core (this or `pydantic-ai`) | [Slim install](install.md#slim-install) |
| Pydantic AI Harness | The official capability library and harness: memory, guardrails, sub-agents, planning, context management, up to a complete coding agent. A Python library that runs in your process, not a hosted service | `pip install pydantic-ai-harness` | Optional package | [Harness](https://pydantic.dev/docs/ai/harness/) |
| Pydantic Graph | Typed graph control flow | Comes with `pydantic-ai` and `pydantic-ai-slim`; on its own, `pip install pydantic-graph` | Comes with core; using it is optional | [Graphs](graph.md) |
| Pydantic Evals | Evaluate any Python function, agents included | Comes with `pydantic-ai`; with slim, the `evals` extra; on its own, `pip install pydantic-evals` | Optional package | [Evals](evals.md) |
| Pydantic Logfire | AI-first, full-stack observability that sees your whole app, not just the LLM calls | Hosted, with a free Personal plan ([pricing](https://pydantic.dev/pricing)); self-hosted on the Enterprise plan. The `logfire` SDK comes with `pydantic-ai`, or with the slim `logfire` extra | Optional, hosted; any OpenTelemetry backend works instead | [Logfire](logfire.md) |
| Pydantic AI Gateway | One key for every model, with real-time cost monitoring and budget control | Hosted, as part of Logfire (sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev/)); self-hosting [is available](https://pydantic.dev/ai-gateway). No package: use the `gateway/` model prefix | Optional, hosted | [Gateway](gateway.md) |
| Pydantic | The validation layer underneath all of it | Comes with `pydantic-ai` and `pydantic-ai-slim` | Comes with core | [Pydantic Validation](https://pydantic.dev/docs/validation/latest/) |
| genai-prices | Model pricing data, kept current; it prices your runs and fills in context window sizes | Comes with `pydantic-ai` and `pydantic-ai-slim` | Comes with core | [GitHub](https://github.com/pydantic/genai-prices) |
| Monty | A sandboxed Python interpreter for model-written code | `pip install pydantic-monty`; the Harness `code-mode` and `dynamic-workflow` extras pull it in | Only for Harness [Code Mode](https://pydantic.dev/docs/ai/harness/code-mode/) and [dynamic workflows](https://pydantic.dev/docs/ai/harness/dynamic-workflow/) | [GitHub](https://github.com/pydantic/monty) |

Every package in the table is MIT licensed. What you can pay for is Logfire and the Gateway, and neither is needed to build, run, test or deploy an agent.

## What `pydantic-ai` pulls in

`pip install pydantic-ai` installs `pydantic-ai-slim` with the libraries for the OpenAI, Anthropic and Google models, plus the [CLI](cli.md), [MCP](mcp/client.md), [Evals](evals.md), [Web UI](ui/overview.md) and [Logfire](logfire.md) integrations. That is the right install for trying things out and for most applications.

If you ship Pydantic AI inside something else, like a library of your own, a small container image or a tool other people install, depend on `pydantic-ai-slim` with the extras you use: the same code at the same version, and nothing you did not ask for. Pydantic AI Harness and Pydantic Evals do exactly this; both depend on `pydantic-ai-slim` and leave the providers to you. The [installation guide](install.md#slim-install) lists every extra.
