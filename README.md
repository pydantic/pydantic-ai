<div align="center">
  <a href="https://ai.pydantic.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://pydantic.dev/docs/ai/img/pydantic-ai-dark.svg">
      <img src="https://pydantic.dev/docs/ai/img/pydantic-ai-light.svg" alt="Pydantic AI">
    </picture>
  </a>
</div>
<div align="center">
  <h3>How Python does AI</h3>
</div>
<div align="center">
  <a href="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain"><img src="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push" alt="CI"></a>
  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai"><img src="https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg" alt="Coverage"></a>
  <a href="https://pypi.python.org/pypi/pydantic-ai"><img src="https://img.shields.io/pypi/v/pydantic-ai.svg" alt="PyPI"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/pypi/pyversions/pydantic-ai.svg" alt="versions"></a>
  <a href="https://github.com/pydantic/pydantic-ai/blob/main/LICENSE"><img src="https://img.shields.io/github/license/pydantic/pydantic-ai.svg?v" alt="license"></a>
  <a href="https://logfire.pydantic.dev/docs/join-slack/"><img src="https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack" alt="Join Slack" /></a>
</div>

---

**Documentation**: [ai.pydantic.dev](https://ai.pydantic.dev/)

---

**Pydantic AI** is the Python agent framework: a typed agent loop, [every model](https://ai.pydantic.dev/models/overview) behind one API, and [every interface](https://ai.pydantic.dev/ui/overview) — terminal, web, your own frontend, even [voice](https://ai.pydantic.dev/realtime). **[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness)** is its official capability library and harness: 50+ capabilities across core and Harness that you snap onto an agent — from [web search](https://ai.pydantic.dev/capabilities/web-search/) to a complete coding agent.

Whatever you came to build — a one-off LLM call in a script, an AI feature inside your product, a [realtime voice agent](https://ai.pydantic.dev/realtime), [embeddings](https://ai.pydantic.dev/embeddings), [image generation](https://ai.pydantic.dev/capabilities/image-generation/), or your own Claude-Code-style coding agent — this is the right place, and it's the same library at every step.

The simplest agent is three lines:

```python
from pydantic_ai import Agent

agent = Agent('anthropic:claude-fable-5', instructions='Be concise.')
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
```

A full coding agent in your terminal is five:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])
agent.to_cli_sync()
```

Everything between those two is composition, not rewriting: add [tools](https://ai.pydantic.dev/tools), typed [structured outputs](https://ai.pydantic.dev/output), and [capabilities](https://ai.pydantic.dev/capabilities/overview/) one at a time, and run the result anywhere — headless, [as a web app](https://ai.pydantic.dev/web), [inside your UI](https://ai.pydantic.dev/ui/overview), or as a [realtime voice agent](https://ai.pydantic.dev/realtime). No framework bloat: it drops into a single function as easily as it powers a product.

## Why Pydantic AI

- **Any model, one API.** [Virtually every model and provider](https://ai.pydantic.dev/models/overview) — OpenAI, Anthropic, Google, Bedrock, Azure AI Foundry, Groq, Mistral, xAI, Ollama, and dozens more — swappable with a string. No flagship feature is locked to one vendor.

- **Typed end to end.** [Structured outputs](https://ai.pydantic.dev/output), typed [dependency injection](https://ai.pydantic.dev/dependencies), typed tools: your IDE, type checker, and AI coding agent all know what your agent returns, moving whole classes of errors from runtime to write-time.

- **Batteries, composably.** One primitive — the [capability](https://ai.pydantic.dev/capabilities/overview/) — bundles tools, instructions, hooks, and model settings into reusable units. Core ships the fundamentals ([web search](https://ai.pydantic.dev/capabilities/web-search/), [thinking](https://ai.pydantic.dev/capabilities/thinking/), [MCP](https://ai.pydantic.dev/capabilities/mcp/), [tool search](https://ai.pydantic.dev/capabilities/tool-search/)); the [Harness](https://github.com/pydantic/pydantic-ai-harness) ships everything else — code execution, memory, sub-agents, guardrails, compaction — plus complete presets like `Coder`. Build up from blocks or take a preset apart; it's the same primitive either way. Or skip code entirely with [YAML/JSON agent specs](https://ai.pydantic.dev/agent-spec).

- **Every interface.** One agent definition runs as a CLI, a [built-in web chat](https://ai.pydantic.dev/web), [your own frontend](https://ai.pydantic.dev/ui/overview) (AG-UI and Vercel AI protocols), an editor agent, or a [voice agent](https://ai.pydantic.dev/realtime) — realtime is just another frontend, on OpenAI Realtime, Gemini Live, Azure, and xAI Grok Voice.

- **Measured, not vibes.** OpenTelemetry-native [instrumentation](https://ai.pydantic.dev/logfire) works with any OTel backend — one line lights up [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, tracing, and cost tracking — and [Pydantic Evals](https://ai.pydantic.dev/evals) tests agent behavior the way pytest tests code.

- **Durable by choice.** First-party, co-maintained [durable execution](https://ai.pydantic.dev/durable_execution/overview/) on Temporal, DBOS, or Prefect — your agents survive restarts and run for days, on the engine you already operate, with [human-in-the-loop approval](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval) built in.

Built by the [Pydantic](https://docs.pydantic.dev) team: Pydantic Validation is the validation layer of the OpenAI SDK, the Anthropic SDK, the Google ADK, LangChain, and most of the AI ecosystem — and the foundation FastAPI was built on. Pydantic AI brings that same feeling to agents.

## Show me real code

A typed agent with a tool and a guaranteed-structured result:

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

class Triage(BaseModel):
    severity: int
    summary: str

agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=str,  # dependency-inject anything: DB pools, API clients, user info
    output_type=Triage,  # the run returns a validated Triage, typed as such
    instructions='Triage the incident report.',
)

@agent.tool
async def get_runbook(ctx: RunContext[str], service: str) -> str:
    """Fetch the runbook for a service."""
    return f'Runbook for {service} (requested by {ctx.deps})'

result = agent.run_sync('Payments API is returning 500s', deps='oncall@example.com')
print(result.output)
#> severity=2 summary='Payments API outage: 500s on all requests'
```

For the full version with a real database, dynamic instructions, and Logfire tracing, see the [bank support example](https://ai.pydantic.dev/#tools-dependency-injection-example) in the docs.

## Next Steps

- [Install Pydantic AI](https://ai.pydantic.dev/install) and follow the [examples](https://ai.pydantic.dev/examples/setup) — no API key needed to start (there's a built-in [`'test'` model](https://ai.pydantic.dev/testing#unit-testing-with-testmodel)).
- Read the [docs](https://ai.pydantic.dev/agents/) and the [API reference](https://ai.pydantic.dev/api/agent/).
- Give your agent its batteries: [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness).
- Join [Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Part of the Pydantic Stack

Everything you need to ship production-grade AI agents:

- [Pydantic AI](https://pydantic.dev/pydantic-ai?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) — the type-safe agent framework
- [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) — the official capability library, from single batteries to complete harnesses
- [Pydantic Logfire](https://pydantic.dev/logfire?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) — AI-first, full-stack observability
- [Logfire AI Gateway](https://pydantic.dev/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) — unified LLM proxy
- [Pydantic Evals](https://ai.pydantic.dev/evals) — evaluate any Python function, agents included
- [Pydantic Graph](https://ai.pydantic.dev/graph) — typed graph control flow
- [genai-prices](https://github.com/pydantic/genai-prices) — model pricing data, kept current
