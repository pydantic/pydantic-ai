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

**Pydantic AI** is the Python AI SDK: a typed, [instrumented](https://ai.pydantic.dev/logfire) agent loop, [every model](https://ai.pydantic.dev/models/overview) behind one Python API, and [every interface](https://ai.pydantic.dev/interfaces) — [terminal](https://ai.pydantic.dev/cli), [web](https://ai.pydantic.dev/web), [your own frontend](https://ai.pydantic.dev/ui/overview), even [voice](https://ai.pydantic.dev/realtime). It does [embeddings](https://ai.pydantic.dev/embeddings) and [image generation](https://ai.pydantic.dev/capabilities/image-generation/) too. **[Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness)** is its official capability library and harness: everything an agent needs for complex, long-running work, [capabilities](https://ai.pydantic.dev/capabilities/overview/) you snap on — from [web search](https://ai.pydantic.dev/capabilities/web-search/) and [MCP](https://ai.pydantic.dev/capabilities/mcp/) to a complete [coding agent](https://pydantic.dev/docs/ai/harness/coder/).

Whatever you came to build — a one-off LLM call extracting typed data in a script, an AI feature inside your product, a [realtime voice agent](https://ai.pydantic.dev/realtime) you can talk to, or your very own [coding agent](https://pydantic.dev/docs/ai/harness/coder/) in your terminal — you've come to the right place.

The simplest agent is three lines:

```python
from pydantic_ai import Agent

agent = Agent('anthropic:claude-fable-5', instructions='Be concise.')
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

A full coding agent in your terminal is five:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Coder

agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])
agent.to_cli_sync()
```

Run that file and you're chatting with it in your terminal — or skip the file entirely and run the exported [`coder_agent`](https://pydantic.dev/docs/ai/harness/coder/) with [`clai`](https://ai.pydantic.dev/cli#custom-agents) (the Pydantic AI CLI), via [`uvx`](https://docs.astral.sh/uv/guides/tools/):

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
```

## What are you building?

A coding agent is the five lines above — and Pydantic AI and [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) have the rest covered too:

### Structured output & tools

```python
from typing import Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent


class Sentiment(BaseModel):
    label: Literal['positive', 'negative', 'neutral']
    score: float = Field(ge=-1, le=1)


agent = Agent('openai:gpt-5.6-sol', output_type=Sentiment)


@agent.tool_plain
def recent_reviews(product: str) -> list[str]:
    """Fetch recent review snippets for a product."""
    return ['The new release fixed everything I complained about!']


result = agent.run_sync('How are people feeling about the Extract app?')
print(result.output)
#> label='positive' score=0.9
```

The [`@agent.tool`](https://ai.pydantic.dev/tools) function's signature and docstring become the tool schema, its arguments are validated before your code runs, and the run is guaranteed to return a `Sentiment` — your IDE, type checker, and the LLM all agree on the shape. **Build this →** [Agents](https://ai.pydantic.dev/agents/), [Function Tools](https://ai.pydantic.dev/tools), and [Structured Output](https://ai.pydantic.dev/output)

### Realtime voice

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent(
    instructions='You are a helpful voice assistant.',
    capabilities=[MCP('https://internal.example.com/mcp')],  # capabilities work in voice too
)

@agent.tool_plain
def order_status(order_id: str) -> str:
    """Look up the status of an order."""
    return f'Order {order_id}: shipped, arriving Thursday.'

async with agent.realtime('openai:gpt-realtime-2.1').session() as session:
    microphone = asyncio.create_task(stream_microphone(session))  # chunks → session.send_audio()
    speaker = asyncio.create_task(play_audio(session.stream_audio()))  # model audio → your speaker
    async for part in session.stream_transcripts():
        print(f'{part.speaker}: {part.transcript}')
```

The model calls your [tools](https://ai.pydantic.dev/realtime/tools) mid-conversation while it keeps talking, [capabilities](https://ai.pydantic.dev/realtime/capabilities) attach the same way as in any run, and every session is [instrumented](https://ai.pydantic.dev/logfire) — voice is just another frontend, on OpenAI Realtime, Gemini Live, Azure, and xAI Grok Voice. **Build this →** [Realtime Voice](https://ai.pydantic.dev/realtime)

### Image generation

```python
from pathlib import Path

from pydantic_ai import Agent, BinaryImage

agent = Agent('openai:gpt-5.6-sol', output_type=BinaryImage)
result = agent.run_sync('Generate a minimalist logo for a coffee shop called Extract.')
Path('logo.png').write_bytes(result.output.data)
```

Ask for an image and the run's typed output *is* the image: [provider-native generation](https://ai.pydantic.dev/native-tools#image-generation-tool) where the model supports it, a subagent fallback everywhere else, and a [standalone image API](https://github.com/pydantic/pydantic-ai/pull/5357) on the way. **Build this →** [Image Generation](https://ai.pydantic.dev/capabilities/image-generation/)

### Embeddings

```python
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')
result = embedder.embed_query_sync('What is machine learning?')
print(len(result.embeddings[0]))
#> 1536
```

Embedding your documents for semantic search or a [RAG pipeline](https://ai.pydantic.dev/examples/rag)? Seven providers behind one typed API, [instrumented](https://ai.pydantic.dev/logfire) like everything else — and it lives next to the agent that will use the results. **Build this →** [Embeddings](https://ai.pydantic.dev/embeddings)

## Why Pydantic AI

- **Any model, one Python API.** [Virtually every model and provider](https://ai.pydantic.dev/models/overview) — OpenAI, Anthropic, Google, Bedrock, Azure AI Foundry, Groq, Mistral, xAI, Ollama, and dozens more — swappable with a string. No flagship feature is locked to one vendor.

- **Typed end to end.** [Structured outputs](https://ai.pydantic.dev/output), typed [dependency injection](https://ai.pydantic.dev/dependencies), typed tools: your IDE, type checker, and AI coding agent all know what your agent returns, moving whole classes of errors from runtime to write-time.

- **Batteries, composably.** One primitive — the [capability](https://ai.pydantic.dev/capabilities/overview/) — bundles [tools](https://ai.pydantic.dev/tools), [instructions](https://ai.pydantic.dev/agents/#instructions), [hooks](https://ai.pydantic.dev/hooks), and [model settings](https://ai.pydantic.dev/agents/#model-run-settings) into reusable units. Core ships the fundamentals; the [Harness](https://github.com/pydantic/pydantic-ai-harness) ships everything else — code execution, memory, sub-agents, guardrails, compaction — plus complete harnesses like [Coder](https://pydantic.dev/docs/ai/harness/coder/) and [Researcher](https://pydantic.dev/docs/ai/harness/researcher/) that are themselves just capabilities composed, so you can take them apart. Or skip code entirely with [YAML/JSON agent specs](https://ai.pydantic.dev/agent-spec).

- **[Every interface](https://ai.pydantic.dev/interfaces).** One agent definition runs as a [CLI](https://ai.pydantic.dev/cli), a [built-in web chat](https://ai.pydantic.dev/web), [your own frontend](https://ai.pydantic.dev/ui/overview) (AG-UI and Vercel AI protocols), an [editor agent](https://pydantic.dev/docs/ai/harness/acp/) *(experimental)*, or a [voice agent](https://ai.pydantic.dev/realtime).

- **Measured, not vibes.** OpenTelemetry-native [instrumentation](https://ai.pydantic.dev/logfire) works with any OTel backend — one line lights up [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, tracing, and cost tracking — and [Pydantic Evals](https://ai.pydantic.dev/evals) tests agent behavior the way pytest tests code.

- **Durable by choice.** First-party, co-maintained [durable execution](https://ai.pydantic.dev/durable_execution/overview/) on Temporal, DBOS, or Prefect — your agents survive restarts and run for days, on the engine you already operate, with [human-in-the-loop approval](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval) built in.

Built by the [Pydantic](https://docs.pydantic.dev) team: Pydantic Validation is the validation layer of the OpenAI SDK, the Anthropic SDK, the Google ADK, LangChain, and most of the AI ecosystem — and the foundation FastAPI was built on. Pydantic AI brings that same feeling to agents.

## Tools & dependency injection

A typed support agent for a bank, with the customer-facing behavior packaged as a reusable [capability](https://ai.pydantic.dev/capabilities/overview/):

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # inject any client: DB pools, HTTP APIs, user info
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


customer_context = Capability[SupportDependencies](  # a reusable unit of tools + instructions
    id='customer-context',
    description="Who the customer is and what's on their account.",
)


@customer_context.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@customer_context.tool  # signature and docstring become the tool schema the LLM sees
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


refunds = Capability[SupportDependencies](  # deferred: loads on demand, like a skill
    id='refunds',
    description='Refund eligibility and refund status.',
    defer_loading=True,
)


@refunds.tool
async def refund_status(ctx: RunContext[SupportDependencies]) -> str:
    """Look up the refund status for the customer's most recent charge."""
    return await ctx.deps.db.refund_status(id=ctx.deps.customer_id)


support_agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=SupportDependencies,
    output_type=SupportOutput,  # the run returns a validated SupportOutput, typed as such
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
    capabilities=[customer_context, refunds],
)


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)
    print(result.output)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """

    result = await support_agent.run(  # the model loads `refunds` on demand, then answers
        'Was I refunded for the duplicate charge on my last statement?', deps=deps
    )
    print(result.output)
    """
    support_advice='Good news, John: the duplicate charge on your last statement was refunded on 2026-05-01.' block_card=False risk=1
    """
```

For the annotated walkthrough and Logfire tracing, see the [same example in the docs](https://ai.pydantic.dev/#tools-dependency-injection-example).

## Next Steps

- [Install Pydantic AI](https://ai.pydantic.dev/install) and follow the [examples](https://ai.pydantic.dev/examples/setup) — no API key needed to start (there's a built-in [`'test'` model](https://ai.pydantic.dev/testing#unit-testing-with-testmodel)).
- Read the [docs](https://ai.pydantic.dev/agents/) and the [API reference](https://ai.pydantic.dev/api/agent/).
- Give your agent its batteries: [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness).
- Join [Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues).

## Part of the Pydantic Stack

Everything you need to ship production-grade AI agents:

- [Pydantic AI](https://pydantic.dev/pydantic-ai?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) — the type-safe AI SDK
- [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness) — the official capability library and harness, from single capabilities to complete agents
- [Pydantic Logfire](https://pydantic.dev/logfire?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) — AI-first, full-stack observability
- [Logfire AI Gateway](https://pydantic.dev/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) — unified LLM proxy
- [Pydantic Evals](https://ai.pydantic.dev/evals) — evaluate any Python function, agents included
- [Pydantic Graph](https://ai.pydantic.dev/graph) — typed graph control flow
- [genai-prices](https://github.com/pydantic/genai-prices) — model pricing data, kept current
