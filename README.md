<div align="center">
  <a href="https://ai.pydantic.dev/">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://pydantic.dev/docs/ai/img/pydantic-ai-dark.svg">
      <img src="https://pydantic.dev/docs/ai/img/pydantic-ai-light.svg" alt="Pydantic AI">
    </picture>
  </a>
</div>
<div align="center">
  <h3>The agent harness for Python — typed, composable, and measured</h3>
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

### <em>Pydantic AI is the Python agent harness: the framework around the model that determines whether your agent actually gets the job done.</em>

An agent is only as good as the loop around its model: how context is assembled, how tool results and failures flow back, what gets retried, when the run is allowed to stop, and what it all costs. As models have gotten stronger, this layer — the harness — has replaced the model as the bottleneck: the same model in two different harnesses can differ more than two model generations in the same harness. Pydantic AI makes the harness layer explicit and typed. Agents are built from composable [capabilities](https://ai.pydantic.dev/capabilities) — [web search](https://ai.pydantic.dev/capabilities#provider-adaptive-tools), [thinking](https://ai.pydantic.dev/capabilities#thinking), or ones you write yourself — the way web apps are built from middleware, with heavier batteries like sandboxed code execution available in the [Pydantic AI Harness](https://ai.pydantic.dev/harness/overview) library.

A harness you can't observe is just a promise. Every Pydantic AI run is instrumented end to end with OpenTelemetry — every model request, tool call, retry, and token — viewable in [Pydantic Logfire](https://pydantic.dev/logfire) or any OTel backend, and testable with [Pydantic Evals](https://ai.pydantic.dev/evals). You don't have to trust that your agent behaves: you can watch it, measure it, and pin it down in a test.

Underneath is the type-safe agent framework built by the Pydantic team — [Pydantic Validation](https://docs.pydantic.dev) is the validation layer inside virtually every AI SDK. FastAPI brought that foundation and ergonomics to web development; Pydantic AI was built with one simple aim: to bring that same feeling to agents.

## Why use Pydantic AI

1. **The Harness Layer, Made Explicit**:
Build agents from composable [capabilities](https://ai.pydantic.dev/capabilities) that bundle tools, hooks, instructions, and model settings into reusable, typed units. Use built-in capabilities for [web search](https://ai.pydantic.dev/capabilities#provider-adaptive-tools), [thinking](https://ai.pydantic.dev/capabilities#thinking), and [MCP](https://ai.pydantic.dev/capabilities#provider-adaptive-tools); pick batteries like planning, compaction, sandboxed code execution, and sub-agents from the [Pydantic AI Harness](https://ai.pydantic.dev/harness/overview) library; build your own; or install [third-party capability packages](https://ai.pydantic.dev/extensibility). Define agents entirely in [YAML/JSON](https://ai.pydantic.dev/agent-spec) — no code required.

2. **Measured by Default**:
Every run is traced end to end via OpenTelemetry and tightly [integrates](https://ai.pydantic.dev/logfire) with [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, evals-based performance monitoring, and behavior, tracing, and cost tracking — or [any other OTel backend](https://ai.pydantic.dev/logfire#alternative-observability-backends). What your harness does is never a black box: it's in the trace.

3. **Powerful Evals**:
Systematically test and [evaluate](https://ai.pydantic.dev/evals) the performance and accuracy of the agentic systems you build with Pydantic Evals, and monitor them over time in Pydantic Logfire — the same scorers work in CI and against production traces.

4. **Fully Type-safe**:
Designed to give your IDE or AI coding agent as much context as possible for auto-completion and [type checking](https://ai.pydantic.dev/agents#static-type-checking), moving entire classes of errors from runtime to write-time for a bit of that Rust "if it compiles, it works" feel.

5. **Model-agnostic**:
Supports virtually every [model](https://ai.pydantic.dev/models/overview) and provider: OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, and Perplexity; Azure AI Foundry, Amazon Bedrock, Google Cloud, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks AI, Cerebras, Hugging Face, GitHub, Heroku, Vercel, Nebius, OVHcloud, Alibaba Cloud, SambaNova, and Z.AI. If your favorite model or provider is not listed, you can easily implement a [custom model](https://ai.pydantic.dev/models/overview#custom-models). Swapping models — or falling back between them mid-run — doesn't mean rebuilding your harness.

6. **Structured & Streamed Outputs**:
Model responses are validated against Pydantic models — if validation fails, the model is prompted to retry — and structured output can be [streamed](https://ai.pydantic.dev/output#streamed-results) continuously with immediate validation.

7. **MCP and UI**:
Integrates the [Model Context Protocol](https://ai.pydantic.dev/mcp/overview) and various [UI event stream](https://ai.pydantic.dev/ui/overview) standards to give your agent access to external tools and data and build interactive applications with streaming event-based communication.

8. **Production-grade Runs**:
Build [durable agents](https://ai.pydantic.dev/durable_execution/overview/) that preserve progress across transient failures and restarts, flag tool calls that [require human approval](https://ai.pydantic.dev/deferred-tools#human-in-the-loop-tool-approval), and enforce [usage limits](https://ai.pydantic.dev/agents#usage-limits) on tokens, requests, and cost.

9. **Graph Support**:
Define [graphs](https://ai.pydantic.dev/graph) using type hints, for complex applications where standard control flow can degrade to spaghetti code.

10. **Built by the Pydantic Team**:
[Pydantic Validation](https://docs.pydantic.dev/latest/) is the validation layer of the OpenAI SDK, the Google ADK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more. _Why use the derivative when you can go straight to the source?_ :smiley:

Realistically though, no list is going to be as convincing as [giving it a try](#next-steps) and seeing how it makes you feel!

## Hello World Example

Here's a minimal example of Pydantic AI:

```python
from pydantic_ai import Agent

# Define a very simple agent including the model to use, you can also set the model when running the agent.
agent = Agent(
    'anthropic:claude-sonnet-4-6',
    # Register static instructions using a keyword argument to the agent.
    # For more complex dynamically-generated instructions, see the example below.
    instructions='Be concise, reply with one sentence.',
)

# Run the agent synchronously, conducting a conversation with the LLM.
result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

_(This example is complete, it can be run "as is", assuming you've [installed the `pydantic_ai` package](https://ai.pydantic.dev/install))_

The exchange will be very short: Pydantic AI will send the instructions and the user prompt to the LLM, and the model will return a text response.

Not very interesting yet, but we can easily add [tools](https://ai.pydantic.dev/tools), [dynamic instructions](https://ai.pydantic.dev/agents#instructions), [structured outputs](https://ai.pydantic.dev/output), or composable [capabilities](https://ai.pydantic.dev/capabilities) to build more powerful agents.

Here's the same agent with [thinking](https://ai.pydantic.dev/capabilities#thinking) and [web search](https://ai.pydantic.dev/capabilities#provider-adaptive-tools) capabilities:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking, WebSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Be concise, reply with one sentence.',
    capabilities=[Thinking(), WebSearch()],
)

result = agent.run_sync('What was the mass of the largest meteorite found this year?')
print(result.output)
```

## Tools & Dependency Injection Example

Here is a concise example using Pydantic AI to build a support agent for a bank:

**(Better documented example [in the docs](https://ai.pydantic.dev/#tools-dependency-injection-example))**

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


# SupportDependencies is used to pass data, connections, and logic into the model that will be needed when running
# instructions and tool functions. Dependency injection provides a type-safe way to customise the behavior of your agents.
@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


# This Pydantic model defines the structure of the output returned by the agent.
class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


# This agent will act as first-tier support in a bank.
# Agents are generic in the type of dependencies they accept and the type of output they return.
# In this case, the support agent has type `Agent[SupportDependencies, SupportOutput]`.
support_agent = Agent(
    'openai:gpt-5.2',
    deps_type=SupportDependencies,
    # The response from the agent will be guaranteed to be a SupportOutput,
    # if validation fails the agent is prompted to try again.
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


# Dynamic instructions can make use of dependency injection.
# Dependencies are carried via the `RunContext` argument, which is parameterized with the `deps_type` from above.
# If the type annotation here is wrong, static type checkers will catch it.
@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


# The `tool` decorator let you register functions which the LLM may call while responding to a user.
# Again, dependencies are carried via `RunContext`, any other arguments become the tool schema passed to the LLM.
# Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
@support_agent.tool
async def customer_balance(
        ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    # The docstring of a tool is also passed to the LLM as the description of the tool.
    # Parameter descriptions are extracted from the docstring and added to the parameter schema sent to the LLM.
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return balance


...  # In a real use case, you'd add more tools and a longer system prompt


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    # Run the agent asynchronously, conducting a conversation with the LLM until a final response is reached.
    # Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.
    result = await support_agent.run('What is my balance?', deps=deps)
    # The `result.output` will be validated with Pydantic to guarantee it is a `SupportOutput`. Since the agent is generic,
    # it'll also be typed as a `SupportOutput` to aid with static type checking.
    print(result.output)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```

## Next Steps

To try Pydantic AI for yourself, [install it](https://ai.pydantic.dev/install) and follow the instructions [in the examples](https://ai.pydantic.dev/examples/setup).

Read the [docs](https://ai.pydantic.dev/agents/) to learn more about building applications with Pydantic AI.

Read the [API Reference](https://ai.pydantic.dev/api/agent/) to understand Pydantic AI's interface.

Join [Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [GitHub](https://github.com/pydantic/pydantic-ai/issues) if you have any questions.

## Part of the Pydantic Stack

The Pydantic Stack is everything you need to ship production-grade AI agents:

- [Pydantic AI](https://pydantic.dev/pydantic-ai?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) - Type-safe agent harness
- [Pydantic AI Harness](https://ai.pydantic.dev/harness/overview) - Batteries-included capability library
- [Pydantic Logfire](https://pydantic.dev/logfire?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) - AI-first, full-stack observability
- [Logfire AI Gateway](https://pydantic.dev/ai-gateway?utm_source=github&utm_medium=readme&utm_campaign=pydantic-ai) - Unified LLM proxy
