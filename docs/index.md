---
title: Pydantic AI
description: "How Python does AI: agents, realtime voice, image generation, embeddings. Every model, every interface, typed end to end."
---

# Pydantic AI {.hide}

<div style="text-align: center">
  <img class="off-glb only-dark" src="./img/pydantic-ai-dark.svg" alt="Pydantic AI" />
</div>
<div style="text-align: center">
  <img class="off-glb only-light" src="./img/pydantic-ai-light.svg" alt="Pydantic AI" />
</div>
<p style="text-align: center">
  <em>How Python does AI</em>
</p>
<p style="text-align: center">
  <a href="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push" alt="CI" />
  </a>
  <a href="https://coverage-badge.samuelcolvin.workers.dev/redirect/pydantic/pydantic-ai">
    <img src="https://coverage-badge.samuelcolvin.workers.dev/pydantic/pydantic-ai.svg" alt="Coverage" />
  </a>
  <a href="https://pypi.python.org/pypi/pydantic-ai">
    <img src="https://img.shields.io/pypi/v/pydantic-ai.svg" alt="PyPI" />
  </a>
  <a href="https://github.com/pydantic/pydantic-ai">
    <img src="https://img.shields.io/pypi/pyversions/pydantic-ai.svg" alt="versions" />
  </a>
  <a href="https://github.com/pydantic/pydantic-ai/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/pydantic/pydantic-ai.svg" alt="license" />
  </a>
  <a href="https://logfire.pydantic.dev/docs/join-slack/">
    <img src="https://img.shields.io/badge/Slack-Join%20Slack-4A154B?logo=slack" alt="Join Slack" />
  </a>
</p>

<p style="text-align: center; font-size: 1.15em">
  Agents, realtime voice, image generation, embeddings. Every model, every interface, typed end to end.
</p>

**Pydantic AI** is the Python AI SDK: a typed, [extensible](extensibility.md) agent loop with [every model](models/overview.md) a string swap away. The same agent [runs everywhere you need it](interfaces.md): behind a [web frontend](ui/overview.md), in the [terminal](cli.md), on a [voice call](realtime/overview.md), on a [durable background queue](durable_execution/overview.md), or as a plain object you call [`run()`](agent.md#running-agents) on. [Image generation](capabilities/image-generation.md) and [embeddings](embeddings.md) come in the same box.

**[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/)** has everything an agent needs for complex, long-running work, snapped on as [capabilities](capabilities/overview.md), from [memory](https://pydantic.dev/docs/ai/harness/memory/), [sub-agents](https://pydantic.dev/docs/ai/harness/subagents/), and [context management](https://pydantic.dev/docs/ai/harness/compaction/) to a complete [coding agent](https://pydantic.dev/docs/ai/harness/coder/).

## What are you building?

From simple typed data extraction to complex, long-running multi-agent collaboration, Pydantic AI and [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/) have got you covered.

=== "Coding agent"

    A complete coding agent in your terminal: workspace-rooted [file access](https://pydantic.dev/docs/ai/harness/filesystem/), allowlisted [shell](https://pydantic.dev/docs/ai/harness/shell/), [repo orientation](https://pydantic.dev/docs/ai/harness/repo-context/), [planning](https://pydantic.dev/docs/ai/harness/planning/), and [context management](https://pydantic.dev/docs/ai/harness/compaction/) that survives long sessions. Here with [web search](capabilities/web-search.md) and a second-opinion [advisor](https://pydantic.dev/docs/ai/harness/advisor/) snapped on alongside:

    ```bash
    uv add pydantic-ai pydantic-ai-harness
    ```

    ```python {test="skip" lint="skip"}
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import WebSearch
    from pydantic_ai_harness import Advisor, Coder

    agent = Agent(
        'anthropic:claude-fable-5',
        capabilities=[
            Coder(),  # files, shell, repo context, planning, sub-agents, context management
            WebSearch(),  # look up docs and error messages on the web
            Advisor('openai:gpt-5.6-sol'),  # a second opinion from another model when stuck
        ],
    )
    agent.to_cli_sync()
    ```

    [`Coder`](https://pydantic.dev/docs/ai/harness/coder/) is a regular [combined capability](capabilities/custom.md#composition-and-middleware-semantics), not a black box: use it whole, or use the blocks it bundles directly; the two are equivalent:

    ```python {test="skip" lint="skip"}
    capabilities = [
        FileSystem('.'), Shell(cwd='.'), RepoContext(), Planning(), SubAgents(...),
        ClearToolResults(), WarnNearLimits(), ToolOutputLimits(),
    ]
    ```

    Run the file and you're chatting with the agent in your terminal. To try it before writing any code, run the exported [`coder_agent`](https://pydantic.dev/docs/ai/harness/coder/#api-reference) with [`clai`](cli.md#custom-agents) (the Pydantic AI CLI), via [`uvx`](https://docs.astral.sh/uv/guides/tools/):

    ```bash
    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
    ```

    **Build this →** [Coder](https://pydantic.dev/docs/ai/harness/coder/), from the [Harness](https://pydantic.dev/docs/ai/harness/)

=== "Data extraction"

    Give the agent an [output type](output.md) and [tools](tools.md), and every run comes back validated and typed:

    ```bash
    uv add pydantic-ai
    ```

    ```python {title="review_sentiment.py"}
    from typing import Literal

    from pydantic import BaseModel, Field

    from pydantic_ai import Agent, RunContext


    class Sentiment(BaseModel):
        label: Literal['positive', 'negative', 'neutral']
        score: float = Field(ge=-1, le=1)


    agent = Agent('openai:gpt-5.6-sol', output_type=Sentiment)


    @agent.tool
    def recent_reviews(ctx: RunContext[None], product: str) -> list[str]:
        """Fetch recent review snippets for a product."""
        return ['The new release fixed everything I complained about!']


    result = agent.run_sync('How are people feeling about the Extract app?')
    print(result.output)
    #> label='positive' score=0.9
    ```

    The [`@agent.tool`](tools.md) function receives a [`RunContext`][pydantic_ai.tools.RunContext] that carries your [dependencies](dependencies.md) in; the rest of its signature and its docstring become the tool schema, arguments are [validated](tools.md#function-tools-and-schema) before your code runs, and the run is guaranteed to return a `Sentiment`, so your IDE, type checker, and the LLM all agree on the returned type.

    **Build this →** [Agents](agent.md), [Function Tools](tools.md), and [Structured Output](output.md)

=== "Realtime voice"

    Put the same agent on a live voice session, [tools](realtime/tools.md) and [capabilities](realtime/capabilities.md) included:

    ```bash
    uv add "pydantic-ai[openai-realtime]"
    ```

    ```python {test="skip" lint="skip"}
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
        microphone = asyncio.create_task(session.send_audio(microphone_chunks()))  # your microphone → the model
        speaker = asyncio.create_task(play_audio(session.stream_audio()))  # model audio → your speaker
        async for part in session.stream_transcripts():
            print(f'{part.speaker}: {part.transcript}')
    ```

    The model calls your tools mid-conversation while it keeps talking, and every session is [instrumented](logfire.md); voice is just another frontend, on OpenAI Realtime, Gemini Live, Azure, and xAI Grok Voice.

    **Build this →** [Realtime Voice](realtime/overview.md), starting from the [voice assistant example](examples/realtime-voice.md)

=== "Durable background agent"

    Attach [`TemporalDurability`](durable_execution/temporal.md) and the same agent runs inside a [Temporal](durable_execution/temporal.md) workflow: every model and tool call becomes a durable activity, so a run working through a background queue survives restarts, failures, and long waits:

    ```bash
    uv add "pydantic-ai[temporal]"
    ```

    ```python {title="durable_research.py"}
    from temporalio import workflow

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import WebFetch, WebSearch
    from pydantic_ai.durable_exec.temporal import PydanticAIWorkflow, TemporalDurability

    agent = Agent(
        'openai:gpt-5.6-sol',
        instructions='Research the topic and write a structured brief.',
        name='researcher',
        capabilities=[WebSearch(), WebFetch(), TemporalDurability()],
    )


    @workflow.defn
    class ResearchWorkflow(PydanticAIWorkflow):
        __pydantic_ai_agents__ = [agent]

        @workflow.run
        async def run(self, topic: str) -> str:
            result = await agent.run(f'Write a brief on: {topic}')
            return result.output
    ```

    [DBOS](durable_execution/dbos.md) and [Prefect](durable_execution/prefect.md) attach the same way, first-party and co-maintained, with [Restate, Kitaru, and Airflow](durable_execution/overview.md) integrations besides.

    **Build this →** [Durable Execution](durable_execution/overview.md)

=== "Image generation"

    Ask for an image and make it the run's typed [output](output.md):

    ```bash
    uv add pydantic-ai
    ```

    ```python {title="logo_generation.py"}
    from pathlib import Path

    from pydantic_ai import Agent, BinaryImage

    agent = Agent('openai:gpt-5.6-sol', output_type=BinaryImage)
    result = agent.run_sync('Generate a minimalist logo for a coffee shop called Extract.')
    Path('logo.png').write_bytes(result.output.data)
    ```

    [Provider-native generation](native-tools.md#image-generation-tool) on models that support it (like this one), a [subagent fallback](capabilities/image-generation.md) you can configure for the rest, and a [standalone image API](https://github.com/pydantic/pydantic-ai/pull/5357) on the way.

    **Build this →** [Image Generation](capabilities/image-generation.md)

=== "Embeddings"

    Embed documents and queries for semantic search or a [RAG pipeline](examples/rag.md):

    ```python {title="embeddings_quickstart.py"}
    from pydantic_ai import Embedder

    embedder = Embedder('openai:text-embedding-3-small')
    result = embedder.embed_query_sync('What is machine learning?')
    print(len(result.embeddings[0]))
    #> 1536
    ```

    Seven providers behind one typed API, [instrumented](logfire.md) like everything else. It lives next to the agent that will use the results.

    **Build this →** [Embeddings](embeddings.md), then the [RAG example](examples/rag.md)

!!! tip "No API key yet?"
    You don't need a provider API key to try any of this. Pass the built-in [`'test'` model](testing.md#unit-testing-with-testmodel) (`Agent('test')`), which runs entirely offline without calling an LLM, so you can exercise your agent, tools, and outputs first. When you're ready for a real model, see [Models and Providers](models/overview.md) to pick a provider and set its API key.

## Why Pydantic AI

- **Any model, one Python API.** [Virtually every model and provider](models/overview.md) (OpenAI, Anthropic, Google, Bedrock, Azure AI Foundry, Groq, Mistral, xAI, Ollama, and dozens more), swappable with a string, or through the [Pydantic AI Gateway](gateway.md): one key for all of them, with failover and cost monitoring built in. No flagship feature is locked to one vendor.

- **Typed end to end.** [Structured outputs](output.md), typed [dependency injection](dependencies.md), [typed tools](tools.md): your IDE, type checker, and coding agent all know what your agent returns, moving whole classes of errors from runtime to write-time. When plain control flow isn't enough, [Pydantic Graph](graph.md) brings the same typing to graph-based workflows.

- **Measured, not vibes.** OpenTelemetry-native [instrumentation](logfire.md) works with any OTel backend; one line lights up [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, tracing, and cost tracking backed by [genai-prices](https://github.com/pydantic/genai-prices). [Pydantic Evals](evals.md) tests agent behavior the way pytest tests code.

- **Batteries, composably.** One primitive, the [capability](capabilities/overview.md), bundles [tools](tools.md), [instructions](agent.md#instructions), [hooks](hooks.md), and [model settings](agent.md#model-run-settings) into reusable units. Core ships fundamentals like [MCP](capabilities/mcp.md) and [web search](capabilities/web-search.md), the [Harness](https://pydantic.dev/docs/ai/harness/) ships everything else, and complete agents like [Coder](https://pydantic.dev/docs/ai/harness/coder/) and [Researcher](https://pydantic.dev/docs/ai/harness/researcher/) are just capabilities composed: they come apart the way they went together. Or skip code entirely with [YAML/JSON agent specs](agent-spec.md).

- **[Every interface](interfaces.md).** One agent definition runs as a [CLI](cli.md), a [built-in web chat](web.md), or [realtime speech](realtime/overview.md); [UI event streams](ui/overview.md) (AG-UI, Vercel AI) connect it to your own frontend or anything else; and [ACP](https://pydantic.dev/docs/ai/harness/acp/) *(experimental)* serves it as an editor agent.

- **Durable execution.** First-party, co-maintained [durable execution](durable_execution/overview.md) on Temporal, DBOS, or Prefect, with [Restate, Kitaru, and Airflow](durable_execution/overview.md) integrations and more coming. Agents survive restarts and run for days on the engine you already operate, with [human-in-the-loop approval](deferred-tools.md#human-in-the-loop-tool-approval) built in.

Built by the [Pydantic](https://docs.pydantic.dev) team: [Pydantic Validation](https://pydantic.dev/docs/) is the validation layer of the OpenAI SDK, the Anthropic SDK, the Google ADK, LangChain, and most of the AI ecosystem (and the foundation FastAPI was built on). Pydantic AI brings that same feeling to agents.

**Sign up for our newsletter, *The Pydantic Stack*, with updates & tutorials on Pydantic AI, Logfire, and Pydantic:**

  <form method="POST" action="https://eu.customerioforms.com/forms/submit_action?site_id=53d2086c3c4214eaecaa&form_id=14b22611745b458&success_url=https://pydantic.dev/docs/ai/" class="md-typeset" style="display: flex; align-items: center; gap: 0.5rem; width: 100%;">
      <input
      type="email"
      id="email_input"
      name="email"
      class="md-input md-input--stretch"
      style="flex: 1; background: var(--md-default-bg-color); color: var(--md-default-fg-color);"
      required
      placeholder="Email"
      data-1p-ignore
      data-lpignore="true"
      data-protonpass-ignore="true"
      data-bwignore="true"
      />
      <input type="hidden" id="source_input" name="source" value="pydantic-ai" />
      <button type="submit" class="md-button md-button--primary">Subscribe</button>
  </form>

## Putting it together: a bank support agent

Here's a support agent for a bank, showing several features working together: [dependency injection](dependencies.md) carrying a database connection into instructions and tools, [function tools](tools.md) the model calls, [structured output](output.md) validated on every run, a reusable [capability](capabilities/overview.md) bundling the customer context, and an [on-demand capability](capabilities/on-demand.md) the model loads only when the conversation calls for it:

```python {title="bank_support.py"}
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, Capability, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (1)!
    customer_id: int
    db: DatabaseConn  # (2)!


class SupportOutput(BaseModel):  # (3)!
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


customer_context = Capability[SupportDependencies](  # (4)!
    id='customer-context',
    description="Who the customer is and what's on their account.",
)


@customer_context.instructions  # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@customer_context.tool  # (6)!
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  # (7)!
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


refunds = Capability[SupportDependencies](  # (8)!
    id='refunds',
    description='Refund eligibility and refund status.',
    defer_loading=True,
)


@refunds.tool
async def refund_status(ctx: RunContext[SupportDependencies]) -> str:
    """Look up the refund status for the customer's most recent charge."""
    return await ctx.deps.db.refund_status(id=ctx.deps.customer_id)


support_agent = Agent(  # (9)!
    'openai:gpt-5.6-sol',  # (10)!
    deps_type=SupportDependencies,
    output_type=SupportOutput,  # (11)!
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
    capabilities=[customer_context, refunds],  # (12)!
)


...  # (13)!


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)  # (14)!
    print(result.output)
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """

    result = await support_agent.run(  # (15)!
        'Was I refunded for the duplicate charge on my last statement?', deps=deps
    )
    print(result.output)
    """
    support_advice='Good news, John: the duplicate charge on your last statement was refunded on 2026-05-01.' block_card=False risk=1
    """
```

1. The `SupportDependencies` dataclass is used to pass data, connections, and logic into the model that will be needed when running [instructions](agent.md#instructions) and [tool](tools.md) functions. Pydantic AI's system of [dependency injection](dependencies.md) provides a [type-safe](agent.md#static-type-checking) way to customise the behavior of your agents, and can be especially useful when running [unit tests](testing.md) and evals.
2. This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
3. This [Pydantic](https://docs.pydantic.dev) model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.
4. A [`Capability`][pydantic_ai.capabilities.Capability] bundles related instructions and tools into one reusable unit: the same primitive behind [built-in capabilities](capabilities/overview.md) like [web search](capabilities/web-search.md) and everything in the [Harness](https://pydantic.dev/docs/ai/harness/). This one carries the customer context; you could drop it into any other agent's `capabilities` list as-is.
5. Dynamic [instructions](agent.md#instructions) can make use of dependency injection. Dependencies are carried via the [`RunContext`][pydantic_ai.tools.RunContext] argument, which is parameterized with the `deps_type` from above. If the type annotation here is wrong, static type checkers will catch it.
6. The [`tool`](tools.md) decorator registers a function whose signature becomes a tool the LLM may call while responding to a user. Again, dependencies are carried via [`RunContext`][pydantic_ai.tools.RunContext]; any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
7. The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are [extracted](tools.md#function-tools-and-schema) from the docstring and added to the parameter schema sent to the LLM.
8. `defer_loading=True` makes this an [on-demand capability](capabilities/on-demand.md), like an [Agent Skill](capabilities/on-demand.md#loading-skills-from-markdown-files). It collapses to a one-line catalog entry in the prompt, and its tools stay hidden until the model decides it's relevant and loads it with the framework-managed `load_capability` tool.
9. This [agent](agent.md) will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has type `#!python Agent[SupportDependencies, SupportOutput]`.
10. Here we configure the agent to use [OpenAI's GPT-5.6 Sol](api/models/openai.md) model; you can also set the model when running the agent.
11. The response from the agent will be guaranteed to be a `SupportOutput`. Since the agent is generic, it'll also be typed as a `SupportOutput` to aid with static type checking. If validation fails, the agent is [prompted to try again](agent.md#reflection-and-self-correction).
12. Mount the capabilities on the agent. More [capabilities](capabilities/overview.md), like [web search](capabilities/web-search.md) or anything from the [Harness](https://pydantic.dev/docs/ai/harness/), snap on alongside them in the same list.
13. In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
14. [Run the agent](agent.md#running-agents) asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.
15. This turn exercises the deferred capability: the model sees the `refunds` catalog entry, calls `load_capability` with `id='refunds'`, and only then gets the `refund_status` tool to answer with: [on-demand loading](capabilities/on-demand.md) in action.

The [dependencies](dependencies.md) dataclass carries the database connection into [instructions](agent.md#instructions) and [tools](tools.md) with full type safety: swap in a test double and the same agent runs in [unit tests](testing.md) and evals. And because the customer context is a [capability](capabilities/overview.md), it composes: the same unit drops into a voice agent or a web app unchanged.

!!! tip "Complete `bank_support.py` example"
    The code included here is incomplete for the sake of brevity (the definition of `DatabaseConn` is missing); you can find the complete `bank_support.py` example [here](examples/bank-support.md).

## Instrumentation with Pydantic Logfire

Pydantic AI is [OpenTelemetry](https://opentelemetry.io/)-native: the [Instrumentation](capabilities/instrumentation.md) capability emits standard OTel spans for every model call and tool call, and [any OTLP backend works](logfire.md#alternative-observability-backends). The easiest setup is the [`logfire` SDK](logfire.md#using-logfire), which speaks plain OpenTelemetry and can point at [Pydantic Logfire](https://pydantic.dev/logfire) or any other backend.

Even a simple agent with just a handful of tools can result in a lot of back-and-forth with the LLM, making it nearly impossible to be confident of what's going on just from reading the code. To watch the runs above in action, [set up Logfire](logfire.md#using-logfire) and add the following to the code:

```python {title="bank_support_with_logfire.py" hl_lines="6-10" test="skip" lint="skip"}
...
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn

import logfire

logfire.configure()  # (1)!
logfire.instrument_pydantic_ai()  # (2)!
logfire.instrument_sqlite3()  # (3)!

...

support_agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
    capabilities=[customer_context],
)
```

1. Configure the Logfire SDK, this will fail if project is not set up.
2. This will instrument all Pydantic AI agents used from here on out. To instrument only a specific agent, add an [`Instrumentation`][pydantic_ai.capabilities.Instrumentation] entry to the agent's `capabilities=[...]`.
3. In our demo, `DatabaseConn` uses [`sqlite3`][] to connect to a PostgreSQL database, so [`logfire.instrument_sqlite3()`](https://logfire.pydantic.dev/docs/integrations/databases/sqlite3/)
   is used to log the database queries.

That's enough to get the following view of your agent in action:

/// public-trace | https://logfire-eu.pydantic.dev/public-trace/a2957caa-b7b7-4883-a529-777742649004?spanId=31aade41ab896144
    title: 'Logfire instrumentation for the bank agent'
///

See [Monitoring and Performance](logfire.md) to learn more.

## `llms.txt`

The Pydantic AI documentation is available in the [llms.txt](https://llmstxt.org/) format.
This format is defined in Markdown and suited for LLMs and AI coding assistants and agents.

Two formats are available:

- [`llms.txt`](https://pydantic.dev/docs/ai/llms.txt): a file containing a brief description
  of the project, along with links to the different sections of the documentation. The structure
  of this file is described in details [here](https://llmstxt.org/#format).
- [`llms-full.txt`](https://pydantic.dev/docs/ai/llms-full.txt): Similar to the `llms.txt` file,
  but every link content is included. Note that this file may be too large for some LLMs.

As of today, these files are not automatically leveraged by IDEs or coding agents, but they will use it if you provide a link or the full text.


## Next steps

**Run something right now.** One command puts a complete [coding agent](https://pydantic.dev/docs/ai/harness/coder/) in your terminal:

```bash
uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent -m anthropic:claude-fable-5
```

Or [install Pydantic AI](install.md), pick a [model](models/overview.md), and put your own coding agent to work: install the [Pydantic AI skill](coding-agent-skills.md) to give it up-to-date framework knowledge, point it at the [examples](examples/setup.md) and the [Harness index](https://pydantic.dev/docs/ai/harness/), and tell it what you'd like to build.

**See what your agent did.** [Instrument it](logfire.md): one line of setup, and every model call and tool call shows up. It's standard OpenTelemetry: [Pydantic Logfire](https://pydantic.dev/logfire) is the easiest way to look, any OTLP backend works.

**Go deeper.** The [Agents guide](agent.md) is the core walkthrough; the [API Reference](api/agent.md) covers the full interface; the [Harness](https://pydantic.dev/docs/ai/harness/) has the batteries.

**Get help.** Join [Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [:simple-github: GitHub](https://github.com/pydantic/pydantic-ai/issues).
