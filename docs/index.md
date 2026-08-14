---
title: Pydantic AI
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
  Agents, realtime voice, embeddings, image generation — every model, every interface, typed end to end.
</p>

**Pydantic AI** is the Python agent framework: a typed agent loop, [every model](models/overview.md) behind one API, and [every interface](interfaces.md) — terminal, [web](web.md), [your own frontend](ui/overview.md), even [voice](realtime/overview.md). **[Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/)** is its official capability library and harness — everything around the model that turns it into an agent: 50+ capabilities across core and Harness that you snap on, from [web search](capabilities/web-search.md) to a complete coding agent.

Whatever you came to build — a one-off LLM call in a script, an AI feature inside your SaaS, a voice agent, or your own Claude-Code-style coding agent — this is the right place, and it's the same library at every step. Start small:

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent(  # (1)!
    'anthropic:claude-fable-5',
    instructions='Be concise, reply with one sentence.',  # (2)!
)

result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

1. We configure the agent to use [Anthropic's Claude Fable 5](api/models/anthropic.md) model, but you can also set the model when running the agent.
2. Register static [instructions](agent.md#instructions) using a keyword argument to the agent.
3. [Run the agent](agent.md#running-agents) synchronously, starting a conversation with the LLM.

_(This example is complete, it can be run "as is", assuming you've [installed the `pydantic_ai` package](install.md))_

## What are you building?

Text agents are the start — the same library, with the same typed API, is a complete coding agent, a voice agent, embeddings, and image generation. No separate SDKs, no framework bloat: each one drops into a single function.

=== "Agents"

    ```python {test="skip" lint="skip"}
    from typing import Literal

    from pydantic import BaseModel, Field
    from pydantic_ai import Agent

    class Sentiment(BaseModel):
        label: Literal['positive', 'negative', 'neutral']
        score: float = Field(ge=-1, le=1)

    agent = Agent('openai:gpt-5.6-sol', output_type=Sentiment)
    result = agent.run_sync('The new release fixed everything I complained about!')
    print(result.output)
    #> label='positive' score=0.9
    ```

    The result is validated against the model and typed as `Sentiment` — your IDE, type checker, and the LLM all agree on the shape.

    **Build this →** [Agents](agent.md) and [Structured Output](output.md)

=== "Coding agent"

    ```python {test="skip" lint="skip"}
    from pydantic_ai import Agent
    from pydantic_ai_harness import Coder

    agent = Agent('anthropic:claude-fable-5', capabilities=[Coder()])
    agent.to_cli_sync()
    ```

    Run that file and you're chatting with a complete coding agent in your terminal — workspace-rooted file access, allowlisted shell, repo orientation, planning, context management that survives long sessions. Or skip the file entirely:

    ```bash
    uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent
    ```

    **Build this →** [Coder](https://pydantic.dev/docs/ai/harness/coder/), from the [Harness](https://pydantic.dev/docs/ai/harness/)

=== "Realtime voice"

    ```python {test="skip"}
    from pydantic_ai import Agent

    agent = Agent(instructions='You are a helpful voice assistant.')

    async with agent.realtime('openai:gpt-realtime').session() as session:
        ...  # stream microphone audio in, play session.stream_audio() out
    ```

    Same tools, same capabilities, same observability as any other agent — voice is just another frontend, on OpenAI Realtime, Gemini Live, Azure, and xAI Grok Voice.

    **Build this →** [Realtime Voice](realtime/overview.md)

=== "Embeddings"

    ```python {test="skip" lint="skip"}
    from pydantic_ai import Embedder

    embedder = Embedder('openai:text-embedding-3-small')
    result = embedder.embed_query_sync('What is machine learning?')
    print(len(result.embeddings[0]))
    #> 1536
    ```

    Seven providers, one typed API, instrumented like everything else.

    **Build this →** [Embeddings](embeddings.md)

=== "Image generation"

    ```python {test="skip" lint="skip"}
    from pathlib import Path

    from pydantic_ai import Agent, BinaryImage

    agent = Agent('openai-responses:gpt-5.4', output_type=BinaryImage)
    result = agent.run_sync('Generate a minimalist logo for a coffee shop called Extract.')
    Path('logo.png').write_bytes(result.output.data)
    ```

    Provider-native where supported, with a fallback path everywhere else — and the image is the typed output of the run.

    **Build this →** [Image Generation](capabilities/image-generation.md)

!!! tip "No API key yet?"
    You don't need a provider API key to try any of this. Pass the built-in [`'test'` model](testing.md#unit-testing-with-testmodel) — `Agent('test')` — which runs entirely offline without calling an LLM, so you can exercise your agent, tools, and outputs first. When you're ready for a real model, see [Models and Providers](models/overview.md) to pick a provider and set its API key.

Everything between these is composition, not rewriting: add [tools](tools.md), [structured outputs](output.md), and [capabilities](capabilities/overview.md) one at a time. Here's the three-liner grown a typed output and [web search](capabilities/web-search.md):

```python {title="hello_world_capabilities.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch


class PythonRelease(BaseModel):
    version: str
    release_month: str
    source_url: str


agent = Agent(
    'anthropic:claude-fable-5',
    output_type=PythonRelease,
    capabilities=[WebSearch(local='duckduckgo')],
)

result = agent.run_sync('When is the next Python feature release due?')
print(result.output)
"""
version='3.15.0' release_month='October 2026' source_url='https://peps.python.org/pep-0790/'
"""
```

## Why Pydantic AI

- **Any model, one API.** [Virtually every model and provider](models/overview.md) — OpenAI, Anthropic, Google, Bedrock, Azure AI Foundry, Groq, Mistral, xAI, Ollama, and dozens more — swappable with a string. No flagship feature is locked to one vendor.

- **Typed end to end.** [Structured outputs](output.md), typed [dependency injection](dependencies.md), typed tools: your IDE, type checker, and AI coding agent all know what your agent returns, moving whole classes of errors from runtime to write-time.

- **Batteries, composably.** One primitive — the [capability](capabilities/overview.md) — bundles tools, instructions, hooks, and model settings into reusable units. Core ships the fundamentals; the [Harness](https://pydantic.dev/docs/ai/harness/) ships everything else — code execution, memory, sub-agents, guardrails, compaction — plus complete harnesses like `Coder` that are themselves just capabilities composed, so you can take them apart. Or skip code entirely with [YAML/JSON agent specs](agent-spec.md).

- **[Every interface](interfaces.md).** One agent definition runs as a CLI, a [built-in web chat](web.md), [your own frontend](ui/overview.md) (AG-UI and Vercel AI protocols), an editor agent, or a [voice agent](realtime/overview.md).

- **Measured, not vibes.** OpenTelemetry-native [instrumentation](logfire.md) works with any OTel backend — one line lights up [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, tracing, and cost tracking — and [Pydantic Evals](evals.md) tests agent behavior the way pytest tests code.

- **Durable by choice.** First-party, co-maintained [durable execution](durable_execution/overview.md) on Temporal, DBOS, or Prefect — agents that survive restarts and run for days, on the engine you already operate, with [human-in-the-loop approval](deferred-tools.md#human-in-the-loop-tool-approval) built in.

Built by the [Pydantic](https://docs.pydantic.dev) team: Pydantic Validation is the validation layer of the OpenAI SDK, the Anthropic SDK, the Google ADK, LangChain, and most of the AI ecosystem — and the foundation FastAPI was built on. Pydantic AI brings that same feeling to agents.

## Tools & Dependency Injection Example

Here is a concise example using Pydantic AI to build a support agent for a bank:

```python {title="bank_support.py"}
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:
    customer_id: int
    db: DatabaseConn


class SupportOutput(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(
    'openai:gpt-5.6-sol',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


@support_agent.instructions
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


...


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
```

The [dependencies](dependencies.md) dataclass carries the database connection into [instructions](agent.md#instructions) and [tools](tools.md) with full type safety — swap in a test double and the same agent runs in [unit tests](testing.md) and evals. The [`@support_agent.tool`](tools.md) function's signature and docstring become the tool schema the LLM sees, with arguments validated by Pydantic. And the run is guaranteed to return a `SupportOutput` — if validation fails, the agent is [prompted to try again](agent.md#reflection-and-self-correction).

!!! tip "Complete `bank_support.py` example"
    The code included here is incomplete for the sake of brevity (the definition of `DatabaseConn` is missing); you can find the complete `bank_support.py` example [here](examples/bank-support.md).

## Instrumentation with Pydantic Logfire

Even a simple agent with just a handful of tools can result in a lot of back-and-forth with the LLM, making it nearly impossible to be confident of what's going on just from reading the code.
To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.

To do this, we need to [set up Logfire](logfire.md#using-logfire), and add the following to our code:

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

See [Monitoring and Performance](logfire.md) to learn more. It's standard OpenTelemetry, so [any OTLP backend works](logfire.md#alternative-observability-backends) — Logfire is simply the easiest way to see it during development.

## `llms.txt`

The Pydantic AI documentation is available in the [llms.txt](https://llmstxt.org/) format.
This format is defined in Markdown and suited for LLMs and AI coding assistants and agents.

Two formats are available:

- [`llms.txt`](https://ai.pydantic.dev/llms.txt): a file containing a brief description
  of the project, along with links to the different sections of the documentation. The structure
  of this file is described in details [here](https://llmstxt.org/#format).
- [`llms-full.txt`](https://ai.pydantic.dev/llms-full.txt): Similar to the `llms.txt` file,
  but every link content is included. Note that this file may be too large for some LLMs.

As of today, these files are not automatically leveraged by IDEs or coding agents, but they will use it if you provide a link or the full text.


## Next steps

**Run something right now.** `uvx --with pydantic-ai-harness clai -a pydantic_ai_harness.coder:coder_agent` puts a complete coding agent in your terminal — or [install Pydantic AI](install.md), pick a [model](models/overview.md), and run the [examples](examples/setup.md) on your own machine.

**See what your agent did.** [Instrument it](logfire.md) — one line of setup, and every model call and tool call shows up. It's standard OpenTelemetry: Logfire is the easiest way to look, any OTLP backend works.

**Go deeper.** The [Agents guide](agent.md) is the core walkthrough; the [API Reference](api/agent.md) covers the full interface; the [Harness](https://pydantic.dev/docs/ai/harness/) has the batteries.

**Get help.** Join [Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [:simple-github: GitHub](https://github.com/pydantic/pydantic-ai/issues).

**Sign up for our newsletter, *The Pydantic Stack*, with updates & tutorials on Pydantic AI, Logfire, and Pydantic:**

  <form method="POST" action="https://eu.customerioforms.com/forms/submit_action?site_id=53d2086c3c4214eaecaa&form_id=14b22611745b458&success_url=https://ai.pydantic.dev/" class="md-typeset" style="display: flex; align-items: center; gap: 0.5rem; width: 100%;">
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
