---
title: Pydantic AI
---

# Pydantic AI {.hide}

--8<-- "docs/.partials/index-header.html"

FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic Validation](https://docs.pydantic.dev) and modern Python features like type hints.

Yet despite virtually every Python agent framework and LLM library using Pydantic Validation, when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.

We built Pydantic AI with one simple aim: to bring that FastAPI feeling to GenAI app and agent development.

## Why use Pydantic AI

1. **Built by the Pydantic Team**:
[Pydantic Validation](https://docs.pydantic.dev/latest/) is the validation layer of the OpenAI SDK, the Google ADK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more. _Why use the derivative when you can go straight to the source?_ :smiley:

2. **Model-agnostic**:
Supports virtually every [model](models/overview.md) and provider: OpenAI, Anthropic, Gemini, DeepSeek, Grok, Cohere, Mistral, and Perplexity; Azure AI Foundry, Amazon Bedrock, Google Vertex AI, Ollama, LiteLLM, Groq, OpenRouter, Together AI, Fireworks AI, Cerebras, Hugging Face, GitHub, Heroku, Vercel, Nebius, OVHcloud, Alibaba Cloud, SambaNova, and Outlines. If your favorite model or provider is not listed, you can easily implement a [custom model](models/overview.md#custom-models).

3. **Seamless Observability**:
Tightly [integrates](logfire.md) with [Pydantic Logfire](https://pydantic.dev/logfire), our general-purpose OpenTelemetry observability platform, for real-time debugging, evals-based performance monitoring, and behavior, tracing, and cost tracking. If you already have an observability platform that supports OTel, you can [use that too](logfire.md#alternative-observability-backends).

4. **Fully Type-safe**:
Designed to give your IDE or AI coding agent as much context as possible for auto-completion and [type checking](agents.md#static-type-checking), moving entire classes of errors from runtime to write-time for a bit of that Rust "if it compiles, it works" feel.

5. **Powerful Evals**:
Enables you to systematically test and [evaluate](evals.md) the performance and accuracy of the agentic systems you build, and monitor the performance over time in Pydantic Logfire.

6. **MCP, A2A, and UI**:
Integrates the [Model Context Protocol](mcp/overview.md), [Agent2Agent](a2a.md), and various [UI event stream](ui/overview.md) standards to give your agent access to external tools and data, let it interoperate with other agents, and build interactive applications with streaming event-based communication.

7. **Human-in-the-Loop Tool Approval**:
Easily lets you flag that certain tool calls [require approval](deferred-tools.md#human-in-the-loop-tool-approval) before they can proceed, possibly depending on tool call arguments, conversation history, or user preferences.

8. **Durable Execution**:
Enables you to build [durable agents](durable_execution/overview.md) that can preserve their progress across transient API failures and application errors or restarts, and handle long-running, asynchronous, and human-in-the-loop workflows with production-grade reliability.

9. **Streamed Outputs**:
Provides the ability to [stream](output.md#streamed-results) structured output continuously, with immediate validation, ensuring real time access to generated data.

10. **Graph Support**:
Provides a powerful way to define [graphs](graph.md) using type hints, for use in complex applications where standard control flow can degrade to spaghetti code.

Realistically though, no list is going to be as convincing as [giving it a try](#next-steps) and seeing how it makes you feel!

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

## Hello World Example

Here's a minimal example of Pydantic AI:

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent(  # (1)!
    'anthropic:claude-sonnet-4-0',
    instructions='Be concise, reply with one sentence.',  # (2)!
)

result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

1. We configure the agent to use [Anthropic's Claude Sonnet 4.0](api/models/anthropic.md) model, but you can also set the model when running the agent.
2. Register static [instructions](agents.md#instructions) using a keyword argument to the agent.
3. [Run the agent](agents.md#running-agents) synchronously, starting a conversation with the LLM.

_(This example is complete, it can be run "as is", assuming you've [installed the `pydantic_ai` package](install.md))_

The exchange will be very short: Pydantic AI will send the instructions and the user prompt to the LLM, and the model will return a text response.

Not very interesting yet, but we can easily add [tools](tools.md), [dynamic instructions](agents.md#instructions), and [structured outputs](output.md) to build more powerful agents.

## Tools & Dependency Injection Example

Here is a concise example using Pydantic AI to build a support agent for a bank:

```python {title="bank_support.py"}
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (3)!
    customer_id: int
    db: DatabaseConn  # (12)!


class SupportOutput(BaseModel):  # (13)!
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  # (1)!
    'openai:gpt-5',  # (2)!
    deps_type=SupportDependencies,
    output_type=SupportOutput,  # (9)!
    instructions=(  # (4)!
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


@support_agent.instructions  # (5)!
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.tool  # (6)!
async def customer_balance(
    ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """Returns the customer's current account balance."""  # (7)!
    return await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )


...  # (11)!


async def main():
    deps = SupportDependencies(customer_id=123, db=DatabaseConn())
    result = await support_agent.run('What is my balance?', deps=deps)  # (8)!
    print(result.output)  # (10)!
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.output)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```

1. This [agent](agents.md) will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of output they return. In this case, the support agent has type `#!python Agent[SupportDependencies, SupportOutput]`.
2. Here we configure the agent to use [OpenAI's GPT-5 model](api/models/openai.md), you can also set the model when running the agent.
3. The `SupportDependencies` dataclass is used to pass data, connections, and logic into the model that will be needed when running [instructions](agents.md#instructions) and [tool](tools.md) functions. Pydantic AI's system of dependency injection provides a [type-safe](agents.md#static-type-checking) way to customise the behavior of your agents, and can be especially useful when running [unit tests](testing.md) and evals.
4. Static [instructions](agents.md#instructions) can be registered with the [`instructions` keyword argument][pydantic_ai.agent.Agent.__init__] to the agent.
5. Dynamic [instructions](agents.md#instructions) can be registered with the [`@agent.instructions`][pydantic_ai.agent.Agent.instructions] decorator, and can make use of dependency injection. Dependencies are carried via the [`RunContext`][pydantic_ai.tools.RunContext] argument, which is parameterized with the `deps_type` from above. If the type annotation here is wrong, static type checkers will catch it.
6. The [`@agent.tool`](tools.md) decorator let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via [`RunContext`][pydantic_ai.tools.RunContext], any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
7. The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are [extracted](tools.md#function-tools-and-schema) from the docstring and added to the parameter schema sent to the LLM.
8. [Run the agent](agents.md#running-agents) asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve an output.
9. The response from the agent will be guaranteed to be a `SupportOutput`. If validation fails [reflection](agents.md#reflection-and-self-correction), the agent is prompted to try again.
10. The output will be validated with Pydantic to guarantee it is a `SupportOutput`, since the agent is generic, it'll also be typed as a `SupportOutput` to aid with static type checking.
11. In a real use case, you'd add more tools and longer instructions to the agent to extend the context it's equipped with and support it can provide.
12. This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
13. This [Pydantic](https://docs.pydantic.dev) model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.

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
    'openai:gpt-5',
    deps_type=SupportDependencies,
    output_type=SupportOutput,
    instructions=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)
```

1. Configure the Logfire SDK, this will fail if project is not set up.
2. This will instrument all Pydantic AI agents used from here on out. If you want to instrument only a specific agent, you can pass the [`instrument=True` keyword argument][pydantic_ai.agent.Agent.__init__] to the agent.
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

- [`llms.txt`](https://ai.pydantic.dev/llms.txt): a file containing a brief description
  of the project, along with links to the different sections of the documentation. The structure
  of this file is described in details [here](https://llmstxt.org/#format).
- [`llms-full.txt`](https://ai.pydantic.dev/llms-full.txt): Similar to the `llms.txt` file,
  but every link content is included. Note that this file may be too large for some LLMs.

As of today, these files are not automatically leveraged by IDEs or coding agents, but they will use it if you provide a link or the full text.


## Next Steps

To try Pydantic AI for yourself, [install it](install.md) and follow the instructions [in the examples](examples/setup.md).

Read the [docs](agents.md) to learn more about building applications with Pydantic AI.

Read the [API Reference](api/agent.md) to understand Pydantic AI's interface.

Join [:simple-slack: Slack](https://logfire.pydantic.dev/docs/join-slack/) or file an issue on [:simple-github: GitHub](https://github.com/pydantic/pydantic-ai/issues) if you have any questions.
