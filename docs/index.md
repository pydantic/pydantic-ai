# Introduction {.hide}

--8<-- "docs/.partials/index-header.html"

FastAPI revolutionized web development by offering an innovative and ergonomic design, built on the foundation of [Pydantic](https://docs.pydantic.dev).

Similarly, virtually every agent framework and LLM library in Python uses Pydantic, yet when we began to use LLMs in [Pydantic Logfire](https://pydantic.dev/logfire), we couldn't find anything that gave us the same feeling.

We built PydanticAI with one simple aim: to bring that FastAPI feeling to GenAI app development.

## Why use PydanticAI

* __Built by the Pydantic Team__:
Built by the team behind [Pydantic](https://docs.pydantic.dev/latest/) (the validation layer of the OpenAI SDK, the Anthropic SDK, LangChain, LlamaIndex, AutoGPT, Transformers, CrewAI, Instructor and many more).

* __Model-agnostic__:
Supports OpenAI, Anthropic, Gemini, Deepseek, Ollama, Groq, Cohere, and Mistral, and there is a simple interface to implement support for [other models](models.md).

* __Pydantic Logfire Integration__:
Seamlessly [integrates](logfire.md) with [Pydantic Logfire](https://pydantic.dev/logfire) for real-time debugging, performance monitoring, and behavior tracking of your LLM-powered applications.

* __Type-safe__:
Designed to make [type checking](agents.md#static-type-checking) as powerful and informative as possible for you.

* __Python-centric Design__:
Leverages Python's familiar control flow and agent composition to build your AI-driven projects, making it easy to apply standard Python best practices you'd use in any other (non-AI) project.

* __Structured Responses__:
Harnesses the power of [Pydantic](https://docs.pydantic.dev/latest/) to [validate and structure](results.md#structured-result-validation) model outputs, ensuring responses are consistent across runs.

* __Dependency Injection System__:
Offers an optional [dependency injection](dependencies.md) system to provide data and services to your agent's [system prompts](agents.md#system-prompts), [tools](tools.md) and [result validators](results.md#result-validators-functions).
This is useful for testing and eval-driven iterative development.

* __Streamed Responses__:
Provides the ability to [stream](results.md#streamed-results) LLM outputs continuously, with immediate validation, ensuring rapid and accurate results.

* __Graph Support__:
[Pydantic Graph](graph.md) provides a powerful way to define graphs using typing hints, this is useful in complex applications where standard control flow can degrade to spaghetti code.

## Hello World Example

Here's a minimal example of PydanticAI:

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent(  # (1)!
    'google-gla:gemini-1.5-flash',
    system_prompt='Be concise, reply with one sentence.',  # (2)!
)

result = agent.run_sync('Where does "hello world" come from?')  # (3)!
print(result.data)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

1. We configure the agent to use [Gemini 1.5's Flash](api/models/gemini.md) model, but you can also set the model when running the agent.
2. Register a static [system prompt](agents.md#system-prompts) using a keyword argument to the agent.
3. [Run the agent](agents.md#running-agents) synchronously, conducting a conversation with the LLM.

_(This example is complete, it can be run "as is")_

The exchange should be very short: PydanticAI will send the system prompt and the user query to the LLM, the model will return a text response.

Not very interesting yet, but we can easily add "tools", dynamic system prompts, and structured responses to build more powerful agents.

## Tools & Dependency Injection Example

Here is a concise example using PydanticAI to build a support agent for a bank:

```python {title="bank_support.py"}
from dataclasses import dataclass

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (3)!
    customer_id: int
    db: DatabaseConn  # (12)!


class SupportResult(BaseModel):  # (13)!
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description="Whether to block the customer's card")
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  # (1)!
    'openai:gpt-4o',  # (2)!
    deps_type=SupportDependencies,
    result_type=SupportResult,  # (9)!
    system_prompt=(  # (4)!
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
)


@support_agent.system_prompt  # (5)!
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
    print(result.data)  # (10)!
    """
    support_advice='Hello John, your current account balance, including pending transactions, is $123.45.' block_card=False risk=1
    """

    result = await support_agent.run('I just lost my card!', deps=deps)
    print(result.data)
    """
    support_advice="I'm sorry to hear that, John. We are temporarily blocking your card to prevent unauthorized transactions." block_card=True risk=8
    """
```

1. This [agent](agents.md) will act as first-tier support in a bank. Agents are generic in the type of dependencies they accept and the type of result they return. In this case, the support agent has type `#!python Agent[SupportDependencies, SupportResult]`.
2. Here we configure the agent to use [OpenAI's GPT-4o model](api/models/openai.md), you can also set the model when running the agent.
3. The `SupportDependencies` dataclass is used to pass data, connections, and logic into the model that will be needed when running [system prompt](agents.md#system-prompts) and [tool](tools.md) functions. PydanticAI's system of dependency injection provides a [type-safe](agents.md#static-type-checking) way to customise the behavior of your agents, and can be especially useful when running [unit tests](testing-evals.md) and evals.
4. Static [system prompts](agents.md#system-prompts) can be registered with the [`system_prompt` keyword argument][pydantic_ai.Agent.__init__] to the agent.
5. Dynamic [system prompts](agents.md#system-prompts) can be registered with the [`@agent.system_prompt`][pydantic_ai.Agent.system_prompt] decorator, and can make use of dependency injection. Dependencies are carried via the [`RunContext`][pydantic_ai.tools.RunContext] argument, which is parameterized with the `deps_type` from above. If the type annotation here is wrong, static type checkers will catch it.
6. [`tool`](tools.md) let you register functions which the LLM may call while responding to a user. Again, dependencies are carried via [`RunContext`][pydantic_ai.tools.RunContext], any other arguments become the tool schema passed to the LLM. Pydantic is used to validate these arguments, and errors are passed back to the LLM so it can retry.
7. The docstring of a tool is also passed to the LLM as the description of the tool. Parameter descriptions are [extracted](tools.md#function-tools-and-schema) from the docstring and added to the parameter schema sent to the LLM.
8. [Run the agent](agents.md#running-agents) asynchronously, conducting a conversation with the LLM until a final response is reached. Even in this fairly simple case, the agent will exchange multiple messages with the LLM as tools are called to retrieve a result.
9. The response from the agent will, be guaranteed to be a `SupportResult`, if validation fails [reflection](agents.md#reflection-and-self-correction) will mean the agent is prompted to try again.
10. The result will be validated with Pydantic to guarantee it is a `SupportResult`, since the agent is generic, it'll also be typed as a `SupportResult` to aid with static type checking.
11. In a real use case, you'd add more tools and a longer system prompt to the agent to extend the context it's equipped with and support it can provide.
12. This is a simple sketch of a database connection, used to keep the example short and readable. In reality, you'd be connecting to an external database (e.g. PostgreSQL) to get information about customers.
13. This [Pydantic](https://docs.pydantic.dev) model is used to constrain the structured data returned by the agent. From this simple definition, Pydantic builds the JSON Schema that tells the LLM how to return the data, and performs validation to guarantee the data is correct at the end of the run.

!!! tip "Complete `bank_support.py` example"
    The code included here is incomplete for the sake of brevity (the definition of `DatabaseConn` is missing); you can find the complete `bank_support.py` example [here](examples/bank-support.md).

## Instrumentation with Pydantic Logfire

To understand the flow of the above runs, we can watch the agent in action using Pydantic Logfire.

To do this, we need to set up logfire, and add the following to our code:

```python {title="bank_support_with_logfire.py" hl_lines="6-9 21" test="skip" lint="skip"}
...
from pydantic_ai import Agent, RunContext

from bank_database import DatabaseConn

import logfire

logfire.configure()  # (1)!
logfire.instrument_asyncpg()  # (2)!

...

support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    result_type=SupportResult,
    system_prompt=(
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query.'
    ),
    instrument=True,
)
```

1. Configure logfire, this will fail if project is not set up.
2. In our demo, `DatabaseConn` uses [`asyncpg`]() to connect to a PostgreSQL database, so [`logfire.instrument_asyncpg()`](https://magicstack.github.io/asyncpg/current/) is used to log the database queries.

That's enough to get the following view of your agent in action:

{{ video('9078b98c4f75d01f912a0368bbbdb97a', 25, 55) }}

See [Monitoring and Performance](logfire.md) to learn more.

## Next Steps

To try PydanticAI yourself, follow the instructions [in the examples](examples/index.md).

Read the [docs](agents.md) to learn more about building applications with PydanticAI.

Read the [API Reference](api/agent.md) to understand PydanticAI's interface.
