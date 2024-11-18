# Introduction {.hide}

--8<-- "docs/.partials/index-header.html"

When I first found FastAPI, I got it immediately, I was excited to find something genuinely innovative and ergonomic built on Pydantic.

Virtually every Agent Framework and LLM library in Python uses Pydantic, but when we came to use Gen AI in [Pydantic Logfire](https://pydantic.dev/logfire), I couldn't find anything that gave me the same feeling.

PydanticAI is a Python AI Framework designed to make it less painful to build production grade applications with Generative AI.

## Why use PydanticAI

* Built by the team behind Pydantic (the validation layer of the OpenAI SDK, the Anthropic SDK, Langchain, LlamaIndex, AutoGPT, Transformers, Instructor and many more)
* Typesafe
* Multi-model — currently with OpenAI and Gemini are support, Anthropic [coming soon](https://github.com/pydantic/pydantic-ai/issues/63), simply interface to implement other models or adapt existing ones
* Built on tried and tested best practices in Python
* Structured response validation with Pydantic
* Streamed responses, including validation of streamed structured responses with Pydantic
* Novel, typesafe dependency injection system
* Logfire integration

!!! example "In Beta"
    PydanticAI is in early beta, the API is subject to change and there's a lot more to do.
    [Feedback](https://github.com/pydantic/pydantic-ai/issues) is very welcome!

## Example — Hello World

Here's a very minimal example of PydanticAI.

```py title="hello_world.py"
from pydantic_ai import Agent

agent = Agent(
    'gemini-1.5-flash',
    system_prompt='Be concise, reply with one sentence.'
)

result = agent.run_sync('Where does "hello world" come from?')
print(result.data)
```

Not very interesting yet, but we can easily add retrievers, dynamic system prompts and structured responses to build more powerful agents.

## Example — Retrievers and Dependency Injection

Small but complete example of using PydanticAI to build a support agent for a bank.

```py title="bank_support.py"
from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, CallContext

from bank_database import DatabaseConn


@dataclass
class SupportDependencies:  # (3)!
    customer_id: int
    db: DatabaseConn


class SupportResult(BaseModel):
    support_advice: str = Field(description='Advice returned to the customer')
    block_card: bool = Field(description='Whether to block their')
    risk: int = Field(description='Risk level of query', ge=0, le=10)


support_agent = Agent(  # (1)!
    'openai:gpt-4o',  # (2)!
    deps_type=SupportDependencies,
    result_type=SupportResult,  # (9)!
    system_prompt=(  # (4)!
        'You are a support agent in our bank, give the '
        'customer support and judge the risk level of their query. '
        "Reply using the customer's name to make your response feel personalised. "
    ),
)


@support_agent.system_prompt  # (5)!
async def add_customer_name(ctx: CallContext[SupportDependencies]) -> str:
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    return f"The customer's name is {customer_name!r}"


@support_agent.retriever_context  # (6)!
async def customer_balance(ctx: CallContext[SupportDependencies], include_pending: bool) -> str:
    """Returns the customer's current account balance"""  # (7)!
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,
    )
    return f'${balance:.2f}'


deps = SupportDependencies(customer_id=123, db=DatabaseConn())
result = support_agent.run_sync('What is my balance?', deps=deps)  # (8)!
print(result.data)  # (10)!

result = support_agent.run_sync('I just lost my card!', deps=deps)
print(result.data)
```

1. An [agent](agents.md) that acts as first-tier support in a bank, agents are generic in the type of dependencies they take and the type of result they return, in this case `Deps` and `SupportResult`.
2. Here we configure the agent to use [OpenAI's GPT-4o model](api/models/openai.md), you can also customise the model when running the agent.
3. The `SupportDependencies` dataclass is used to pass data and connections into the model that will be needed when running [system prompts](agents.md#system-prompts) and [retrievers](agents.md#retrievers). PydanticAI's system of dependency injection provides a powerful, type safe way to customise the behaviour of your agents, including for unit tests and evals.
4. Static [system prompts](agents.md#system-prompts) can be registered as keyword arguments to the agent
5. dynamic [system prompts](agents.md#system-prompts) can be registered with the `@agent.system_prompot` decorator and benefit from dependency injection.
6. [Retrievers](agents.md#retrievers) let you register "tools" which the LLM may call while responding to a user. You inject dependencies into the retriever with [`CallContext`][pydantic_ai.dependencies.CallContext], any other arguments become the tool schema passed to the LLM, Pydantic is used to validate these arguments, errors are passed back to the LLM so it can retry.
7. The docstring is also passed to the LLM as a description of the tool.
8. [Run the agent](agents.md#running-agents) synchronously, conducting a conversation with the LLM until a final response is reached.
9. The response from the agent will, be guaranteed to be a `SupportResult`, if validation fails [reflection](agents.md#reflection-and-self-correction) will mean the agent is prompted to try again.
10. The result will be validated with Pydantic to guarantee it is a `SupportResult`, since the agent is generic, it'll also be typed as a `SupportResult` to aid with static type checking.

!!! tip "Complete `weather_agent.py` example"
    This example is incomplete for the sake of brevity; you can find a complete `weather_agent.py` example [here](examples/weather-agent.md).

## Example — Result Validation

TODO
