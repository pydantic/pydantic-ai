## Introduction

Agents are PydanticAI's primary interface for interacting with LLMs.

In some use cases a single Agent will control an entire application or component,
but multiple agents can also interact to embody more complex workflows.

The [`Agent`][pydantic_ai.Agent] class is well documented, but in essence you can think of an agent as a container for:

* A [system prompt](#system-prompts) — a set of instructions for the LLM written by the developer
* One or more [retrievers](#retrievers) — functions that the LLM may call to get information while generating a response
* An optional structured [result type](results.md) — the structured datatype the LLM must return at the end of a run
* A [dependency](dependencies.md) type constraint — system prompt functions, retrievers and result validators may all use dependencies when they're run
* Agents may optionally also have a default [model](models/index.md) associated with them, the model to use can also be defined when running the agent

In typing terms, agents are generic in their dependency and result types, e.g. an agent which required `#!python Foobar` dependencies and returned data of type `#!python list[str]` results would have type `#!python Agent[Foobar, list[str]]`.

Here's a toy example of an agent that simulates a roulette wheel:

```py title="roulette_wheel.py"
from pydantic_ai import Agent, CallContext

roulette_agent = Agent(  # (1)!
    'openai:gpt-4o',
    deps_type=int,
    result_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.retriever_context
async def roulette_wheel(ctx: CallContext[int], square: int) -> str:  # (2)!
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18  # (3)!
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.data)  # (4)!
#> True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.data)
#> False
```

1. Create an agent, which expects an integer dependency and returns a boolean result, this agent will ahve type of `#!python Agent[int, bool]`.
2. Define a retriever that checks if the square is a winner, here [`CallContext`][pydantic_ai.dependencies.CallContext] is parameterized with the dependency type `int`, if you got the dependency type wrong you'd get a typing error.
3. In reality, you might want to use a random number here e.g. `random.randint(0, 36)` here.
4. `result.data` will be a boolean indicating if the square is a winner, Pydantic performs the result validation, it'll be typed as a `bool` since its type is derived from the `result_type` generic parameter of the agent.

!!! tip "Agents are Singletons, like FastAPI"
    Agents are a singleton instance, you can think of them as similar to a small [`FastAPI`][fastapi.FastAPI] app or an [`APIRouter`][fastapi.APIRouter].

## Running Agents

There are three ways to run an agent:

1. [`#!python agent.run()`][pydantic_ai.Agent.run] — a coroutine which returns a result containing a completed response, returns a [`RunResult`][pydantic_ai.result.RunResult]
2. [`#!python agent.run_sync()`][pydantic_ai.Agent.run_sync] — a plain function which returns a result containing a completed response (internally, this just calls `#!python asyncio.run(self.run())`), returns a [`RunResult`][pydantic_ai.result.RunResult]
3. [`#!python agent.run_stream()`][pydantic_ai.Agent.run_stream] — a coroutine which returns a result containing methods to stream a response as an async iterable, returns a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult]

Here's a simple example demonstrating all three:

```python title="run_agent.py"
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.data)
#> Rome


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.data)
    #> Paris

    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_data())
        #> London
```
_(This example is complete, it can be run "as is")_

You can also pass messages from previous runs to continue a conversation or provide context, as described in [Messages and Chat History](message-history.md).

## Runs vs. Conversations

An agent **run** might represent an entire conversation — there's no limit to how many messages can be exchanged in a single run. However, a **conversation** might also be composed of multiple runs, especially if you need to maintain state between separate interactions or API calls.

Here's an example of a conversation comprised of multiple runs:

```python title="conversation_example.py"
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.data)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?', message_history=result1.new_messages()  # (1)!
)
print(result2.data)
#> Albert Einstein's most famous equation is (E = mc^2).
```
1. Continue the conversation, without `message_history` the model would not know who "he" was referring to.

## System Prompts

System prompts might seem simple at first glance since they're just strings (or sequences of strings that are concatenated), but crafting the right system prompt is key to getting the model to behave as you want.

Generally, system prompts fall into two categories:

1. **Static system prompts**: These are known when writing the code and can be defined via the `system_prompt` parameter of the `Agent` constructor.
2. **Dynamic system prompts**: These aren't known until runtime and should be defined via functions decorated with `@agent.system_prompt`.

You can add both to a single agent; they're concatenated in the order they're defined at runtime.

Here's an example using both types of system prompts:

```python title="system_prompts.py"
from datetime import date

from pydantic_ai import Agent, CallContext

agent = Agent(
    'openai:gpt-4o',
    deps_type=str,  # (1)!
    system_prompt="Use the customer's name while replying to them.",  # (2)!
)


@agent.system_prompt  # (3)!
def add_the_users_name(ctx: CallContext[str]) -> str:
    return f"The user's named is {ctx.deps}."


@agent.system_prompt
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.data)
#> Hello Frank, the date today is 2032-01-02.
```

1. The agent expects a string dependency.
2. Static system prompt defined at agent creation time.
3. Dynamic system prompt defined via a decorator.
4. Another dynamic system prompt, system prompts don't have to have the `CallContext` parameter.

## Retrievers

* two different retriever decorators (`retriver_plain` and `retriever_context`) depending on whether you want to use the context or not, show an example using both
* retriever parameters are extracted and used to build the schema for the tool, then validated with pydantic
* if a retriever has a single "model like" parameter (e.g. pydantic mode, dataclass, typed dict), the schema for the tool will but just that type
* docstrings are parsed to get the tool description, thanks to griffe docs for each parameter are extracting using Google, numpy or sphinx docstring styling
* You can raise `ModelRetry` from within a retriever to suggest to the model it should retry
* the return type of retriever can either be `str` or a JSON object typed as `dict[str, Any]` as some models (e.g. Gemini) support structured return values, some expect text (OpenAI) but seem to be just as good at extracting meaning from the data

## Reflection and self-correction

* validation errors from both retrievers parameter validation and structured result validation can be passed back to the with a request to retry
* as described above, you can also raise `ModelRetry` from within a retriever or result validator to tell the model it should retry
* the default retry count is 1, but can be altered both on a whole agent, or on a per-retriever basis and result validator basis
* you can access the current retry count from within a retriever or result validator via `ctx.retry`

## Model errors

* If models behave unexpectedly, e.g. the retry limit is exceed, agent runs will raise `UnexpectedModelBehaviour` exceptions
* If you use PydanticAI in correctly, we try to raise a `UserError` with a helpful message
* show an except of a `UnexpectedModelBehaviour` being raised
* if a `UnexpectedModelBehaviour` is raised, you may want to access the [`.last_run_messages`][pydantic_ai.Agent.last_run_messages] attribute of an agent to see the messages exchanged that led to the error, show an example of accessing `.last_run_messages` in an except block to get more details

## API Reference

::: pydantic_ai.Agent
    options:
      members:
        - __init__
        - run
        - run_sync
        - run_stream
        - model
        - override_deps
        - override_model
        - last_run_messages
        - system_prompt
        - retriever_plain
        - retriever_context
        - result_validator

::: pydantic_ai.exceptions
