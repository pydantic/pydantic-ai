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

1. [`agent.run()`][pydantic_ai.Agent.run] — a coroutine which returns a result containing a completed response, returns a [`RunResult`][pydantic_ai.result.RunResult]
2. [`agent.run_sync()`][pydantic_ai.Agent.run_sync] — a plain function which returns a result containing a completed response (internally, this just calls `asyncio.run(self.run())`), returns a [`RunResult`][pydantic_ai.result.RunResult]
3. [`agent.run_stream()`][pydantic_ai.Agent.run_stream] — a coroutine which returns a result containing methods to stream a response as an async iterable, returns a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult]

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

1. **Static system prompts**: These are known when writing the code and can be defined via the `system_prompt` parameter of the [`Agent` constructor][pydantic_ai.Agent.__init__].
2. **Dynamic system prompts**: These aren't known until runtime and should be defined via functions decorated with [`@agent.system_prompt`][pydantic_ai.Agent.system_prompt].

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

Retrievers provide a mechanism for models to request extra information to help them generate a response.

They're useful when it is impractical or impossible to put all the context an agent might need into the system prompt, or when you want to make agents' behavior more deterministic by deferring some of the logic required to generate a response to another tool.

!!! info "Retrievers vs. RAG"
    Retrievers are basically the "R" of RAG (Retrieval-Augmented Generation) — they augment what the model can do by letting it request extra information.

    The main semantic difference between PydanticAI Retreivers and RAG is RAG is synonymous with vector search, while PydanticAI retrievers are more general purpose. (Note: we might add support for some vector search functionality in the future, particuarly an API for generating embeddings, see [#58](https://github.com/pydantic/pydantic-ai/issues/58))

There are two different decorator functions to register retrievers:

1. [`@agent.retriever_plain`][pydantic_ai.Agent.retriever_plain] — for retrievers that don't need access to the agent [context][pydantic_ai.dependencies.CallContext]
2. [`@agent.retriever_context`][pydantic_ai.Agent.retriever_context] — for retrievers that do need access to the agent [context][pydantic_ai.dependencies.CallContext]

Here's an example using both:

```py title="dice_game.py"
import random

from pydantic_ai import Agent, CallContext

agent = Agent(
    'gemini-1.5-flash',  # (1)!
    deps_type=str,  # (2)!
    system_prompt=(
        "You're a dice game, you should roll the dice and see if the number "
        "you got back matches the user's guess, if so tell them they're a winner. "
        "Use the player's name in the response."
    ),
)


@agent.retriever_plain  # (3)!
def roll_dice() -> str:
    """Roll a six-sided dice and return the result."""
    return str(random.randint(1, 6))


@agent.retriever_context  # (4)!
def get_player_name(ctx: CallContext[str]) -> str:
    """Get the player's name."""
    return ctx.deps


dice_result = agent.run_sync('My guess is 4', deps='Adam')  # (5)!
print(dice_result.data)
#> Congratulations Adam, you guessed correctly! You're a winner!
```

1. This is a pretty simple task, so we can use the fast and cheap Gemini flash model.
2. We pass the user's name as the dependency, to keep things simple we use just the name as a string as the dependency.
3. This retriever doesn't need any context, it just returns a random number. You could probably use a dynamic system prompt in this case.
4. This retriever needs the player's name, so it uses `CallContext` to access dependencies which are just the player's name.
5. Run the agent, passing the player's name as the dependency.

_(This example is complete, it can be run "as is")_

Let's print the messages from that game to see what happened:

```python title="dice_game_messages.py"
from dice_game import dice_result

print(dice_result.all_messages())
"""
[
    SystemPrompt(
        content="You're a dice game, you should roll the dice and see if the number you got back matches the user's guess, if so tell them they're a winner. Use the player's name in the response.",
        role='system',
    ),
    UserPrompt(
        content='My guess is 4',
        timestamp=datetime.datetime(...),
        role='user',
    ),
    ModelStructuredResponse(
        calls=[
            ToolCall(
                tool_name='roll_dice', args=ArgsObject(args_object={}), tool_id=None
            )
        ],
        timestamp=datetime.datetime(...),
        role='model-structured-response',
    ),
    ToolReturn(
        tool_name='roll_dice',
        content='4',
        tool_id=None,
        timestamp=datetime.datetime(...),
        role='tool-return',
    ),
    ModelStructuredResponse(
        calls=[
            ToolCall(
                tool_name='get_player_name',
                args=ArgsObject(args_object={}),
                tool_id=None,
            )
        ],
        timestamp=datetime.datetime(...),
        role='model-structured-response',
    ),
    ToolReturn(
        tool_name='get_player_name',
        content='Adam',
        tool_id=None,
        timestamp=datetime.datetime(...),
        role='tool-return',
    ),
    ModelTextResponse(
        content="Congratulations Adam, you guessed correctly! You're a winner!",
        timestamp=datetime.datetime(...),
        role='model-text-response',
    ),
]
"""
```

We can represent that as a flow diagram, thus:

![Dice game flow diagram](./img/dice-diagram-light.svg#only-light)
![Dice game flow diagram](./img/dice-diagram-dark.svg#only-dark)

### Retrievers, tools, and schema

Under the hood, retrievers use the model's "tools" or "functions" API to let the model know what retrievers are available to call. Tools or functions are also used to define the schema(s) for structured responses, thus a model might have access to many tools, some of which call retrievers while others end the run and return a result.

Function parameters are extracted from the function signature, and all parameters except `CallContext` are used to build the schema for that tool call.

Even better, PydanticAI extracts the docstring from retriever functions and (thanks to [griffe](https://mkdocstrings.github.io/griffe/)) extracts parameter descriptions from the docstring and add them to the schema.

[Griffe supports](https://mkdocstrings.github.io/griffe/reference/docstrings/#docstrings) extracting parameter descriptions from `google`, `numpy` and `sphinx` style docstrings, PydanticAI will infer the format to use based on the docstring. We'll add support in future to explicitly set the style to use, and warn/error if not all parameters are documented, see [#59](https://github.com/pydantic/pydantic-ai/issues/59).

To demonstrate retriever schema, here we use [`FunctionModel`][pydantic_ai.models.function.FunctionModel] to print the schema a model would receive:

```py title="retriever_schema.py"
from pydantic_ai import Agent
from pydantic_ai.messages import Message, ModelAnyResponse, ModelTextResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

agent = Agent()


@agent.retriever_plain
def foobar(a: int, b: str, c: dict[str, list[float]]) -> str:
    """Get me foobar.

    Args:
        a: apple pie
        b: banana cake
        c: carrot smoothie
    """
    return f'{a} {b} {c}'


def print_schema(messages: list[Message], info: AgentInfo) -> ModelAnyResponse:
    retriever = info.retrievers['foobar']
    print(retriever.description)
    #> Get me foobar.
    print(retriever.json_schema)
    """
    {
        'description': 'Get me foobar.',
        'properties': {
            'a': {'description': 'apple pie', 'title': 'A', 'type': 'integer'},
            'b': {'description': 'banana cake', 'title': 'B', 'type': 'string'},
            'c': {
                'additionalProperties': {'items': {'type': 'number'}, 'type': 'array'},
                'description': 'carrot smoothie',
                'title': 'C',
                'type': 'object',
            },
        },
        'required': ['a', 'b', 'c'],
        'type': 'object',
        'additionalProperties': False,
    }
    """
    return ModelTextResponse(content='foobar')


agent.run_sync('hello', model=FunctionModel(print_schema))
```

_(This example is complete, it can be run "as is")_

The return type of retriever can either be `str` or a JSON object typed as `dict[str, Any]` as some models (e.g. Gemini) support structured return values, some expect text (OpenAI) but seem to be just as good at extracting meaning from the data.

## Reflection and self-correction

Validation errors from both retriever parameter validation and [structured result validation](results.md#structured-result-validation) can be passed back to the model with a request to retry.

You can also raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] from within a [retriever](#retrievers) or [result validators functions](results.md#result-validators-functions) to tell the model it should retry.

- The default retry count is **1** but can be altered for the [entire agent][pydantic_ai.Agent.__init__], a [specific retriever][pydantic_ai.Agent.retriever_context], or a [result validator][pydantic_ai.Agent.__init__].
- You can access the current retry count from within a retriever or result validator via [`ctx.retry`][pydantic_ai.dependencies.CallContext].

Here's an example:

```py title="retriever_retry.py"
from fake_database import DatabaseConn
from pydantic import BaseModel

from pydantic_ai import Agent, CallContext, ModelRetry


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    'openai:gpt-4o',
    deps_type=DatabaseConn,
    result_type=ChatResult,
)


@agent.retriever_context(retries=2)
def get_user_by_name(ctx: CallContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    #> John
    #> John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(
            f'No user found with name {name!r}, remember to provide their full name'
        )
    return user_id


result = agent.run_sync(
    'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.data)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""
```

## Model errors

If models behave unexpectedly (e.g., the retry limit is exceeded, or their api returns `503`), agent runs will raise [`UnexpectedModelBehaviour`][pydantic_ai.exceptions.UnexpectedModelBehaviour].

In these cases, [`agent.last_run_messages`][pydantic_ai.Agent.last_run_messages] can be used to access the messages exchanged during the run to help diagnose the issue.

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehaviour

agent = Agent('openai:gpt-4o')


@agent.retriever_plain
def calc_volume(size: int) -> int:  # (1)!
    if size == 42:
        return size**3
    else:
        raise ModelRetry('Please try again.')


try:
    result = agent.run_sync('Please get me the volume of a box with size 6.')
except UnexpectedModelBehaviour as e:
    print('An error occurred:', e)
    #> An error occurred: Retriever exceeded max retries count of 1
    print('cause:', repr(e.__cause__))
    #> cause: ModelRetry('Please try again.')
    print('messages:', agent.last_run_messages)
    """
    messages:
    [
        UserPrompt(
            content='Please get me the volume of a box with size 6.',
            timestamp=datetime.datetime(...),
            role='user',
        ),
        ModelStructuredResponse(
            calls=[
                ToolCall(
                    tool_name='calc_volume',
                    args=ArgsObject(args_object={'size': 6}),
                    tool_id=None,
                )
            ],
            timestamp=datetime.datetime(...),
            role='model-structured-response',
        ),
        RetryPrompt(
            content='Please try again.',
            tool_name='calc_volume',
            tool_id=None,
            timestamp=datetime.datetime(...),
            role='retry-prompt',
        ),
        ModelStructuredResponse(
            calls=[
                ToolCall(
                    tool_name='calc_volume',
                    args=ArgsObject(args_object={'size': 6}),
                    tool_id=None,
                )
            ],
            timestamp=datetime.datetime(...),
            role='model-structured-response',
        ),
    ]
    """
else:
    print(result.data)
```
1. Define a retriever that will raise `ModelRetry` repeatedly in this case.

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
