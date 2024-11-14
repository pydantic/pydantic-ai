from write_docs import result

# Dependencies

PydanticAI uses a dependency injection system to provide data and services to your agent's [system prompts](system-prompt.md), [retrievers](retrievers.md) and [result validators](result-validation.md#TODO).

Dependencies provide a scalable, flexible, type safe and testable way to build agents.

Matching PydanticAI's design philosophy, our dependency system tries to use existing best practice in Python development rather than inventing esoteric "magic", this should make dependencies type-safe, understandable and ultimately easier to deploy in production.

## Defining Dependencies

Dependencies can be any python type, while in simple cases you might be able to pass a single object
as a dependency (e.g. an HTTP connection), [dataclasses][] are generally a convenient container when your dependencies included multiple objects.

!!! note "Asynchronous vs Synchronous dependencies"
    System prompt functions, retriever functions and result validator functions which are not coroutines (e.g. `async def`)
    are called with [`run_in_executor`][asyncio.loop.run_in_executor] in a thread pool, it's therefore marginally preferable
    to use `async` methods where dependencies perform IO, although synchronous dependencies should work fine too.

Here's an example of defining an agent that requires dependencies.

(**Note:** dependencies aren't actually used in this example, see [Accessing Dependencies](#accessing-dependencies) below)

```python title="unused_dependencies.py"
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent


@dataclass
class MyDeps:  # (1)!
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,  # (2)!
)


async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run(
        'Tell me a joke.',
        deps=deps,  # (3)!
    )
    print(result.data)
```

1. Define a dataclass to hold dependencies.
2. Pass the dataclass type to the `deps_type` argument of the [`Agent` constructor][pydantic_ai.Agent.__init__]. **Note**: we're passing the type here, NOT an instance, this parameter is not actually used at runtime, it's here so we can get full type checking of the agent.
3. When running the agent, pass an instance of the dataclass to the `deps` parameter.

_(This example is complete, it can be run "as is" inside an async context)_

## Accessing Dependencies

Dependencies are accessed through the [`CallContext`][pydantic_ai.dependencies.CallContext] type, this should be the first parameter of system prompt functions etc.


```python title="System prompt with dependencies" hl_lines="20-27"
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, CallContext


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)


@agent.system_prompt  # (1)!
async def get_system_prompt(ctx: CallContext[MyDeps]) -> str:  # (2)!
    response = await ctx.deps.http_client.get(  # (3)!
        'https://example.com',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'}  # (4)!
    )
    response.raise_for_status()
    return f'Prompt: {response.text}'


async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run('Tell me a joke.', deps=deps)
    print(result.data)
```

1. [`CallContext`][pydantic_ai.dependencies.CallContext] may optionally be passed to a [`system_prompt`][pydantic_ai.Agent.system_prompt] function as the only argument.
2. [`CallContext`][pydantic_ai.dependencies.CallContext] is parameterized with the type of the dependencies, if this type is incorrect, static type checkers will raise an error.
3. Access dependencies through the [`.deps`][pydantic_ai.dependencies.CallContext.deps] attribute.
4. Access dependencies through the [`.deps`][pydantic_ai.dependencies.CallContext.deps] attribute.

_(This example is complete, it can be run "as is" inside an async context)_

## Full Example

As well as system prompts, dependencies can be used in [retrievers](retrievers.md) and [result validators](result-validation.md#TODO).

```python title="full_example.py" hl_lines="27-34 38-48"
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, CallContext, ModelRetry


@dataclass
class MyDeps:
    api_key: str
    http_client: httpx.AsyncClient


agent = Agent(
    'openai:gpt-4o',
    deps_type=MyDeps,
)


@agent.system_prompt
async def get_system_prompt(ctx: CallContext[MyDeps]) -> str:
    response = await ctx.deps.http_client.get('https://example.com')
    response.raise_for_status()
    return f'Prompt: {response.text}'


@agent.retriever_context  # (1)!
async def get_joke_material(ctx: CallContext[MyDeps], subject: str) -> str:
    response = await ctx.deps.http_client.get(
        'https://example.com#jokes',
        params={'subject': subject},
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    response.raise_for_status()
    return response.text


@agent.result_validator  # (2)!
async def validate_result(ctx: CallContext[MyDeps], final_response: str) -> str:
    response = await ctx.deps.http_client.post(
        'https://example.com#validate',
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
        params={'query': final_response},
    )
    if response.status_code == 400:
        raise ModelRetry(f'invalid response: {response.text}')
    response.raise_for_status()
    return final_response


async with httpx.AsyncClient() as client:
    deps = MyDeps('foobar', client)
    result = await agent.run('Tell me a joke.', deps=deps)
    print(result.data)
```

1. To pass `CallContext` and to a retriever, us the [`retriever_context`][pydantic_ai.Agent.retriever_context] decorator.
2. `CallContext` may optionally be passed to a [`result_validator`][pydantic_ai.Agent.result_validator] function as the first argument.

## Overriding Dependencies

## Agents as dependencies of other Agents
