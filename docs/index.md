<p align="center">
  <img src="./img/pydantic-ai-dark.svg" alt="PydanticAI">
</p>
<p align="center">
  <em>Agent Framework / shim to use Pydantic with LLMs</em>
</p>
<p align="center">
  <a href="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://github.com/pydantic/pydantic-ai/actions/workflows/ci.yml/badge.svg?event=push" alt="CI">
  </a>
  <a href="https://pypi.python.org/pypi/pydantic-ai">
    <img src="https://img.shields.io/pypi/v/pydantic-ai.svg" alt="PyPI">
  </a>
  <a href="https://github.com/pydantic/pydantic-ai">
    <img src="https://img.shields.io/pypi/pyversions/pydantic-ai.svg" alt="versions">
  </a>
  <a href="https://github.com/pydantic/pydantic-ai/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/pydantic/pydantic-ai.svg" alt="license">
  </a>
</p>

You can think of PydanticAI as an Agent Framework or a shim to use Pydantic with LLMs — they're the same thing.

PydanticAI tries to make working with LLMs feel similar to building a web application.

!!! example "In Beta"
    PydanticAI is in early beta, the API is subject to change and there's a lot more to do.
    [Feedback](https://github.com/pydantic/pydantic-ai/issues) is very welcome!

## Retrievers and Dependency Injection

Partial example of using retrievers to help an LLM respond to a user's query about the weather:

```py title="weather_agent.py"
import httpx
from pydantic_ai import Agent, CallContext

weather_agent = Agent(  # (1)!
    'openai:gpt-4o',  # (2)!
    deps_type=httpx.AsyncClient,  # (3)!
    system_prompt='Be concise, reply with one sentence.',  # (4)!
)


@weather_agent.retriever_context  # (5)!
async def get_location(
    ctx: CallContext[httpx.AsyncClient],
    location_description: str,
) -> dict[str, float]:
    """Get the latitude and longitude of a location by its description."""  # (6)!
    response = await ctx.deps.get('https://api.geolocation...')
    ...


@weather_agent.retriever_context  # (7)!
async def get_weather(
    ctx: CallContext[httpx.AsyncClient],
    lat: float,
    lng: float,
) -> dict[str, str]:
    """Get the weather at a location by its latitude and longitude."""
    response = await ctx.deps.get('https://api.weather...')
    ...


async def main():
    async with httpx.AsyncClient() as client:
        result = await weather_agent.run(  # (8)!
            'What is the weather like in West London and in Wiltshire?',
            deps=client,
        )
        print(result.data)  # (9)!
        # > 'The weather in West London is raining, while in Wiltshire it is sunny.'

        print(result.all_messages())  # (10)!
```

1. An agent that can tell users about the weather in a particular location. Agents combine a system prompt, a response type (here `str`) and "retrievers" (aka tools).
2. Here we configure the agent to use OpenAI's GPT-4o model, you can also customise the model when running the agent.
3. We specify the type dependencies for the agent, in this case an HTTP client, which retrievers will use to make requests to external services. PydanticAI's system of dependency injection provides a powerful, type safe way to customise the behaviour of your agents, including for unit tests and evals.
4. Static system prompts can be registered as key word arguments to the agent, dynamic system prompts can be registered with the `@agent.system_prompot` decorator and benefit from dependency injection.
5. Retrievers let you register "tools" which the LLM may call while to respond to a user. You inject dependencies into the retriever with `CallContext`, any other arguments become the tool schema passed to the LLM, Pydantic is used to validate these arguments, errors are passed back to the LLM so it can retry.
6. This docstring is also passed to the LLM as a description of the tool.
7. Multiple retrievers can be registered with the same agent, the LLM can choose which (if any) retrievers to call in order to respond to a user.
8. Run the agent asynchronously, conducting a conversation with the LLM until a final response is reached. You can also run agents synchronously with `run_sync`. Internally agents are all async, so `run_sync` is a helper using `asyncio.run` to call `run()`.
9. The response from the LLM, in this case a `str`, Agents are generic in both the type of `deps` and `result_type`, so calls are typed end-to-end.
10. `result.all_messages()` includes details of messages exchanged, this is useful both to understand the conversation that took place and useful if you want to continue the conversation later — messages can be passed back to later `run/sync_run` calls.

!!! tip "Complete `weather_agent.py` example"
    The above `weather_agent.py` example is complete for the sake of brevity, but you can find a complete example [here](examples/weather_agent.md).
