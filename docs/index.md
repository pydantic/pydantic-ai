# PydanticAI

You can think of PydanticAI as an Agent Framework or a shim to use Pydantic with LLMs — they're the same thing.

PydanticAI trys to make working with LLMs feel similar to building a web application.

## Example — Retrievers

Partial example of using retrievers to help an LLM respond to a user's query about the weather:

```py title="weather_agent.py"
import httpx
from pydantic_ai import Agent, CallContext

weather_agent = Agent(  # (1)!
    'openai:gpt-4o',  # (2)!
    deps=httpx.AsyncClient,  # (3)!
    system_prompt='Be concise, reply with one sentence.',  # (4)!
)


@weather_agent.retriever_context  # (5)!
async def get_location(ctx: CallContext[httpx.AsyncClient], location_description: str) -> dict[str, float]:
    """Get the latitude and longitude of a location by its description."""  # (6)!
    ...


@weather_agent.retriever_context  # (7)!
async def get_weather(ctx: CallContext[httpx.AsyncClient], lat: float, lng: float) -> dict[str, str]:
    """Get the weather at a location by its latitude and longitude."""
    ...


result = weather_agent.run_sync('What is the weather like in West London and in Wiltshire?')  # (8)!
print(result.data)  # (9)!
# > 'The weather in West London is raining, while in Wiltshire it is sunny.'

print(result.all_messages())  # (10)!
```

1. An agent that can tell users about the weather in a particular location. Agents combine a system prompt, a response type (here `str`) and "retrievers" (aka tools).
2. Here we configure the agent to use OpenAI's GPT-4o model, you can also customise the model when running the agent.
3. We specify a dependency for the agent, in this case an HTTP client, which retrievers will use to make requests to external services. PydanticAI's system of dependency injection provides a powerful, type safe way to customise the behaviour of your agents, including for unit tests and evals.
4. Static system prompts can be registered as key word arguments to the agent, dynamic system prompts can be registered with the `@agent.system_prompot` decorator and benefit from dependency injection.
5. Retrievers let you register "tools" which the LLM can call while trying to respond to a user. You inject dependencies into the retriever with `CallContext`, any other arguments become the tool schema passed to the LLM, Pydantic is used to validate these arguments, errors are passed back to the LLM so it can retry.
6. This docstring is also passed to the LLM as a description of the tool.
7. Multiple retrievers can be registered with the same agent, the LLM can choose which (if any) retrievers to call in order to respond to a user.
8. Run the agent synchronously, conducting a conversation with the LLM until a final response is reached. (internally agents are all async, `run_sync` is a helper using `asyncio.run` to call `run()`)
9. The response from the LLM, in this case a `str`, Agents are generic in both the type of `deps` and `result_type`, so calls are typed end-to-end.
10. `result.all_messages()` includes details of messages exchanged, this is useful both to understand teh conversation that tool place and useful if you want to continue the conversation later — messages can be passed back to later `run/sync_run` calls.

!!! tip "Complete `weather_agent.py` example"
    The above `weather_agent.py` example is complete for the sake of brevity, but you can find a complete example [here](#TODO).
