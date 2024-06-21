# PydanticAI

Shim to use Pydantic with LLMs.

## Example of usage

```py
from pydantic_ai import Agent, CallInfo


# An agent that can tell users about the weather in a particular location.
# Agents combine a system prompt, a response type (here `str`) and one or more
# "retrievers" (aka tools). They can be independent of the LLM used.
weather_agent = Agent(
    'openai:gpt-4o',
    system_prompt='Be concise, reply with one sentence.',
)


# retrievers let you register "tools" which the LLM can call while trying to respond to a user.
@weather_agent.retriever(retries=2)
async def get_location(_: CallInfo[None], location_description: str) -> str:
    """
    Get the latitude and longitude of a location by its description.

    In a real-world application, this function would use a geocoding API,
    hence it's async, you could pass an HTTP client via `CallInfo.context`.

    Arguments are validated via Pydantic, and validation errors are returned to the LLM
    `retries` times.

    This docstring is also passed to the LLM as a description for This.
    """
    import json

    if 'london' in location_description.lower():
        lat_lng = {'lat': 51.1, 'lng': -0.1}
    elif 'wiltshire' in location_description.lower():
        lat_lng = {'lat': 51.1, 'lng': -2.11}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@weather_agent.retriever
async def get_weather(_: CallInfo[None], lat: float, lng: float):
    """
    Get the weather at a location by its latitude and longitude.
    """
    # In a real-world application, this function would use a weather API.
    if abs(lat - 51.1) < 0.1 and abs(lng + 0.1) < 0.1:
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


# Run the agent synchronously, conducting a conversation with the LLM until a final response is reached.
# (internally agents are all async, `run_sync` is a helper using `asyncio.run`)
result = weather_agent.run_sync('What is the weather like in West London and in Wiltshire?')
print(result.response)
#> 'The weather in West London is raining, while in Wiltshire it is sunny.'

# `result.message_history` details of messages exchanged, useful if you want to continue
# the conversation later, via the `message_history` argument of `run_sync`.
print(result.message_history)
```
