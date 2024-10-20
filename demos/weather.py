from devtools import debug

from pydantic_ai import Agent

try:
    import logfire
except ImportError:
    pass
else:
    logfire.configure()

weather_agent: Agent[None, str] = Agent('openai:gpt-4o', system_prompt='Be concise, reply with one sentence.')


@weather_agent.retriever_plain
async def get_lat_lng(location_description: str) -> dict[str, float]:
    """
    Get the latitude and longitude of a location.

    Args:
        location_description: A description of a location.
    """
    if 'london' in location_description.lower():
        return {'lat': 51.1, 'lng': -0.1}
    elif 'wiltshire' in location_description.lower():
        return {'lat': 51.1, 'lng': -2.11}
    else:
        return {'lat': 0, 'lng': 0}


@weather_agent.retriever_plain
async def get_whether(lat: float, lng: float) -> str:
    """
    Get the weather at a location.

    Args:
        lat: Latitude of the location.
        lng: Longitude of the location.
    """
    if abs(lat - 51.1) < 0.1 and abs(lng + 0.1) < 0.1:
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


if __name__ == '__main__':
    result = weather_agent.run_sync('What is the weather like in West London and in Wiltshire?')
    debug(result)
