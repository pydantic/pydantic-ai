import json

from devtools import debug

from pydantic_ai import Agent

prompt = """
Your job is to help the end user by telling them the weather in a particular location.
Be concise, reply with one sentence.

To do this, use the `get_location` function to look up the latitude and longitude of a location,
then use the `get_weather` function to look up the weather at that latitude and longitude.
"""

weather_agent: Agent[None, str] = Agent('gemini-1.5-pro', system_prompt=prompt)


@weather_agent.retriever_plain
async def get_location(location_description: str) -> str:
    """
    Get the latitude and longitude of named location or description of place.

    Args:
        location_description: The name or description of the location.

    Returns:

    """
    if 'london' in location_description.lower():
        lat_lng = {'lat': 51.1, 'lng': -0.1}
    elif 'wiltshire' in location_description.lower():
        lat_lng = {'lat': 51.1, 'lng': -2.11}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@weather_agent.retriever_plain
async def get_weather(lat: float, lng: float) -> str:
    """
    Get the weather at a latitude and longitude.

    Args:
        lat: Latitude to check the weather for, float.
        lng: Longitude to check the weather for, float.

    Returns: The weather at the given latitude and longitude, as a string.
    """
    if abs(lat - 51.1) < 0.1 and abs(lng + 0.1) < 0.1:
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


if __name__ == '__main__':
    result = weather_agent.run_sync('What is the weather like in West London and in Wiltshire?')
    debug(result)
