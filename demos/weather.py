import json

from devtools import debug

from pydantic_ai import Agent

weather_agent: Agent[None, str] = Agent('openai:gpt-4o', system_prompt='Be concise, reply with one sentence.')


@weather_agent.retriever_plain
async def get_location(location_description: str) -> str:
    if 'london' in location_description.lower():
        lat_lng = {'lat': 51.1, 'lng': -0.1}
    elif 'wiltshire' in location_description.lower():
        lat_lng = {'lat': 51.1, 'lng': -2.11}
    else:
        lat_lng = {'lat': 0, 'lng': 0}
    return json.dumps(lat_lng)


@weather_agent.retriever_plain
async def get_whether(lat: float, lng: float):
    if abs(lat - 51.1) < 0.1 and abs(lng + 0.1) < 0.1:
        # it always rains in London
        return 'Raining'
    else:
        return 'Sunny'


if __name__ == '__main__':
    result = weather_agent.run_sync('What is the weather like in West London and in Wiltshire?')
    debug(result)
