"""Example of SQL Mode: letting the model orchestrate tool calls by writing SQL.

SQL Mode (from the [Pydantic AI Harness](https://github.com/pydantic/pydantic-ai-harness))
registers your tools as DuckDB functions and gives the model a single `run_sql`
tool. Instead of one model round-trip per tool call, the model writes one SQL
query that calls the tools, navigates the nested JSON they return, pipes
sub-objects from one tool into another, and aggregates the results.

Run with:

    uv run -m pydantic_ai_examples.sql_mode
"""

from pydantic import BaseModel
from pydantic_ai_harness import SQLModeBuilder

from pydantic_ai import Agent


class Coordinates(BaseModel):
    lat: float
    lon: float


class Address(BaseModel):
    country: str
    region: str


class Place(BaseModel):
    coordinates: Coordinates
    address: Address
    population: int


class Forecast(BaseModel):
    high_c: float
    low_c: float
    summary: str


# Canned data keeps the example self-contained; real tools would call an API.
_PLACES = {
    'Paris': Place(
        coordinates=Coordinates(lat=48.85, lon=2.35),
        address=Address(country='France', region='Ile-de-France'),
        population=2_103_000,
    ),
    'Tokyo': Place(
        coordinates=Coordinates(lat=35.68, lon=139.69),
        address=Address(country='Japan', region='Kanto'),
        population=13_960_000,
    ),
    'Lima': Place(
        coordinates=Coordinates(lat=-12.05, lon=-77.04),
        address=Address(country='Peru', region='Lima'),
        population=9_752_000,
    ),
    'Cairo': Place(
        coordinates=Coordinates(lat=30.04, lon=31.24),
        address=Address(country='Egypt', region='Cairo Governorate'),
        population=10_100_000,
    ),
    'Oslo': Place(
        coordinates=Coordinates(lat=59.91, lon=10.75),
        address=Address(country='Norway', region='Ostlandet'),
        population=709_000,
    ),
}

_UNKNOWN = Place(
    coordinates=Coordinates(lat=0.0, lon=0.0),
    address=Address(country='Unknown', region='Unknown'),
    population=0,
)


def geocode(city: str) -> Place:
    """Look up a city's coordinates, structured address, and population."""
    return _PLACES.get(city, _UNKNOWN)


async def get_forecast(coordinates: Coordinates) -> Forecast:
    """Get today's forecast for a latitude/longitude pair."""
    high = round(34.0 - abs(coordinates.lat) * 0.45, 1)
    return Forecast(high_c=high, low_c=round(high - 8.0, 1), summary='clear skies')


# `geocode` returns a nested object: `coordinates` and `address` are sub-objects.
# `get_forecast` takes only the `Coordinates`, so in SQL the model has to navigate
# into the geocode result with `->` and pipe the sub-object across:
#     get_forecast(geocode(city) -> 'coordinates')
sql_mode = SQLModeBuilder().register_tool(geocode).register_tool(get_forecast).build()

agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[sql_mode])


if __name__ == '__main__':
    result = agent.run_sync(
        'For Paris, Tokyo, Lima, Cairo and Oslo, show each city with its country, '
        'population, and forecast high today, sorted warmest first. Which country '
        'has the warmest city, and what is the combined population of all five? '
        'Use a single SQL query.'
    )
    print(result.output)
