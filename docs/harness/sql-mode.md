# SQL Mode

SQL Mode is one of the capabilities in [**Pydantic AI Harness**](overview.md), the official capability library for Pydantic AI. The full docs live in the [harness repo](https://github.com/pydantic/pydantic-ai-harness) -- this page is a short intro.

[`SQLModeBuilder`](https://github.com/pydantic/pydantic-ai-harness/blob/main/pydantic_ai_harness/sql_mode/README.md) registers your tools as DuckDB functions and exposes a single `run_sql` tool. The model writes one SQL query that calls the tools, navigates the JSON they return, pipes values between them, and joins, filters, and aggregates the results -- all in one tool call, against a locked-down in-memory DuckDB.

Standard tool calling requires one model round-trip per tool call. SQL Mode collapses that into one, and lets the model use SQL -- a contained, well-defined language -- as the orchestration layer.

## Usage

```bash
uv add "pydantic-ai-harness[sql-mode]"
```

```python {test="skip" noqa="I001"}
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai_harness import SQLModeBuilder


class Coordinates(BaseModel):
    lat: float
    lon: float


class Place(BaseModel):
    coordinates: Coordinates
    address: dict[str, str]
    population: int


def geocode(city: str) -> Place:
    """Look up a city's coordinates, address, and population."""
    ...


async def get_forecast(coordinates: Coordinates) -> dict[str, float]:
    """Get today's forecast for a latitude/longitude pair."""
    ...


sql_mode = SQLModeBuilder().register_tool(geocode).register_tool(get_forecast).build()
agent = Agent('anthropic:claude-sonnet-4-6', toolsets=[sql_mode])

result = agent.run_sync('How warm is Tokyo today, and what country is it in?')
print(result.output)
```

Each tool's pydantic JSON Schema is rendered into the `run_sql` description, so the model knows `geocode` returns a nested object -- and writes one query that drills into it with `->`/`->>`, pipes the `coordinates` sub-object into `get_forecast`, and sorts the result:

```sql
WITH geocoded AS (
    SELECT city, geocode(city) AS place
    FROM (SELECT unnest(['Paris', 'Tokyo']) AS city)
)
SELECT
    city,
    place->'address'->>'country'                  AS country,
    get_forecast(place->'coordinates')->>'high_c' AS high_c
FROM geocoded
ORDER BY high_c DESC;
```

## Full documentation

See the [SQL Mode README](https://github.com/pydantic/pydantic-ai-harness/blob/main/pydantic_ai_harness/sql_mode/README.md) in the harness repo for sync and async tools, JSON/pydantic typing, the DuckDB lockdown, result handling, the full API, and limitations.
