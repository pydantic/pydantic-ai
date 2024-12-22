Example of PydanticAI with multiple tools which the LLM needs to call in turn to answer a question.

Demonstrates:

* [tools](../tools.md)
* [agent dependencies](../dependencies.md)
* [streaming text responses](../results.md#streaming-text)

In this case the idea is a "weather" agent — the user can ask for the weather in multiple locations,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather for those locations.

## Running the Example

To run this example properly, you might want to add two extra API keys **(Note if either key is missing, the code will fall back to dummy data, so they're not required)**:

* A weather API key from [tomorrow.io](https://www.tomorrow.io/weather-api/) set via `WEATHER_API_KEY`
* A geocoding API key from [geocode.maps.co](https://geocode.maps.co/) set via `GEO_API_KEY`

With [dependencies installed and environment variables set](./index.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.weather_agent
```

## Example Code

```python {title="pydantic_ai_examples/weather_agent.py"}
#! examples/pydantic_ai_examples/weather_agent.py
```
