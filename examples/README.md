# Pydantic AI Examples

Examples of how to use Pydantic AI and what it can do.

## Usage

To run the examples, run:

```bash
uv run -m examples.<example_module_name>
```

## Examples

### `pydantic_model.py`

(Demonstrates: custom `result_type`)

Simple example of using Pydantic AI to construct a Pydantic model from a text input.

```bash
uv run --extra examples -m examples.pydantic_model
```

This examples uses `openai:gpt-4o` by default but it works well with other modesl, e.g. you can run it
with Gemini using:

```bash
PYDANTIC_AI_MODEL=gemini-1.5-pro uv run --extra examples -m examples.pydantic_model
```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash...`)

### `sql_gen.py`

(Demonstrates: custom `result_type`, dynamic system prompt, result validation, agent deps)

Example demonstrating how to use Pydantic AI to generate SQL queries based on user input.

```bash
uv run --extra examples -m examples.sql_gen
```

or to use a custom prompt:

```bash
uv run --extra examples -m examples.sql_gen "find me whatever"
```

This model uses `gemini-1.5-flash` by default since Gemini is good at single shot queries.

### `weather.py`

(Demonstrates: retrievers, multiple retrievers, agent deps)

Example of Pydantic AI with multiple tools which the LLM needs to call in turn to answer a question.

In this case the idea is a "weather" agent — the user can ask for the weather in multiple cities,
the agent will use the `get_lat_lng` tool to get the latitude and longitude of the locations, then use
the `get_weather` tool to get the weather.

To run this example properly, you'll need two extra API keys:
* A weather API key from [tomorrow.io](https://www.tomorrow.io/weather-api/) set via `WEATHER_API_KEY`
* A geocoding API key from [geocode.maps.co](https://geocode.maps.co/) set via `GEO_API_KEY`

**(Note if either key is missing, the code will fall back to dummy data.)**

```bash
uv run --extra examples -m examples.weather
```

This example uses `openai:gpt-4o` by default. Gemini seems to be unable to handle the multiple tool
calls.
