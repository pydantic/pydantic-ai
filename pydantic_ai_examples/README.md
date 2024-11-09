# Pydantic AI Examples

Examples of how to use Pydantic AI and what it can do.

## Usage

These examples are distributed with `pydantic-ai` so you can run them either by cloning the [pydantic-ai repo](https://github.com/pydantic/pydantic-ai) or by simply installing `pydantic-ai` from PyPI with `pip` or `uv`.

Either way you'll need to install extra dependencies to run some examples, you just need to install the `examples` optional dependency group.

If you've cloned the repo, add the extra dependencies with:

```bash
uv sync --extra examples
```

If you've installed `pydantic-ai` via pip/uv, you can install the extra dependencies with:

```bash
pip install 'pydantic-ai[examples]'
# of if you're using uv
uv add 'pydantic-ai[examples]'
```

To run the examples, run:

```bash
python -m pydantic_ai_examples.<example_module_name>
```
(replace `python` with just `uv run` if you're using `uv`)

But you'll probably want to edit examples as well as just running them, so you can copy the examples to a new directory with:

```bash
python -m pydantic_ai_examples --copy-to examples/
```

### Setting model environment variables

All these examples will need to set either:

* `OPENAI_API_KEY` to use OpenAI models, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find how to generate an API key
* `GEMINI_API_KEY` to use Gemini/Google models, go to [aistudio.google.com](https://aistudio.google.com/) and do the same to generate an API key

Then set the API key as an environment variable with:

```bash
export OPENAI_API_KEY=your-api-key
# or
export GEMINI_API_KEY=your-api-key
```

## Examples

### `pydantic_model.py`

(Demonstrates: custom `result_type`)

Simple example of using Pydantic AI to construct a Pydantic model from a text input.

```bash
(uv run/python) -m pydantic_ai_examples.pydantic_model
```

This examples uses `openai:gpt-4o` by default, but it works well with other models, e.g. you can run it
with Gemini using:

```bash
PYDANTIC_AI_MODEL=gemini-1.5-pro (uv run/python) -m pydantic_ai_examples.pydantic_model
```

(or `PYDANTIC_AI_MODEL=gemini-1.5-flash ...`)

### `sql_gen.py`

(Demonstrates: custom `result_type`, dynamic system prompt, result validation, agent deps)

Example demonstrating how to use Pydantic AI to generate SQL queries based on user input.

The resulting SQL is validated by running it as an `EXPLAIN` query on PostgreSQL. To run the example, you first need to run PostgreSQL, e.g. via Docker:

```bash
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 postgres
```
_(we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running)_

Then to run the code

```bash
(uv run/python) -m pydantic_ai_examples.sql_gen
```

or to use a custom prompt:

```bash
(uv run/python) -m pydantic_ai_examples.sql_gen "find me whatever"
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
(uv run/python) -m pydantic_ai_examples.weather
```

This example uses `openai:gpt-4o` by default. Gemini seems to be unable to handle the multiple tool
calls.

### `rag.py`

(Demonstrates: retrievers, agent deps, RAG search)

RAG search example. This demo allows you to ask question of the [logfire](https://pydantic.dev/logfire) documentation.

This is done by creating a database containing each section of the markdown documentation, then registering
the search tool as a retriever with the Pydantic AI agent.

Logic for extracting sections from markdown files and a JSON file with that data is available in
[this gist](https://gist.github.com/samuelcolvin/4b5bb9bb163b1122ff17e29e48c10992).

[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) is used as the search database, the easiest way to download and run pgvector is using Docker:

```bash
mkdir postgres-data
docker run --rm -e POSTGRES_PASSWORD=postgres -p 54320:5432 -v `pwd`/postgres-data:/var/lib/postgresql/data pgvector/pgvector:pg17
```

As above, we run postgres on port `54320` to avoid conflicts with any other postgres instances you may have running.
We also mount the postgresql `data` directory locally to persist the data if you need to stop and restart the container.

With that running, we can build the search database with (**WARNING**: this requires the `OPENAI_API_KEY` env variable and will calling the OpenAI embedding API around 300 times to generate embeddings for each section of the documentation):

```bash
(uv run/python) -m pydantic_ai_examples.rag build
```

(Note building the database doesn't use Pydantic AI right now, instead it uses the OpenAI SDK directly.)

You can then ask the agent a question with:

```bash
(uv run/python) -m pydantic_ai_examples.rag search "How do I configure logfire to work with FastAPI?"
```

### `chat_app.py`

(Demonstrates: reusing chat history, serializing messages)

**TODO**: stream responses

Simple chat app example build with FastAPI.

This demonstrates storing chat history between requests and using it to give the model context for new responses.

Most of the complex logic here is in `chat_app.html` which includes the page layout and JavaScript to handle the chat.

Run the app with:

```bash
(uv run/python) -m pydantic_ai_examples.chat_app
```

Then open the app at [localhost:8000](http://localhost:8000).

### `stream_markdown.py`

(Demonstrates: streaming text responses)

This example shows how to stream markdown from an agent, using the `rich` library to display the markdown.

Run with:

```bash
(uv run/python) -m pydantic_ai_examples.stream_markdown
```

### `stream_whales.py`

(Demonstrates: streaming structured responses)

Information about whales — an example of streamed structured response validation.

This script streams structured responses from GPT-4 about whales, validates the data
and displays it as a dynamic table using Rich as the data is received.

Run with:

```bash
(uv run/python) -m pydantic_ai_examples.stream_whales
```
