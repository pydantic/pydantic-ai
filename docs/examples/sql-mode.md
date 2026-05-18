Example of [SQL Mode](../harness/sql-mode.md) from the Pydantic AI Harness: the model orchestrates tool calls by writing a single SQL query.

Demonstrates:

- [SQL Mode](../harness/sql-mode.md)
- [capabilities](../capabilities.md)

The `SQLMode` capability exposes the agent's `geocode` and `get_forecast` tools as DuckDB functions inside a single `run_sql` tool. `geocode` returns a nested object — coordinates as a sub-object, a structured address, and population. Asked to build a table across five cities, the model writes one SQL query that fans `geocode` over the list with `unnest`, navigates the nested JSON it returns (`->`/`->>`) to read the country, population, and coordinates, feeds the coordinates into `get_forecast`, and aggregates the result — one model round-trip instead of ten tool calls.

## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.sql_mode
```

## Example Code

```snippet {path="/examples/pydantic_ai_examples/sql_mode.py"}```
