# Data Analyst

Sometimes in an agent workflow, the agent does not need to know the exact tool
output, but still needs to process the tool output in some ways. This is
especially common in data analytics: the agent needs to know that the result of a
query tool is a `DataFrame` with certain named columns, but not
necessarily the content of every single row.

With Pydantic AI, you can use a [dependencies object](../dependencies.md) to
store the result from one tool and use it in another tool.

In this example, we'll build an agent that analyzes the [Rotten Tomatoes movie review dataset from Cornell](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes).


Demonstrates:

- [agent dependencies](../dependencies.md)


## Running the Example

With [dependencies installed and environment variables set](./setup.md#usage), run:

```bash
python/uv-run -m pydantic_ai_examples.data_analyst
```


Output (debug):


> Based on my analysis of the Cornell Movie Review dataset (rotten_tomatoes), there are **4,265 negative comments** in the training split. These are the reviews labeled as 'neg' (represented by 0 in the dataset).



## Example Code

```snippet {path="/examples/pydantic_ai_examples/data_analyst.py"}```


## Appendix

### Choosing a Model

This example requires using a model that understands DuckDB SQL. You can check with `clai`:

```sh
> clai -m bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
clai - Pydantic AI CLI v0.0.1.dev920+41dd069 with bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
clai ➤ do you understand duckdb sql?
# DuckDB SQL

Yes, I understand DuckDB SQL. DuckDB is an in-process analytical SQL database
that uses syntax similar to PostgreSQL. It specializes in analytical queries
and is designed for high-performance analysis of structured data.

Some key features of DuckDB SQL include:

 • OLAP (Online Analytical Processing) optimized
 • Columnar-vectorized query execution
 • Standard SQL support with PostgreSQL compatibility
 • Support for complex analytical queries
 • Efficient handling of CSV/Parquet/JSON files

I can help you with DuckDB SQL queries, schema design, optimization, or other
DuckDB-related questions.
```
