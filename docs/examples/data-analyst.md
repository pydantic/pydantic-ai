# Data Analyst (Referencing Variables in `RunContext` like Jupyter)

Sometimes in an agent workflow, the agent does not need to know exact the tool
output, but they need to process the tool output in some ways. This is
especially common in data analytics. The agent needs to know the result of a
query (tool result) is a DataFrame with certain named columns, but not
necessarily the content of every single row.

With `pydantic-ai`, [`RunContext`][pydantic_ai.tools.RunContext] can be used to
store output and pass the reference to the other tool. Here is an example of
building an agent to analyze [Rotten Tomatoes movie review dataset from Cornell](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
on Hugging Face Hub.


Demonstrates:
- [structured `output_type`](../output.md#structured-output)
- [agent dependencies](../dependencies.md)


## Running the Example


```bash
python/uv-run -m pydantic_ai_examples.data_analyst
```


## Example Code

```snippet {path="/examples/pydantic_ai_examples/data_analyst.py"}```


Output (debug):

```
examples/pydantic_ai_examples/data_analyst.py:103 <module>
    result.output: (
        'Based on my analysis of the cornell-movie-review-data/rotten_tomatoes dataset, there are **4,265 negative com'
        'ments** in the training split. The dataset is balanced, with an equal number of positive comments.'
    ) (str) len=207
```
## Appendix

### Choosing a Model

This example requires using a model that understands DuckDB SQL. I am using
Anthropic Claude 3.7. You can use other models but check with `clai`

```sh
> clai -m bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
clai - Pydantic AI CLI v0.0.1.dev920+41dd069 with bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
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
