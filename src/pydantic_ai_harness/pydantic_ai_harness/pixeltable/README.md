# Pixeltable

`Pixeltable` gives an agent read-only tools over existing [Pixeltable](https://pixeltable.com) tables
and views: `list_tables`, `describe_table`, `query_table` (equality filters), and `similarity_search`
over an embedding index. `PixeltableMemoryStore` keeps [Memory](../memory/) in a Pixeltable table, so
the agent's notes sit in the same catalog as the data it searches. Each works without the other.

[Catalog source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pixeltable/)
and [memory store source](https://github.com/pydantic/pydantic-ai/blob/main/src/pydantic_ai_harness/pydantic_ai_harness/memory/_pixeltable.py)

## The problem

Pixeltable keeps tables, views, document chunks, and media together with the embedding indexes and
computed columns derived from them. An agent that should answer from that data needs tools that list
and describe tables, filter rows, and run similarity search, with limits on what it can reach and on
how much text comes back. Writing those tools per project repeats the same allowlist, bounding, and
error handling.

## Usage

Install the `pixeltable` extra (Python 3.11 or later). The example uses an OpenAI model and OpenAI
embeddings, so it also pulls in the `openai` provider, plus `spec` for the YAML agent spec below:

uv:

```bash
uv add "pydantic-ai-harness[pixeltable]" "pydantic-ai-slim[openai,spec]"
```

pip:

```bash
pip install "pydantic-ai-harness[pixeltable]" "pydantic-ai-slim[openai,spec]"
```

A table with an embedding index, created once (in an application, declare it on a Pixeltable
`TableModel` and apply it with `pxt schema update`):

```python
import pixeltable as pxt
from pixeltable.functions.openai import embeddings

pxt.create_dir('hr', if_exists='ignore')
handbook = pxt.create_table('hr.handbook', {'topic': pxt.String, 'text': pxt.String}, if_exists='ignore')
handbook.insert([{'topic': 'expenses', 'text': 'Meals during business travel are reimbursed up to 60 EUR per day.'}])
handbook.add_embedding_index('text', embedding=embeddings.using(model='text-embedding-3-small'), if_exists='ignore')
```

An agent that searches it and keeps notes across runs:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Memory
from pydantic_ai_harness.pixeltable import Pixeltable
from pydantic_ai_harness.memory import PixeltableMemoryStore

agent = Agent(
    'openai:gpt-5.6-sol',
    capabilities=[
        Pixeltable(['hr.handbook']),
        Memory(PixeltableMemoryStore(table_name='hr.memory')),
    ],
)
result = agent.run_sync('Can I expense a 75 EUR dinner? Remember that I travel monthly.')
print(result.output)
#> ...
```

## Catalog tools

| Tool | What it does |
|---|---|
| `list_tables` | The allowed table and view paths. |
| `describe_table` | Kind, comment, columns (type, `is_computed`, `is_stored`), and indexes. |
| `query_table` | Rows matching equality filters (`{"status": "open"}`); timestamp, date, and UUID values are ISO strings. |
| `similarity_search` | Nearest rows by `column.similarity(string=query)`, with a similarity score in the result. |

- `tables` is a required allowlist of table paths or directory prefixes; `['*']` allows the whole
  catalog, including any memory table. A view inside an allowed directory exposes its base table's
  columns. Version handles (`'dir.tbl:3'`) are refused, since an old version keeps rows deleted and
  columns dropped since.
- Default columns skip media, array, and binary columns, including when searching an indexed media
  column. Computed columns that are not stored rerun their function (possibly a model call) on every
  read, so the tools skip them by default and reject them in `columns` and `where`. Explicit media
  columns are rejected because their local file paths are not usable by a remote model.
- `max_rows` (default 20) and `max_chars` (default 8000) bound the results of `query_table` and
  `similarity_search`. An oversized string is cut to end in `...`, any other oversized value becomes
  `null`, and `truncated` is set. Both return the `{"table", "rows", "truncated"}` envelope.
  `list_tables` and `describe_table` return their own shapes, sized by the allowlist and the schema.
- Invalid requests, missing tables, and transient provider errors become
  [`ModelRetry`](https://ai.pydantic.dev/tools-toolsets/tools-advanced/#tool-retries); infrastructure
  and authorization failures propagate to the application.
- The default instructions tell the model to describe unfamiliar tables first and to treat table
  contents as untrusted data. Pass `guidance='...'` to replace them, or `guidance=''` to add none.

## Memory store

`PixeltableMemoryStore(table_name='harness.memory')` implements `MemoryStore` and
`SearchableMemoryStore`. The table is created on first use.

- Writes use compare-and-set versions and operation receipts, so a retry can recover or replay a
  mutation without applying it twice. Paths are limited to 255 characters and operation ids to 248;
  the `__op__` and `__meta__` path roots are reserved.
- `search_memory` uses the same lexical scoring and prefix isolation as the other stores.
- `store.table` supports queries, joins, and computed columns; filter to `kind == 'file'` for memory
  rows. Route writes and deletes through the store to preserve versions and receipts.
- Pixeltable keeps old row versions for every update and delete, and receipts are not pruned, so the
  table grows with history.

## Multiple instances

Two `Pixeltable` capabilities on one agent share `id='pixeltable'` and merge by intersecting their
allowlists, keeping the smaller `max_rows` and `max_chars`; a disjoint merge raises `ValueError`. A
capability passed to a single run replaces the agent's instead of merging with it, so it can widen
what that run reaches.

## Telemetry

The package adds no spans; core records tool calls and `Memory` emits `memory.*` spans.

## Agent spec (YAML/JSON)

`Pixeltable` works with Pydantic AI's [agent spec](https://ai.pydantic.dev/core-concepts/agent-spec/). The allowlist is
its one positional argument, so the short form passes it alone:

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - Pixeltable: ['hr.handbook']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.pixeltable import Pixeltable

agent = Agent.from_file('agent.yaml', custom_capability_types=[Pixeltable])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate it.
`PixeltableMemoryStore` is Python-only; the `Memory` spec backends are `memory`, `file`, and `sqlite`.
