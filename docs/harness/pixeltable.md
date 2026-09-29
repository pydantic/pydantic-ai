---
title: Pixeltable
description: Give a Pydantic AI agent read-only, allowlisted tools over existing Pixeltable tables and embedding indexes, and keep Harness Memory in a Pixeltable table.
---

# Pixeltable

`Pixeltable` gives an agent read-only tools over existing [Pixeltable](https://pixeltable.com) tables
and views: `list_tables`, `describe_table`, `query_table` (equality filters), and `similarity_search`
over an embedding index. `PixeltableMemoryStore` keeps [Memory](memory.md) in a Pixeltable table, so
the agent's notes sit in the same catalog as the data it searches. Each works without the other.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/pixeltable/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Pixeltable keeps tables, views, document chunks, and media together with the embedding indexes and
computed columns derived from them. An agent that should answer from that data needs tools that list
and describe tables, filter rows, and run similarity search, with limits on what it can reach and on
how much text comes back. Writing those tools per project repeats the same allowlist, bounding, and
error handling.

## Usage

Install the `pixeltable` extra (Python 3.11 or later). The example uses an OpenAI model and OpenAI
embeddings, so it also pulls in the `openai` provider, plus `spec` for the YAML agent spec below:

```bash
pip/uv-add "pydantic-ai-harness[pixeltable]" "pydantic-ai-slim[openai,spec]"
```

A table with an embedding index, created once (in an application, declare it on a Pixeltable
`TableModel` and apply it with `pxt schema update`):

```python {test="skip" py="3.11"}
import pixeltable as pxt
from pixeltable.functions.openai import embeddings

pxt.create_dir('hr', if_exists='ignore')
handbook = pxt.create_table('hr.handbook', {'topic': pxt.String, 'text': pxt.String}, if_exists='ignore')
handbook.insert([{'topic': 'expenses', 'text': 'Meals during business travel are reimbursed up to 60 EUR per day.'}])
handbook.add_embedding_index('text', embedding=embeddings.using(model='text-embedding-3-small'), if_exists='ignore')
```

An agent that searches it and keeps notes across runs:

```python {test="skip" py="3.11"}
from pydantic_ai import Agent
from pydantic_ai_harness import Memory
from pydantic_ai_harness.pixeltable import Pixeltable, PixeltableMemoryStore

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
| `similarity_search` | Nearest rows by `column.similarity(string=query)`, with a `score` field. If the table has a column by that name, the field takes `similarity_` prefixes until it is free (`similarity_score`, then `similarity_similarity_score`). |

- `tables` is a required allowlist of table paths or directory prefixes; `['*']` allows the whole
  catalog, including any memory table. A view inside an allowed directory exposes its base table's
  columns. Version handles (`'dir.tbl:3'`) are refused, since an old version keeps rows deleted and
  columns dropped since.
- Default columns skip media, array, and binary columns. Computed columns that are not stored rerun
  their function (possibly a model call) on every read, so the tools skip them by default and reject
  them in `columns` and `where`. A named media column returns a file URL.
- `max_rows` (default 20) and `max_chars` (default 8000) bound the results of `query_table` and
  `similarity_search`. An oversized string is cut to end in `...`, any other oversized value becomes
  `null`, and `truncated` is set. Both return the `{"table", "rows", "truncated"}` envelope.
  `list_tables` and `describe_table` return their own shapes, sized by the allowlist and the schema.
- Pixeltable errors become [`ModelRetry`](../tools-advanced.md#tool-retries) with a hint
  such as "Call describe_table", so the model can correct its call.
- The default instructions tell the model to describe unfamiliar tables first and to treat table
  contents as untrusted data. Pass `guidance='...'` to replace them, or `guidance=''` to add none.

## Memory store

`PixeltableMemoryStore(table_name='harness.memory')` implements `MemoryStore` and
`SearchableMemoryStore`. The table is created on first use.

- Each memory path is a `kind == 'file'` row. Writes are compare-and-set on a UUID version, enforced by
  a conditional `update` or `delete` and the primary key on `path`.
- A write that carries an operation id journals its intent as an `__op__/<id>` row before applying
  it. A crash rolls forward on replay; a writer that loses the compare-and-set withdraws an intent that
  no peer claimed, so a retry does not apply it twice. The path roots `__op__` and `__meta__` are
  reserved, paths are at most 255 characters, and operation ids at most 248.
- `search_memory` uses the same lexical scoring as the other stores, over the files under the
  tenant's prefix. Listing and search order paths with the `"C"` collation in SQL, so the database
  applies the bound in code point order whatever its default collation is.
- `store.table` is an ordinary Pixeltable table: query it, join it with application data, or add an
  embedding index on `content`. A direct `update` of `content` is not a Memory write and leaves the
  version unchanged.
- Pixeltable keeps old row versions for every update and delete, and receipts are not pruned, so the
  table grows with history.

## Multiple instances

Two `Pixeltable` capabilities on one agent share `id='pixeltable'` and merge by intersecting their
allowlists, keeping the smaller `max_rows` and `max_chars`; a disjoint merge raises `ValueError`. A
capability passed to a single run replaces the agent's instead of merging with it, so it can widen
what that run reaches.

## Telemetry

The package adds no spans. Core's tool-call spans already record each catalog tool call, and
`Memory` emits the `memory.*` spans for store operations.

## Agent spec (YAML/JSON)

`Pixeltable` works with Pydantic AI's [agent spec](../agent-spec.md). The allowlist is
its one positional argument, so the short form passes it alone:

```yaml
# agent.yaml
model: openai:gpt-5.6-sol
capabilities:
  - Pixeltable: ['hr.handbook']
```

```python {test="skip" py="3.11"}
from pydantic_ai import Agent
from pydantic_ai_harness.pixeltable import Pixeltable

agent = Agent.from_file('agent.yaml', custom_capability_types=[Pixeltable])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate it.
`PixeltableMemoryStore` is Python-only; the `Memory` spec backends are `memory`, `file`, and `sqlite`.

## API reference

::: pydantic_ai_harness.pixeltable.Pixeltable

::: pydantic_ai_harness.pixeltable.PixeltableToolset

::: pydantic_ai_harness.pixeltable.PixeltableMemoryStore
