---
title: Memory
description: Persistent, namespaced agent notebooks with bounded prompt injection, on-demand search, and concurrency-safe stores.
---

# Memory

Give an agent a persistent notebook that it can update, search, and reuse across runs without loading every stored file into every prompt.

> [!NOTE]
> Import this capability from its submodule. It is not re-exported from `pydantic_ai_harness`:
>
> ```python
> from pydantic_ai_harness.memory import Memory
> ```

Memory is a released, non-experimental capability. Pydantic AI Harness is still on 0.x releases, so the API may change between minor releases. See the [version policy](index.md#version-policy).

## Notebook model

Memory gives each agent a notebook made of Markdown files:

- `MEMORY.md` is the main notebook. By default, a bounded excerpt and the names of other files are added to the current request as delimited user-role context.
- Other files hold longer or focused notes. The model reads them on demand or finds them with bounded text search.

The model gets four tools:

| Tool | Purpose |
| --- | --- |
| `write_memory` | Append to a file or replace one unique text fragment. Writes use optimistic concurrency and an idempotency identifier derived from the run and tool call. |
| `read_memory` | Read a bounded prefix of one memory file. |
| `delete_memory` | Delete a file. The main notebook is protected. |
| `search_memory` | Search across notebook files, subject to configured result, character, and file-scan limits. |

```python
from pydantic_ai import Agent
from pydantic_ai_harness.memory import FileStore, Memory

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Memory(FileStore('.agent-memory'))],
    defer_model_check=True,
)
```

The namespace is resolved by application code, not supplied to the tools. The model therefore cannot select another user's namespace in a tool call.

## Injection modes and limits

Automatic injection is enabled by default. Trusted usage guidance remains in model instructions, while model-written memory is enclosed in `<memory>` delimiters in a user-role part on the current request. Together, the guidance, main notebook, and file listing share a finite `max_tokens` budget, estimated at four characters per token. The default is 2,000 approximate tokens. `max_lines` is an additional limit on the main notebook. Backend reads are limited by `max_memory_size`, and the number of requested paths is derived from the prompt budget, so the capability never requests an unbounded file or listing. Content that does not fit is omitted with a prompt directing the model to use `read_memory` or `search_memory`.

```python
from pydantic_ai_harness.memory import FileStore, Memory

memory = Memory(
    FileStore('.agent-memory'),
    max_tokens=2_000,
    max_lines=200,
)
```

Only the current request retains the injected user-role part, so copies do not accumulate in message history. Each model request receives the latest bounded snapshot, including after `write_memory` or an external update changes `MEMORY.md`.

Set `inject_memory=False` for cache-stable prompts or durable workflows. The tools remain available, and the model can fetch memory only when it needs it:

```python
from pydantic_ai_harness.memory import FileStore, Memory

memory = Memory(FileStore('.agent-memory'), inject_memory=False)
```

With `injection_errors='ignore'` (the default), a store failure skips automatic injection and emits content-safe telemetry. Spans record the backend type and a hash of the resolved scope; successful injection records counts, and failures record the exception type. They do not record memory content. Set `injection_errors='raise'` when a run must fail rather than proceed without injected memory. Namespace and store resolver failures always propagate. Tool failures are still returned as tool errors; this setting controls automatic injection only.

## Persistence and concurrency

The store contract includes optimistic compare-and-swap mutations and idempotency. A write based on a stale revision fails with a conflict instead of overwriting a concurrent edit. Replaying the same run and tool call does not apply its mutation twice; reusing that operation identifier with different arguments raises `MemoryOperationConflictError`. These guarantees belong to the mutation operation, so custom stores must implement them atomically rather than composing separate read and write calls.

Every `MemoryStore.read` call includes a finite `max_chars`, and every `list_paths` call includes a finite `limit`. A store returns the bounded prefix plus `MemoryFile.truncated=True` when more content exists, while its version still represents the complete file. `read_memory` marks that bounded result as truncated. `write_memory` refuses to append or edit an oversized externally supplied file because doing so would derive new content from a partial read; remediate or replace it through the backing store first. A custom `SearchableMemoryStore.search` must likewise honor `max_file_chars` as well as the result limits.

| Store | Persistence and concurrency boundary |
| --- | --- |
| `InMemoryStore()` | Process lifetime; atomic across tasks using that store instance. |
| `FileStore(directory)` | Local filesystem; atomic Markdown replacement plus a hidden SQLite journal provide recovery, cross-process compare-and-swap, and durable idempotency receipts. |
| `SqliteMemoryStore(database=...)` | Durable single-host storage; compare-and-swap and idempotency are enforced in database transactions. |
| `PostgresMemoryStore(pool)` | Durable shared storage; compare-and-swap and idempotency are enforced in database transactions. The caller owns the pool lifecycle. |

```python
from pydantic_ai_harness.memory import FileStore, Memory, SqliteMemoryStore

local_memory = Memory(FileStore('.agent-memory'))
sqlite_memory = Memory(SqliteMemoryStore(database='.agent-memory.db'))
```

`SqliteMemoryStore` can instead use a caller-owned `sqlite3.Connection`. Because operations run off the event loop, create that connection with `check_same_thread=False` and manage its lifecycle in the application. The connection must be dedicated to the store and idle at the start of every operation; a call fails rather than commit or roll back an active caller transaction.

`FileStore` keeps the journal at `.memory-store.sqlite3` inside its root. Keep it with the Markdown files when copying or backing up the store. Editing a Markdown file outside the capability changes its content version and can produce a conflict with a prepared operation; the journal recovers operations interrupted between transaction preparation and filesystem replacement.

`PostgresMemoryStore` accepts the driver-neutral `PostgresPool` protocol, so the harness does not require a particular PostgreSQL driver. Install and manage the driver in your application (for example, `uv add asyncpg`):

```python
import asyncpg

from pydantic_ai_harness.memory import Memory, PostgresMemoryStore


async def build_memory() -> tuple[Memory[None], asyncpg.Pool]:
    pool = await asyncpg.create_pool('postgres://localhost/app')
    memory = Memory(PostgresMemoryStore(pool))
    return memory, pool
```

Call `build_memory` during application startup and close the returned pool during shutdown. The store does not manage it.

## Namespaces

Use a namespace resolver when one `Agent` serves multiple users. It runs once per run from your typed dependencies, and its result is hidden from the model-facing tool schema.

```python
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai_harness.memory import FileStore, Memory


@dataclass
class AppDeps:
    user_id: str


agent = Agent(
    'anthropic:claude-sonnet-4-6',
    deps_type=AppDeps,
    capabilities=[
        Memory(
            FileStore('/var/lib/myapp/memory'),
            namespace=lambda ctx: ctx.deps.user_id,
        )
    ],
    defer_model_check=True,
)
```

Namespace isolation controls which records the capability addresses. It is not an authorization system for a custom or shared backing store. Validate the identity in application dependencies, restrict backend credentials, and ensure custom stores cannot escape the resolved namespace.

## Search

`search_memory` performs literal text search and always applies three bounds:

- `max_search_results` limits returned matches, default 10.
- `max_search_result_chars` limits the combined scope-relative filename and snippet text, default 4,000 characters.
- `max_search_files` limits how many files a fallback scan may inspect, default 1,000.

The bundled stores implement `SearchableMemoryStore`. For a custom store that implements only `MemoryStore`, `search_memory` requests at most `max_search_files + 1` paths, scans at most `max_search_files`, and performs bounded reads. Lexical scoring uses only each scope-relative filename and its bounded content; tenant namespaces and agent names never affect relevance. Implement the optional search protocol for an indexed or semantic backend while preserving the same tenant boundary and result limits. Semantic ranking is not built in.

Before backend dispatch, queries are limited to 1,000 characters and 32 unique whitespace-separated terms. Repeated case-insensitive terms are collapsed so they cannot inflate scoring or scan work.

## Configuration

```python
from pydantic_ai_harness.memory import FileStore, Memory

Memory(
    FileStore('.agent-memory'),
    store_resolver=None,               # optional per-run store resolver
    agent_name='main',                 # agent segment inside the namespace
    namespace='',                      # string or per-run resolver
    inject_memory=True,                # False keeps prompts cache-stable
    max_tokens=2_000,                  # finite approximate total injection budget
    max_lines=200,                     # additional main-notebook line limit
    max_memory_size=65_536,            # per-file read, search, and write boundary
    max_search_results=10,
    max_search_result_chars=4_000,
    max_search_files=1_000,
    injection_errors='ignore',         # or 'raise'
    guidance=None,                     # None uses the default notebook guidance
)
```

## Agent specs

Register `Memory` as a custom capability type when constructing an agent from a Python spec:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.memory import Memory

agent = Agent.from_spec(
    {
        'model': 'anthropic:claude-sonnet-4-6',
        'capabilities': [
            {'Memory': {'backend': 'file', 'directory': '.agent-memory'}},
        ],
    },
    custom_capability_types=[Memory],
    defer_model_check=True,
)
```

The serializable backends are `memory`, `file`, and `sqlite`. A namespace callable and a live PostgreSQL pool must be configured in Python.

## Durable execution compatibility

| Execution mode | Support |
| --- | --- |
| Normal `Agent.run` calls | Supported with automatic injection or on-demand tools. |
| Temporal and Prefect | Use `inject_memory=False` with a statically configured store and on-demand tools. Automatic injection performs backend I/O in a model-request hook and is not workflow-safe. |
| DBOS | Normal execution works, but ordinary `FunctionToolset` calls are not DBOS-durable. Wrap memory operations in application-provided DBOS steps when durability is required. |

The memory backend and the workflow state backend are independent. Durable execution does not make an in-memory notebook persistent.

## Security and provenance

Memory is model-written, untrusted content that can re-enter future prompts. Keeping it in a delimited user-role part lowers its authority relative to model instructions, but this is not a hard prompt-injection boundary. Use `inject_memory=False` when less-trusted actors can write to the store, and expose memory only through application-controlled retrieval when stronger isolation is required. Do not store secrets unless the backend, retention policy, and access controls are appropriate. Sanitize content before rendering it into another trust domain.

Memory records do not carry source citations or verified provenance. If an application needs auditable facts, store provenance in the note itself or implement a custom store and schema. Optimistic concurrency prevents lost updates; it does not establish that a remembered claim is true.

## API reference

- [`pydantic_ai_harness.memory` source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/memory/)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Pydantic AI hooks](/ai/core-concepts/hooks/)

The public module exports `Memory`, `MemoryToolset`, the bundled stores, the store protocols, mutation and search result models, and conflict exceptions. Import them from `pydantic_ai_harness.memory`.

::: pydantic_ai_harness.memory.Memory

::: pydantic_ai_harness.memory.MemoryToolset
