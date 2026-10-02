# Knowledge and Memory

Capabilities that give an agent knowledge beyond its prompt: `Memory` (a persistent Markdown
notebook the model writes and searches across runs), `ConversationSearch` (BM25 recall over
history `StepPersistence` already stored, including turns compaction dropped), `Skills` (Agent
Skills `SKILL.md` files loaded on demand), and `PydanticAIDocs` (fetch Pydantic AI docs pages on
demand). Only `Skills` needs an extra.

| Need | Use |
|---|---|
| Facts/preferences that persist across runs or sessions | `Memory` |
| Exact details from earlier turns or past runs of a conversation | `ConversationSearch` + `StepPersistence` |
| Specialized instructions the model pulls in only when relevant | `Skills` |
| Agent that writes Pydantic AI code and needs current docs | `PydanticAIDocs` |

## Memory

Gives the model four tools: `write_memory(content, file='MEMORY.md', old_text=None)` (append, or
replace one unique fragment), `read_memory(file)`, `delete_memory(file)` (`MEMORY.md` is
protected), and `search_memory(query)`. By default a bounded excerpt of `MEMORY.md` plus the list
of other files is injected into each request as a delimited user-role part.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import InMemoryStore

agent = Agent('test', capabilities=[Memory(InMemoryStore())])
```

The first positional argument is the store; it defaults to `InMemoryStore()` (process lifetime
only).

### Stores

| Store | Persistence | Notes |
|---|---|---|
| `InMemoryStore()` | Process lifetime | Tests, ephemeral agents |
| `FileStore(directory, *, workspace=None)` | Markdown files in the run's workspace | Needs a workspace or `workspace=` backend; one writer per directory |
| `SqliteMemoryStore(database=... or connection=...)` | Durable, single host | A shared `connection` needs `check_same_thread=False` and must be dedicated to the store |
| `PostgresMemoryStore(pool, *, table='agent_memory')` | Durable, shared | Driver-neutral `PostgresPool` protocol (for example an `asyncpg` pool); you own the pool lifecycle; no harness extra |

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import FileStore

agent = Agent(
    'test',
    capabilities=[LocalWorkspace('.'), Memory(FileStore('.agent-memory'))],
)
```

```python {test="skip"}
import asyncpg

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import PostgresMemoryStore


async def build_memory() -> tuple[Memory[None], asyncpg.Pool]:
    pool = await asyncpg.create_pool('postgres://localhost/app')
    return Memory(PostgresMemoryStore(pool)), pool  # close the pool on shutdown
```

Gotchas:
- `FileStore` without an attached workspace fails the run at start (`UserError`). To keep memory
  on the host while the agent works in a sandbox, pass
  `FileStore('.', workspace=LocalWorkspaceBackend('/var/lib/app/memory'))` (a backend, not the
  `LocalWorkspace` capability). With a read-only run workspace, write/delete tools are hidden.
- Writes use optimistic compare-and-swap plus an idempotency id derived from run and tool call;
  a stale write returns a conflict to the model instead of overwriting. Custom stores must
  implement `MemoryStore` mutations atomically.
- `FileStore` keeps replay receipts in `.memory-operations.json`; for concurrent writers across
  processes use `SqliteMemoryStore` or `PostgresMemoryStore`.

### Namespaces (multi-user)

`namespace` is a string or a per-run resolver from `RunContext`. It is resolved by your code and
never exposed in the tool schema, so the model cannot pick another user's namespace. It is not an
authorization system: validate identity in deps and scope backend credentials.

```python
from dataclasses import dataclass

from pydantic_ai import Agent

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import InMemoryStore


@dataclass
class AppDeps:
    user_id: str


agent = Agent(
    'test',
    deps_type=AppDeps,
    capabilities=[Memory(InMemoryStore(), namespace=lambda ctx: ctx.deps.user_id)],
)
```

`store_resolver=` (a `RunContext -> MemoryStore` callable) picks a store per run. Under Temporal or
Prefect both `store_resolver` and a callable `namespace` must be deterministic and do no backend
I/O.

### Bounded injection and search

- `max_tokens=2_000` (approx, 4 chars/token) bounds guidance + notebook + file list together;
  `max_lines=200` additionally limits `MEMORY.md`; `max_memory_size=65_536` bounds per-file read,
  search, and write. Overflow is omitted with a pointer to `read_memory` / `search_memory`.
- `inject_memory=False` keeps the prompt cache-stable; the tools remain. Also use it when
  less-trusted actors can write to the store (memory is untrusted, model-written content).
- `injection_errors='ignore'` (default) skips injection on a store failure; `'raise'` fails the run.
- `search_memory` is literal, bounded text search: `max_search_results=10`,
  `max_search_result_chars=4_000`, `max_search_files=1_000`. Queries are capped at 1,000 characters
  and 32 unique terms. No semantic ranking is built in; implement `SearchableMemoryStore.search`
  for an indexed backend.
- Other options: `agent_name='main'` (storage segment, never shown to the model), `heading=''`
  (labels the guidance and injected block), `guidance=None` (replace the default usage text).

### Several memories on one agent

Give each instance a distinct `agent_name` or `namespace` (otherwise their injections replace each
other), a distinct `heading`, and a distinct `id` (all default to `'memory'`; duplicates raise
`UserError` at `Agent` construction), and wrap all but one with `.prefix_tools(...)` so tool
names do not collide:

```python
from pydantic_ai import Agent

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import InMemoryStore

store = InMemoryStore()
agent = Agent(
    'test',
    capabilities=[
        Memory(store, heading='Your notes'),
        Memory(
            store, agent_name='org', heading='Org notes', id='org_memory'
        ).prefix_tools('org'),
    ],
)
```

Agent specs: `{'Memory': {'backend': 'file', 'directory': '.agent-memory'}}` with
`custom_capability_types=[Memory]`. Serializable backends are `memory`, `file`, `sqlite`; Postgres
and callable namespaces need Python. Keep the default `id='memory'` for durable recovery.

## ConversationSearch

Adds one tool, `search_conversation_history(query, run_id=None)`, that BM25-ranks history a
persistence capability already stored. It persists nothing itself. Pair it with `StepPersistence`
on the same store instance via `SnapshotHistorySource`.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import (
    ConversationSearch,
    SlidingWindowCompaction,
    StepPersistence,
)
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import InMemoryStepStore

store = InMemoryStepStore()
agent = Agent(
    'test',
    capabilities=[
        StepPersistence(store=store),
        ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
        SlidingWindowCompaction(max_messages=80, keep_messages=40),
    ],
)
```

Key parameters: `source` (required; a `HistorySource`), `scope` (`'conversation'` or `'all'`),
`max_matches=10`, `context_lines=5`, `bm25_k1=1.5`, `bm25_b=0.75`, `add_instructions=True`,
`tool_id='conversation-search'`.

Gotchas:
- Always set `scope` explicitly. Unset resolves to `'conversation'` and emits a
  `HarnessDeprecationWarning` once per instance. There is no per-run store resolver or tenant
  filter: to search one tenant's earlier conversations, give each tenant its own step store and pass
  both capabilities per run with `scope='all'`:
  `agent.run(q, capabilities=[StepPersistence(store=s), ConversationSearch(SnapshotHistorySource(s), scope='all')])`.
- `'conversation'` matches runs by `conversation_id`. Pass an authenticated, tenant-scoped
  `agent.run(..., conversation_id=...)`; follow-up runs that thread `message_history` inherit it,
  a run with neither gets a fresh id and searches only itself.
- `SnapshotHistorySource(store)` raises `TypeError` at construction if the store lacks
  `list_runs` / `list_snapshots`; the shipped `InMemoryStepStore`, `FileStepStore`,
  `SqliteStepStore`, `MongoStepStore` all work. A custom `HistorySource` must populate
  `conversation_id` on its `RunRecord`s.
- Recovery of compaction-dropped messages depends on pre-compaction snapshots still being
  retained (`max_snapshots_per_run` can prune them). The corpus is rebuilt on every call, so cost
  grows with in-scope history.

## Skills

Loads Agent Skills (`<library>/<skill>/SKILL.md`) as deferred capabilities: the model sees each
skill's name and description and loads the body by calling core's `load_capability` tool with
`{'id': '<skill name>'}`.

```bash
uv add "pydantic-ai-harness[skills]"
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import Skills

agent = Agent(
    'test',
    capabilities=[LocalWorkspace('.'), Skills('.agents/skills', include=['code-review'])],
)
```

Signature: `Skills(directories, *, include=None, exclude=None, workspace=None)`. `directories` is
one path or a sequence of library directories (not individual skill folders).

Gotchas:
- Without PyYAML (the `skills` extra) importing the loader raises `ImportError`.
- Reads from the run's workspace; no workspace fails the run at start. Use
  `workspace=LocalWorkspaceBackend('/app')` to read skills shipped with the app while the agent
  works in a sandbox.
- No auto-discovery of `.agents`, `.claude`, or `~`: pass every library explicitly. Only immediate
  child directories with a `SKILL.md` count.
- `include` and `exclude` are mutually exclusive; unknown names, duplicate selected names across
  libraries, and missing library paths fail at run start. Malformed frontmatter or a name not
  matching its directory warns and skips that skill.
- `description` is required. Bundled `references/`, `scripts/`, `assets/` are never read or run,
  and `${CLAUDE_SKILL_DIR}` is not substituted. Fields like `allowed-tools`, `model`, `hooks`
  are accepted but not implemented (one aggregated `UserWarning`).
- Skill bodies become model instructions: load only trusted libraries. `include`/`exclude` are
  catalog selection, not access control.
- For instructions defined in Python, use core
  `Capability(id=..., description=..., instructions=..., defer_loading=True)` instead.
- In agent specs: `Skills: {directories: .agents/skills, include: [...]}` with
  `custom_capability_types=[Skills]`.

## PydanticAIDocs

Adds one tool, `read_pyai_docs(topic)`, returning a Pydantic AI docs page verbatim, plus a short
cache-stable instruction to read docs before writing capabilities, hooks, tools, or toolsets. It
serves only these six Pydantic AI pages (`PydanticAIDocsTopic`), not your own docs: `capabilities`,
`hooks`, `tools`, `tools-advanced`, `toolsets`, `agent`. The `capabilities` topic maps to
`docs/capabilities.md`, which the docs no longer have (the page is now `docs/capabilities/overview.md`),
so that topic only works from a local path that provides the file.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import PydanticAIDocs

agent = Agent('test', capabilities=[PydanticAIDocs()])
```

Parameters: `local_docs_path: Path | None = None`, `cache: bool = True` (memoize per run).
Resolution per call: `{local_docs_path}/{topic}.md` in the run workspace (falling back to the
`PYDANTIC_AI_HARNESS_DOCS_PATH` env var), then
`https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/{topic}.md`, else an error.

Gotchas: the local path is read through the run workspace only, with no `workspace=` override (in
a sandbox the checkout must be in the sandbox); with a local path set and no workspace, the run fails at start. With no
local path, the first read of each topic in a run fetches it from GitHub (every read with
`cache=False`). `~` is not expanded. It never runs git, so keep a local checkout current yourself. Use `pydantic_ai_harness.pydantic_ai_docs`; the old `docs`
module and `PyaiDocs` names are deprecated (keep `PyaiDocs` only to load old specs).

## See also

- https://pydantic.dev/docs/ai/harness/memory/
- https://pydantic.dev/docs/ai/harness/conversation-search/
- https://pydantic.dev/docs/ai/harness/skills/
- https://pydantic.dev/docs/ai/harness/pydantic-ai-docs/
- https://pydantic.dev/docs/ai/harness/step-persistence/
- https://pydantic.dev/docs/ai/capabilities/on-demand/
