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
protected), and `search_memory(query)`. With a `FileStore` in the run's workspace (no `workspace=`,
relative `directory`), `FileSystem` on the agent and `inject_memory=True` (the default), `read_memory` is left out
(`unless_tool_kind=READ_FILE_KIND`, fixed for the run) and an instruction block named `directory`
(`on_change='append'`, present and empty from the first request until a file reader is on the run; its text
counts against `max_tokens`) gives the scope's directory, e.g. `.agent-memory/main/`. The guidance names no reader, so it stays static. By default a bounded excerpt of `MEMORY.md` plus the list
of other files is recorded in message history as a delimited user-role `<memory>` part, with the
user's prompt on a run's first request, and only when memory changed since the latest snapshot in
history (after the latest compaction boundary). History stays append-only: earlier snapshots are kept,
a later one for the same memory supersedes them, and a
change made during a run (including `write_memory`) reaches the model with the next user prompt,
not after tool results (the exception: re-enqueuing the run's own snapshot when compaction drops
it mid tool loop). It is only added before the model's first response in a run, so a deferred `Memory`
loaded mid-run waits for the next prompt. Snapshots carry `TextContent.metadata`
`{'origin': 'pydantic-ai-harness.memory.v1', 'scope': ..., 'id': ...}`; list compaction before `Memory`.

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

### Write metadata (who wrote a file)

When the app's UI also edits memory, record provenance with the write instead of in the note.
`InMemoryStore`, `SqliteMemoryStore` and `PostgresMemoryStore` implement `MetadataMemoryStore`:
`store.write_with_metadata(path, content, expected_version=..., metadata={'updated_by': 'ui'})`
stores a `Mapping[str, str]` (at most 4,096 characters as JSON) in the content's transaction
(Postgres: a queryable `write_metadata JSONB` column), and
`store.read(...).metadata` returns it. `Memory(write_metadata=...)` (a mapping or a
`RunContext -> Mapping | None` callable, resolved per write, never a tool argument) records it
with the model's writes.

- Every write replaces the metadata; a plain `write`, including the model's writes when
  `write_metadata` is unset, leaves none, so `updated_by='ui'` cannot outlive an agent edit.
  Deletes remove it; a replayed tool call returns its receipt and leaves the stored metadata as is.
- Files live at `{namespace}/{agent_name}/{file}`, so a UI edits `{user_id}/main/MEMORY.md`.
- `write_metadata` with a store that is not a `MetadataMemoryStore` (`FileStore`, or a custom
  store implementing only `MemoryStore`) raises `UserError` at construction or run start.
- A custom store opts in by implementing `write_with_metadata` (checked structurally, like
  `search`), writing metadata atomically with the content; its `write` must record none.

### Bounded injection and search

- `max_tokens=2_000` (approx, 4 chars/token) bounds guidance + notebook + file list together;
  `max_lines=200` additionally limits `MEMORY.md`; `max_memory_size=65_536` bounds per-file read,
  search, and write. Overflow is omitted with a pointer to `read_memory` (or `read_file` when it replaces it) / `search_memory`.
- `inject_memory=False` keeps memory out of the conversation; the tools remain. Also use it when
  less-trusted actors can write to the store (memory is untrusted, model-written content).
- `injection_errors='ignore'` (default) skips injection on a store failure; `'raise'` fails the run.
- `search_memory` is literal, bounded text search: `max_search_results=10`,
  `max_search_result_chars=4_000`, `max_search_files=1_000`. Queries are capped at 1,000 characters
  and 32 unique terms. No semantic ranking is built in; implement `SearchableMemoryStore.search`
  for an indexed backend.
- Other options: `agent_name='main'` (storage segment, never shown to the model), `heading=''`
  (labels the guidance and injected block), `guidance=None` (replace the default usage text).

### Several memories on one agent

Give each instance a distinct `agent_name` or `namespace` (the storage scope: instances with the same
scope and store share one notebook, though each still tracks its own snapshots by `id`), a distinct `heading`, and a distinct `id` (all default to `'memory'`; duplicates raise
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
  `SqliteStepStore`, `MongoStepStore`, `PostgresStepStore` all work. A custom `HistorySource` must
  populate
  `conversation_id` on its `RunRecord`s.
- Recovery of compaction-dropped messages depends on pre-compaction snapshots still being
  retained (`max_snapshots_per_run` can prune them). The corpus is rebuilt on every call, so cost
  grows with in-scope history.
- History a run inherits through `message_history` is indexed under the run that first
  persisted it; a request a later run adds an instruction change to is indexed again with that
  change, so its other parts can match twice. Excerpts are clipped to 500 characters; when the matched term lies past the cutoff,
  the excerpt shows the text around it. The `\w+` tokenizer cannot match a word inside unspaced
  CJK or Thai text.

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

Signature: `Skills(directories, *, include=None, exclude=None, missing_directories='error',
duplicate_names='error', unknown_names='error', workspace=None)`. `directories` is one path or a sequence of library
directories (not individual skill folders), in precedence order.

To layer project and personal libraries as coding agents do, pass
`missing_directories='skip'` (absent folders are left out quietly; a file, or a skill package, in
their place is left out with a `UserWarning`) and `duplicate_names='keep_first'` (the earlier directory's valid skill wins, the other is skipped with
a `UserWarning`).

A host that lets a person invoke a skill (a `/code-review src/app.py` command) reads the catalog
with `catalog = await skills.load(LocalWorkspaceBackend('.'))` and sends
`skill.render(arguments)` as the prompt: `$ARGUMENTS` in the body is substituted, else
`ARGUMENTS: ...` is appended (indexed forms like `$0` and `$ARGUMENTS[0]` are left as-is).
`catalog.skipped` lists the `SKILL.md` files and library paths left out as `SkippedSkill(path,
reason, message)`, `reason` one of `'invalid'`, `'duplicate_name'`, `'not_a_directory'`,
`'skill_package'`, for the host to show; `load` emits no warnings itself.

Frontmatter `disable-model-invocation: true` keeps a skill out of the model's catalog
(`SkillDefinition.model_invocable` is `False`) but `load` still returns it, for a host command.
`user-invocable: false` sets `SkillDefinition.user_invocable` to `False`, for a host to offer no
command; the model still sees it. Either field with a value other than `true` or `false` skips the skill.

Gotchas:
- Without PyYAML (the `skills` extra) importing the loader raises `ImportError`.
- Reads from the run's workspace; no workspace fails the run at start. Use
  `workspace=LocalWorkspaceBackend('/app')` to read skills shipped with the app while the agent
  works in a sandbox.
- No auto-discovery of `.agents`, `.claude`, or `~`: pass every library explicitly, and spell out
  a home path (workspace paths never expand `~`). Only immediate child directories with a
  `SKILL.md` count. A skill found twice (symlink or byte-identical `SKILL.md`; bundled files are
  not compared) counts once.
- `include` and `exclude` are mutually exclusive; unknown names fail at run start, and so do
  duplicate selected names and missing library paths with the default `'error'` options.
  `unknown_names='skip'` ignores names no library has, for a selection shared across projects.
  Malformed frontmatter or a name not matching its directory warns and skips that skill.
- With no skills found, `Skills` adds no instructions and no `load_capability` tool.
- `description` is required. The loaded instructions name the skill's directory in the run's
  workspace (not for a `workspace=` library), so a model with file or shell tools can follow
  `references/` or run `scripts/`; `Skills` itself never reads or
  runs them, and `${CLAUDE_SKILL_DIR}` is not substituted. Fields like `allowed-tools`, `model`,
  `hooks` are accepted but not implemented (one aggregated `UserWarning`).
- Skill bodies become model instructions: load only trusted libraries. `include`/`exclude` are
  catalog selection, not access control.
- For instructions defined in Python, use core
  `Capability(id=..., description=..., instructions=..., defer_loading=True)` instead.
- In agent specs: `Skills: {directories: .agents/skills, include: [...]}` with
  `custom_capability_types=[Skills]`.

## PydanticAIDocs

Adds one tool, `read_pyai_docs(topic)`, returning a Pydantic AI docs page with its model aliases
replaced by the models they name, plus a short cache-stable instruction to read docs before writing
capabilities, hooks, tools, or toolsets. It serves only these six Pydantic AI pages
(`PydanticAIDocsTopic`), not your own docs: `capabilities`, `hooks`, `tools`, `tools-advanced`,
`toolsets`, `agent`.

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
