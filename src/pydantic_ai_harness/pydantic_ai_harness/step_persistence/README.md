# Step Persistence

> [!NOTE]
> The `media` stores are not re-exported at the top level -- import them from their submodule:
>
> ```python
> from pydantic_ai_harness import StepPersistence
> from pydantic_ai_harness.media import S3MediaStore
> ```
>
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

`StepPersistence` records what an agent did at each boundary, separate from
whether the run can be safely resumed. It is the persistence substrate for
orchestrators that delegate to sub-agents (e.g. an AICA orchestrator spawning
a `code_librarian` to investigate one symbol, then continuing that delegate's
investigation with a follow-up question).

It is not a full graph-state checkpoint. Capability-state restore, workspace
snapshots, and graph-node resume are out of scope and tracked separately
(see `pydantic-ai-harness` issues #149 and #196).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/step_persistence/)

## What it gives you

1. **Append-only step events** -- every interesting boundary (run start/end,
   model request, tool call, failure) appends a `StepEvent`. A run killed
   mid-tool-call still leaves a usable event trail.
2. **Continuable snapshots** -- a `ContinuableSnapshot` is saved at settled
   node boundaries, and a failing run saves its live at-failure history.
   Each snapshot carries a `state`: `complete` when every `ToolCallPart`
   has a matching result, `interrupted` when the capture holds unsettled
   tool work (e.g. a crash mid-tool-cycle). `latest_snapshot` and
   `continue_run` return only `complete` snapshots unless the caller passes
   `include_interrupted=True`. Pass the snapshot's `messages` back to
   `Agent.run(message_history=...)` to continue or fork.
3. **Tool-effect ledger** -- every tool call's lifecycle (`started`,
   `completed`, `failed`) is recorded against `(run_id, tool_call_id)`.
   After a crash, a tool with a `started` record and no terminal update
   should be treated as `unknown_after_crash`: the side effect may or may
   not have happened.
4. **Lineage metadata** -- `conversation_id` (sequence) and `parent_run_id`
   (hierarchy) are independent axes. See [Three-level identity](#three-level-identity).

## Quick start

```python
from pydantic_ai import Agent
from pydantic_ai_harness import StepPersistence
from pydantic_ai_harness.step_persistence import InMemoryStepStore

store = InMemoryStepStore()
librarian = Agent(
    'openai:gpt-5',
    capabilities=[StepPersistence(store=store, agent_name='code_librarian')],
)

await librarian.run('Find ThinkingPartDelta and confirm the callable allowance')
```

That is the whole setup. **`run_id` is always per-`Agent.run` call**,
matching pydantic_ai's `RunContext.run_id`. For multi-turn logical
grouping use `conversation_id=` -- that is the pydantic_ai-native
primitive for it (see [Three-level identity](#three-level-identity)).

`run_id` resolution per call:

- **Explicit `run_id='libr-1'`** becomes the id for this one call.
  Single-shot use cases (deterministic id for testing, replay, debugging,
  a one-off scripted run). Reusing one capability instance with the same
  explicit `run_id` across multiple `.run()` calls raises `ValueError`
  in `before_run` -- the tool-effect ledger is keyed by
  `(run_id, tool_call_id)` and providers reuse deterministic tool-call
  ids, so a silent collision would erase the `unknown_after_crash`
  signal. Use `conversation_id=` for multi-turn grouping instead.
- **`agent_name` set, `run_id` unset** derives
  a path-safe base64url encoding of the complete `(agent_name, ctx.run_id)`
  pair. The encoding is injective within `FileStepStore`'s 200-character
  limit, so replay addresses the same stored run without collisions between
  distinct accepted context ids. A longer derived id raises `ValueError`
  before backend selection, including with the memory, SQLite, and Mongo
  stores.
- **Neither set** uses `ctx.run_id` unchanged. A missing context run id raises
  `RuntimeError` because inventing one would disconnect replayed writes.

## Durable execution

`StepPersistence` has the stable capability id `step_persistence`, so it can
be attached alongside a Pydantic AI durability capability without passing
`id=`. Pass an explicit id only when the same agent has more than one
`StepPersistence` instance.

Six store boundaries are durable operations: registration identity, run
registration, event append, snapshot save, tool-effect start, and tool-effect
completion or failure. Every persisted timestamp is read inside one of those
operations, so replay uses the journaled value instead of reading the workflow
wall clock again. The journaled registration identity makes a retried
registration idempotent while a distinct reuse of the same run id still fails.

Events and snapshots written by the capability carry deterministic per-run
idempotency keys. Every built-in store suppresses a key it already applied,
while records created directly with `idempotency_key=None` retain append
behavior. Snapshot keys use a per-run save sequence together with
`step_index` and `state`. Replay produces the same sequence, while distinct
snapshots at the same step and state retain their write order and newer history.

The orchestrator pattern -- one logical agent serving many turns -- uses
`conversation_id`, not a shared `run_id`:

```python
orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[StepPersistence(store=store, agent_name='orchestrator')],
)

for turn in turns:
    await orchestrator.run(turn, conversation_id='orch-conv')

# All turns of this orchestrator, chronological:
records = await store.list_runs(conversation_id='orch-conv')
```

## Three-level identity

The capability mirrors pydantic_ai's identity stack:

| Concept | Definition | Granularity |
| --- | --- | --- |
| `conversation_id` | The dialogue. Resolved by pydantic_ai from the `conversation_id=` argument to `Agent.run`, or the most recent `conversation_id` on `message_history`, or a fresh UUID7. | sequence of runs |
| `run_id` | One `Agent.run` invocation. | one step in the sequence |
| `step_index` | Graph-node count *within* a run (`ctx.run_step`). | one node within one run |

`StepEvent.conversation_id` and `RunRecord.conversation_id` are populated
from `ctx.conversation_id`. So three `.run()` calls sharing one
`conversation_id` produce three distinct `run_id`s, all queryable as a
group:

```python
runs = await store.list_runs(conversation_id='conv-abc')  # 3 records, chronological
```

## Continuing a delegate's investigation

pydantic_ai already has `message_history=` for "carry on with this prior
context". `StepPersistence` does not introduce a parallel mechanism -- it
exposes one helper that loads the most recent settled snapshot:

```python
from pydantic_ai_harness.step_persistence import continue_run

# Earlier: tag the first turn with a conversation id so the follow-up can find it.
await librarian.run(
    'Find ThinkingPartDelta and confirm the callable allowance',
    conversation_id='libr-conv',
)

# Later (possibly a different process):
prior_run = (await store.list_runs(conversation_id='libr-conv'))[-1].run_id
history = await continue_run(store, run_id=prior_run)
await librarian.run(
    'Read _apply_provider_details_delta and check the path',
    message_history=history,
    conversation_id='libr-conv',   # keep the conversation grouping
)
```

`fork_run(store, run_id=...)` returns the same shape but is intended when
the caller wants a branched logical run from that snapshot point (the new
run gets a fresh `run_id` and probably a fresh `conversation_id`).

### What "safe to continue from" means

By default `continue_run` returns the messages of the latest `complete`
snapshot for that `run_id` -- a point whose tool work was fully settled
when captured. Snapshots are written at these boundaries:

- after every `CallToolsNode` whose tool calls all returned -- the pending
  tool-return request is folded in, so the point is durable the moment the
  tool completes, before the next model request is even sent,
- at `after_run`, when the run ended past that boundary (a run that reached
  no boundary at all, or an `Agent.run_stream` whose closing response lands
  after the last one), and
- when a run *fails*: the live history at failure time is saved, whatever
  its shape -- a model request that raises after a clean tool cycle
  produces a `complete` snapshot; a crash mid-tool-cycle produces an
  `interrupted` one carrying every completed cycle.

An `interrupted` snapshot is sendable on resume -- pydantic-ai (>= 2.10)
repairs broken tool-call/result pairing before every model request -- but
not necessarily *safe*: a pending tool call may be re-executed (resuming
without a new prompt) or closed out with a synthesized `interrupted`
return, and neither says whether the original side effect happened. That
is the tool-effect ledger's job. So the default read path skips
`interrupted` snapshots; pass `include_interrupted=True` to
`continue_run` / `fork_run` / `latest_snapshot` after checking
`list_unresolved_tool_effects`. If no matching snapshot exists,
`continue_run` raises `LookupError`.

## Run lineage -- `parent_run_id`

`parent_run_id` is a lineage label, not a functional dependency. It does
two things:

- Every `StepEvent` and `RunRecord` carries it, so you can filter / group.
- `store.list_runs(parent_run_id='orch-1')` returns every delegate run
  pointing at that orchestrator.

It is auto-inferred for in-process delegation: when an orchestrator's
tool synchronously calls a delegate's `Agent.run(...)`, the delegate's
`StepPersistence` picks up the orchestrator's `run_id` via a `ContextVar`
that the orchestrator's `wrap_run` set. No threading required:

```python
orchestrator = Agent(
    'openai:gpt-5',
    capabilities=[StepPersistence(store=store, agent_name='orchestrator')],
)
librarian = Agent(
    'openai:gpt-5',
    capabilities=[StepPersistence(store=store, agent_name='code_librarian')],
)

@orchestrator.tool_plain
async def ask_librarian(question: str) -> str:
    result = await librarian.run(question)   # parent_run_id auto-filled
    return result.output

# Tag the orchestrator turn so the lookup below can find its run_id.
await orchestrator.run(
    'Where is ThinkingPartDelta defined?',
    conversation_id='orch-conv',
)

# All librarian runs now point at the orchestrator's run_id:
orch_run_id = (await store.list_runs(conversation_id='orch-conv'))[-1].run_id
delegates = await store.list_runs(parent_run_id=orch_run_id)
```

Set `parent_run_id=` explicitly to override (e.g. cross-process delegation
where `ContextVar`s do not propagate).

`parent_run_id` is distinct from `conversation_id`. The orchestrator
and delegate usually live in *different* conversations (the orchestrator
talks to a user; the delegate talks to itself). But they share a
parent-child link.

## Inspecting a run tree

`list_runs` returns matches sorted by `started_at` ascending across both
backends -- pick the most recent with `[-1]`.

```python
# Every delegate of one orchestrator run (chronological)
delegates = await store.list_runs(parent_run_id='orch-3f2a')

# Every run in one dialogue (multi-turn conversation across many .run() calls)
turns = await store.list_runs(conversation_id='conv-abc')
latest_turn = turns[-1]

# Filters combine (AND):
focused = await store.list_runs(
    parent_run_id='orch-3f2a',
    conversation_id='libr-conv',
)

# Detail per run:
events = await store.list_events(run_id=delegates[0].run_id)
snapshot = await store.latest_snapshot(run_id=delegates[0].run_id)
unresolved = await store.list_unresolved_tool_effects(run_id=delegates[0].run_id)
```

## Failure recovery

```python
# An earlier delegate run died mid-investigation.
events = await store.list_events(run_id='libr-3f2a')
unresolved = await store.list_unresolved_tool_effects(run_id='libr-3f2a')
for record in unresolved:
    # status == 'started' with no terminal update -- unknown_after_crash.
    print(f'tool {record.tool_name} ({record.tool_call_id}) may or may not have run')
    print(f'  idempotency_key={record.idempotency_key}  '
          f'effect_summary={record.effect_summary}')

# Decide whether to resume or branch:
history = await continue_run(store, run_id='libr-3f2a')
# If the unresolved tools were read-only and safe to redo:
await librarian.run('continue investigating', message_history=history,
                    conversation_id='libr-conv')
# If side effects might have happened and the orchestrator wants a fresh attempt:
history = await fork_run(store, run_id='libr-3f2a')
# ... pass to a new delegate run with a different agent_name / conversation_id.

# To resume from the interrupted frontier itself (the crashed cycle included),
# after checking the unresolved effects above:
history = await continue_run(store, run_id='libr-3f2a', include_interrupted=True)
```

Side-effect deduplication is the orchestrator's responsibility. Tools that
write external state should annotate their in-flight `ToolEffectRecord`
via `annotate_tool_effect`:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness.step_persistence import annotate_tool_effect

@orchestrator.tool
async def set_label(ctx: RunContext[Deps], issue: int, label: str) -> str:
    await annotate_tool_effect(
        store,
        ctx,
        idempotency_key=f'issue-{issue}::label::{label}',
        effect_summary=f'set label {label!r} on issue #{issue}',
    )
    await github.set_label(issue, label)   # the actual side effect
    return 'ok'
```

The helper reads the active `run_id` from the `StepPersistence`
`ContextVar` and `tool_call_id` / `tool_name` from `ctx`, then merges the
metadata into the prior record. It is a no-op when called outside a
step-persistence-wrapped tool call. `after_tool_execute` preserves both
fields when it writes the terminal `completed` / `failed` entry.

## Compaction receipt handles

`StepPersistence.compaction_transcript_handle()` exposes the current `run_id` to compaction
receipts. It is an identifier for this store's persisted run history, not a promise that the
pre-compaction transcript remains available: snapshots can already contain compacted history and
configured retention can delete older snapshots.

## Backends

- `InMemoryStepStore` -- process-local; great for tests.
- `FileStepStore(directory)` -- directory layout under `<directory>/<run_id>/`:
    - `run.json` -- `RunRecord` (lineage)
    - `events.jsonl` -- append-only `StepEvent`s
    - `tool_effects.jsonl` -- append-only `ToolEffectRecord`s, scoped to this run
    - `snapshot-keys.jsonl` -- replay-suppression keys retained independently
      of snapshot pruning
    - `snapshots/{seq}.json` -- `ContinuableSnapshot`s, named by a per-run
      monotonic counter (NOT `step_index`, which would collide when the
      same `run_id` is reused across `Agent.run` calls -- `ctx.run_step`
      resets to 0 each call).
- `SqliteStepStore(database='runs.db')` -- single SQLite file with tables
  `runs`, `events`, `snapshots`, `snapshot_idempotency_keys`, `tool_effects`, and a sibling `media`
  table for externalized blobs (see [Persisting media](#persisting-media)
  below). WAL mode is enabled; `tool_effects` upserts per
  `(run_id, tool_call_id)` so the latest state wins; snapshots use
  `AUTOINCREMENT seq` to mirror `FileStepStore._next_snapshot_seq`.
  Databases created before the snapshot `state` column existed gain it
  automatically on open (existing rows read as `complete`).
  Pass `connection=` instead of `database=` to share a `sqlite3.Connection`
  with the rest of your application; the connection must be opened with
  `check_same_thread=False` because hook calls are dispatched onto a
  worker thread.
- `MongoStepStore(client= or db_url=, database=...)` -- MongoDB collections
  `runs`, `events`, `snapshots`, `snapshot_idempotency_keys`, `tool_effects`,
  and `counters` (atomic `$inc` for monotonic `seq`). Run registration uses an
  atomic insert by `runs._id = run_id`; duplicate ids raise `ValueError`.
  Needs the `mongodb` extra
  (`pip install pydantic-ai-harness[mongodb]`, which installs
  `pymongo>=4.17.0`); pass a shared `AsyncMongoClient` as `client=`, or a
  connection string as `db_url=` (the store then owns the client -- call
  `await store.aclose()` to release it).
  Externalizes individual parts at or above `media_threshold_bytes` by
  default to a `MongoMediaStore` on the same client. That is a per-value
  offload, not an aggregate cap: a snapshot of many below-threshold parts can
  still exceed MongoDB's 16 MiB document limit and fail on insert -- lower the
  threshold if that is a risk for your workload.

All implement the same async `StepStore` protocol, so capability hooks never
block the event loop on the file/sqlite backends (I/O is dispatched via
`anyio.to_thread`); the Mongo backend is natively async.

`FileStepStore` validates `run_id` against `[A-Za-z0-9_.-]{1,200}` (and
rejects `..`) to prevent path traversal -- callers passing user-controlled
IDs should still sanitise first.

### What `MongoStepStore` creates on first write

The store issues `createIndex` on its first write, for ten indexes:
`conversation_id` and `parent_run_id` (both sparse) plus `started_at` on
`runs`; `(run_id, seq)` and unique keyed `(run_id, idempotency_key)` on
`events`; `(run_id, seq)`, unique keyed `(run_id, idempotency_key)`, and
`(run_id, state, seq)` on `snapshots`; and a unique `(run_id, tool_call_id)`
plus `(run_id, status)` on `tool_effects`. The idempotency indexes include only
documents whose key is a string, so `None` retains append behavior. Its default
`MongoMediaStore` adds one more (see the [media docs](../media/)). Three
consequences worth knowing before pointing the store at an existing
deployment:

- The connecting user needs the privilege to create indexes. A restricted
  Atlas role without it fails on the first write, not at construction.
- The unique index build fails if an existing `tool_effects` collection
  already holds duplicate `(run_id, tool_call_id)` pairs.
- Index builds against already-populated collections cost time and I/O on
  that first call.

`RunRecord.metadata` and `StepEvent.metadata` are stored as nested documents,
so their keys become BSON field names: keys containing `.` or starting with
`$` need [MongoDB 5.0 or later](https://www.mongodb.com/docs/manual/core/dot-dollar-considerations/),
and a key containing a NULL byte is rejected by the BSON encoder before it
reaches the server. CI exercises both Mongo backends against `mongo:8`.

## Bounding snapshot growth

Each step writes a new full-history snapshot keyed by an incrementing `seq`,
and nothing is pruned by default. Within one long `Agent.run` the snapshot
count equals the number of settled tool-call steps, so a long single run pays a
growing storage cost.

All four stores -- `InMemoryStepStore`, `FileStepStore`, `SqliteStepStore`, and
`MongoStepStore` -- accept an opt-in `max_snapshots_per_run: int | None`
(default `None`, unbounded -- byte-for-byte the prior behavior). When set to
`N >= 1`, each `save_snapshot` prunes the run down to a retain set:

- the newest `N` snapshots by `seq`,
- the newest snapshot overall (serves `latest_snapshot(include_interrupted=True)`),
- the newest `complete` snapshot (serves the default read path).

The last two keep both read modes correct even when the newest `N` snapshots
are all `interrupted` and the newest resumable `complete` sits below that
window, so the retain set can exceed `N`. `from_spec(..., max_snapshots_per_run=N)`
forwards the bound to the store it constructs (`backend='memory'`, `'file'`, or
`'sqlite'`; a Mongo store is built directly, not from a spec).

```python
from pydantic_ai_harness.step_persistence import FileStepStore

store = FileStepStore('runs', max_snapshots_per_run=8)
```

Pruning a snapshot never deletes its externalized media: blobs are
content-addressed and may be shared across snapshots and runs, so orphaned-blob
GC is out of scope (see the non-goals below). Age-based (TTL) expiry is out of
scope too -- it belongs at whole-run granularity, not per snapshot.

Bounded retention discards older per-step snapshots, including pre-compaction
ones. Any downstream that reconstructs history by unioning a run's retained
snapshots -- snapshot search or a compaction receipt keyed on `run_id` -- can
only see what is retained. With a tight bound (for example
`max_snapshots_per_run=1`) the older, pre-compaction states are gone, so treat
the bound as a hard limit on how far back such recovery can reach. Leave the
bound at `None`, or set it high enough to cover the history you need to recover,
when historical reconstruction matters.

## Persisting media

`BinaryContent` payloads (images, audio, documents, video) inline as
base64 inside a snapshot would balloon every file/row containing the
message; a large text part (e.g. a big tool-return string) does the same and
can push a `MongoStepStore` snapshot past MongoDB's 16 MiB document cap
([#440](https://github.com/pydantic/pydantic-ai-harness/issues/440)). The
file/sqlite/mongo backends externalize any `BinaryContent.data`, and any
part whose string `content` is at or above 64 KiB, through a configured
`MediaStore`, leaving a URI reference in the snapshot. The same
`media_threshold_bytes` governs both binary and text; there is no separate
text knob. Round-trip is transparent -- `latest_snapshot(...).messages[*]`
returns the original `BinaryContent` bytes and text.

Text externalization is not Mongo-only and has no opt-out short of
`media_store=None`: the walker is shared, so an existing `FileStepStore` or
`SqliteStepStore` deployment starts writing blobs for large text parts as well
as binary ones from this release on. Snapshots written before it still restore
-- the reader recognises the older binary marker shape. This compatibility is
upgrade-only: a release that predates text externalization treats every marker
as binary, so it cannot validate a snapshot containing an externalized text
marker. Keep a current reader for persisted snapshots that contain those
markers.

| StepStore           | Default `media_store`                  | Where blobs live                      |
| ------------------- | --------------------------------------- | ------------------------------------- |
| `InMemoryStepStore` | _(not applicable)_                     | bytes stay in the in-memory snapshot  |
| `FileStepStore`     | `DiskMediaStore(<root>/media/)`        | `<root>/media/<sha256>.bin`           |
| `SqliteStepStore`   | `SqliteMediaStore(database=<same db>)` | sibling `media` table in the same DB  |
| `MongoStepStore`    | `MongoMediaStore(client=<same client>)` | sibling `media` + `media_chunks` collections |

Override the destination by passing your own `MediaStore`:

```python
from pydantic_ai_harness.media import S3MediaStore
from pydantic_ai_harness.step_persistence import FileStepStore

store = FileStepStore(
    'runs',
    media_store=S3MediaStore(
        bucket='my-bucket',
        endpoint='https://<account>.r2.cloudflarestorage.com',
        region='auto',
        access_key_id=...,
        secret_access_key=...,
    ),
    media_threshold_bytes=64 * 1024,  # raise/lower if you want
)
```

Opt out entirely (keep bytes inline in the snapshot JSON/row):

```python
FileStepStore('runs', media_store=None)
SqliteStepStore(database='runs.db', media_store=None)
```

URIs are `media+sha256://<hex>`, content-addressed. The same blob written
through any `MediaStore` resolves the same way -- dedup is automatic and
moving the underlying storage is a one-line swap. The shipped
implementations are:

- `DiskMediaStore(directory)` -- one file per blob at
  `<directory>/<sha256>.bin`.
- `SqliteMediaStore(database=...)` or `SqliteMediaStore(connection=...)` --
  one row per blob (`INSERT OR IGNORE` for content-addressed dedup).
- `S3MediaStore(bucket=, endpoint=, region=, access_key_id=, secret_access_key=)`
  -- path-style URLs + handrolled SigV4. Compatible with AWS S3, Cloudflare
  R2 (`region='auto'`), MinIO, and other S3-compatible providers. PUT/GET/HEAD
  only -- no multipart, lifecycle, or listing in v1.
- `MongoMediaStore(client= or db_url=, database=...)` -- MongoDB, needs the
  `mongodb` extra. Stores each blob as sha256-addressed chunks across a
  `media` manifest document and a sibling `media_chunks` collection (manual
  chunking, not GridFS -- see the [media docs](../media/) for why), so a blob
  larger than one BSON document still stores and reads back. Chunking bounds
  the document, not memory: there is no streaming API, so each blob is held
  whole in process memory on both `put` and `get`. The manifest holds
  `MediaContext.metadata` inline and is not chunked, so keep per-blob metadata
  small. `collection=` renames both collections and `chunk_size_bytes=`
  (default 8 MiB) sets the split size.

### Exposing externalized bytes as URLs

Each store accepts a `public_url=` callable that turns the canonical
`media+sha256://<hex>` URI into a URL the model can fetch directly. The
forthcoming `MediaExternalizer` capability will use this to swap
`BinaryContent` parts for `ImageUrl` / `AudioUrl` / etc. before the
model sees the message -- letting providers fetch big media over the wire
without re-encoding bytes into the request body.

Static base URL (public R2 bucket, CDN):

```python
from pydantic_ai_harness.media import S3MediaStore, make_static_public_url

store = S3MediaStore(
    bucket='my-bucket',
    endpoint='https://<acc>.r2.cloudflarestorage.com',
    region='auto',
    access_key_id=..., secret_access_key=...,
    key_prefix='media/',
    public_url=make_static_public_url('https://pub-abc.r2.dev', key_prefix='media/'),
)
```

Presigned / rotating-signature URL -- pass any async callable that takes
`(uri, MediaContext)`:

```python
from pydantic_ai_harness.media import MediaContext, S3MediaStore

async def presign(uri: str, ctx: MediaContext) -> str:
    key = 'media/' + uri.removeprefix('media+sha256://') + '.bin'
    return await my_signer.generate(key, ttl=3600, content_type=ctx.media_type)

store = S3MediaStore(..., public_url=presign)
```

### `MediaContext`, extensible per-operation bag

Every `MediaStore` method (`put`, `get`, `exists`, `public_url`,
`get_metadata`) and both user-supplied callables (`PublicUrlResolver`,
`KeyStrategy`) accept a `MediaContext`:

```python
@dataclass(frozen=True, kw_only=True)
class MediaContext:
    media_type: str | None = None      # e.g. 'image/png'
    filename: str | None = None        # original filename, when known
    metadata: Mapping[str, str] = {}   # user-supplied tags
```

All fields default; new fields are added non-breakingly as use cases
emerge. Pass what you have, ignore the rest.

**Persistence by store.** `get_metadata(uri)` round-trips the
user-supplied `metadata` mapping on all four stores. `media_type` is
also persisted but is not part of what `get_metadata` returns (it is
stored for the byte payload itself, e.g. as the `Content-Type`).

- `SqliteMediaStore` writes `metadata` to a JSON column and `media_type`
  to a dedicated column
- `S3MediaStore` sends `metadata` as signed `x-amz-meta-*` headers
  (ASCII alphanumeric + dash key names) and `media_type` as
  `Content-Type`; `get_metadata` reads the `x-amz-meta-*` values back
  from the HEAD response
- `DiskMediaStore` writes a sidecar JSON file (`<resolved>.meta.json`)
  alongside each blob, atomic via tmp + rename. Sidecars are absent
  only when the put carried no metadata
- `MongoMediaStore` writes `metadata` as a JSON string and `media_type`
  as a dedicated field on the blob's manifest document (the `media`
  collection by default); `get_metadata` decodes the JSON string back.
  Because the mapping is one JSON string rather than nested fields,
  metadata keys are not subject to BSON field-name rules here

### `key_strategy` -- controlling the backend storage path

Default is `<sha256>.bin`. `DiskMediaStore` and `S3MediaStore` accept
overrides to fit existing layouts; `SqliteMediaStore` and `MongoMediaStore`
do not (the digest is their primary key, so a user-chosen key would either
break dedup or be a no-op -- use `table=` / `collection=` to move the rows
or documents):

```python
from pydantic_ai_harness.media import DiskMediaStore, MediaContext

def by_media_type(uri: str, ctx: MediaContext) -> str:
    digest = uri.removeprefix('media+sha256://')
    ext = {'image/png': '.png', 'image/jpeg': '.jpg'}.get(ctx.media_type or '', '.bin')
    return f'images/{digest}{ext}'

store = DiskMediaStore('runs', key_strategy=by_media_type)
```

**Caveat**: if your strategy depends on `context.media_type` (e.g. to
pick an extension), `get(uri)` and `exists(uri)` won't find the blob
unless the same context is supplied at read time. For pure
path-organisation strategies (no context dependency) the constraint
doesn't apply.

`DiskMediaStore` rejects strategies that produce absolute paths or paths
containing `..` segments, to prevent escaping the store directory.

Separately, all four stores accept a `public_url=` resolver, useful
when a CDN, local HTTP server, or signed-URL service fronts the bytes.
Without it `public_url(...)` returns `None` (the model never sees a URL
unless a resolver is configured and it returns a string).

pyai providers transparently download bytes from a URL when the target
model doesn't natively accept that URL type, so emitting a URL is
always safe -- you only ever lose wire savings, never correctness.

> **Note on the future `MediaExternalizer` capability.** When it lands,
> the composition will be
> `Agent(capabilities=[MediaExternalizer(store), StepPersistence(...)])`
> and `StepPersistence` will see already-URL-ified messages -- the
> externalize walk becomes a no-op. The existing API does not change.

### Persisting unsupported backends

DynamoDB, Postgres, Redis, GCS, and other backends are out of scope for
this release. Write your own `StepStore` (about ten methods on a Protocol) or
your own `MediaStore` (five methods: `put`, `get`, `exists`, `public_url`,
`get_metadata`) and pass it via `store=` / `media_store=`. Please open an issue if you ship one -- we want to feed
the eventual shared adapter layer with N >= 3 real implementations before
abstracting.

## What this capability does not do

- It does not restore capability per-run state, graph-node state, retry
  counters, or in-flight streaming responses.
- It does not deduplicate replayed side effects automatically. Tools that
  write artifacts, labels, PRs, or external state should call
  `annotate_tool_effect(store, ctx, ...)` (see [Failure recovery](#failure-recovery))
  so the orchestrator can decide whether replay is safe.
- It does not prune events, and by default does not prune snapshots.
  Retention is the caller's responsibility; snapshot growth can be bounded
  opt-in with `max_snapshots_per_run` (see [Bounding snapshot growth](#bounding-snapshot-growth)).
- It does not garbage-collect externalized media. Pruning a snapshot leaves
  its content-addressed blobs in place, since they may be shared across
  snapshots and runs.
- It does not emit OpenTelemetry spans. pydantic_ai's `Instrumentation`
  capability already spans `agent run` / `chat` / `running tool` and
  populates `gen_ai.agent.name`, `gen_ai.agent.call.id`,
  `gen_ai.conversation.id` via baggage. A future change may add
  step-persistence attributes to the active span; that is tracked as a
  follow-up issue.
