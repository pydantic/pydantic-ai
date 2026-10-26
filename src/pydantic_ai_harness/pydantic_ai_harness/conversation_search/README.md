# Conversation Search

Give the model a `search_conversation_history` tool that BM25-ranks the history a `StepPersistence` capability already persists -- earlier turns that compaction dropped from the live context, and past runs in the same conversation by default.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/conversation_search/)

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ConversationSearch, SlidingWindowCompaction, StepPersistence
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.step_persistence import SqliteStepStore

store = SqliteStepStore(database='sessions.db')
agent = Agent(
    'openai:gpt-5',
    capabilities=[
        StepPersistence(store=store),
        ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
        SlidingWindowCompaction(max_messages=40),
    ],
)


async def ask(question: str, conversation_id: str) -> str:
    result = await agent.run(question, conversation_id=conversation_id)
    return result.output
```

## Why

Compaction keeps the live history inside the context window, so an exact detail (a file path, a value, a decision) can end up only in a summary's paraphrase, and nothing from earlier runs is in context at all. `ConversationSearch` writes nothing itself. It reads what `StepPersistence` stored, through a `HistorySource`, and lets the model pull exact excerpts back on demand. Give both capabilities the same store instance; they can be listed in any order.

- Ranking is BM25, implemented in pure Python with no extra dependencies. Rare terms score higher, and each word of a multi-word query is scored independently.
- Each result names its run (`run: ... | conversation: ...`). The tool's optional `run_id` argument searches one run's full record, so a run a compaction receipt references is directly resolvable.
- The tool reads the store when called, so it sees everything persisted so far, including earlier steps of the current run.
- History a run inherits through `message_history` is indexed under the run that first persisted it, so threaded runs do not return the same message once per run (with the instruction-change exception under [Limitations](#limitations)).
- Instruction changes recorded in history (`on_change='append'`) are searchable as system text; the initial instructions are not indexed.
- Long messages are clipped to 500 characters for display (200 for system text and tool-call arguments). When the matched term lies past the cutoff, the excerpt shows the text around it instead of the start. The index always holds the full text.

## How recovery works

`StepPersistence` saves the full history at every step. A compaction that persists its edits (like `SummarizingCompaction`) changes later snapshots only, so the earlier snapshots of the same run still hold the originals.

`SnapshotHistorySource` replays a run's `complete` snapshots in order, drops summary artifacts, and appends only what each snapshot adds past its longest overlap with what it has collected so far. `interrupted` snapshots (unsettled tool work, synthesized tool returns) stay out. Overlap is matched on a content hash of each message, because durable executors (Temporal, DBOS) re-instantiate messages between steps. The hash leaves out what core adds to a message after it is first saved (timestamps, run ids, instruction changes appended by `on_change='append'`), and the later copy is kept.

`HistorySource` is the seam: a substrate that keeps an append-only entry log can implement it by replay and replace the snapshot adapter without touching the search layer.

## Scope

`scope='conversation'` restricts the corpus to runs whose `conversation_id` matches the calling run's. The tool description tells the model so, and the `run_id` argument cannot reach past it: an out-of-scope run gets the same "no persisted history" answer as one that does not exist.

A run's `conversation_id` is the explicit `conversation_id=` argument, else the most recent one on `message_history`, else a fresh UUID7 (`conversation_id='new'` forces a fresh one). Threading `message_history` therefore keeps runs in one corpus, and a run with neither reaches only itself. A run whose `RunContext` has no `conversation_id` searches nothing and the tool says why, rather than pooling every unlabelled run in the store.

The value is trusted as given. UI adapters set it from client input (AG-UI's thread ID, Vercel AI's chat ID), so in a multi-tenant app derive it on the server from the authenticated user; see the [threat model](https://github.com/pydantic/pydantic-ai/blob/main/THREAT_MODEL.md#persistent-state-and-tenant-identity):

```python
async def ask(question: str, user_id: str, chat_id: str) -> str:
    # Bind the conversation to the authenticated user, not a raw client-supplied id.
    result = await agent.run(question, conversation_id=f'{user_id}:{chat_id}')
    return result.output
```

Scoping applies to the `RunRecord`s a `HistorySource` returns, so a custom source must populate `conversation_id` on them. `scope='all'` searches every run the source enumerates and returns verbatim excerpts from any of them; use it only when the store holds a single principal's history.

### Migrating from the `scope='all'` default

`scope` used to default to `'all'` and now defaults to `'conversation'`. Leaving it unset emits one `HarnessDeprecationWarning` per instance; the warning will be removed in the next breaking release. Pass `scope='all'` to keep store-wide search or `scope='conversation'` to accept the new default. To silence every harness deprecation at once:

```python
import warnings

from pydantic_ai_harness import HarnessDeprecationWarning

warnings.filterwarnings('ignore', category=HarnessDeprecationWarning)
```

## Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `source` | (required) | Where the corpus comes from. Use `SnapshotHistorySource(store)` over the store `StepPersistence` writes to. |
| `scope` | `'conversation'` (unset warns) | Restricts search to the calling run's `conversation_id`. Use `'all'` only for a store isolated to one principal. |
| `max_matches` | `10` | Maximum matching excerpts the search tool returns. |
| `context_lines` | `5` | Messages shown around each match (within the match's run). |
| `bm25_k1` | `1.5` | BM25 term-frequency saturation. Lucene's `BM25Similarity` uses `1.2`. |
| `bm25_b` | `0.75` | BM25 length normalization (Lucene default). |
| `add_instructions` | `True` | Emit a short note telling the model the recall tool exists. |
| `tool_id` | `conversation-search` | Toolset id for the search tool. |

## Telemetry

`ConversationSearch` emits no spans of its own. Each search is a tool call, so core's tool-execution span already records its duration, the query, and the result (content behind the usual instrumentation settings).

## Two of them

`ConversationSearch` declares the default `id` `'conversation_search'`. The `id` keys the capability's instructions, so they
can be addressed, and lets `defer_loading=True` work without one. Two `ConversationSearch`s on one agent that
share it are one search stated twice: they become one when they agree, and the agent raises a
`UserError` when it's built if they differ. Merging them instead could widen the corpus one of them was given. The `id` is separate from `tool_id`, which names the toolset. To keep two, give each its own `id` and wrap them
in [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/), since both register `search_conversation_history`.

Earlier releases declared no default `id`, so two always failed on the tool name when the run
started. Two identical ones now become one, and two that differ fail when the agent is built.

## Limitations

- Search only reaches what was persisted: history inherited from runs that never ran with `StepPersistence` (for example a long `message_history` from an unpersisted session) is lost if compaction drops it before the first snapshot.
- Recovering compaction-dropped originals needs the pre-compaction snapshots. A store with bounded snapshot retention (a per-run snapshot cap) can prune them; the search then returns what the surviving snapshots carry.
- If a snapshot that dropped a prefix also repeats the exchange that ends the accumulated history, the repeat is absorbed into the overlap and indexed once.
- An inherited request that gets an instruction change in a later run is indexed again with that change, so its other parts can match twice. A run that branches from before such a change, followed by one that continues past it, indexes from the changed request onward once more.
- The tokenizer splits on word characters, so scripts written without spaces (Chinese, Japanese, Thai) match only a whole unbroken run, not a word inside it.
- The corpus is rebuilt on each call from every in-scope run's snapshots, and every run record in the store is listed to apply the scope. A large store or history makes each search proportionally slower ([#9361](https://github.com/pydantic/pydantic-ai/issues/9361)).
- Reading snapshots restores externalized media even though the index only names files; stores with remote media backends pay that fetch per search.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Step Persistence](../step_persistence/) -- the substrate this capability reads
- [Compaction](../compaction/) -- the capabilities whose drops this one recovers from
