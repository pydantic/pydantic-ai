# Knowledge

Give the model a `search_knowledge` tool over your documents: retrieval-augmented generation (RAG) for answers drawn from a manual, a policy set, or a `docs/` folder rather than from the model's training data. Point it at workspace files and directories, inline documents, or a source of your own; it chunks and embeds them with any Pydantic AI [embedding model](https://pydantic.dev/docs/ai/guides/embeddings/) and returns the passages closest in meaning to the model's query.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/knowledge/)

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Knowledge

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        LocalWorkspace('.'),
        Knowledge(['docs'], embedder='openai:text-embedding-3-small', description='the product manual'),
    ],
)
```

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Installation

`Knowledge` needs no extra of its own. Install the extra for your embedding provider, such as `openai`:

uv:

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install pydantic-ai-harness "pydantic-ai-slim[openai]"
```

## How it works

Each call to `search_knowledge(query)` does two things:

1. **Sync.** It reads every source, splits each document into chunks with the `chunker`, and compares a fingerprint of the embedding model and the chunk texts with the one stored for that document. Only new and changed documents are embedded, in requests of up to 64 chunks that span documents; documents the sources no longer produce are removed. An unchanged corpus costs a read of the sources and no embedding requests.
2. **Search.** It embeds the query and returns the `max_results` (default 5) most similar chunks, most similar first, as a list of `KnowledgeMatch(document_id, text, score)`.

Because every search syncs first, the model searches the documents as they are when it calls the tool, including a file the agent itself edited earlier in the run. A search holds a lock while it syncs and searches the store, so parallel calls in one response embed a new corpus once, and each search runs against the documents of its own run's workspace.

The instructions tell the model to search before answering a question the documents may cover, and to say which document the answer comes from. `description` names the documents in the instructions and in the tool description (`'the product manual'`); without it they say "the knowledge base".

## Sources

`sources` is a sequence, in order, of:

- **A path** (`str` or `Path`): a file or directory in the run's workspace, relative to its working directory. A directory contributes every file below it whose suffix is in `file_suffixes` (Markdown, reStructuredText, plain text, CSV/TSV, JSON/JSONL, YAML, TOML and HTML by default; see `DEFAULT_FILE_SUFFIXES`), skipping hidden files and directories, and skipping a symlink whose target is outside both that directory and the working directory, so a link in the documents cannot index a file from elsewhere on the machine. A file listed by path is read whatever its suffix. The document id is the path as configured, joined with the file's path below it (`docs/guide/setup.md`). A path that does not exist fails the run with a `UserError`.
- **A `KnowledgeDocument(id, text)`**: indexed as given.
- **A `KnowledgeSource`**: any object with `async def load(self, workspace: Workspace) -> Sequence[KnowledgeDocument]`, called on every sync. Use it for content that is not a text file in a workspace, converting formats such as PDF to text:

```python {test="skip" typecheck="skip"}
from collections.abc import Sequence

from pypdf import PdfReader

from pydantic_ai.workspaces import Workspace
from pydantic_ai_harness import Knowledge
from pydantic_ai_harness.knowledge import KnowledgeDocument


class Manuals:
    async def load(self, workspace: Workspace) -> Sequence[KnowledgeDocument]:
        reader = PdfReader('manuals/router.pdf')
        text = '\n\n'.join(page.extract_text() for page in reader.pages)
        return [KnowledgeDocument('manuals/router.pdf', text)]


knowledge = Knowledge([Manuals(), 'docs'], embedder='openai:text-embedding-3-small')
```

Document ids must be unique across all sources; a repeated id fails the sync with a `UserError`.

Path sources are read from the run's workspace, so a run without one fails at its start. To read documents shipped with your app while the agent works in a sandbox, pass `workspace=LocalWorkspaceBackend('/app')` (a backend, not the `LocalWorkspace` capability). A `KnowledgeSource` receives the same workspace, which is unattached when there is none.

## Stores

`store` holds the embedded chunks. Without one, `Knowledge` keeps them in an `InMemoryVectorStore()`. It ranks by exact cosine similarity over every chunk, which suits corpora up to tens of thousands of chunks, and lasts as long as the capability: a new process embeds the corpus again on its first sync.

For a durable or larger index, implement the `VectorStore` protocol over a vector database. It has four methods, all keyed by document id:

```python {test="skip" lint="skip"}
class VectorStore(Protocol):
    async def fingerprints(self) -> Mapping[str, str]: ...
    async def replace_document(self, document_id: str, *, fingerprint: str, chunks: Sequence[EmbeddedChunk]) -> None: ...
    async def remove_documents(self, document_ids: Collection[str]) -> None: ...
    async def search(self, vector: Sequence[float], *, limit: int) -> Sequence[KnowledgeMatch]: ...
```

`replace_document` replaces all of a document's chunks at once, so a store never merges chunk lists. A store holds one corpus: a sync removes every document its sources did not produce, so give each `Knowledge` its own store (or collection).

The fingerprint names the embedding model by its provider and model name. A setting that changes the embedding size under the same name, such as `dimensions`, is not part of it: start a new store when you change one. `InMemoryVectorStore` raises a `UserError` when the query and stored vectors differ in size. Retry and fallback for a remote store belong in your implementation; an error it raises fails the run.

## Chunking

The default `RecursiveChunker(chunk_size=1_000, chunk_overlap=200)` splits at the coarsest separator that keeps chunks within `chunk_size` characters (paragraphs, then lines, then sentences, then words, then a hard cut), and starts each chunk with up to `chunk_overlap` characters from the end of the previous one. Pass your own to tune it, or any object with `def chunk(self, text: str) -> Sequence[str]`. Changing the chunking changes the fingerprints, so the next sync embeds every document again.

## Syncing ahead of a run

The first search of a large corpus waits while it is embedded. To embed it when your app starts instead, call `sync`, passing the workspace path sources are read from:

```python
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness import Knowledge

knowledge = Knowledge(['docs'], embedder='openai:text-embedding-3-small')


async def warm_up() -> None:
    result = await knowledge.sync(LocalWorkspaceBackend('.'))
    print(f'{result.embedded} of {result.documents} documents embedded')
```

`sync` returns a `KnowledgeSyncResult` with `documents`, `embedded`, `removed`, and `chunks_embedded`.

## Key options

| Option | Default | Purpose |
| --- | --- | --- |
| `sources` | (required) | Paths, `KnowledgeDocument`s, and `KnowledgeSource`s to index. |
| `embedder` | (required) | An `Embedder`, an `EmbeddingModel`, or a model name such as `'openai:text-embedding-3-small'`. |
| `store` | `None` (an `InMemoryVectorStore`) | Where embedded chunks are kept. |
| `chunker` | `RecursiveChunker()` | How documents are split into chunks. |
| `description` | `None` | What the documents are, named in the instructions and tool description. |
| `max_results` | `5` | Passages returned per search. |
| `file_suffixes` | `DEFAULT_FILE_SUFFIXES` | Suffixes read from directory sources, case-insensitive. |
| `workspace` | `None` | A backend to read path sources from instead of the run's workspace. |

## Agent specs

```yaml
capabilities:
  - Knowledge:
      sources: [docs]
      embedder: openai:text-embedding-3-small
      max_results: 3
```

Load it with `Agent.from_file(..., custom_capability_types=[Knowledge])`. A spec takes `sources` (workspace paths), `embedder` (a model name), `description`, `max_results`, `file_suffixes`, `id`, and `chunk_size` and `chunk_overlap` for the `RecursiveChunker`. Custom stores, chunkers, sources and `workspace=` need Python.

## Two of them

`Knowledge` declares the default `id` `'knowledge'`. Two `Knowledge`s on one agent that share it are one knowledge base stated twice: they become one when they agree, and the agent raises a `UserError` when it is built if they differ, since a merge would mix two corpora. To keep two, give each its own `id` and `description`, and wrap all but one in [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/), since both register `search_knowledge`:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace, PrefixTools
from pydantic_ai_harness import Knowledge

agent = Agent(
    'openai:gpt-5',
    capabilities=[
        LocalWorkspace('.'),
        Knowledge(['handbook'], embedder='openai:text-embedding-3-small', description='the support handbook'),
        PrefixTools(
            Knowledge(['api'], embedder='openai:text-embedding-3-small', description='the API reference', id='api'),
            prefix='api',
        ),
    ],
)
```

## Telemetry

Each sync is a `knowledge_sync` span on the run's tracer, with `knowledge.documents`, `knowledge.documents_embedded`, `knowledge.documents_removed` and `knowledge.chunks_embedded`: what the store changed and what embedding cost the search incurred. Inside a durable unit, such as a Temporal activity, where the run's tracer is not carried, the span goes to the global OpenTelemetry tracer provider. `sync()` outside a run takes `tracer=`, and emits nothing without one. The search itself is a tool call, which core's tool-execution span already records. Embedding requests get their own spans when the embedder is instrumented (`Embedder(..., instrument=True)` or `Embedder.instrument_all()`). They are not counted in the run's usage, so `UsageLimits` and `SpendLimits` do not see them.

## Limitations

- Every search reads every source again to find what changed. That is cheap for local files, but grows with the corpus, and on a remote sandbox each file read is a round trip; sync with a `workspace=` backend on the host, or a `KnowledgeSource` that knows what changed, for large corpora.
- Retrieved passages are document content returned as tool output. Load only documents you would let the model read, since text in them can try to steer the model like any other tool result.
- Search is by embedding similarity only: no keyword or hybrid search and no reranking. Rerank in a custom `VectorStore.search` if you need it; see [two-stage retrieval](https://pydantic.dev/docs/ai/guides/embeddings/#two-stage-retrieval-with-rerankers).
- The store holds one corpus. An agent whose runs use different workspaces with different documents re-embeds what differs on each switch, and searches wait for each other. For per-tenant workspaces, pass a `Knowledge` with that tenant's store to the run: `agent.run(..., capabilities=[Knowledge(['docs'], embedder=..., store=tenant_store)])`.
- Searches are serialized per capability on one event loop. Two processes sharing a durable store can embed the same changes twice; the result is the same.

## Further reading

- [Embeddings](https://pydantic.dev/docs/ai/guides/embeddings/) -- the models `embedder` accepts, and RAG pipeline guidance
- [RAG example](https://pydantic.dev/docs/ai/examples/data-analytics/rag/) -- a retrieval tool over Postgres with pgvector
- [Memory](../memory/) -- notes the model writes itself, rather than documents you give it
