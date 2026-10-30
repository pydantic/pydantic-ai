# Knowledge Base

`KnowledgeBase` gives an agent retrieval from a knowledge base: a corpus of documents your
application provides, such as manuals, policies or support articles. The agent searches it with
`search_knowledge_base`, opens whole documents with `read_document` where the backend can return
them, and cites the passages it used. Where the documents live and how a search matches them is
the job of a pluggable backend: an in-memory index, an Amazon Bedrock Knowledge Base, an OpenAI
vector store, or your own.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/knowledge_base/)

## Installation

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

The capability and the in-memory backend need nothing else. The hosted backends use the client
library of their service, installed through Pydantic AI's provider extras. For
`BedrockKnowledgeBase` (`boto3`):

uv:

```bash
uv add "pydantic-ai-slim[bedrock]"
```

pip:

```bash
pip install "pydantic-ai-slim[bedrock]"
```

For `OpenAIVectorStore` (`openai`):

uv:

```bash
uv add "pydantic-ai-slim[openai]"
```

pip:

```bash
pip install "pydantic-ai-slim[openai]"
```

## The problem

Retrieval-augmented generation splits into two jobs. Building the index (chunking, embedding,
storing) usually happens outside the agent, in a pipeline or a managed service. Using the index
is the agent's side: a tool around the search call, a filter that scopes a shared index to the
current tenant, passages rendered so the model can cite them, a way to read past a passage into
the whole document, and guidance on when to search. That second job looks the same whichever
service holds the index, and every agent rewrites it.

Model providers that search uploaded files natively, through
[`FileSearchTool`](https://pydantic.dev/docs/ai/tools-toolsets/native-tools/#file-search-tool), cover
part of it, but only for files stored with that provider and only on that provider's models.

## The solution

`KnowledgeBase` is the agent's side as one [capability](https://pydantic.dev/docs/ai/capabilities/overview/),
and the index is a backend you pass to it:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import Document, InMemoryKnowledgeBase

docs = InMemoryKnowledgeBase(
    [
        Document('returns', 'Items can be returned within 30 days of delivery.', source='https://shop.example/returns'),
        Document('shipping', 'Orders ship within two business days.', source='https://shop.example/shipping'),
    ]
)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[KnowledgeBase(docs, description='the shop policies')],
)

result = agent.run_sync('Can I still return something I got three weeks ago?')
print(result.output)
```

Swapping the backend changes where the documents live, not how the agent uses them.

## Backends

| Backend | What it searches | Reads documents | Native search |
|---|---|---|---|
| `InMemoryKnowledgeBase(documents)` | Documents held in process, BM25 keyword ranking over paragraph chunks | Yes | No |
| `BedrockKnowledgeBase(knowledge_base_id)` | An [Amazon Bedrock Knowledge Base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html), with hybrid search and reranking | No | No |
| `OpenAIVectorStore(vector_store_id)` | An [OpenAI vector store](https://platform.openai.com/docs/guides/retrieval) | Yes | On OpenAI Responses models |

`InMemoryKnowledgeBase` keeps nothing on disk and has no embeddings, so it matches words, not
meaning. Use it for tests, examples, and small corpora that ship with the application.

### Amazon Bedrock

```python
from pydantic_ai import Agent
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import BedrockKnowledgeBase

agent = Agent(
    'bedrock:anthropic.claude-sonnet-4-6',
    capabilities=[
        KnowledgeBase(
            BedrockKnowledgeBase(
                'KB123',
                region_name='eu-central-1',
                search_type='HYBRID',
                reranking_model='amazon.rerank-v1:0',
                fetch_results=50,
            ),
            number_of_results=5,
        ),
    ],
)
```

Credentials come from the usual boto3 sources (environment, profile, instance role); pass
`client=boto3.client('bedrock-agent-runtime', ...)` to set them explicitly. `search_type='HYBRID'`
matches on embeddings and raw text, which helps with product names and part numbers; it needs an
OpenSearch Serverless store with a filterable text field. With a `reranking_model`, the vector
search fetches `fetch_results` passages and the reranker keeps the best `number_of_results`. A bare
model id resolves to an ARN in `region_name` or the default client's region; with a `client` you
pass, set `region_name` too or give the full ARN.

`retrieval_configuration` takes a complete Bedrock `retrievalConfiguration` for anything the fields
don't cover: implicit filtering, Bedrock-native filters such as ranges or `startsWith`, reranker
metadata selection, or a managed knowledge base's `managedSearchConfiguration` (boto3 1.43 or
later). The capability's metadata filter is still applied on top of it, combined with any filter it
carries. Bedrock has no API to read a document back, so this backend offers no `read_document`.

Throttling, quota, internal, dependency, gateway and model-not-ready errors and network failures
become retry prompts; access, resource and validation errors raise.

### OpenAI vector stores

```python
from pydantic_ai import Agent
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import OpenAIVectorStore

agent = Agent(
    'openai:gpt-5.2',
    capabilities=[KnowledgeBase(OpenAIVectorStore('vs_abc123'), native=True)],
)
```

Upload and attribute files with the OpenAI API; the backend only queries the store. A file's id is
its document id, so `read_document` opens the whole file. Searches return at most 50 passages, the
API maximum. The default client reads `OPENAI_API_KEY`; pass `client=AsyncOpenAI(...)` to configure
it. Rate limits, server errors and connection failures become retry prompts.

## Native file search

With `native=True` and a backend that a model provider can also search itself (`OpenAIVectorStore`),
models that support it search the index with the provider's native
[`FileSearchTool`](https://pydantic.dev/docs/ai/tools-toolsets/native-tools/#file-search-tool), and every
other model uses `search_knowledge_base` as usual. One configuration then works on any model and
uses the provider's own search where it can. `read_document` stays a local tool either way.

`FileSearchTool` has no metadata filter, so `native=True` can't be combined with `filter`: a
native search over a shared store would not be scoped to the run's tenant. With native search the
passages come back as the provider's own tool result, not in `metadata['results']`.

`FileSearchTool` on its own remains the right tool when you only target one provider and want it to
manage the files. `KnowledgeBase` is for an index the application owns, searched from any model.

## Tools

| Tool | Purpose |
|---|---|
| `search_knowledge_base` | Search for passages relevant to a query and return the best `number_of_results`, each with its source and, where the backend reads documents, its document id. |
| `read_document` | Return a whole document by the id a search showed, cut at `max_document_chars`, if it matches the run's filter. Only offered when the backend can read documents. |

Each passage is rendered under a `Source:` line with its score when the backend reports one, and a
`Document:` line when it can be opened. A query with no matches returns
`No results found for {query!r}.`; an unknown document id is returned to the model as a
[retry prompt](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-retries).

## Scoping a shared knowledge base

A metadata filter scopes every search. It maps a metadata key to a value (equal) or a list of
values (any of), and every key must match. Each backend translates it to its own filter syntax. A
static `filter` applies to every run; a callable builds one per run from the `RunContext`, for
example from the tenant in `ctx.deps`:

```python
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import BedrockKnowledgeBase, MetadataFilter


@dataclass
class Deps:
    tenant_id: str


def tenant_filter(ctx: RunContext[Deps]) -> MetadataFilter:
    return {'tenant_id': ctx.deps.tenant_id, 'source': ['manuals', 'faq']}


agent = Agent(
    'bedrock:anthropic.claude-sonnet-4-6',
    deps_type=Deps,
    capabilities=[KnowledgeBase(BedrockKnowledgeBase('KB123', region_name='eu-central-1'), filter=tenant_filter)],
)
```

The callable may be async, and returning `None` searches unfiltered. `read_document` applies the
same filter, so a document outside it reads as missing even when the model knows its id. The filter
is resolved before the backend call, so under durable execution a recovered run replays the same
scoped request.

## Structured citations

The search tool returns a
[`ToolReturn`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns):
`return_value` carries the text the model sees, and `metadata['results']` carries each passage as a
`KnowledgeBaseResult` (`source`, `score`, `document_id`, `metadata`). `read_document` puts the
document's id, source, metadata and whether it was cut under `metadata['document']`. Metadata is
never sent to the model; read it from the `ToolReturnPart` in the message history to render
citations without parsing text:

```python
from pydantic_ai.messages import ModelRequest, ToolReturnPart


def citations(messages: list[ModelRequest]) -> list[str]:
    return [
        str(result['source'])
        for message in messages
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'search_knowledge_base' and part.metadata
        for result in part.metadata['results']
    ]
```

`show_sources=False` keeps the `Source:` lines out of the text the model sees while
`metadata['results']` keeps them, for an application that renders citations itself or must not
show the model internal storage paths.

## Untrusted content

Passages come from the documents the knowledge base indexed, so they can carry instructions aimed
at the model. Sources and document ids are rendered on one line, so they can't fake another
passage's labels; passage and document text are passed through as returned. To screen tool output,
add [`PromptInjectionDefender`](https://pydantic.dev/docs/ai/harness/prompt-injection-defender/).

## Instructions

The capability adds short guidance to the system prompt: search before answering a question the
knowledge base could cover, base the answer on the passages and cite their sources, phrase queries
as statements or keyword phrases, resolve references to earlier turns first, run separate searches
for unrelated questions, open a whole document when a passage is not enough (with a readable
backend), and say so when nothing relevant comes back. `description` is woven in, so the model
knows what the knowledge base holds. Replace it with `guidance='...'`, or set `guidance=''` to
contribute no instructions.

With [`defer_loading=True`](https://pydantic.dev/docs/ai/capabilities/on-demand/) the model sees a
one-line catalog entry instead ("Search a knowledge base of ... for passages before answering
questions it could cover."), and the tools and guidance arrive once it calls `load_capability`.

## Configuration

Every field of `KnowledgeBase` with its default:

```python
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import InMemoryKnowledgeBase

KnowledgeBase(
    InMemoryKnowledgeBase(),  # backend, required
    number_of_results=5,  # passages per search (1 to 100; OpenAI caps at 50)
    filter=None,  # MetadataFilter, or a callable of RunContext building one per run
    show_sources=True,  # Source: lines in the text; off keeps provenance in metadata only
    max_document_chars=20_000,  # how much of a document read_document returns
    native=False,  # native FileSearchTool where the backend and model allow it
    description=None,  # what the knowledge base holds, for the tools and the instructions
    guidance=None,  # None = default instructions, '' = none, str = custom
    id='knowledge_base',  # durable-execution identity
)
```

An out-of-range `number_of_results`, a non-positive `max_document_chars`, and `native=True` with a
backend that has no native search or together with `filter` raise at construction.

## Your own backend

Any object with an async `search` method is a backend. Add an async `read` method to get
`read_document`, and `file_search_tool` to offer native search:

```python
from collections.abc import Sequence

from pydantic_ai_harness.knowledge_base import Document, MetadataFilter, Passage, matches

PAGES = {'deploys': Document('deploys', 'Deploys run every weekday at 10:00.', metadata={'team': 'platform'})}


class Wiki:
    async def search(self, query: str, *, limit: int, filter: MetadataFilter | None) -> Sequence[Passage]:
        return [
            Passage(page.text, source=f'wiki/{page.document_id}', document_id=page.document_id)
            for page in PAGES.values()
            if matches(page.metadata, filter)
        ][:limit]

    async def read(self, document_id: str, *, max_chars: int, filter: MetadataFilter | None) -> Document | None:
        page = PAGES.get(document_id)
        return page if page is not None and matches(page.metadata, filter) else None
```

A backend owns its transport and failure policy: raise `ModelRetry` for a failure the model can
recover from by searching again, such as throttling, and let anything else propagate. `read` gets
the run's filter too and must return `None` for a document outside it: otherwise a tenant-scoped
agent could open another tenant's document by guessing its id. `matches` applies a neutral filter
to a metadata mapping.

## Multiple instances

Two instances register the same tools and share the default `id`, so two that differ raise an
error. To search two knowledge bases from one agent, give the second a distinct `id` and wrap it in
core's `PrefixTools` capability, which prefixes its tool names:

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools
from pydantic_ai_harness import KnowledgeBase
from pydantic_ai_harness.knowledge_base import BedrockKnowledgeBase

agent = Agent(
    'bedrock:anthropic.claude-sonnet-4-6',
    capabilities=[
        KnowledgeBase(BedrockKnowledgeBase('KB123', region_name='eu-central-1'), description='product manuals'),
        PrefixTools(
            wrapped=KnowledgeBase(
                BedrockKnowledgeBase('KB456', region_name='eu-central-1'), guidance='', id='policies_knowledge_base'
            ),
            prefix='policies',
        ),  # policies_search_knowledge_base
    ],
)
```

Set `guidance=''` on the wrapped instance (or replace it with text that tells the model when to use
the prefixed tool), since each instance otherwise contributes the same default guidance.

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/),
each search and each document read is recorded, so a recovered run reuses the result instead of
querying again. The records are named after the capability's `id`, which defaults to
`knowledge_base`, so durable execution needs no configuration. Changing the `id` renames the
records, which in-flight runs then cannot find.

## Agent spec (YAML/JSON)

`KnowledgeBase` works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/) for the Bedrock and OpenAI backends:

```yaml
# agent.yaml
model: bedrock:anthropic.claude-sonnet-4-6
capabilities:
  - KnowledgeBase:
      backend: bedrock
      knowledge_base_id: KB123
      region_name: eu-central-1
      search_type: HYBRID
      filter:
        tenant_id: acme
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import KnowledgeBase

agent = Agent.from_file('agent.yaml', custom_capability_types=[KnowledgeBase])
```

`knowledge_base_id` is the Bedrock knowledge base id or the OpenAI vector store id. The Bedrock
options (`region_name`, `search_type`, `reranking_model`, `fetch_results`) only apply to
`backend: bedrock`. A spec takes a static `filter`; a callable filter, a custom client and an
in-memory backend are Python-only.

## Telemetry

`KnowledgeBase` emits no spans of its own. Core's tool-call spans cover each search and read, and
the retrieval options are fixed per capability, so there is no decision to record.

## Further reading

- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/capabilities/overview/)
- [Native tools: file search](https://pydantic.dev/docs/ai/tools-toolsets/native-tools/#file-search-tool)
- [Embeddings](https://pydantic.dev/docs/ai/guides/embeddings/), for building your own vector index
- [Memory](https://pydantic.dev/docs/ai/harness/memory/), for knowledge the agent writes itself
