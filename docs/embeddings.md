# Embeddings

Embeddings are vector representations of text that capture semantic meaning. They're essential for building:

- **Semantic search** — Find documents based on meaning, not just keyword matching
- **RAG (Retrieval-Augmented Generation)** — Retrieve relevant context for your AI agents
- **Similarity detection** — Find similar documents, detect duplicates, or cluster content
- **Classification** — Use embeddings as features for downstream ML models

Pydantic AI provides a unified interface for generating embeddings across multiple providers.

## Quick Start

The [`Embedder`][pydantic_ai.embeddings.Embedder] class is the high-level interface for generating embeddings:

```python {title="embeddings_quickstart.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    # Embed a search query
    result = await embedder.embed_query('What is machine learning?')
    print(f'Embedding dimensions: {len(result.embeddings[0])}')
    #> Embedding dimensions: 1536

    # Embed multiple documents at once
    docs = [
        'Machine learning is a subset of AI.',
        'Deep learning uses neural networks.',
        'Python is a programming language.',
    ]
    result = await embedder.embed_documents(docs)
    print(f'Embedded {len(result.embeddings)} documents')
    #> Embedded 3 documents
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

!!! tip "Queries vs Documents"
    Some embedding models optimize differently for queries and documents. Use
    [`embed_query()`][pydantic_ai.embeddings.Embedder.embed_query] for search queries and
    [`embed_documents()`][pydantic_ai.embeddings.Embedder.embed_documents] for content you're indexing.

## Embedding Result

All embed methods return an [`EmbeddingResult`][pydantic_ai.embeddings.EmbeddingResult] containing the embeddings along with useful metadata.

For convenience, you can access embeddings either by index (`result[0]`) or by the original input text (`result['Hello world']`).

```python {title="embedding_result.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    result = await embedder.embed_query('Hello world')

    # Access embeddings - each is a sequence of floats
    embedding = result.embeddings[0]  # By index via .embeddings
    embedding = result[0]  # Or directly via __getitem__
    embedding = result['Hello world']  # Or by original input text
    print(f'Dimensions: {len(embedding)}')
    #> Dimensions: 1536

    # Check usage
    print(f'Tokens used: {result.usage.input_tokens}')
    #> Tokens used: 2

    # Calculate cost (requires `genai-prices` to have pricing data for the model)
    cost = result.cost()
    print(f'Cost: ${cost.total_price:.6f}')
    #> Cost: $0.000000
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Chunking Strategies

Before embedding documents for RAG (Retrieval-Augmented Generation) or semantic search, you need to break them into smaller pieces called **chunks**. This section covers why chunking matters, the main strategies, and how to implement them effectively.

### Why Chunking Matters

Chunking directly impacts your RAG system's performance:

- **Chunks too large**: The embedding becomes a "noisy average" that doesn't clearly represent any single topic. Subtopics get lost or muddled, and retrieval becomes less precise.
- **Chunks too small**: Context is lost, and your system can't answer questions that require understanding relationships between concepts.

Research shows chunking strategy can create **up to a 9% gap in recall performance** between best and worst approaches. Getting this right is worth the effort.

!!! tip "The Core Trade-off"
    Include too much in a chunk and the vector loses the ability to be specific to anything it discusses. Include too little and you lose the context of the data.

### Recommended Starting Point

For most use cases, start with **recursive character splitting** at 400-512 tokens:

```python {title="chunking_quickstart.py" test="skip"}
from chonkie import RecursiveChunker

chunker = RecursiveChunker(chunk_size=512)

document = '''
Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without explicit programming.

Deep learning is a subset of machine learning.
It uses neural networks with many layers to learn complex patterns.
'''

chunks = chunker.chunk(document)
for i, chunk in enumerate(chunks):
    print(f'Chunk {i}: {chunk.text[:50]}...')
```

This approach works well for 80% of RAG applications. Only reach for more complex strategies if you have specific requirements or see poor retrieval quality.

### Main Chunking Strategies

#### Fixed-Size Chunking

The simplest approach: split text into equally sized pieces based on character or token count.

**How it works**: Split text at fixed intervals (e.g., every 512 tokens), adding overlap between chunks to preserve boundary context.

**Pros**:

- Computationally cheap and simple
- No NLP libraries required
- Predictable chunk sizes
- Works with any text

**Cons**:

- Can cut in the middle of sentences or words
- Ignores semantic structure
- May split related concepts

**When to use**: Default choice when you need simplicity and predictability.

```python {title="fixed_size_chunking.py" test="skip"}
from chonkie import TokenChunker

chunker = TokenChunker(
    chunk_size=512,
    chunk_overlap=50,
)

document = 'Your document text here...'
chunks = chunker.chunk(document)
```

#### Recursive Character Splitting

A hierarchical approach that splits text using a priority list of separators.

**How it works**: Uses a rules-based hierarchy. First attempts to split by paragraphs (`\n\n`), then lines (`\n`), then sentence boundaries, then whitespace. Keeps larger semantic units together when possible.

**Pros**:

- Balances structure awareness with simplicity
- Respects paragraph and sentence boundaries when possible
- More semantically meaningful than pure fixed-size

**Cons**:

- Still character/token-based at core
- May still break mid-sentence for very long paragraphs

**When to use**: **Recommended starting point** for most applications.

```python {title="recursive_chunking.py" test="skip"}
from chonkie import RecursiveChunker

# RecursiveChunker uses a rules-based system with default separators
# ('\n\n', '\n', sentence boundaries, whitespace)
chunker = RecursiveChunker(chunk_size=512)

document = 'Your document text here...'
chunks = chunker.chunk(document)
```

#### Sentence-Based Chunking

Splits text respecting sentence boundaries, then groups sentences into chunks.

**How it works**: Uses NLP to detect sentence boundaries. Groups consecutive sentences until chunk size limit. Never breaks mid-sentence.

**Pros**:

- Maintains grammatical coherence
- Natural reading units
- Better for conversational/narrative content

**Cons**:

- Requires NLP library for sentence detection
- Sentence detection can fail on unusual formatting
- Individual sentences may be very short or very long

**When to use**: Narrative text, conversational data, documentation.

```python {title="sentence_chunking.py" test="skip"}
from chonkie import SentenceChunker

chunker = SentenceChunker(
    chunk_size=512,
    chunk_overlap=50,
)

document = 'Your document text here...'
chunks = chunker.chunk(document)
```

#### Semantic Chunking

Uses embedding similarity to determine where to split text.

**How it works**:

1. Embed each sentence in the document
2. Calculate similarity between consecutive sentences
3. Split where similarity drops significantly (topic shift)
4. Group semantically similar sentences together

**Pros**:

- Groups related content together
- Splits at natural topic boundaries
- Can capture themes that span multiple paragraphs

**Cons**:

- **Expensive**: Every sentence needs an embedding (API calls or local inference)
- 10,000-word document = 200-300 embeddings just for chunking
- Research shows benefits are **highly task-dependent** and often don't justify cost

!!! warning "Cost Consideration"
    Semantic chunking requires embedding every sentence before you can even start indexing. For a 10,000-word document, that's 200-300 embedding calls just for chunking. Evaluate carefully whether the improved chunk quality justifies this cost.

**When to use**: When you have budget for embeddings AND initial testing shows clear improvement over recursive splitting. Not recommended as a default.

```python {title="semantic_chunking.py" test="skip"}
from chonkie import SemanticChunker

# Uses sentence-transformers by default
chunker = SemanticChunker(
    chunk_size=512,
    threshold=0.5,  # Split when similarity drops below this
)

document = 'Your document text here...'
chunks = chunker.chunk(document)
```

#### Document-Structure Aware Chunking

Leverages document structure (headers, sections, code blocks) to determine chunk boundaries.

**How it works**: Parse document to identify structural elements. Keep sections, code blocks, tables as atomic units. Split based on headers and hierarchy.

**Variants**:

- **Markdown splitting**: Respects markdown headers, maintains hierarchy in metadata
- **HTML splitting**: Parses HTML structure
- **Page-level**: Uses page breaks (for PDFs)

**Pros**:

- Preserves document organization
- Headers provide natural topic boundaries
- Metadata includes section hierarchy for filtering

**Cons**:

- Requires well-structured documents
- Section sizes vary (may need secondary splitting)
- Format-specific implementations needed

**When to use**: Documentation, reports, research papers, markdown/HTML content with clear structure.

```python {title="markdown_chunking.py" test="skip"}
from chonkie import RecursiveChunker
from chonkie.types.recursive import RecursiveLevel, RecursiveRules

# Custom rules for markdown headers
markdown_rules = RecursiveRules(
    levels=[
        RecursiveLevel(delimiters=['\n## ', '\n### ', '\n#### ']),
        RecursiveLevel(delimiters=['\n\n']),
        RecursiveLevel(delimiters=['\n']),
        RecursiveLevel(whitespace=True),
    ]
)

chunker = RecursiveChunker(chunk_size=512, rules=markdown_rules)

markdown_document = '# My Doc\n\n## Section 1\n\nContent here...'
chunks = chunker.chunk(markdown_document)
```

#### Code Chunking

Syntax-aware chunking that respects code structure.

**How it works**: Uses syntax parsing (tree-sitter) to identify code boundaries. Splits at function/class boundaries. Keeps imports with related code.

**Pros**:

- Never breaks mid-function or mid-class
- Preserves semantic code units
- Multi-language support

**Cons**:

- Requires syntax parser for each language
- Function sizes vary widely

**When to use**: Code repositories, documentation with code samples.

```python {title="code_chunking.py" test="skip"}
from chonkie import CodeChunker

chunker = CodeChunker(
    language='python',
    chunk_size=1500,  # Larger for code
)

code = '''
def calculate_total(items):
    """Calculate total price of items."""
    return sum(item.price for item in items)

def apply_discount(total, discount_percent):
    """Apply percentage discount to total."""
    return total * (1 - discount_percent / 100)
'''

chunks = chunker.chunk(code)
```

#### Agentic Chunking (LLM-Based)

Uses an LLM to determine optimal chunk boundaries based on content understanding.

**How it works**:

1. LLM analyzes entire document structure, content type, and density
2. Agent decides which chunking strategy to apply (or mix of strategies)
3. LLM may process each sentence and assign to chunks based on semantic meaning

**Pros**:

- Most context-aware approach
- Adapts strategy per document type
- Handles complex, mixed-content documents

**Cons**:

- **Very expensive**: Every sentence may require an LLM call
- **Slow**: Significant latency per document
- Cost scales with document size

**When to use**:

- One-time processing of high-value content
- As benchmark to evaluate simpler methods
- When cost/latency are not constraints

```python {title="agentic_chunking.py" test="skip"}
from chonkie import SlumberChunker

# Uses an LLM to determine chunk boundaries
# Requires a 'genie' (LLM interface) to be configured
chunker = SlumberChunker(chunk_size=512)

document = 'Your document text here...'
chunks = chunker.chunk(document)
```

### Key Parameters

#### Chunk Size

| Query Type | Optimal Chunk Size | Notes |
|------------|-------------------|-------|
| Factoid/Lookup | 256-512 tokens | Precise, focused answers |
| Analytical/Reasoning | 1024+ tokens | Needs broader context |
| General purpose | 400-512 tokens | Good starting point |

#### Overlap

- **Recommended**: 10-20% of chunk size
- **Example**: For 512-token chunks, use 50-100 token overlap
- **Purpose**: Preserves context at boundaries, prevents information loss

#### Embedding Model Token Limits

Different embedding models have different maximum input lengths. Chunks exceeding these limits will be truncated or cause errors:

| Model | Max Tokens | Notes |
|-------|-----------|-------|
| OpenAI text-embedding-3-small/large | 8,191 | ~6,000-6,500 English words |
| Cohere Embed v3 | 512 | Recommend chunking to <512 tokens |
| Cohere Embed v4 | ~128,000 | Large context, but smaller chunks often better for RAG |
| Sentence Transformers (typical) | 384-512 | Varies by model |
| Jina v2/v3 | 8,192 | Good for late chunking |

!!! tip "Check Your Model's Limits"
    Use [`embedder.max_input_tokens()`][pydantic_ai.embeddings.Embedder.max_input_tokens] to check your model's limit programmatically.

### Best Practices

#### Strategy Selection by Content Type

| Content Type | Recommended Strategy |
|-------------|---------------------|
| General text | RecursiveChunker (400-512 tokens) |
| Markdown/Docs | Markdown-aware separators + recursive |
| Code | CodeChunker with appropriate language |
| PDFs (formatted) | Page-level + structure extraction |
| Research papers | Structure-aware (sections, abstracts) |
| Conversational | SentenceChunker |

#### Metadata Preservation

Always attach metadata to chunks for filtering and context:

```python {title="chunking_with_metadata.py"}
from dataclasses import dataclass


@dataclass
class ChunkWithMetadata:
    text: str
    source: str
    page: int | None
    section: str | None
    chunk_index: int
    total_chunks: int


def chunk_with_metadata(document: str, source: str) -> list[ChunkWithMetadata]:
    from chonkie import RecursiveChunker

    chunker = RecursiveChunker(chunk_size=512)
    chunks = chunker.chunk(document)

    return [
        ChunkWithMetadata(
            text=chunk.text,
            source=source,
            page=None,
            section=None,
            chunk_index=i,
            total_chunks=len(chunks),
        )
        for i, chunk in enumerate(chunks)
    ]
```

**Key metadata fields**:

- Source file/URL
- Page number (for PDFs)
- Section headers/hierarchy
- Chunk position in document
- Document type
- Creation/modification date

#### Evaluation

Always evaluate chunking strategies on your specific use case:

1. Create a test set of queries and expected retrieved chunks
2. Measure retrieval metrics (recall@k, precision@k)
3. Compare 2-3 strategies before committing
4. Test with your actual embedding model

### Using chonkie

We recommend [chonkie](https://github.com/chonkie-inc/chonkie) for chunking. It's lightweight, fast (33x faster with C extensions), and provides all the strategies covered above.

#### Installation

```bash
pip install chonkie
# Or with SIMD acceleration:
pip install chonkie[fast]
```

#### Quick Reference

```python {title="chonkie_overview.py" test="skip" lint="skip"}
from chonkie import (
    TokenChunker,       # Fixed-size token chunks
    SentenceChunker,    # Sentence-based
    RecursiveChunker,   # Hierarchical with separators
    SemanticChunker,    # Embedding-based similarity
    CodeChunker,        # Syntax-aware for code
    SlumberChunker,     # LLM-based (agentic)
)

# All chunkers share the same interface (some support chunk_overlap)
chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
chunks = chunker.chunk(document)

for chunk in chunks:
    print(f'Text: {chunk.text[:50]}...')
    print(f'Tokens: {chunk.token_count}')
```

#### Pipeline Support

Chain multiple chunking strategies:

```python {title="chonkie_pipeline.py" test="skip" lint="skip"}
from chonkie import Pipeline

pipeline = Pipeline([
    ('recursive', {'chunk_size': 1024}),   # First pass: large chunks
    ('sentence', {'chunk_size': 512}),     # Second pass: refine
])

chunks = pipeline.chunk(document)
```

### Advanced Techniques (2024-2025)

#### Late Chunking

**What it is**: Instead of chunking first and then embedding, late chunking embeds the entire document through the transformer first, then applies chunking to the token embeddings.

**Key benefit**: Chunk embeddings retain awareness of surrounding document context. Solves the "lost reference" problem where "the city" in one chunk can't be linked to "Paris" mentioned earlier.

**Performance**: Traditional chunking shows 70-75% similarity to target terms; late chunking achieves 82-84%.

**Requirements**: Long-context embedding model (e.g., jina-embeddings-v3 with 8,192 tokens)

```python {title="late_chunking.py" test="skip" lint="skip"}
from chonkie import LateChunker

chunker = LateChunker(
    chunk_size=512,
    embedding_model='jinaai/jina-embeddings-v3',
)

chunks = chunker.chunk(document)
# Each chunk.embedding is context-aware
```

See [Jina AI's late chunking paper](https://arxiv.org/abs/2409.04701) for details.

#### Contextual Retrieval

**What it is**: Prepends a context snippet to each chunk before embedding, explaining where the chunk fits in the document.

**The problem**: A chunk might say "The company's revenue grew by 3%" without specifying which company or time period.

**Solution**:

1. For each chunk, use an LLM to generate a brief context snippet
2. Prepend context to chunk before embedding and indexing
3. Context explains document source, section, and relevant details

**Performance** (from [Anthropic's research](https://www.anthropic.com/news/contextual-retrieval)):

- Contextual embeddings alone: **35% reduction** in retrieval failure
- With BM25: **49% reduction**
- With reranking: **67% reduction** (5.7% failure down to 1.9%)

```python {title="contextual_retrieval.py"}
from pydantic_ai import Agent

# Use an agent to generate context for each chunk
context_agent = Agent('openai:gpt-5-mini')


async def add_context(chunk: str, full_document: str) -> str:
    result = await context_agent.run(
        f'''Generate a brief context (1-2 sentences) for this chunk.
        Explain what document it's from and what section/topic it covers.

        Full document (first 1000 chars): {full_document[:1000]}...

        Chunk: {chunk}
        '''
    )
    return f'{result.output}\n\n{chunk}'
```

#### Hybrid Retrieval

Combine dense embeddings with sparse retrieval (BM25) for better coverage:

1. **Dense retrieval**: Semantic similarity via embeddings
2. **Sparse retrieval**: Keyword matching via BM25/TF-IDF
3. **Rank fusion**: Combine results using reciprocal rank fusion

This catches both semantic matches and exact keyword matches that embeddings might miss.

### Quick Reference

#### Decision Flowchart

```
Is your document <500 pages and can fit in context?
├── Yes → Include full document in prompt, skip RAG
└── No → Continue to chunking

What's your content type?
├── General text → RecursiveChunker (400-512 tokens, 10-20% overlap)
├── Markdown/Docs → Markdown-aware separators + recursive
├── Code → CodeChunker with appropriate language
├── PDFs → Structure extraction + chunking
└── Mixed/Complex → Consider agentic chunking if budget allows

Evaluate on your test set
├── Retrieval quality good → Done!
└── Retrieval quality poor → Try:
    ├── Late chunking (with Jina embeddings)
    ├── Contextual retrieval (prepend context)
    ├── Hybrid BM25 + dense retrieval
    └── Experiment with chunk sizes
```

#### Key Numbers to Remember

| Parameter | Value | Notes |
|-----------|-------|-------|
| Default chunk size | 400-512 tokens | Good starting point |
| Overlap | 10-20% | 50-100 tokens for 512 chunk |
| OpenAI embedding limit | 8,191 tokens | text-embedding-3-* |
| Cohere v3 limit | 512 tokens | Optimal performance threshold |
| Top-k retrieval | 5-20 chunks | Start with 5, increase if needed |

## Providers

### OpenAI

[`OpenAIEmbeddingModel`][pydantic_ai.embeddings.openai.OpenAIEmbeddingModel] works with OpenAI's embeddings API and any [OpenAI-compatible provider](models/openai.md#openai-compatible-models).

#### Install

To use OpenAI embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

#### Configuration

To use `OpenAIEmbeddingModel` with the OpenAI API, go to [platform.openai.com](https://platform.openai.com/) and follow your nose until you find the place to generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key'
```

You can then use the model:

```python {title="openai_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1536
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See [OpenAI's embedding models](https://platform.openai.com/docs/guides/embeddings) for available models.

#### Dimension Control

OpenAI's `text-embedding-3-*` models support dimension reduction via the `dimensions` setting:

```python {title="openai_dimensions.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

embedder = Embedder(
    'openai:text-embedding-3-small',
    settings=EmbeddingSettings(dimensions=256),
)


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 256
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

#### OpenAI-Compatible Providers {#openai-compatible}

Since [`OpenAIEmbeddingModel`][pydantic_ai.embeddings.openai.OpenAIEmbeddingModel] uses the same provider system as [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], you can use it with any [OpenAI-compatible provider](models/openai.md#openai-compatible-models):

```python {title="openai_compatible_embeddings.py"}
# Using Azure OpenAI
from openai import AsyncAzureOpenAI

from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

azure_client = AsyncAzureOpenAI(
    azure_endpoint='https://your-resource.openai.azure.com',
    api_version='2024-02-01',
    api_key='your-azure-key',
)
model = OpenAIEmbeddingModel(
    'text-embedding-3-small',
    provider=OpenAIProvider(openai_client=azure_client),
)
embedder = Embedder(model)


# Using any OpenAI-compatible API
model = OpenAIEmbeddingModel(
    'your-model-name',
    provider=OpenAIProvider(
        base_url='https://your-provider.com/v1',
        api_key='your-api-key',
    ),
)
embedder = Embedder(model)
```

For providers with dedicated provider classes (like [`OllamaProvider`][pydantic_ai.providers.ollama.OllamaProvider] or [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider]), you can use the shorthand syntax:

```python
from pydantic_ai import Embedder

embedder = Embedder('azure:text-embedding-3-small')
embedder = Embedder('ollama:nomic-embed-text')
```

See [OpenAI-compatible Models](models/openai.md#openai-compatible-models) for the full list of supported providers.

### Google

[`GoogleEmbeddingModel`][pydantic_ai.embeddings.google.GoogleEmbeddingModel] works with Google's embedding models via the Gemini API (Google AI Studio) or Vertex AI.

#### Install

To use Google embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `google` optional group:

```bash
pip/uv-add "pydantic-ai-slim[google]"
```

#### Configuration

To use `GoogleEmbeddingModel` with the Gemini API, go to [aistudio.google.com](https://aistudio.google.com/) and generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export GOOGLE_API_KEY='your-api-key'
```

You can then use the model:

```python {title="google_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('google-gla:gemini-embedding-001')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 3072
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [Google Embeddings documentation](https://ai.google.dev/gemini-api/docs/embeddings) for available models.

##### Vertex AI

To use Google's embedding models via Vertex AI instead of the Gemini API, use the `google-vertex` provider prefix:

```python {title="google_vertex_embeddings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.providers.google import GoogleProvider

# Using provider prefix
embedder = Embedder('google-vertex:gemini-embedding-001')

# Or with explicit provider configuration
model = GoogleEmbeddingModel(
    'gemini-embedding-001',
    provider=GoogleProvider(vertexai=True, project='my-project', location='us-central1'),
)
embedder = Embedder(model)
```

See the [Google provider documentation](models/google.md#vertex-ai-enterprisecloud) for more details on Vertex AI authentication options, including application default credentials, service accounts, and API keys.

#### Dimension Control

Google's embedding models support dimension reduction via the `dimensions` setting:

```python {title="google_dimensions.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

embedder = Embedder(
    'google-gla:gemini-embedding-001',
    settings=EmbeddingSettings(dimensions=768),
)


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 768
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

#### Google-Specific Settings

Google models support additional settings via [`GoogleEmbeddingSettings`][pydantic_ai.embeddings.google.GoogleEmbeddingSettings]:

```python {title="google_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.google import GoogleEmbeddingSettings

embedder = Embedder(
    'google-gla:gemini-embedding-001',
    settings=GoogleEmbeddingSettings(
        dimensions=768,
        google_task_type='SEMANTIC_SIMILARITY',  # Optimize for similarity comparison
    ),
)
```

See [Google's task type documentation](https://ai.google.dev/gemini-api/docs/embeddings#task-types) for available task types. By default, `embed_query()` uses `RETRIEVAL_QUERY` and `embed_documents()` uses `RETRIEVAL_DOCUMENT`.

### Cohere

[`CohereEmbeddingModel`][pydantic_ai.embeddings.cohere.CohereEmbeddingModel] provides access to Cohere's embedding models, which offer multilingual support and various model sizes.

#### Install

To use Cohere embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `cohere` optional group:

```bash
pip/uv-add "pydantic-ai-slim[cohere]"
```

#### Configuration

To use `CohereEmbeddingModel`, go to [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys) and follow your nose until you find the place to generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export CO_API_KEY='your-api-key'
```

You can then use the model:

```python {title="cohere_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('cohere:embed-v4.0')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1024
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [Cohere Embed documentation](https://docs.cohere.com/docs/cohere-embed) for available models.

#### Cohere-Specific Settings

Cohere models support additional settings via [`CohereEmbeddingSettings`][pydantic_ai.embeddings.cohere.CohereEmbeddingSettings]:

```python {title="cohere_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.cohere import CohereEmbeddingSettings

embedder = Embedder(
    'cohere:embed-v4.0',
    settings=CohereEmbeddingSettings(
        dimensions=512,
        cohere_truncate='END',  # Truncate long inputs instead of erroring
        cohere_max_tokens=256,  # Limit tokens per input
    ),
)
```

### VoyageAI

[`VoyageAIEmbeddingModel`][pydantic_ai.embeddings.voyageai.VoyageAIEmbeddingModel] provides access to VoyageAI's embedding models, which are optimized for retrieval with specialized models for code, finance, and legal domains.

#### Install

To use VoyageAI embedding models, you need to install `pydantic-ai-slim` with the `voyageai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[voyageai]"
```

#### Configuration

To use `VoyageAIEmbeddingModel`, go to [dash.voyageai.com](https://dash.voyageai.com/) to generate an API key. Once you have the API key, you can set it as an environment variable:

```bash
export VOYAGE_API_KEY='your-api-key'
```

You can then use the model:

```python {title="voyageai_embeddings.py"}
from pydantic_ai import Embedder

embedder = Embedder('voyageai:voyage-3.5')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1024
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [VoyageAI Embeddings documentation](https://docs.voyageai.com/docs/embeddings) for available models.

#### VoyageAI-Specific Settings

VoyageAI models support additional settings via [`VoyageAIEmbeddingSettings`][pydantic_ai.embeddings.voyageai.VoyageAIEmbeddingSettings]:

```python {title="voyageai_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingSettings

embedder = Embedder(
    'voyageai:voyage-3.5',
    settings=VoyageAIEmbeddingSettings(
        dimensions=512,  # Reduce output dimensions
        voyageai_input_type='document',  # Override input type for all requests
    ),
)
```

### Bedrock

[`BedrockEmbeddingModel`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingModel] provides access to embedding models through AWS Bedrock, including Amazon Titan, Cohere, and Amazon Nova models.

#### Install

To use Bedrock embedding models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `bedrock` optional group:

```bash
pip/uv-add "pydantic-ai-slim[bedrock]"
```

#### Configuration

Authentication with AWS Bedrock uses standard AWS credentials. See the [Bedrock provider documentation](models/bedrock.md#environment-variables) for details on configuring credentials via environment variables, AWS credentials file, or IAM roles.

Ensure your AWS account has access to the Bedrock embedding models you want to use. See [AWS Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) for details.

#### Basic Usage

```python {title="bedrock_embeddings.py" test="skip"}
from pydantic_ai import Embedder

# Using Amazon Titan
embedder = Embedder('bedrock:amazon.titan-embed-text-v2:0')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 1024
```

_(This example requires AWS credentials configured)_

#### Supported Models

Bedrock supports three families of embedding models. See the [AWS Bedrock documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html) for the full list of available models.

**Amazon Titan:**

- `amazon.titan-embed-text-v1` — 1536 dimensions (fixed), 8K tokens
- `amazon.titan-embed-text-v2:0` — 256/384/1024 dimensions (configurable, default: 1024), 8K tokens

**Cohere Embed:**

- `cohere.embed-english-v3` — English-only, 1024 dimensions (fixed), 512 tokens
- `cohere.embed-multilingual-v3` — Multilingual, 1024 dimensions (fixed), 512 tokens
- `cohere.embed-v4:0` — 256/512/1024/1536 dimensions (configurable, default: 1536), 128K tokens

**Amazon Nova:**

- `amazon.nova-2-multimodal-embeddings-v1:0` — 256/384/1024/3072 dimensions (configurable, default: 3072), 8K tokens

#### Titan-Specific Settings

Titan v2 supports vector normalization for direct similarity calculations via `bedrock_titan_normalize` (default: `True`). Titan v1 does not support this setting.

```python {title="bedrock_titan.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

embedder = Embedder(
    'bedrock:amazon.titan-embed-text-v2:0',
    settings=BedrockEmbeddingSettings(
        dimensions=512,
        bedrock_titan_normalize=True,
    ),
)
```

!!! note
    Titan models do not support the `truncate` setting. The `dimensions` setting is only supported by Titan v2.

#### Cohere-Specific Settings

Cohere models on Bedrock support additional settings via [`BedrockEmbeddingSettings`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingSettings]:

- `bedrock_cohere_input_type` — By default, `embed_query()` uses `'search_query'` and `embed_documents()` uses `'search_document'`. Also accepts `'classification'` or `'clustering'`.
- `bedrock_cohere_truncate` — Fine-grained truncation control: `'NONE'` (default, error on overflow), `'START'`, or `'END'`. Overrides the base `truncate` setting.
- `bedrock_cohere_max_tokens` — Limits tokens per input (default: 128000). Only supported by Cohere v4.

```python {title="bedrock_cohere.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

embedder = Embedder(
    'bedrock:cohere.embed-v4:0',
    settings=BedrockEmbeddingSettings(
        dimensions=512,
        bedrock_cohere_max_tokens=1000,
        bedrock_cohere_truncate='END',
    ),
)
```

!!! note
    The `dimensions` and `bedrock_cohere_max_tokens` settings are only supported by Cohere v4. Cohere v3 models have fixed 1024 dimensions.

#### Nova-Specific Settings

Nova models on Bedrock support additional settings via [`BedrockEmbeddingSettings`][pydantic_ai.embeddings.bedrock.BedrockEmbeddingSettings]:

- `bedrock_nova_truncate` — Fine-grained truncation control: `'NONE'` (default, error on overflow), `'START'`, or `'END'`. Overrides the base `truncate` setting.
- `bedrock_nova_embedding_purpose` — By default, `embed_query()` uses `'GENERIC_RETRIEVAL'` and `embed_documents()` uses `'GENERIC_INDEX'`. Also accepts `'TEXT_RETRIEVAL'`, `'CLASSIFICATION'`, or `'CLUSTERING'`.

```python {title="bedrock_nova.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

embedder = Embedder(
    'bedrock:amazon.nova-2-multimodal-embeddings-v1:0',
    settings=BedrockEmbeddingSettings(
        dimensions=1024,
        bedrock_nova_embedding_purpose='TEXT_RETRIEVAL',
        truncate=True,
    ),
)
```

#### Concurrency Settings

Models that don't support batch embedding (Titan and Nova) make individual API requests for each input text. By default, these requests run concurrently with a maximum of 5 parallel requests.

You can adjust this with the `bedrock_max_concurrency` setting:

```python {title="bedrock_concurrency.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingSettings

# Increase concurrency for faster throughput
embedder = Embedder(
    'bedrock:amazon.titan-embed-text-v2:0',
    settings=BedrockEmbeddingSettings(bedrock_max_concurrency=10),
)

# Or reduce concurrency to avoid rate limits
embedder = Embedder(
    'bedrock:amazon.nova-2-multimodal-embeddings-v1:0',
    settings=BedrockEmbeddingSettings(bedrock_max_concurrency=2),
)
```

#### Regional Prefixes (Cross-Region Inference)

Bedrock supports cross-region inference using geographic prefixes like `us.`, `eu.`, or `apac.`:

```python {title="bedrock_regional.py"}
from pydantic_ai import Embedder

embedder = Embedder('bedrock:us.amazon.titan-embed-text-v2:0')
```

#### Using a Custom Provider

For advanced configuration like explicit credentials or a custom boto3 client, you can create a [`BedrockProvider`][pydantic_ai.providers.bedrock.BedrockProvider] directly. See the [Bedrock provider documentation](models/bedrock.md#provider-argument) for more details.

```python {title="bedrock_provider.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.bedrock import BedrockEmbeddingModel
from pydantic_ai.providers.bedrock import BedrockProvider

provider = BedrockProvider(
    region_name='us-west-2',
    aws_access_key_id='your-access-key',
    aws_secret_access_key='your-secret-key',
)

model = BedrockEmbeddingModel('amazon.titan-embed-text-v2:0', provider=provider)
embedder = Embedder(model)
```

!!! note "Token Counting"
    Bedrock embedding models do not support the `count_tokens()` method because AWS Bedrock's token counting API only works with text generation models (Claude, Llama, etc.), not embedding models. Calling `count_tokens()` will raise `NotImplementedError`.

### Sentence Transformers (Local)

[`SentenceTransformerEmbeddingModel`][pydantic_ai.embeddings.sentence_transformers.SentenceTransformerEmbeddingModel] runs embeddings locally using the [sentence-transformers](https://www.sbert.net/) library. This is ideal for:

- **Privacy** — Data never leaves your infrastructure
- **Cost** — No API charges for high-volume workloads
- **Offline use** — No internet connection required after model download

#### Install

To use Sentence Transformers embedding models, you need to install `pydantic-ai-slim` with the `sentence-transformers` optional group:

```bash
pip/uv-add "pydantic-ai-slim[sentence-transformers]"
```

#### Usage

```python {title="sentence_transformers_embeddings.py"}
from pydantic_ai import Embedder

# Model is downloaded from Hugging Face on first use
embedder = Embedder('sentence-transformers:all-MiniLM-L6-v2')


async def main():
    result = await embedder.embed_query('Hello world')
    print(len(result.embeddings[0]))
    #> 384
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

See the [Sentence-Transformers pretrained models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) documentation for available models.

#### Device Selection

Control which device to use for inference:

```python {title="sentence_transformers_device.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingSettings,
)

embedder = Embedder(
    'sentence-transformers:all-MiniLM-L6-v2',
    settings=SentenceTransformersEmbeddingSettings(
        sentence_transformers_device='cuda',  # Use GPU
        sentence_transformers_normalize_embeddings=True,  # L2 normalize
    ),
)
```

#### Using an Existing Model Instance

If you need more control over model initialization:

```python {title="sentence_transformers_instance.py"}
from sentence_transformers import SentenceTransformer

from pydantic_ai import Embedder
from pydantic_ai.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingModel,
)

# Create and configure the model yourself
st_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Wrap it for use with Pydantic AI
model = SentenceTransformerEmbeddingModel(st_model)
embedder = Embedder(model)
```

## Settings

[`EmbeddingSettings`][pydantic_ai.embeddings.EmbeddingSettings] provides common configuration options that work across providers:

- `dimensions`: Reduce the output embedding dimensions (supported by OpenAI, Google, Cohere, Bedrock, VoyageAI)
- `truncate`: When `True`, truncate input text that exceeds the model's context length instead of raising an error (supported by Cohere, Bedrock, VoyageAI)

Settings can be specified at the embedder level (applied to all calls) or per-call:

```python {title="embedding_settings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings

# Default settings for all calls
embedder = Embedder(
    'openai:text-embedding-3-small',
    settings=EmbeddingSettings(dimensions=512),
)


async def main():
    # Override for a specific call
    result = await embedder.embed_query(
        'Hello world',
        settings=EmbeddingSettings(dimensions=256),
    )
    print(len(result.embeddings[0]))
    #> 256
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Token Counting

You can check token counts before embedding to avoid exceeding model limits:

```python {title="token_counting.py"}
from pydantic_ai import Embedder

embedder = Embedder('openai:text-embedding-3-small')


async def main():
    text = 'Hello world, this is a test.'

    # Count tokens in text
    token_count = await embedder.count_tokens(text)
    print(f'Tokens: {token_count}')
    #> Tokens: 7

    # Check model's maximum input tokens (returns None if unknown)
    max_tokens = await embedder.max_input_tokens()
    print(f'Max tokens: {max_tokens}')
    #> Max tokens: 1024
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Testing

Use [`TestEmbeddingModel`][pydantic_ai.embeddings.TestEmbeddingModel] for testing without making API calls:

```python {title="testing_embeddings.py"}
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel


async def test_my_rag_system():
    embedder = Embedder('openai:text-embedding-3-small')
    test_model = TestEmbeddingModel()

    with embedder.override(model=test_model):
        result = await embedder.embed_query('test query')

        # TestEmbeddingModel returns deterministic embeddings
        assert result.embeddings[0] == [1.0] * 8

        # Check what settings were used
        assert test_model.last_settings is not None
```

## Instrumentation

Enable OpenTelemetry instrumentation for debugging and monitoring:

```python {title="instrumented_embeddings.py"}
import logfire

from pydantic_ai import Embedder

logfire.configure()

# Instrument a specific embedder
embedder = Embedder('openai:text-embedding-3-small', instrument=True)

# Or instrument all embedders globally
Embedder.instrument_all()
```

See the [Debugging and Monitoring guide](logfire.md) for more details on using Logfire with Pydantic AI.

## Building Custom Embedding Models

To integrate a custom embedding provider, subclass [`EmbeddingModel`][pydantic_ai.embeddings.EmbeddingModel]:

```python {title="custom_embedding_model.py"}
from collections.abc import Sequence

from pydantic_ai.embeddings import EmbeddingModel, EmbeddingResult, EmbeddingSettings
from pydantic_ai.embeddings.result import EmbedInputType


class MyCustomEmbeddingModel(EmbeddingModel):
    @property
    def model_name(self) -> str:
        return 'my-custom-model'

    @property
    def system(self) -> str:
        return 'my-provider'

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: EmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        inputs, settings = self.prepare_embed(inputs, settings)

        # Call your embedding API here
        embeddings = [[0.1, 0.2, 0.3] for _ in inputs]  # Placeholder

        return EmbeddingResult(
            embeddings=embeddings,
            inputs=inputs,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
        )
```

Use [`WrapperEmbeddingModel`][pydantic_ai.embeddings.WrapperEmbeddingModel] if you want to wrap an existing model to add custom behavior like caching or logging.
