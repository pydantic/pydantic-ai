# Chunking Demo

A standalone demonstration of different chunking strategies for RAG applications.

Demonstrates:

- Comparing chunking strategies (token, recursive, sentence-based)
- Embedding chunks using pydantic-ai's [`Embedder`][pydantic_ai.embeddings.Embedder]
- Simple in-memory similarity search

This example doesn't require any external database — it's designed to help you understand and compare chunking strategies before building a full RAG system.

## Running the Demo

With [dependencies installed and environment variables set](./setup.md#usage):

```bash
python/uv-run -m pydantic_ai_examples.chunking_demo
```

!!! tip "No Database Required"
    This demo uses in-memory vectors for similarity search, so you don't need Docker or any external database to run it.

## What It Shows

The demo processes a sample document about machine learning through three different chunking strategies:

1. **Token Chunking** — Fixed-size token splits (fast, simple)
2. **Recursive Chunking** — Hierarchical splitting by separators (recommended default)
3. **Sentence Chunking** — NLP-based sentence boundary detection

For each strategy, it shows:

- Number of chunks produced
- Preview of chunk content
- Token count per chunk

Then it demonstrates similarity search by embedding queries and finding the most relevant chunks across all strategies.

## Example Output

```
=== Chunking Strategies Demo ===

--- TOKEN CHUNKING ---
Number of chunks: 8
  Chunk 1: "# Introduction to Machine Learning  Machine learning is a subset of artificial..."
    Tokens: 256
  Chunk 2: "commonly used in applications like spam detection, image recognition, and medical..."
    Tokens: 256

--- RECURSIVE CHUNKING ---
Number of chunks: 6
  Chunk 1: "# Introduction to Machine Learning  Machine learning is a subset of artificial..."
    Tokens: 248
  Chunk 2: "## Types of Machine Learning  ### Supervised Learning  Supervised learning uses..."
    Tokens: 251

--- SENTENCE CHUNKING ---
Number of chunks: 7
  Chunk 1: "# Introduction to Machine Learning Machine learning is a subset of artificial..."
    Tokens: 243

=== SIMILARITY SEARCH DEMO ===

Query: "What is supervised learning?"
  Top 3 results:
    [recursive] (sim=0.891) "## Types of Machine Learning  ### Supervised Learning..."
    [sentence] (sim=0.887) "Supervised learning uses labeled datasets to train..."
    [token] (sim=0.842) "commonly used in applications like spam detection..."
```

## Key Takeaways

1. **Token chunking** produces the most predictable chunk sizes but may split mid-sentence
2. **Recursive chunking** respects paragraph/sentence boundaries when possible — best default
3. **Sentence chunking** guarantees grammatical coherence but may produce uneven chunk sizes

See [Chunking Strategies](../embeddings.md#chunking-strategies) for detailed guidance on choosing the right strategy for your use case.

## Example Code

```snippet {path="/examples/pydantic_ai_examples/chunking_demo.py"}```
