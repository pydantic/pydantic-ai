"""Chunking strategies demo — comparing different approaches for splitting documents.

This example demonstrates:
- Different chunking strategies using chonkie
- How to embed chunks using pydantic-ai's Embedder
- Simple in-memory similarity search (no external database needed)

Run with:

    uv run -m pydantic_ai_examples.chunking_demo
"""

from __future__ import annotations as _annotations

import asyncio
from dataclasses import dataclass

from chonkie import RecursiveChunker, SentenceChunker, TokenChunker

from pydantic_ai import Embedder

# Sample document for demonstration
SAMPLE_DOCUMENT = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed. It focuses on developing
computer programs that can access data and use it to learn for themselves.

## Types of Machine Learning

### Supervised Learning

Supervised learning uses labeled datasets to train algorithms to classify data or predict
outcomes accurately. As input data is fed into the model, it adjusts its weights until
the model has been fitted appropriately. This occurs as part of the cross validation process.

Common algorithms include linear regression, logistic regression, decision trees, and
support vector machines. Supervised learning is widely used in applications like spam
detection, image recognition, and medical diagnosis.

### Unsupervised Learning

Unsupervised learning uses machine learning algorithms to analyze and cluster unlabeled
datasets. These algorithms discover hidden patterns or data groupings without the need
for human intervention.

Popular techniques include clustering (K-means, hierarchical), dimensionality reduction
(PCA, t-SNE), and association rule learning. Applications include customer segmentation,
anomaly detection, and recommendation systems.

### Reinforcement Learning

Reinforcement learning is a type of machine learning where an agent learns to make
decisions by performing actions in an environment to maximize cumulative reward.
The agent learns from trial and error, receiving feedback in the form of rewards
or penalties.

Key concepts include the agent, environment, state, action, and reward. It's used
in robotics, game playing (like AlphaGo), autonomous vehicles, and resource management.

## Getting Started

To get started with machine learning, you'll need:

1. A solid understanding of mathematics (linear algebra, calculus, statistics)
2. Programming skills (Python is the most popular language)
3. Familiarity with ML libraries (scikit-learn, TensorFlow, PyTorch)
4. Access to datasets for practice

Start with simple projects and gradually increase complexity as you build understanding.
"""


@dataclass
class Chunk:
    """A chunk with its embedding for similarity search."""

    text: str
    embedding: list[float]
    strategy: str


async def main():
    """Demonstrate different chunking strategies and similarity search."""
    print('=== Chunking Strategies Demo ===\n')

    # Initialize the embedder
    embedder = Embedder('openai:text-embedding-3-small')

    # Demonstrate different chunking strategies
    # Note: RecursiveChunker doesn't support chunk_overlap (uses rules instead)
    strategies = {
        'token': TokenChunker(chunk_size=256, chunk_overlap=25),
        'recursive': RecursiveChunker(chunk_size=256),
        'sentence': SentenceChunker(chunk_size=256, chunk_overlap=25),
    }

    all_chunks: list[Chunk] = []

    for name, chunker in strategies.items():
        print(f'\n--- {name.upper()} CHUNKING ---')
        chunks = chunker.chunk(SAMPLE_DOCUMENT)
        print(f'Number of chunks: {len(chunks)}')

        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            preview = chunk.text[:100].replace('\n', ' ')
            print(f'  Chunk {i + 1}: "{preview}..."')
            print(f'    Tokens: {chunk.token_count}')

        # Embed the chunks
        texts = [c.text for c in chunks]
        result = await embedder.embed_documents(texts)

        # Store chunks with embeddings
        for chunk, emb in zip(chunks, result.embeddings, strict=False):
            all_chunks.append(
                Chunk(text=chunk.text, embedding=list(emb), strategy=name)
            )

        print(f'  Embedded {len(chunks)} chunks')

    # Simple similarity search demo
    print('\n\n=== SIMILARITY SEARCH DEMO ===')

    queries = [
        'What is supervised learning?',
        'How does reinforcement learning work?',
        'What do I need to start with ML?',
    ]

    for query in queries:
        print(f'\nQuery: "{query}"')

        # Embed the query
        query_result = await embedder.embed_query(query)
        query_embedding = query_result.embeddings[0]

        # Find most similar chunks (simple cosine similarity)
        similarities = []
        for chunk in all_chunks:
            sim = cosine_similarity(query_embedding, chunk.embedding)
            similarities.append((sim, chunk))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[0], reverse=True)

        # Show top 3 results
        print('  Top 3 results:')
        for sim, chunk in similarities[:3]:
            preview = chunk.text[:80].replace('\n', ' ')
            print(f'    [{chunk.strategy}] (sim={sim:.3f}) "{preview}..."')


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


if __name__ == '__main__':
    asyncio.run(main())
