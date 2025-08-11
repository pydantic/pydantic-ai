# LanceDB Multimodal (E‑commerce RAG)

This example turns LanceDB into a simple, local product catalog for multimodal RAG. It fetches products from a live API, stores images and vector embeddings together, and exposes a tool the agent uses to perform semantic search with SQL-like filters. Results render as a quick collage for visual feedback.

- Object storage: store images next to vectors
- Embedded: fast local prototyping with transactional storage

Demonstrates:

- [tools](../tools.md)
- [agent dependencies](../dependencies.md)
- Vector search with LanceDB (CLIP embeddings)
- Object storage of product images in LanceDB
- SQL-like metadata filtering (category, price)


## Installation

```bash
pip install lancedb sentence-transformers torch httpx pandas Pillow
```

Set your Google API key (agent text generation):

```bash
export GOOGLE_API_KEY=your_api_key_here
```

## Usage

Build the product database from the live API:

```bash
uv run -m pydantic_ai_examples.lancedb_multimodal build
```

Search for products (hybrid search):

```bash
uv run -m pydantic_ai_examples.lancedb_multimodal search "a cool t-shirt in men's clothing under 20 dollars"
```

```bash
uv run -m pydantic_ai_examples.lancedb_multimodal search "An external SSD with 1TB or more storage"
```

## Architecture

### Data Schema (Pydantic model)

```python {test="skip"}
from lancedb.pydantic import LanceModel, Vector


class ProductVector(LanceModel):
    id: int
    title: str
    price: float
    description: str
    category: str
    image: bytes
    embedding: Vector(512)  # CLIP 'clip-ViT-B-32' embedding size
```
