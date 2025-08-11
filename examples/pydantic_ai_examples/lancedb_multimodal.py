"""Multimodal RAG with LanceDB.

What this does:
- Fetches a product catalog from a live API (fakestoreapi.com).
- Embeds product descriptions using a CLIP model (clip-ViT-B-32) and stores them in LanceDB.
- Implements a RAG agent with a `find_products` tool that can:
    1. Perform semantic search (e.g., "something for the cold weather").
    2. Perform metadata filtering (e.g., "show me all electronics").
       (e.g., "a cool t-shirt" from the "men's clothing" category under $20).
- Generates and displays an image collage of the results for instant visual feedback.
- Creates logfire tracing dashboard if api key is set

Install dependencies:
    pip install lancedb sentence-transformers torch httpx pandas Pillow logfire[httpx]

Set your Google API key (for the agent's text generation):
    export GOOGLE_API_KEY=your_api_key_here

Usage:
    # First, build the product database from the live API
    python lancedb_multimodal.py build

    # Then, ask for a recommendation:
    python lancedb_multimodal.py search "a cool t-shirt in men's clothing under 20 dollars"
"""

import asyncio
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast
from pydantic_ai import Agent, RunContext

import httpx

try:
    import lancedb
    import logfire
    from lancedb.pydantic import LanceModel, Vector
    from PIL import Image
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        """Missing dependencies. To run this example, please install the required packages by running:
pip install lancedb sentence-transformers torch httpx pandas Pillow logfire[httpx]"""
    )
    sys.exit(0)


# ruff: noqa
# pyright: reportMissingImports=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUntypedBaseClass=false

# 'if-token-present' means nothing will be sent if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

DATA_DIR = Path('./lancedb_data/products')


# LanceDB schema
class ProductVector(LanceModel):
    id: int
    title: str
    price: float
    description: str
    category: str
    image: bytes
    embedding: Vector(512)  # CLIP 'clip-ViT-B-32' model produces 512-dim vectors


@dataclass
class Deps:
    db: lancedb.DBConnection
    embedding_model: SentenceTransformer


agent = Agent(
    'google-gla:gemini-2.5-flash',
    deps_type=Deps,
    system_prompt=(
        'You are a helpful AI Shopping Assistant. Your goal is to help users find the perfect product '
        'by using the `find_products` tool. You can search by a text query, filter by category and '
        'price, or combine all three for a powerful search. '
        'After getting the results, present them clearly to the user and mention that you are '
        'displaying a collage of the findings.'
    ),
)


async def _generate_image_collage(image_bytes_list: list[bytes], title: str):
    if not image_bytes_list:
        return

    with logfire.span(
        'generate image collage', num_images=len(image_bytes_list), title=title
    ):
        images = [
            Image.open(io.BytesIO(image_bytes)) for image_bytes in image_bytes_list
        ]

        if not images:
            print('Could not create any images from bytes to create a collage.')
            return

        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)
        collage = Image.new('RGB', (total_width, max_height))
        x_offset = 0
        for img in images:
            collage.paste(img, (x_offset, 0))
            x_offset += img.width

        collage.show(title=title)


@agent.tool
async def find_products(
    context: RunContext[Deps],
    query: Optional[str] = None,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    top_k: int = 4,
) -> str:
    """Finds products using semantic search, metadata filtering, or both (hybrid search)."""
    table = context.deps.db.open_table('products')

    query_embedding = None
    if query and query.strip():
        with logfire.span('encode semantic query', query=query):
            query_embedding = context.deps.embedding_model.encode(
                query, convert_to_tensor=False
            )

    # Use vector search when a semantic query is provided, otherwise fall back to metadata-only search
    searcher = (
        table.search(query_embedding) if query_embedding is not None else table.search()
    )

    conditions = []
    if category:
        safe_category = category.replace("'", "''")
        conditions.append(f"category = '{safe_category}'")
    if min_price is not None:
        conditions.append(f'price >= {min_price}')
    if max_price is not None:
        conditions.append(f'price <= {max_price}')

    if conditions:
        where_clause = ' AND '.join(conditions)
        searcher = searcher.where(where_clause)

    with logfire.span('search products', top_k=top_k, conditions=conditions):
        results_df = searcher.limit(top_k).to_pandas()

    if results_df.empty:
        return 'No products found matching your criteria.'

    logfire.info('Displaying collage of results ({n})', n=len(results_df))
    await _generate_image_collage(
        cast(list[bytes], results_df['image'].tolist()), title='Product Results'
    )

    # Don't return image byte string in the text response
    results_df_no_image = results_df.drop(columns=['image'])
    results_json = results_df_no_image.to_json(orient='records')
    return f'Found {len(results_df)} products.\n Product details: {results_json}'


async def build_product_database():
    logfire.info('Building product database from Fake Store API...')
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch product data from the API
    with logfire.span('fetch product catalog', url='https://fakestoreapi.com/products'):
        async with httpx.AsyncClient() as client:
            response = await client.get('https://fakestoreapi.com/products', timeout=30)
            response.raise_for_status()
            products_data = response.json()

    # Initialize LanceDB and Embedding Model
    with logfire.span('initialize db and model'):
        db = lancedb.connect(DATA_DIR)
        embedding_model = SentenceTransformer('clip-ViT-B-32')

    # Create embeddings for product descriptions and fetch image bytes
    logfire.info(
        'Creating embeddings and fetching images for {n} products...',
        n=len(products_data),
    )
    product_vectors: list[ProductVector] = []
    async with httpx.AsyncClient() as client:
        for p_data in products_data:
            with logfire.span(
                'embed + download image',
                product_id=p_data.get('id'),
                category=p_data.get('category'),
            ):
                content_to_embed = f'Product Name: {p_data["title"]}\nCategory: {p_data["category"]}\nDescription: {p_data["description"]}'
                embedding = embedding_model.encode(
                    content_to_embed, convert_to_tensor=False
                )

                try:
                    image_response = await client.get(p_data['image'], timeout=30)
                    image_response.raise_for_status()
                    image_bytes = image_response.content
                    p_data['image'] = image_bytes
                    product_vectors.append(ProductVector(**p_data, embedding=embedding))
                except httpx.HTTPStatusError as e:
                    logfire.warning(
                        'Skipping product due to image download error: {e}', e=str(e)
                    )

    # Create a LanceDB table and add the data
    with logfire.span(
        'create lancedb table and add rows', num_rows=len(product_vectors)
    ):
        table = db.create_table('products', schema=ProductVector, mode='overwrite')
        table.add(product_vectors)

    logfire.info(
        'Successfully indexed {n} products into LanceDB.', n=len(product_vectors)
    )


async def run_search(query: str):
    db = lancedb.connect(DATA_DIR)
    embedding_model = SentenceTransformer('clip-ViT-B-32')
    deps = Deps(db=db, embedding_model=embedding_model)

    logfire.info('User Query: {query}', query=query)
    result = await agent.run(query, deps=deps)
    print(result.output)


def main():
    if 'search' in sys.argv and not os.getenv('GOOGLE_API_KEY'):
        raise ValueError(
            "GOOGLE_API_KEY environment variable is required for the 'search' action."
        )

    if len(sys.argv) < 2:
        print(
            'Usage:\n'
            '  python lancedb_multimodal.py build\n'
            '  python lancedb_multimodal.py search <query>',
            file=sys.stderr,
        )
        sys.exit(1)

    action = sys.argv[1]
    if action == 'build':
        asyncio.run(build_product_database())
    elif action == 'search':
        search_query = (
            ' '.join(sys.argv[2:])
            if len(sys.argv) > 2
            else 'An external SSD with 1TB or more storage'
        )
        asyncio.run(run_search(search_query))
    else:
        print(f'Unknown action: {action}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
