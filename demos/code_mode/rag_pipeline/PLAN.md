# RAG Pipeline Demo

## Overview

Demonstrates code mode efficiency gains through a multi-step RAG pipeline:
search → conditional fetch → upsert → rerank → answer.

## Why This Demo

- **Multi-step pipeline**: 5+ operations per query
- **Conditional logic**: exists check determines path
- **Sequential agent**: 5+ LLM turns per query × N queries = many turns
- **Code mode**: single run_code with loop handles all queries

## Expected Code Mode Output

```python
questions = [
    'How do I create a streaming agent in Pydantic AI?',
    'What output types does Pydantic AI support?',
    'How do I add custom tools to an agent?',
    'What is the difference between run() and run_stream()?',
    'How do I handle retries in Pydantic AI?'
]

answers = {}

for q in questions:
    # Step 1: Search existing knowledge
    results = search_records(name='pydantic-kb', namespace='default',
                            query={'topK': 3, 'inputs': {'text': q}})

    # Step 2: Check if good result exists
    top_score = 0
    if results['matches']:
        top_score = results['matches'][0]['score']

    if top_score < 0.7:
        # Step 3: Fetch from web
        web_result = tavily_search(query=q + ' pydantic ai')

        # Step 4: Upsert to index
        upsert_records(name='pydantic-kb', namespace='default',
                      records=[{'_id': q, 'content': web_result['snippet']}])

        # Step 5: Search again
        results = search_records(name='pydantic-kb', namespace='default',
                                query={'topK': 3, 'inputs': {'text': q}})

    # Step 6: Rerank
    docs = []
    for m in results['matches']:
        docs.append(m['fields']['content'])

    reranked = rerank_documents(model='pinecone-rerank-v0',
                                query=q, documents=docs)

    # Step 7: Best answer
    answers[q] = reranked['results'][0]['document']

answers
```

## Sequential vs Code Mode

| Metric | Sequential Agent | Code Mode |
|--------|-----------------|-----------|
| LLM turns per query | ~6 | 1-2 total |
| Total for 5 queries | ~30 turns | 1-2 turns |
| Token overhead | High (tool schemas each turn) | Low |

## Usage

### Full Mode (requires API keys)

```bash
# Set environment variables
export PINECONE_API_KEY=pc-xxx
export TAVILY_API_KEY=tvly-xxx

# One-time: Create index and seed FAQs
source .env && uv run python demos/code_mode/rag_pipeline/setup_index.py

# Run web demo
source .env && uv run python demos/code_mode/rag_pipeline/web.py

# Run batch evals
source .env && uv run python demos/code_mode/rag_pipeline/evals.py
```

### Zero-Setup Mode (no API keys)

```bash
# Run web demo with search_docs only
uv run python demos/code_mode/rag_pipeline/web.py --zero-setup

# Run batch evals
uv run python demos/code_mode/rag_pipeline/evals.py --zero-setup
```

## MCP Servers

**Pinecone MCP** (`@pinecone-database/mcp`):
- `search_records` - vector search with topK
- `upsert_records` - add records to index
- `rerank_documents` - rerank results
- `search_docs` - search Pinecone docs (no key needed)

**Tavily MCP** (`https://mcp.tavily.com/mcp/`):
- `tavily_search` - web search
- `tavily_extract` - extract from URLs

## Monty Compatibility

All operations valid in monty sandbox:
- for loops ✓
- dict/list access ✓
- list.append() ✓
- dict assignment ✓
- conditionals ✓
- no imports needed ✓

Avoided pitfalls:
- No datetime (not needed)
- No slice ops (use index access)
- No string methods beyond `+`
- No comprehensions (explicit for loops)
- No imports (all inline)
