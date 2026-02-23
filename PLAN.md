# Plan: Remove HTTP Client Cache + Add Provider Lifecycle Management

## Context

`cached_async_http_client` uses `@functools.cache` — a process-level global cache with no event-loop awareness. In multi-worker environments (GKE, Celery, serverless), workers share the same process but different event loops. The cached `httpx.AsyncClient` gets bound to one loop and fails on others: `RuntimeError: asyncio.locks.Event object is bound to a different event loop`.

Affects ALL providers — not just Gemini/Google — because they all go through `cached_async_http_client`. Users cannot work around it when using model strings like `google-vertex:...` since the cache is called internally by `infer_provider()` → `GoogleProvider.__init__()`.

- Issue: #3913, triggering report: #3913 comment by @antonacio
- Placeholder PR: #3944 (WIP, only touches OpenAI as sketch)
- Design: DouweM's Jan 7 comment on #3913

## Approach

Three-layer change:

### Layer 1: Remove cache, use factory

Replace `@cache`-backed `cached_async_http_client` with a non-caching `create_async_http_client`. Each Provider instance gets its own `httpx.AsyncClient`. Keep old name as deprecated alias.

Tradeoff: slightly more TCP connections. DouweM: 'the amount of bugs this caching has caused is worth a small hit.' Power users can still pass their own `http_client=`.

### Layer 2: Provider lifecycle (`__aenter__`/`__aexit__`)

Follow the established `entered_count` pattern (already used in Agent, MCPServer, CombinedToolset):

```
Provider.__aenter__ → enter SDK client (AsyncOpenAI, genai.Client, etc.)
Provider.__aexit__  → exit SDK client (closes connections cleanly)
```

- `_entered_count` + `_enter_lock` + `_exit_stack` on Provider base class
- `_own_client: bool` to track whether we created the client (don't manage user-provided clients)
- Each provider subclass enters its specific SDK client via the exit_stack

### Layer 3: Thread through Agent → Model → Provider

- **Model base class**: add `__aenter__`/`__aexit__` that delegates to `self._provider` (no-op default for test models that lack `_provider`)
- **Agent's `__aenter__`**: extend to also enter the model alongside toolsets. Infer the model if still a string (line 1480 pattern: `self.model = models.infer_model(self.model)`)

Result: `async with agent:` enters toolsets AND the model/provider chain.

When Agent is NOT used as a context manager (the common case today), everything still works — httpx.AsyncClient doesn't require entering. The context manager just enables clean connection teardown.

### Test suite optimization

The current conftest fixture (`cleanup_cached_async_http_clients`, line 343) patches `_cached_async_http_client` to track and close clients per test, then clears the cache.

Replace with a fixture that provides **per-provider caching within a test's scope** via monkeypatching `create_async_http_client`:
- One httpx client per provider type per test (same perf as current cache)
- Clean closure after each test (same resource safety as current fixture)
- No process-global state leaking between tests or event loops

### Benchmark anchor

Before starting implementation:
1. Run `time make test` (or `pytest --durations=0`) on `main` and record the wall clock time
2. After implementation, compare. Store the before/after in `local-notes/` as a reference.

## Files to modify

### Core changes
- `pydantic_ai_slim/pydantic_ai/models/__init__.py` — replace cache with factory, deprecate old name
- `pydantic_ai_slim/pydantic_ai/providers/__init__.py` — add `__aenter__`/`__aexit__`/entered_count to Provider base

### Provider import updates (mechanical rename, all 26)
- `providers/openai.py`, `anthropic.py`, `google.py`, `groq.py`, `mistral.py`, `cohere.py`, `azure.py`, `openrouter.py`, `deepseek.py`, `cerebras.py`, `fireworks.py`, `together.py`, `heroku.py`, `moonshotai.py`, `grok.py`, `nebius.py`, `ollama.py`, `alibaba.py`, `sambanova.py`, `litellm.py`, `vercel.py`, `github.py`, `ovhcloud.py`, `gateway.py`, `google_gla.py`, `google_vertex.py`

### Model/Agent lifecycle
- `pydantic_ai_slim/pydantic_ai/models/__init__.py` — add `__aenter__`/`__aexit__` to Model base class
- `pydantic_ai_slim/pydantic_ai/agent/__init__.py` — extend `__aenter__` to also enter the model
- `pydantic_ai_slim/pydantic_ai/_ssrf.py` — update import

### Tests
- `tests/conftest.py` — rewrite `cleanup_cached_async_http_clients` fixture
- `tests/test_mcp.py`, `test_dbos.py`, `test_prefect.py`, `test_temporal.py`, `test_ssrf.py`, `providers/test_litellm.py` — update imports/patches

## Verification

1. `make format && make lint && make typecheck`
2. `make test` — compare wall clock with pre-change baseline
3. Manual multi-event-loop script:
   ```python
   import asyncio
   from pydantic_ai import Agent
   async def worker():
       agent = Agent('google-vertex:gemini-3-flash-preview')
       result = await agent.run('Hello')
       print(result.output)
   for _ in range(3):
       asyncio.run(worker())
   ```
4. Verify `async with agent:` lifecycle with a provider that logs enter/exit

## Not in scope
- Extracting entered_count pattern into shared `EnterOnce` mixin (3+ existing copies — good candidate for follow-up)
- `Provider.__del__` cleanup (let GC handle it for now; full cleanup requires `__del__` scheduling async close on event loop)
