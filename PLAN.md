# Issue #4414: OpenAI WebSocket Mode for Responses API

## Context

OpenAI added WebSocket support for their Responses API — persistent connections with lower latency vs HTTP. The OpenAI Python SDK (v2.22.0+, we have v2.26.0, require >=2.25.0) added native support via `client.responses.connect()`. DouweM (maintainer) provided detailed design guidance in issue comments. We're taking over from madanlalit (assigned but no plan/PR posted).

**Approach**: TDD-first — write cassette infra + test stubs that call not-yet-implemented methods, open PR with plan + tests, then implement.

**References**:
- Issue: https://github.com/pydantic/pydantic-ai/issues/4414
- DouweM design guidance (approach + lifecycle): https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-3987520535
- DouweM on explicit method (not `__aenter__`): https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-4078003956
- DouweM on ModelSetting + ContextVar approach: https://github.com/pydantic/pydantic-ai/pull/4843#discussion_r2991767607
- OpenAI SDK WS support (v2.22.0): https://github.com/openai/openai-python/releases/tag/v2.22.0
- OpenAI WS mode guide: https://developers.openai.com/api/docs/guides/websocket-mode
- PR #4421 (Model lifecycle / `__aenter__`): https://github.com/pydantic/pydantic-ai/pull/4421

---

## Requirements (from DouweM)

> "Conceptually, we'd have a model class [...] whose request and request_stream methods send a response.create message on the websocket, and then passes received messages through the existing OpenAIResponsesStreamedResponse until it hits a response.completed message."
> — [DouweM, Mar 2](https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-3987520535)

> "We could also go a slightly different route and add a `openai_use_websocket` ModelSetting, and use it to determine in `OpenAIResponsesModel.__aenter__` if a new connection should be opened, and then do an `async with model:` from inside `agent.iter` [...] That, combined with the ContextVar approach should ensure one connection per agent run. That's probably the cleanest solution!"
> — [DouweM, Mar 25](https://github.com/pydantic/pydantic-ai/pull/4843#discussion_r2991767607)

---

## Design Decision: Two approaches

### Approach A: Explicit `model.connect()` + ContextVar

```python
model = OpenAIResponsesModel('gpt-4o')
agent = Agent(model)

# Single WS session
async with model.connect():
    result = await agent.run('Hello')  # routes through WS

result2 = await agent.run('Hello')  # back to HTTP

# Parallel — each connect() opens a separate WS connection
async with model.connect() as m1, model.connect() as m2:
    r1, r2 = await asyncio.gather(
        agent.run('Q1', model=m1),
        agent.run('Q2', model=m2),
    )
```

**How it works:**
- `model.connect()` opens a WS connection, stores it in a `ContextVar`, yields `self`
- `request()`/`request_stream()` check the ContextVar: if connection found → WS; else → HTTP
- Each `asyncio.gather` coroutine gets its own ContextVar copy → safe parallelism
- No dependency on PR #4421

**Verified:**
- `AsyncResponsesConnectionManager` is async (`__aenter__`/`__aexit__`), not sync ✅
- Multiple `client.responses.connect()` calls create independent managers ✅
- `ContextVar` properly isolates across `asyncio.gather` (both coroutines and tasks) ✅

### Approach B: `openai_use_websocket` ModelSetting + `__aenter__` + ContextVar

```python
model = OpenAIResponsesModel('gpt-4o', settings={'openai_use_websocket': True})
agent = Agent(model)

# Automatic — agent.run calls `async with model:` internally
result = await agent.run('Hello')  # WS, handled automatically

# Parallel — each run gets its own connection via ContextVar
r1, r2 = await asyncio.gather(
    agent.run('Q1'),
    agent.run('Q2'),
)
```

**How it works:**
- `openai_use_websocket: bool` in `OpenAIResponsesModelSettings` (model-level, not per-request — `__aenter__` runs before any request, so per-request settings aren't available yet)
- `OpenAIResponsesModel.__aenter__` calls `super().__aenter__()` (HTTP lifecycle from #4421) + opens WS if setting is true → stores in ContextVar
- `agent.run`/`agent.iter` must call `async with model:` internally (not yet in #4421)
- Each run is a separate async context → separate ContextVar → separate WS connection

**Blockers:**
1. PR #4421 must merge (adds `Model.__aenter__`/`__aexit__`)
2. `agent.run`/`agent.iter` must call `async with model:` internally — NOT in #4421 yet (currently only `Agent.__aenter__` does this, which requires user to do `async with agent:`)
3. `openai_use_websocket` must be a model-level setting (not per-request) since `__aenter__` runs before requests

### Recommended: Approach A now, evolve to B later

**Phase 1 (this PR):** Implement `model.connect()` + ContextVar. Works immediately, no dependencies.

**Phase 2 (after #4421):** Add `openai_use_websocket` ModelSetting. Override `__aenter__` to open WS when setting is true. When `agent.run`/`agent.iter` calls `async with model:` automatically, the user gets the cleanest API.

Both approaches coexist — `connect()` for explicit control, ModelSetting for "just works."

---

## SDK API Summary

```python
async with client.responses.connect() as connection:
    await connection.response.create(model='gpt-4o', input=..., stream=True, ...)
    async for event in connection:  # ResponsesServerEvent — same 44 classes as HTTP ResponseStreamEvent
        ...
```

Key facts:
- `connection.response.create()` returns `None` (fire-and-forget)
- No multiplexing: one in-flight response at a time; max 60min connection
- **WS events use identical classes** to HTTP streaming → `_get_event_iterator()` reusable as-is
- HTTP-only params not available over WS: `extra_headers`, `extra_body`, `timeout`, `extra_query`
- All `OpenAIResponsesModelSettings` fields (reasoning, truncation, tools, etc.) work fine over WS

---

## Implementation Plan (Phase 1)

### Files to modify

- `pydantic_ai_slim/pydantic_ai/models/openai.py` — `OpenAIResponsesModel` (add `connect()`, modify `_responses_create()`)

### New files

- `tests/websockets/__init__.py`
- `tests/websockets/cassettes.py` — adapted from PR #4375's `tests/realtime/cassettes.py`
- `tests/websockets/conftest.py` — adapted from PR #4375's `tests/realtime/conftest.py`
- `tests/websockets/test_openai_responses.py` — TDD test stubs
- `tests/websockets/cassettes/test_openai_responses/{test_id}.yaml` — cassette files (recorded after implementation)

### 1. `connect()` on `OpenAIResponsesModel` + ContextVar

Module-level ContextVar:
```python
_ws_connection_var: ContextVar[AsyncResponsesConnection | None] = ContextVar(
    'pydantic_ai.openai_ws_connection', default=None
)
```

Method:
```python
@asynccontextmanager
async def connect(self, **kwargs) -> AsyncIterator[Self]:
    async with self.client.responses.connect(**kwargs) as connection:
        token = _ws_connection_var.set(connection)
        try:
            yield self
        finally:
            _ws_connection_var.reset(token)
```

No mutable instance state — ContextVar handles isolation.

### 2. Concurrent request guard

Instead of a lock (silent serialization), raise immediately on concurrent use:

```python
_ws_in_use_var: ContextVar[bool] = ContextVar('pydantic_ai.openai_ws_in_use', default=False)
```

In `_responses_create()`:
```python
conn = _ws_connection_var.get()
if conn is not None:
    if _ws_in_use_var.get():
        raise RuntimeError(
            'This WebSocket connection is already handling a request. '
            'For parallel requests, create separate OpenAIResponsesModel instances '
            'and call connect() on each.'
        )
    _ws_in_use_var.set(True)
    try:
        # ... send over WS
    finally:
        _ws_in_use_var.set(False)
```

### 3. Route `_responses_create()` through WS when connected

When `_ws_connection_var.get()` returns a connection:
- Validate no HTTP-only settings passed (warn/raise for `extra_headers`, `extra_body`, `timeout`)
- Call `await connection.response.create(...)` with shared params
- Return WS event stream adapter

When `_ws_connection_var.get()` is `None`: existing HTTP path unchanged.

### 4. WS event stream adapter

```python
async def _ws_event_stream(self, connection) -> AsyncIterator[responses.ResponseStreamEvent]:
    async for event in connection:
        yield event  # same types as HTTP
        if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)):
            break
```

### 5. Non-streamed `request()` over WS

WS is inherently streaming. For `request()`:
- Send `response.create(stream=True)` → collect events until `ResponseCompletedEvent`
- Extract `Response` from completed event → process with existing `_process_response()`

### 6. Streamed `request_stream()` over WS

- Send `response.create(stream=True)` → feed `_ws_event_stream()` through existing `_process_streamed_response()` / `OpenAIResponsesStreamedResponse`
- No changes to `_get_event_iterator()` (same event types)

---

## Test Infrastructure (from PR #4375)

### Files to adapt from `realtime-voice-session` branch

Source: `/home/claude-handler/pydantic-ai/realtime-voice-session/tests/realtime/`

| Source file | Target file | Changes needed |
|---|---|---|
| `cassettes.py` | `tests/websockets/cassettes.py` | Rename `RealtimeCassette` → `WebSocketCassette`; keep `ReplayWebSocket`, `RecordingWebSocket`, `realtime_cassette_plan` (rename to `ws_cassette_plan`). Core logic is fully generic — no provider-specific code. |
| `conftest.py` | `tests/websockets/conftest.py` | Keep OpenAI fixture pattern only (drop Gemini). Adapt to patch `client.responses.connect` instead of `websockets.connect`. Cassette dir: `cassettes/test_openai_responses/`. |
| `test_cassettes.py` | `tests/websockets/test_cassettes.py` | Keep as-is (tests the generic cassette utilities). |

**What to drop from realtime PR:**
- All Gemini-related code (`gemini_realtime_model` fixture, `test_gemini.py`, Gemini cassettes)
- `_session.py`, `_base.py` (realtime-specific abstractions)
- Realtime event mapping tests (different event model)

### Cassette path convention

Following codebase convention — folder named after test file stem, files named after test ID:
```
tests/websockets/cassettes/test_openai_responses/{test_id}.yaml
```
Same pattern as `tests/models/cassettes/test_openai_responses/*.yaml`.

### How cassettes work

```yaml
version: 1
interactions:
- direction: received
  data:
    type: response.created
    response: { id: resp_xxx, ... }
- direction: sent
  data:
    type: response.create
    model: gpt-4o
    input: [...]
- direction: received
  data:
    type: response.output_text.delta
    delta: "Hello"
- direction: received
  data:
    type: response.completed
    response: { ... }
```

- `ReplayWebSocket` serves pre-recorded `received` events, no-ops on `send()`
- `RecordingWebSocket` wraps real connection, records all frames for later dump
- Record mode controlled by `pytest --record-mode=once|rewrite`

---

## TDD Test Cases

File: `tests/websockets/test_openai_responses.py`

Tests call `model.connect()` and other not-yet-implemented methods. They fail until implementation lands. Cassette-based once recorded.

### Lifecycle
1. **`test_connect_lifecycle`** — enter `connect()` → ContextVar set; exit → ContextVar cleared; yields `Self`
2. **`test_connect_parallel_separate_connections`** — two `model.connect()` contexts in parallel via `asyncio.gather` → each gets isolated connection via ContextVar
3. **`test_no_connect_uses_http`** — without `connect()`, `agent.run()` uses HTTP (existing behavior)

### Requests (cassette-recorded)
4. **`test_ws_simple_text_request`** — `agent.run('Say hello')` over WS → text response
5. **`test_ws_streamed_text_request`** — `agent.run_stream('Say hello')` over WS → streamed text
6. **`test_ws_sequential_requests`** — two `agent.run()` in same WS context → both succeed
7. **`test_ws_request_with_tools`** — agent with tools → tool call + return processed correctly
8. **`test_ws_request_with_structured_output`** — agent with result type → structured output works

### Guards
9. **`test_ws_concurrent_requests_error`** — parallel `agent.run()` on same model in same context → raises `RuntimeError` with actionable message
10. **`test_ws_incompatible_settings_error`** — `extra_body`/`timeout` in WS mode → clear error

### Edge cases
11. **`test_ws_request_after_disconnect`** — `agent.run()` after exiting WS context → HTTP fallback
12. **`test_ws_connection_error_propagates`** — WS connection failure → exception propagates

---

## Verification

1. `uv run pytest tests/websockets/ -v` — tests define expected API (initially fail on `AttributeError`/`NotImplementedError`)
2. After implementation: `uv run pytest tests/websockets/ --record-mode=once` to record cassettes
3. `make format && make lint`
4. `make typecheck 2>&1 | tee /tmp/typecheck-output.txt`

## PR Strategy

1. First commit: plan + test infra + test stubs (TDD baseline)
2. Implementation commits: `connect()`, ContextVar routing, event adapter
3. Record cassettes, fix snapshot assertions
4. Docs (after review confirms logic is correct)
5. Phase 2 (separate PR, after #4421): `openai_use_websocket` ModelSetting + `__aenter__` integration
