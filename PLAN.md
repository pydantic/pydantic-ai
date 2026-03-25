# Issue #4414: OpenAI WebSocket Mode for Responses API

## Context

OpenAI added WebSocket support for their Responses API — persistent connections with lower latency vs HTTP. The OpenAI Python SDK (v2.22.0+, we have v2.26.0, require >=2.25.0) added native support via `client.responses.connect()`. DouweM (maintainer) provided detailed design guidance in issue comments. We're taking over from madanlalit (assigned but no plan/PR posted).

**Approach**: TDD-first — write cassette infra + test stubs that call not-yet-implemented methods, open PR with plan + tests, then implement.

**References**:
- Issue: https://github.com/pydantic/pydantic-ai/issues/4414
- DouweM design guidance (approach + lifecycle): https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-3987520535
- DouweM on explicit method (not `__aenter__`): https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-4078003956
- OpenAI SDK WS support (v2.22.0): https://github.com/openai/openai-python/releases/tag/v2.22.0
- OpenAI WS mode guide: https://developers.openai.com/api/docs/guides/websocket-mode

---

## Requirements (from DouweM)

> "Conceptually, we'd have a model class [...] whose request and request_stream methods send a response.create message on the websocket, and then passes received messages through the existing OpenAIResponsesStreamedResponse until it hits a response.completed message."
> — [DouweM, Mar 2](https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-3987520535)

> "I'd prefer an explicit (asynccontextmanager) method on OpenAIResponsesModel to enter websocket mode."
> — [DouweM, Mar 17](https://github.com/pydantic/pydantic-ai/issues/4414#issuecomment-4078003956)

Summary:
- Add to **existing** `OpenAIResponsesModel`, not a new subclass
- **Explicit `@asynccontextmanager` method** (not `__aenter__`/`__aexit__` — reserved for PR #4421)
- `request()` / `request_stream()` route through WS when connection active, HTTP otherwise
- Lifecycle and parallelism are the user's responsibility
- Fail fast on concurrent reuse; clear errors for WS-incompatible settings

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

## Implementation Plan

### Files to modify

- `pydantic_ai_slim/pydantic_ai/models/openai.py` — `OpenAIResponsesModel` (add `connect_ws()`, modify `_responses_create()`)

### New files

- `tests/websockets/__init__.py`
- `tests/websockets/cassettes.py` — adapted from PR #4375's `tests/realtime/cassettes.py`
- `tests/websockets/conftest.py` — adapted from PR #4375's `tests/realtime/conftest.py`
- `tests/websockets/test_openai_responses.py` — TDD test stubs
- `tests/websockets/cassettes/test_openai_responses/{test_id}.yaml` — cassette files (recorded after implementation)

### 1. `connect_ws()` on `OpenAIResponsesModel`

New fields:
- `_ws_connection: AsyncResponsesConnection | None` (default `None`)
- `_ws_lock: asyncio.Lock` (enforce sequential requests)

```python
@asynccontextmanager
async def connect_ws(self, **kwargs) -> AsyncIterator[Self]:
    if self._ws_connection is not None:
        raise RuntimeError('Already connected via WebSocket')
    async with self.client.responses.connect(**kwargs) as connection:
        self._ws_connection = connection
        try:
            yield self
        finally:
            self._ws_connection = None
```

### 2. Route `_responses_create()` through WS when connected

When `_ws_connection` is set:
- Acquire `_ws_lock`
- Validate no HTTP-only settings passed (warn/raise for `extra_headers`, `extra_body`, `timeout`)
- Call `await self._ws_connection.response.create(...)` with shared params
- Return WS event stream adapter

When `_ws_connection` is `None`: existing HTTP path unchanged.

### 3. WS event stream adapter

```python
async def _ws_event_stream(self) -> AsyncIterator[responses.ResponseStreamEvent]:
    assert self._ws_connection is not None
    async for event in self._ws_connection:
        yield event  # same types as HTTP
        if isinstance(event, (ResponseCompletedEvent, ResponseFailedEvent, ResponseIncompleteEvent)):
            break
```

### 4. Non-streamed `request()` over WS

WS is inherently streaming. For `request()`:
- Send `response.create(stream=True)` → collect events until `ResponseCompletedEvent`
- Extract `Response` from completed event → process with existing `_process_response()`

### 5. Streamed `request_stream()` over WS

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

Tests call `model.connect_ws()` and other not-yet-implemented methods. They fail until implementation lands. No mocks — cassette-based once recorded.

### Lifecycle
1. **`test_connect_ws_lifecycle`** — enter `connect_ws()` → connection active; exit → connection cleared; yields `Self`
2. **`test_connect_ws_double_connect_error`** — nested `connect_ws()` raises `RuntimeError`
3. **`test_no_ws_falls_back_to_http`** — without `connect_ws()`, `agent.run()` uses HTTP (existing behavior)

### Requests (cassette-recorded)
4. **`test_ws_simple_text_request`** — `agent.run('Say hello')` over WS → text response
5. **`test_ws_streamed_text_request`** — `agent.run_stream('Say hello')` over WS → streamed text
6. **`test_ws_sequential_requests`** — two `agent.run()` in same WS context → both succeed
7. **`test_ws_request_with_tools`** — agent with tools → tool call + return processed correctly
8. **`test_ws_request_with_structured_output`** — agent with result type → structured output works

### Guards
9. **`test_ws_concurrent_requests_serialized`** — parallel `agent.run()` on same model → serialized by lock
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
2. Implementation commits: `connect_ws()`, routing, event adapter
3. Record cassettes, fix snapshot assertions
4. Docs (after review confirms logic is correct)
