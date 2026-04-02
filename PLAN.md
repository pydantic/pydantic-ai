# Mid-Stream Fallback for `FallbackModel`

## Problem Statement

`FallbackModel` supports fallback for streaming requests — but only when the exception occurs while **opening** the stream (the `model.request_stream(...)` call itself). Two scenarios are not handled:

1. **Mid-stream exceptions** — The provider connection drops, times out, or errors *during* chunk iteration. Today these propagate as unhandled exceptions to the caller.
2. **Response-based fallback** — A response handler (e.g. checking `finish_reason`) rejects the completed response. Today response handlers are silently ignored for streaming.

The non-streaming `request()` path handles both. This plan brings streaming to parity.

## Background: How Streaming Flows Through the System

Understanding three layers is essential. Here's the full call chain from user code to provider:

```
User code                Agent graph                    FallbackModel              Provider model
─────────                ───────────                    ─────────────              ──────────────
agent.run_stream("hi")
  │
  └─► ModelRequestNode.stream()
        │
        ├─ _streaming_handler(req_ctx):
        │    async with req_ctx.model.request_stream(...) as sr:    ◄── This is FallbackModel
        │      │                                                        │
        │      │                                                        ├─ Opens provider stream
        │      │                                                        └─ yields StreamedResponse
        │      │
        │      agent_stream = _build_agent_stream(sr)
        │      stream_ready.set()
        │      await stream_done.wait()          ◄── Pauses here while user iterates
        │    response = sr.get()                 ◄── Called AFTER user finishes iterating
        │    return response
        │
        └─► User iterates AgentStream
              │
              async for event in agent_stream:   ◄── Pulls from sr.__aiter__()
              output = await result.get_output()
```

### Layer 1: `StreamedResponse` (base class in `models/__init__.py`)

This is the abstract base class all provider-specific streams inherit from.

```python
class StreamedResponse(ABC):
    _parts_manager: ModelResponsePartsManager   # Accumulates parts from events
    _event_iterator: AsyncIterator | None       # Memoized — created once in __aiter__
    final_result_event: FinalResultEvent | None # Set during iteration

    def __aiter__(self):
        if self._event_iterator is None:
            # Wraps _get_event_iterator with two decorators:
            # 1. iterator_with_final_event — detects when output matches result schema
            # 2. iterator_with_part_end — emits PartEndEvent between parts
            self._event_iterator = iterator_with_part_end(
                iterator_with_final_event(self._get_event_iterator())
            )
        return self._event_iterator

    @abstractmethod
    async def _get_event_iterator(self):
        """Provider implements this — yields raw PartStartEvent/PartDeltaEvent."""

    def get(self) -> ModelResponse:
        """Builds ModelResponse from accumulated _parts_manager state."""
```

**Key constraint**: `__aiter__` is **memoized** — the iterator chain is created once and reused. You can't "restart" it.

### Layer 2: `AgentStream` (in `result.py`)

The user-facing wrapper. It holds a reference to the `StreamedResponse` and accesses these attributes directly:

| Attribute | How it's accessed |
|-----------|-------------------|
| `__aiter__()` | `_get_usage_checking_stream_response(self._raw_stream_response, ...)` |
| `get()` | `self._raw_stream_response.get()` |
| `final_result_event` | Checked in `stream_output()` and `validate_response_output()` |
| `usage()` | `self._raw_stream_response.usage()` |
| `timestamp` | `self._raw_stream_response.timestamp` |

Any wrapper we create must correctly proxy **all five** of these.

### Layer 3: `FallbackModel.request_stream` (current implementation)

```python
@asynccontextmanager
async def request_stream(self, messages, model_settings, model_request_parameters, run_context):
    exceptions = []
    for model in self.models:
        async with AsyncExitStack() as stack:
            try:
                response = await stack.enter_async_context(
                    model.request_stream(messages, model_settings, ...)
                )
            except Exception as exc:
                # ✅ Handles exceptions during stream OPENING
                if await self._should_fallback(exc):
                    exceptions.append(exc)
                    continue
                raise

            yield response  # ← Raw StreamedResponse, no wrapping
            return           # ← No post-stream response check

    _raise_fallback_exception_group(exceptions, [])
```

**What's missing**: No wrapping means no hook for mid-stream exceptions or post-stream response evaluation.

## Recommended Solution: `FallbackStreamedResponse` Wrapper

### Design overview

```
FallbackModel.request_stream
  │
  ├─ Try opening first model's stream (existing logic)
  ├─ Wrap it: yield FallbackStreamedResponse(inner=stream, models_remaining=[...])
  └─ return
          │
          └─► Caller iterates FallbackStreamedResponse.__aiter__()
                │
                ├─ Delegates to inner.__aiter__() (provider stream)
                │    │
                │    ├─ On mid-stream exception:
                │    │    if _should_fallback(exc) → open next model, swap inner, continue
                │    │    else → raise
                │    │
                │    └─ On stream completion:
                │         if response handlers reject inner.get():
                │           → open next model, swap inner, continue
                │         else → done
                │
                └─ get() → delegates to self._inner.get()
```

### Why override `__aiter__` (not `_get_event_iterator`)

There are three places we could intercept the stream. Only one works cleanly:

| Approach | Problem |
|----------|---------|
| Override `_get_event_iterator` | Base class's `__aiter__` wraps it with `PartEndEvent`/`FinalResultEvent` logic that uses `self._parts_manager`. On fallback, the parts manager state goes stale and those decorators break. |
| Keep logic in `request_stream` | It's an `@asynccontextmanager` that yields once. Can't yield multiple `StreamedResponse`s. You'd still need a wrapper. |
| **Override `__aiter__`** | Iterate the **inner** stream's `__aiter__` (which handles its own `PartEnd`/`FinalResult` logic). On fallback, swap to a new inner stream whose `__aiter__` handles the new events correctly. No base class state to manage. |

### The wrapper class

```python
@dataclass
class FallbackStreamedResponse(StreamedResponse):
    """StreamedResponse wrapper that supports mid-stream exception and response-based fallback."""

    _inner: StreamedResponse
    _fallback_model: FallbackModel
    _models_remaining: list[Model]

    # Needed to open new streams on fallback
    _messages: list[ModelMessage]
    _model_settings: ModelSettings | None
    _model_request_parameters: ModelRequestParameters
    _run_context: RunContext[Any] | None
    _exit_stack: AsyncExitStack

    # Accumulated across all attempts
    _exceptions: list[Exception]
    _rejected_responses: list[ModelResponse]
```

#### `__aiter__` — the fallback loop

```python
def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
    if self._event_iterator is None:
        self._event_iterator = self._fallback_event_iterator()
    return self._event_iterator

async def _fallback_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    while True:
        # Phase 1: Iterate the current inner stream
        try:
            async for event in self._inner:  # Uses inner's __aiter__ (with PartEnd/FinalResult)
                yield event
        except Exception as exc:
            # Mid-stream exception — check if we should fallback
            if not self._models_remaining or not await self._fallback_model._should_fallback(exc):
                raise
            self._exceptions.append(exc)
            self._inner = await self._open_next_stream()
            continue  # Retry with next model's stream

        # Phase 2: Stream completed — check response handlers
        if self._fallback_model._response_handlers:
            response = self._inner.get()
            if await self._fallback_model._should_fallback(response):
                if not self._models_remaining:
                    self._rejected_responses.append(response)
                    _raise_fallback_exception_group(self._exceptions, self._rejected_responses)
                self._rejected_responses.append(response)
                self._inner = await self._open_next_stream()
                continue  # Retry with next model's stream

        return  # Success — stream completed and response accepted
```

#### Property/method delegation

```python
def get(self) -> ModelResponse:
    return self._inner.get()

def usage(self) -> RequestUsage:
    return self._inner.usage()

# These three are abstract properties that must be implemented:
@property
def model_name(self) -> str:
    return self._inner.model_name

@property
def timestamp(self) -> datetime:
    return self._inner.timestamp

@property
def provider_name(self) -> str | None:
    return self._inner.provider_name

@property
def provider_url(self) -> str | None:
    return self._inner.provider_url

# AgentStream reads this directly — must proxy it
@property
def final_result_event(self) -> FinalResultEvent | None:
    return self._inner.final_result_event
```

**Note on `final_result_event`**: The base class sets this during `__aiter__` via `iterator_with_final_event`. Since we delegate to the inner stream's `__aiter__`, it gets set on `self._inner`, not on `self`. We need a property that proxies it. If fallback switches to a new inner stream, the property automatically points to the new stream's (initially `None`) `final_result_event`, which is correct — the new stream hasn't produced a final result yet.

#### Opening the next model's stream

```python
async def _open_next_stream(self) -> StreamedResponse:
    """Pop the next model and open its stream via the shared exit stack."""
    while self._models_remaining:
        model = self._models_remaining.pop(0)
        try:
            return await self._exit_stack.enter_async_context(
                model.request_stream(
                    self._messages, self._model_settings,
                    self._model_request_parameters, self._run_context
                )
            )
        except Exception as exc:
            # The next model might also fail to open — same logic as init
            if await self._fallback_model._should_fallback(exc):
                self._exceptions.append(exc)
                continue
            raise

    _raise_fallback_exception_group(self._exceptions, self._rejected_responses)
```

**Why `AsyncExitStack`**: Each model's `request_stream` is an async context manager that owns the HTTP connection. We use a shared `AsyncExitStack` so that:
- New model streams can be opened mid-iteration (via `stack.enter_async_context`).
- All opened streams are cleaned up when `FallbackModel.request_stream`'s `async with` block exits.
- We don't need to manually track and close old streams.

### Updated `FallbackModel.request_stream`

```python
@asynccontextmanager
async def request_stream(self, messages, model_settings, model_request_parameters, run_context):
    exceptions = []
    rejected_responses = []

    async with AsyncExitStack() as stack:
        for i, model in enumerate(self.models):
            try:
                _, prepared_parameters = model.prepare_request(model_settings, model_request_parameters)
                response = await stack.enter_async_context(
                    model.request_stream(messages, model_settings, model_request_parameters, run_context)
                )
            except Exception as exc:
                if await self._should_fallback(exc):
                    exceptions.append(exc)
                    continue
                raise

            # Wrap the stream to enable mid-stream fallback
            wrapper = FallbackStreamedResponse(
                model_request_parameters=model_request_parameters,
                _inner=response,
                _fallback_model=self,
                _models_remaining=self.models[i + 1:],
                _messages=messages,
                _model_settings=model_settings,
                _model_request_parameters=model_request_parameters,
                _run_context=run_context,
                _exit_stack=stack,
                _exceptions=exceptions,
                _rejected_responses=rejected_responses,
            )
            self._set_span_attributes(model, prepared_parameters)
            yield wrapper
            return

    _raise_fallback_exception_group(exceptions, rejected_responses)
```

## The "Mixed Events" Problem

When fallback happens mid-stream, the caller has already received events from the failed model:

```
Timeline:
  PartStartEvent(TextPart("The capital of Fra"))  ◄── From model A
  PartDeltaEvent(...)                              ◄── From model A
  💥 ConnectionError                               ◄── Model A dies
  ── fallback ──
  PartStartEvent(TextPart("The capital of"))       ◄── From model B (starts over)
  PartDeltaEvent(...)                              ◄── From model B
  ...complete response...
```

The caller sees text fragments from two different models. For consumers that:
- **Only use `get()` at the end** (most agent workflows): No impact. `get()` returns the final model's complete response.
- **Stream to a frontend** (chat UIs): Would see partial text from model A, then full text from model B — visually broken.

This is inherent to the "transparent retry" approach. Douwe flagged this as needing a flag. Three options for the maintainers:

| Option | Behavior | Tradeoff |
|--------|----------|----------|
| Always-on (recommended) | `fallback_on` works identically for streaming and non-streaming | Frontend consumers see mixed events |
| Opt-in flag (`stream_fallback=True`) | Off by default, explicit opt-in | Another parameter, streaming silently ignores `fallback_on` by default |
| Opt-out flag (`stream_fallback=False` to disable) | On by default, can be disabled | Same as always-on but with escape hatch |

**Recommendation**: Always-on, since the current behavior (silently ignoring `fallback_on`) is itself surprising. The implementation should be structured so adding a flag later is a one-line `if` check.

## Alternatives Considered

### Buffering the full stream before yielding

Buffer all events from the provider, evaluate handlers, then yield events to the caller only if accepted.

**Rejected**: Defeats the purpose of streaming. Adds latency equal to the full response generation time. Would require a fundamentally different `StreamedResponse` that replays buffered events.

### Post-stream evaluation only (no mid-stream exception handling)

Only handle response-based fallback (evaluate after stream completes). Let mid-stream exceptions propagate as today.

**Rejected**: Incomplete — mid-stream exceptions are the more common real-world failure mode (connection drops, timeouts). Response-based fallback alone is less useful.

### Sentinel event to signal "restart"

Emit a new `FallbackRestartEvent` when switching models, so frontend consumers can clear their buffer.

**Interesting but premature**: Requires a new event type (version policy consideration), and no existing consumers know how to handle it. Could be added later if the mixed events problem proves painful.

## Implementation Steps

### Files to modify

| File | What changes |
|------|-------------|
| `pydantic_ai_slim/pydantic_ai/models/fallback.py` | Add `FallbackStreamedResponse` class; update `request_stream` to wrap the response |
| `tests/models/test_fallback.py` | Add streaming fallback tests (see below) |
| `docs/models/overview.md` | Remove "streaming only supports exception-based fallback" caveat; document new behavior |

### Test cases

All tests use `FunctionModel` with `stream_function` to control behavior:

| # | Scenario | Assertion |
|---|----------|-----------|
| 1 | First model raises mid-stream, second succeeds | Output from model B; `all_messages()` shows model B's response |
| 2 | Response handler rejects first model, accepts second | Output from model B; both models fully streamed |
| 3 | All models raise mid-stream | `FallbackExceptionGroup` with all exceptions |
| 4 | All models' responses rejected | `FallbackExceptionGroup` with `ResponseRejected` |
| 5 | Mixed: model A exception, model B rejected, model C succeeds | Output from model C |
| 6 | Custom exception handler callable for mid-stream errors | Handler returning `False` → exception propagates |
| 7 | `get()` returns correct model's response | `model_name`, `usage`, `timestamp` from successful model |
| 8 | `final_result_event` comes from successful model | Validates structured output works across fallback |
| 9 | Stream-open exception (existing) still works | Regression test for current behavior |
| 10 | Async exception/response handlers work mid-stream | `async def handler(exc) -> bool` |

### Documentation updates

1. Remove the note at line 321-322 of `docs/models/overview.md`:
   > "Response-based fallback currently only works with non-streaming requests..."
2. Add a brief note that mid-stream fallback may emit events from multiple models when streaming to frontends.
3. Add a streaming example showing response-based fallback with `run_stream()`.

## Open Questions for Maintainers

1. **Flag or no flag?** Should mid-stream fallback be always-on, opt-in, or opt-out? (See "Mixed Events" section above.)

2. **Usage accounting**: Should `usage()` report only the successful model's usage, or accumulate across all attempts? Non-streaming `request()` returns only the successful model's usage, suggesting consistency. But accumulated usage is arguably more accurate for billing.

3. **Span attributes**: `_set_span_attributes` is called when the initial stream opens. If fallback switches models, the span still reflects the first model. Should the wrapper update span attributes? (It doesn't have span access today — would need a callback or access to the span.)

4. **`_get_event_iterator` stub**: Since we override `__aiter__` and never use the base class's implementation, we still need to implement the abstract `_get_event_iterator`. It can be a no-op stub (`return; yield`), but this is slightly awkward. Acceptable?
