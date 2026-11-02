# Mid-Stream Fallback for `FallbackModel`

> Tracking issue: TBD (to be filed before implementation begins)

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
    provider_response_id: str | None            # Set by providers (e.g. OpenAI)
    provider_details: dict[str, Any] | None     # Set by providers
    finish_reason: FinishReason | None          # Set by providers

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

Any wrapper we create must correctly proxy **all of these**, plus the provider-set fields (`provider_response_id`, `provider_details`, `finish_reason`) which are set during iteration by provider-specific `_get_event_iterator` implementations.

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
                │         if _should_fallback(response) → open next model, swap inner, continue
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

**Consequence of `__aiter__` override**: The wrapper bypasses the base class's `iterator_with_final_event` and `iterator_with_part_end` decorators entirely — those run inside each inner stream's own `__aiter__`. This means:

- **`FinalResultEvent` on fallback**: If model A emits a `FinalResultEvent` before failing, the property proxy (`self._inner.final_result_event`) pointed to model A's event. After swapping `_inner` to model B, it now points to model B's `final_result_event`, which is `None` until model B's stream produces one. `AgentStream.stream_output()` checks `final_result_event is None` on each loop iteration (result.py:73) — after the swap, it correctly skips partial output validation until model B's `FinalResultEvent` arrives. This is correct behavior.

- **Orphaned `PartEndEvent`**: When model A fails mid-stream, its `iterator_with_part_end` never completes, so the last part from model A won't get a `PartEndEvent`. Consumers tracking part boundaries will see an orphaned `PartStartEvent` without a matching end. This is benign for all current consumers (`AgentStream` doesn't track part boundary state), but worth noting in documentation.

### The wrapper class

`FallbackModel` uses `@dataclass(init=False)` with a custom `__init__`. `StreamedResponse` uses `field(init=False)` for internal state. The wrapper should follow the same `@dataclass(init=False)` pattern as `FallbackModel` to avoid exposing `_`-prefixed fields as dataclass constructor parameters:

```python
@dataclass(init=False)
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

    def __init__(
        self,
        *,
        model_request_parameters: ModelRequestParameters,
        inner: StreamedResponse,
        fallback_model: FallbackModel,
        models_remaining: list[Model],
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        run_context: RunContext[Any] | None,
        exit_stack: AsyncExitStack,
        exceptions: list[Exception],
        rejected_responses: list[ModelResponse],
    ) -> None:
        super().__init__(model_request_parameters=model_request_parameters)
        self._inner = inner
        self._fallback_model = fallback_model
        self._models_remaining = models_remaining
        self._messages = messages
        self._model_settings = model_settings
        self._model_request_parameters = model_request_parameters
        self._run_context = run_context
        self._exit_stack = exit_stack
        self._exceptions = exceptions
        self._rejected_responses = rejected_responses
```

**Orphaned base class state**: The wrapper inherits `_parts_manager`, `_usage`, `final_result_event` etc. from `StreamedResponse` as dataclass fields. These are never populated because all delegation goes through `self._inner`. This is an unavoidable cost of subclassing — `AgentStream` type-checks for `StreamedResponse`, so composition without inheritance isn't an option. The orphaned fields are inert and harmless: `_parts_manager` is only accessed inside provider `_get_event_iterator` implementations, never from `result.py` or `_agent_graph.py`. The `get()` override ensures the wrapper never uses its own `_parts_manager`.

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
        # Note: _should_fallback(response) returns False when there are no _response_handlers,
        # so the explicit guard here is a performance optimization to avoid building the
        # ModelResponse via get() when there are no handlers to check it against.
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

#### `_get_event_iterator` — abstract method stub

Since we override `__aiter__` and never use the base class's iteration chain, we still must implement the abstract method. This follows the same pattern as the base class itself (models/__init__.py:1096-1098):

```python
async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
    # Not used — iteration is handled by _fallback_event_iterator via __aiter__ override.
    # This stub satisfies the abstract method contract.
    raise NotImplementedError('FallbackStreamedResponse delegates iteration to inner stream')
    yield  # pragma: no cover  # Make this a generator
```

#### Property/method delegation

All attributes that `AgentStream` or external consumers access must proxy to `self._inner`:

```python
def get(self) -> ModelResponse:
    return self._inner.get()

def usage(self) -> RequestUsage:
    return self._inner.usage()

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

# Fields set during iteration by provider-specific _get_event_iterator:
@property
def final_result_event(self) -> FinalResultEvent | None:
    return self._inner.final_result_event

@property
def provider_response_id(self) -> str | None:
    return self._inner.provider_response_id

@property
def provider_details(self) -> dict[str, Any] | None:
    return self._inner.provider_details

@property
def finish_reason(self) -> FinishReason | None:
    return self._inner.finish_reason
```

**Note on `final_result_event`**: The base class sets this during `__aiter__` via `iterator_with_final_event`. Since we delegate to the inner stream's `__aiter__`, it gets set on `self._inner`, not on `self`. The property proxy means: after a fallback swap, the property automatically points to the new inner stream's (initially `None`) `final_result_event`, which is correct — `AgentStream.stream_output()` will correctly pause partial output validation until the new model's `FinalResultEvent` arrives.

#### Opening the next model's stream

```python
async def _open_next_stream(self) -> StreamedResponse:
    """Open the next model's stream via the shared exit stack.

    Iterates through remaining models, handling stream-open exceptions
    with the same fallback logic as the initial model loop.
    """
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

**Note on `list.pop(0)`**: This is O(n) but the list is tiny (typically 2-3 models). The list is a copy (sliced from `self.models[i + 1:]` in the constructor) so popping doesn't mutate `FallbackModel.models`.

**Note on `prepare_request`/`_set_span_attributes`**: The current code calls `model.prepare_request(...)` and `self._set_span_attributes(model, prepared_parameters)` for the initial model. `_open_next_stream` should also call `prepare_request` on each fallback model to ensure `ModelRequestParameters` are customized correctly. Span attributes are harder — the wrapper doesn't have span access. See open questions.

**Why `AsyncExitStack`**: Each model's `request_stream` is an async context manager that owns the HTTP connection. We use a shared `AsyncExitStack` so that:
- New model streams can be opened mid-iteration (via `stack.enter_async_context`).
- All opened streams are cleaned up when `FallbackModel.request_stream`'s `async with` block exits.
- We don't need to manually track and close old streams.

**Resource lifecycle change**: The current implementation creates a new `AsyncExitStack` *per model attempt* inside the for-loop. The updated design moves it outside the loop so the wrapper can open new streams on the same stack. This means previously-failed streams' resources aren't cleaned up until the outermost `async with` exits, rather than immediately after each failed attempt. In practice this is benign — HTTP connections are lightweight and the stack cleans them all up promptly when `request_stream` exits — but it is a behavioral change worth noting.

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

            wrapper = FallbackStreamedResponse(
                model_request_parameters=model_request_parameters,
                inner=response,
                fallback_model=self,
                models_remaining=self.models[i + 1:],
                messages=messages,
                model_settings=model_settings,
                run_context=run_context,
                exit_stack=stack,
                exceptions=exceptions,
                rejected_responses=rejected_responses,
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
  ── fallback ──                                   ◄── No PartEndEvent for model A's last part
  PartStartEvent(TextPart("The capital of"))       ◄── From model B (starts over)
  PartDeltaEvent(...)                              ◄── From model B
  ...complete response...
```

The caller sees text fragments from two different models. The impact depends on the consumer:

| Consumer | Impact |
|----------|--------|
| `get()` at end (most agent workflows) | **None** — returns final model's complete response |
| `stream_output()` (structured output) | **Recoverable** — `final_result_event` proxy resets to `None` on swap; partial validation pauses until new model's `FinalResultEvent` arrives. But partial outputs from model A were already yielded. |
| `stream_text()` (frontend streaming) | **Broken** — text from model A followed by complete text from model B |
| Part boundary tracking | **Minor** — orphaned `PartStartEvent` without `PartEndEvent` from failed model |
| Response-based fallback (stream completes, then rejected) | **Worst case** — caller receives the *entire* response from model A, then the *entire* response from model B back-to-back |

Douwe explicitly said this "would need to have a flag on `FallbackModel`." Given that guidance:

**Recommendation**: Opt-in via a flag (e.g. `stream_fallback: bool = False`). This respects the maintainer's stated preference, avoids surprising frontend consumers by default, and still allows users who want the behavior to enable it. The implementation should be structured so that the flag is a single `if` check — if `False`, the wrapper is not created and current behavior is preserved.

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
| `pydantic_ai_slim/pydantic_ai/models/fallback.py` | Add `FallbackStreamedResponse` class; update `request_stream` to wrap the response; add `stream_fallback` parameter |
| `tests/models/test_fallback.py` | Add streaming fallback tests (see below) |
| `docs/models/overview.md` | Update fallback docs to describe new streaming behavior |

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
| 11 | Fallback during first chunk (before any events yielded) | Cleaner case — no mixed events, output entirely from model B |
| 12 | Caller cancels early (`break` from `async for` / `aclose()`) | `AsyncExitStack` cleans up; no resource leaks |

### Documentation updates

1. Update `docs/models/overview.md` to describe the `stream_fallback` parameter.
2. Add a note that mid-stream fallback may emit events from multiple models (orphaned `PartStartEvent` without `PartEndEvent` from failed model).
3. Add a streaming example showing `stream_fallback=True` with `run_stream()`.

## Open Questions for Maintainers

1. **Flag name and default**: We propose `stream_fallback: bool = False` on `FallbackModel.__init__`. Is this the right name? Douwe suggested a flag is needed — confirm the default should be opt-in (`False`).

2. **Usage accounting**: Should `usage()` report only the successful model's usage, or accumulate across all attempts? Non-streaming `request()` returns only the successful model's usage, suggesting consistency. But accumulated usage is arguably more accurate for billing. **Current plan**: delegate to `self._inner.usage()` (successful model only) for consistency.

3. **Span attributes on fallback**: `_set_span_attributes` is called when the initial stream opens. If fallback switches models, the span still reflects the first model. The wrapper could accept a callback to update span attributes, or we could pass the span reference. What's the preferred approach?

4. **`InstrumentedModel` interaction**: When models in the fallback list are wrapped in `InstrumentedModel`, each model's `request_stream` has a `finally` block that calls `response_stream.get()` to finish the trace span. If we stop iterating a stream mid-way (on fallback), the `InstrumentedModel` finally-block will call `get()` on a partially-consumed stream, producing a partial/incomplete `ModelResponse` in the trace. Is this acceptable, or should we investigate suppressing the trace for failed streams?

5. **`prepare_request` in `_open_next_stream`**: Should `_open_next_stream` call `model.prepare_request(...)` for each fallback model? The initial loop does this. Currently omitted from the wrapper for simplicity, but could lead to incorrect `ModelRequestParameters` for fallback models that customize them.
