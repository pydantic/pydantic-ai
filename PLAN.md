# Plan: Custom Events from Tool Functions (#2382)

## Context

Users need to emit custom events (progress updates, intermediate results, state deltas) from tool functions **during execution**, not just after completion. Current workaround (`ToolReturn.metadata`) only emits events post-completion, defeating the purpose for long-running tools.

**User pain points** (from issue comments):
- LLM context pollution when returning events as tool results (ElvisTheKing)
- Boilerplate with anyio streams to manually bridge events (kadosh1000, jiachengzz)
- Can't show real-time progress during long operations (kadosh1000, hudyweas)
- Unnatural workarounds with state patches (voorhs)
- 'Monstrosity' code to stream from graphs called by tools (ggozad)

**Douwe's direction** (PR #3114): tools as `AsyncIterator` yielding `CustomEvent` and `Return[T]`. PR is stale but philosophy is sound.

## Proposed API

Two complementary approaches, implementable in phases:

### Approach A: `ctx.emit()` (simpler, recommended Phase 1)

```python
@agent.tool
async def long_task(ctx: RunContext[MyDeps], query: str) -> str:
    await ctx.emit(CustomEvent(name='progress', data={'step': 1, 'total': 3}))
    result = await fetch_data(query)
    await ctx.emit(CustomEvent(name='progress', data={'step': 2, 'total': 3}))
    processed = transform(result)
    await ctx.emit(CustomEvent(name='progress', data={'step': 3, 'total': 3}))
    return processed
```

- No return type changes, works in any async tool
- Simpler implementation (no generator detection, no `Return` sentinel)
- Aligns with kadosh1000's original request and Douwe's early comment about `ctx.deps.event_bus`

### Approach B: AsyncIterator yield (more Pythonic, Phase 2)

```python
@agent.tool_plain
async def long_task(query: str) -> AsyncIterator[CustomEvent | Return[str]]:
    yield CustomEvent(name='progress', data={'step': 1})
    result = await fetch_data(query)
    yield CustomEvent(name='result', data=result)
    yield Return(processed)
```

- Follows Douwe's PR #3114 philosophy
- Cleaner for tools that are 'event-first' (many events, simple return)
- Requires async generator detection + `Return` sentinel type
- Can layer on top of the same queue mechanism from Phase 1

Both produce identical `CustomToolEvent` in the `AgentStreamEvent` stream.

## New Types (`messages.py`)

### `CustomEvent` (~line 846, after `ToolReturn`)

```python
@dataclass(repr=False)
class CustomEvent:
    '''A custom event emitted by a tool during execution.'''

    name: str
    '''Event name for routing/filtering by consumers.'''

    data: Any = None
    '''Event payload. Should be JSON-serializable for UI adapter compatibility.'''

    __repr__ = _utils.dataclasses_no_defaults_repr
```

### `CustomToolEvent` (~line 2372, after `FunctionToolResultEvent`)

```python
@dataclass(repr=False)
class CustomToolEvent:
    '''Event emitted when a tool yields a custom event during execution.'''

    event: CustomEvent
    '''The custom event.'''

    tool_name: str
    '''Name of the tool that emitted this event.'''

    tool_call_id: str
    '''Tool call ID, matching the FunctionToolCallEvent that started execution.'''

    _: KW_ONLY
    event_kind: Literal['custom_tool_event'] = 'custom_tool_event'

    __repr__ = _utils.dataclasses_no_defaults_repr
```

### `Return[T]` (Phase 2 only, ~line 846)

```python
@dataclass(repr=False)
class Return(Generic[ReturnT]):
    '''Sentinel for the final return value of a streaming tool.'''

    value: ReturnT
    '''The tool return value.'''

    _: KW_ONLY
    content: str | Sequence[UserContent] | None = None
    '''Content sent to model as separate UserPromptPart (same semantics as ToolReturn.content).'''

    metadata: Any = None
    '''Additional data accessible by application, not sent to LLM (same as ToolReturn.metadata).'''

    __repr__ = _utils.dataclasses_no_defaults_repr
```

### Update `HandleResponseEvent` union

```python
HandleResponseEvent = Annotated[
    FunctionToolCallEvent
    | FunctionToolResultEvent
    | CustomToolEvent               # NEW
    | BuiltinToolCallEvent
    | BuiltinToolResultEvent,
    pydantic.Discriminator('event_kind'),
]
```

### Exports

Add `CustomEvent`, `CustomToolEvent`, `Return` to `pydantic_ai/__init__.py` and `messages.__all__`.

## Implementation

### Phase 1: `ctx.emit()` + event pipeline

#### 1a. `_run_context.py` — add emit method + queue field

```python
custom_event_queue: asyncio.Queue[Any] | None = field(default=None, init=False, repr=False)

async def emit(self, event: CustomEvent) -> None:
    '''Emit a custom event during tool execution.'''
    if self.custom_event_queue is not None:
        from .messages import CustomToolEvent
        await self.custom_event_queue.put(CustomToolEvent(
            event=event,
            tool_name=self.tool_name or '',
            tool_call_id=self.tool_call_id or '',
        ))
```

`custom_event_queue` is set by the framework in `_call_tools()`, not by users. `init=False` keeps it out of constructors.

#### 1b. `_agent_graph.py` — queue injection + event multiplexing in `_call_tools()`

**Key challenge**: `_call_tool()` is a coroutine used with both direct `await` (sequential) and `asyncio.create_task()` (parallel). Events must be surfaced *during* execution, not after.

**Solution**: shared `asyncio.Queue` + multiplexing in `_call_tools()`.

Add helpers:

```python
def _inject_event_queue(
    validated_calls: dict[str, ValidatedToolCall[DepsT]],
    queue: asyncio.Queue[_messages.CustomToolEvent],
) -> None:
    '''Set the custom event queue on all validated tool call contexts.'''
    for vc in validated_calls.values():
        vc.ctx.custom_event_queue = queue
```

Modify `_call_tools()` — for **sequential mode**:

```python
event_queue: asyncio.Queue[_messages.CustomToolEvent] = asyncio.Queue()
_inject_event_queue(validated_calls, event_queue)

for index, call in enumerate(tool_calls):
    tool_task = asyncio.create_task(
        _call_tool(tool_manager, validated_calls.get(call.tool_call_id, call), ...)
    )

    # Multiplex: drain queue while tool executes
    while not tool_task.done():
        queue_wait = asyncio.ensure_future(event_queue.get())
        done, _ = await asyncio.wait({tool_task, queue_wait}, return_when=asyncio.FIRST_COMPLETED)

        if queue_wait in done and not queue_wait.cancelled():
            yield queue_wait.result()
        elif not queue_wait.done():
            queue_wait.cancel()

    # Drain remaining events
    while not event_queue.empty():
        yield event_queue.get_nowait()

    # Handle tool result as before
    if event := await handle_call_or_result(tool_task, index):
        yield event
```

For **parallel mode** — same queue, multiplex across all tasks:

```python
event_queue: asyncio.Queue[_messages.CustomToolEvent] = asyncio.Queue()
_inject_event_queue(validated_calls, event_queue)

tasks = [asyncio.create_task(_call_tool(...), name=call.tool_name) for call in tool_calls]

if parallel_execution_mode == 'parallel_ordered_events':
    # Wait for all tasks, but drain events as they arrive
    pending = set(tasks)
    while pending:
        queue_wait = asyncio.ensure_future(event_queue.get())
        done, _ = await asyncio.wait(pending | {queue_wait}, return_when=asyncio.FIRST_COMPLETED)
        if queue_wait in done and not queue_wait.cancelled():
            yield queue_wait.result()
        elif not queue_wait.done():
            queue_wait.cancel()
        pending -= (done - {queue_wait})
    # Drain + yield results in order
    while not event_queue.empty():
        yield event_queue.get_nowait()
    for index, task in enumerate(tasks):
        if event := await handle_call_or_result(task, index):
            yield event

else:  # 'parallel' — yield results as they complete
    pending = set(tasks)
    while pending:
        queue_wait = asyncio.ensure_future(event_queue.get())
        done, _ = await asyncio.wait(pending | {queue_wait}, return_when=asyncio.FIRST_COMPLETED)
        if queue_wait in done and not queue_wait.cancelled():
            yield queue_wait.result()
        elif not queue_wait.done():
            queue_wait.cancel()
        for task in done - {queue_wait}:
            pending.discard(task)
            index = tasks.index(task)
            if event := await handle_call_or_result(task, index):
                yield event
    while not event_queue.empty():
        yield event_queue.get_nowait()
```

`_call_tool()` itself needs **no changes** — the queue lives on `RunContext`, and `emit()` puts events on it.

#### 1c. `ui/_event_stream.py` — base dispatch

Add to `handle_event()` dispatch:

```python
case _messages.CustomToolEvent():
    async for e in self.handle_custom_tool_event(event):
        yield e
```

Add base handler:

```python
async def handle_custom_tool_event(
    self, event: _messages.CustomToolEvent
) -> AsyncIterator[EventT]:
    return
    yield  # async generator stub
```

#### 1d. `ui/ag_ui/_event_stream.py` — AG-UI mapping

Override `handle_custom_tool_event`:

```python
async def handle_custom_tool_event(
    self, event: _messages.CustomToolEvent
) -> AsyncIterator[BaseEvent]:
    from ag_ui.core import CustomEvent as AGUICustomEvent
    yield AGUICustomEvent(
        name=event.event.name,
        value=event.event.data,
    )
```

Timestamp is auto-set by existing `handle_event()` override that sets timestamps on all events.

#### 1e. `ui/vercel_ai/_event_stream.py` — Vercel AI mapping

Override `handle_custom_tool_event`:

```python
async def handle_custom_tool_event(
    self, event: _messages.CustomToolEvent
) -> AsyncIterator[BaseChunk]:
    from .response_types import DataChunk
    yield DataChunk(
        type=f'data-{event.event.name}',
        data=event.event.data,
    )
```

### Phase 2: AsyncIterator yield (layered on Phase 1)

#### 2a. `_utils.py` — async generator detection

```python
def is_async_gen_callable(obj: Any) -> bool:
    '''Check if a callable is an async generator function.'''
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.isasyncgenfunction(obj) or (
        callable(obj) and inspect.isasyncgenfunction(obj.__call__)
    )
```

#### 2b. `_function_schema.py` — `is_async_gen` field + iteration

Add field `is_async_gen: bool` to `FunctionSchema`.

Modify `call()`:

```python
async def call(self, args_dict: dict[str, Any], ctx: RunContext[Any]) -> Any:
    args, kwargs = self._call_args(args_dict, ctx)
    if self.is_async_gen:
        return await self._call_async_gen(args, kwargs, ctx)
    elif self.is_async:
        ...  # existing path
    else:
        ...  # existing path

async def _call_async_gen(
    self, args: list[Any], kwargs: dict[str, Any], ctx: RunContext[Any]
) -> Any:
    from .messages import CustomEvent as CE, Return
    function = cast(Callable[..., AsyncIterator[Any]], self.function)
    final_value: Any = None
    async for item in function(*args, **kwargs):
        if isinstance(item, Return):
            final_value = item
            break
        elif isinstance(item, CE):
            await ctx.emit(item)  # Uses the same emit() from Phase 1
        else:
            raise UserError(
                f'Async generator tool yielded {type(item).__name__}. '
                f'Expected CustomEvent or Return.'
            )
    if isinstance(final_value, Return):
        # Convert to ToolReturn for downstream processing
        from .messages import ToolReturn
        return ToolReturn(
            return_value=final_value.value,
            content=final_value.content,
            metadata=final_value.metadata,
        )
    return final_value
```

#### 2c. `function_schema()` factory — detect async gen

```python
is_async_gen = is_async_gen_callable(function)
is_async = is_async_callable(function) or is_async_gen
```

Pass `is_async_gen` to `FunctionSchema(...)`.

## Durable Execution

### Temporal (`durable_exec/temporal/`)

Tools run in activities (separate processes). `RunContext` is serialized via `TemporalRunContext.serialize_run_context()` — the `custom_event_queue` field is `None` after deserialization (queues aren't serializable). `ctx.emit()` silently no-ops when queue is `None`.

**Behavior**: Events silently dropped in Temporal activities. Tool still returns its value. This is acceptable — Temporal activities are meant for durable results, not ephemeral UI events.

**Optional follow-up**: Buffer events in the activity and return them via `ToolReturn.metadata` alongside the result, so they're emitted post-completion rather than dropped entirely. Requires extending `CallToolResult` union in `_toolset.py`.

### DBOS (`durable_exec/dbos/`)

Steps run in-process. The `RunContext` with its queue reference survives through `@DBOS.step()`. **Events work natively, no changes needed.**

On replay: step return value is used without re-execution, so events won't re-emit. Acceptable — events are ephemeral.

### Prefect (`durable_exec/prefect/`)

In-process tasks: events work natively (queue reference survives).
Remote tasks: same degradation as Temporal (queue is `None` after serialization).

**No changes needed** for default in-process case.

## Backward Compatibility

| Concern | Analysis |
|---------|----------|
| **Existing tools** | Unaffected. `FunctionSchema.call()` checks `is_async_gen` first; existing sync/async tools take unchanged paths. `emit()` method on RunContext is additive. |
| **Existing event consumers** | `HandleResponseEvent` is discriminated union. New `custom_tool_event` kind hits `case _: pass` in existing match statements. Base `UIEventStream.handle_event` has default case. |
| **Existing adapters** | Base `handle_custom_tool_event` is a no-op. Third-party adapters silently ignore the event. |
| **Existing `ToolReturn.metadata` pattern** | Fully preserved. Tools can still use `ToolReturn(return_value=..., metadata=[BaseEvent(...)])` for post-completion events. |
| **Frontend compatibility** | AG-UI: `CustomEvent` is already part of the AG-UI protocol. Vercel AI: `DataChunk` is an existing chunk type. No frontend changes needed. |
| **Message history** | `CustomToolEvent` is ephemeral (stream-only). Not persisted in `ModelRequest`/`ModelResponse` message parts. No serialization impact. |
| **Type safety** | `CustomEvent` uses `Any` for `data` initially. Generic `CustomEventDataT` on `Agent` can be added as follow-up without breaking changes. |

## API Stability Considerations

- `CustomEvent` and `Return` become public API (`from pydantic_ai import CustomEvent, Return`)
- `ctx.emit()` becomes public API on `RunContext`
- `CustomToolEvent` in `AgentStreamEvent` union — new event kind, additive
- All additive changes; no existing behavior modified
- `data: Any` on `CustomEvent` is intentionally loose for v1; can be tightened with generics later without breaking existing usage

## Testing Strategy

### Unit tests
- `is_async_gen_callable()` detection for async generators, regular async, sync functions
- `RunContext.emit()` with and without queue
- `CustomEvent` construction and repr
- `_call_async_gen` with mock queue: yields events, collects Return, handles missing Return

### Integration tests
- Sequential mode: `CustomToolEvent` appears in stream interleaved with tool execution
- Parallel mode: events from multiple concurrent tools all appear
- Mixed: one streaming + one regular tool in parallel
- Error mid-stream: tool raises after yielding events — prior events visible, error propagates
- No `Return`: generator finishes without `Return` -> `None` return value
- AG-UI adapter: SSE output includes `CUSTOM` events with correct name/value
- Vercel AI adapter: output includes `data-{name}` chunks
- `event_stream_handler` receives `CustomToolEvent` alongside other events
- Capability hooks: `wrap_run_event_stream` can observe/filter `CustomToolEvent`

### Durable exec tests
- Temporal: streaming tool in activity -> events dropped, return value preserved
- DBOS: streaming tool in step -> events emitted normally

## Documentation Updates

- `docs/tools.md` or `docs/tools-advanced.md`: 'Streaming Custom Events' section
- `docs/ui/ag-ui.md`: custom events mapping
- `docs/ui/vercel-ai.md`: custom events mapping
- API docs: `CustomEvent`, `CustomToolEvent`, `Return`, `RunContext.emit()`
- `docs/durable-exec.md` (if exists): degradation behavior note

## Open Questions for Maintainers

1. **Phase 1 vs Phase 2 priority**: Is `ctx.emit()` alone sufficient for a first PR, or should the generator approach be included from the start?
2. **Strict yield validation**: Should yielding a non-`CustomEvent`/`Return` from a generator raise `UserError`, or auto-wrap in `CustomEvent(name='custom', data=value)` (as Douwe's PR description suggests)?
3. **`CustomEventDataT` generic**: Should this be planned for v1 or explicitly deferred? Default `Never` (strict) vs `object` (permissive)?
4. **Temporal buffering**: Worth implementing event buffering in activities for post-completion emission, or acceptable to silently drop?
5. **Sync generator tools**: Support `def tool() -> Iterator[...]` alongside async? Or async-only for v1?
6. **Event ordering guarantee**: In parallel mode, should `CustomToolEvent` be ordered by emission time, or is arbitrary interleaving acceptable?

## Files to Modify

### Core (Phase 1)
- `pydantic_ai_slim/pydantic_ai/messages.py` — `CustomEvent`, `CustomToolEvent`, `HandleResponseEvent` union
- `pydantic_ai_slim/pydantic_ai/_run_context.py` — `custom_event_queue` field, `emit()` method
- `pydantic_ai_slim/pydantic_ai/_agent_graph.py` — queue injection in `_call_tools()`, event multiplexing
- `pydantic_ai_slim/pydantic_ai/ui/_event_stream.py` — `handle_custom_tool_event` dispatch
- `pydantic_ai_slim/pydantic_ai/ui/ag_ui/_event_stream.py` — AG-UI `CustomEvent` mapping
- `pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_event_stream.py` — Vercel AI `DataChunk` mapping
- `pydantic_ai_slim/pydantic_ai/__init__.py` — exports

### Core (Phase 2, additive)
- `pydantic_ai_slim/pydantic_ai/_utils.py` — `is_async_gen_callable()`
- `pydantic_ai_slim/pydantic_ai/_function_schema.py` — `is_async_gen` field, `_call_async_gen()`
- `pydantic_ai_slim/pydantic_ai/messages.py` — `Return[T]`

### Tests
- `tests/test_streaming.py` or new `tests/test_custom_events.py`
- `tests/test_ag_ui.py`
- `tests/test_vercel_ai.py`

### Docs
- `docs/tools.md` or `docs/tools-advanced.md`
- `docs/ui/ag-ui.md`
- `docs/ui/vercel-ai.md`
