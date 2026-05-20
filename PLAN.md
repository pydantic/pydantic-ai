# Fix: drop stray `PartDeltaEvent` after `PartEndEvent` (#4733)

## Problem

Non-standard OpenAI Responses API endpoints can emit `ResponseFunctionCallArgumentsDeltaEvent` **after** `ResponseOutputItemDoneEvent` for the same tool call. This causes `iterator_with_part_end` to yield a `PartDeltaEvent` for a part that has already received its `PartEndEvent`, which the AG-UI adapter faithfully translates into a `TOOL_CALL_ARGS` event after `TOOL_CALL_END` — violating the AG-UI protocol and crashing `@ag-ui/client`.

**Observed SSE output:**
```
TOOL_CALL_START  (id: mcp_22b...)
TOOL_CALL_ARGS   (id: mcp_22b..., delta: '{"action":"call_tool",...}')
TOOL_CALL_END    (id: mcp_22b...)
TOOL_CALL_ARGS   (id: mcp_22b..., delta: '}')   <- stray delta AFTER end
```

## Root cause

`iterator_with_part_end` (`models/__init__.py`) injects `PartEndEvent` when a new `PartStartEvent` arrives, but does not filter subsequent `PartDeltaEvent`s that reference already-ended part indices. Late deltas from the model layer pass through unchecked.

## Fix

Add an `ended_indices: set[int]` to `iterator_with_part_end`. Track indices as they are ended; drop any `PartDeltaEvent` whose index is already in the set.

Handle index reuse: the `PartStartEvent` docstring says "if multiple `PartStartEvent`s are received with the same index, the new one should fully replace the old one", so clear the index from `ended_indices` when a new `PartStartEvent` arrives.

### Changes to `iterator_with_part_end` (`pydantic_ai_slim/pydantic_ai/models/__init__.py`)

```python
async def iterator_with_part_end(
    iterator: AsyncIterator[ModelResponseStreamEvent],
) -> AsyncIterator[ModelResponseStreamEvent]:
    last_start_event: PartStartEvent | None = None
    ended_indices: set[int] = set()

    # ... part_end_event() unchanged ...

    async for event in iterator:
        if isinstance(event, PartStartEvent):
            if last_start_event:
                end_event = part_end_event(event.part)
                if end_event:
                    ended_indices.add(end_event.index)
                    yield end_event

                event.previous_part_kind = last_start_event.part.part_kind
            ended_indices.discard(event.index)  # handle index reuse
            last_start_event = event
        elif isinstance(event, PartDeltaEvent) and event.index in ended_indices:
            continue  # drop stray delta for already-ended part

        yield event

    end_event = part_end_event()
    if end_event:
        yield end_event
```

### Test (`tests/`)

Add a test (in the appropriate test file — likely `tests/test_streaming.py` or a new focused file) that:

1. Creates a mock async iterator of model events with a `PartDeltaEvent` arriving after its `PartEndEvent` (i.e. after the next `PartStartEvent`)
2. Passes it through `iterator_with_part_end`
3. Asserts the stray delta is dropped and the event sequence is clean

This is a unit test of `iterator_with_part_end` — no VCR cassette or API call needed.

### Scope

- Generic fix: protects all part types (text, thinking, tool call), not just tool calls
- No changes to the AG-UI adapter, model layer, or public API
- No new dependencies

### Documentation

- Update `CLAUDE.md` / `AGENTS.md` if relevant patterns are documented there (unlikely for this change)
