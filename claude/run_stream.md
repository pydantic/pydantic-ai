# Understanding agent.run_stream()

## What It Streams

`agent.run_stream()` streams **token-by-token** updates from the LLM, not event-by-event. It's designed for streaming text generation as it's being produced by the model.

## Key Points

1. **Token streaming**: The stream yields partial model responses as tokens arrive from the LLM
2. **Debouncing**: By default, tokens are grouped/debounced by 100ms (`debounce_by=0.1`) to reduce overhead
3. **Not for tool events**: Tool calls only become available after the model finishes generating them

## What You Can Access While Streaming

```python
async with agent.run_stream('prompt') as stream:
    # Option 1: Stream the validated output (if structured)
    async for partial_output in stream.stream():
        # This is your validated output type, updated as tokens arrive
        print(partial_output)

    # Option 2: Stream raw text (for text-only outputs)
    async for text in stream.stream_text(delta=False):
        # Full text up to this point
        print(text)

    # Option 3: Stream text deltas only
    async for delta in stream.stream_text(delta=True):
        # Just the new text chunk
        print(delta, end='')

    # Option 4: Stream raw ModelResponse objects
    async for response in stream.stream_structured():
        # response is a ModelResponse with parts being built up
        # Tool calls appear here once fully generated
        for part in response.parts:
            if isinstance(part, ToolCallPart):
                # Tool call is complete and ready
                print(f"Tool: {part.tool_name}")
```

## How Token Boundaries Work

- The underlying model API sends token chunks as they're generated
- `PartStartEvent`: Signals a new part (text, tool call, etc.) is starting
- `PartDeltaEvent`: Contains incremental content (text deltas, tool arg deltas)
- Parts accumulate until complete (e.g., a full tool call with all arguments)

## Debouncing

The `debounce_by` parameter controls how tokens are grouped:
- `debounce_by=0.1`: Groups tokens that arrive within 100ms (default)
- `debounce_by=0`: No debouncing, yield immediately
- `debounce_by=None`: No debouncing, yield immediately

Debouncing reduces validation overhead for structured outputs.

## Important Limitations

1. **Tool execution happens after streaming**: Tool calls are streamed as they're generated, but tool execution only happens after the model response completes
2. **Can't access tool returns during streaming**: Tool returns are only available after streaming completes via the `on_complete` callback
3. **For event-by-event access use `agent.iter()`**: If you need tool call/return events as they happen, use `agent.iter()` instead

## When to Use What

- **`agent.run_stream()`**: For streaming text generation to show progress to users
- **`agent.iter()`**: For accessing discrete events (tool calls, tool returns, retries) as they happen
- **`agent.run()`**: When you just need the final result

## Example: Streaming vs Events

```python
# Streaming (token-by-token text)
async with agent.run_stream('Write a story') as stream:
    async for text in stream.stream_text(delta=True):
        print(text, end='')  # Streams: "Once" -> " upon" -> " a" -> " time"...

# Events (tool calls/returns as they complete)
async with agent.iter('Calculate something') as agent_run:
    async for node in agent_run:
        if isinstance(node, CallToolsNode):
            async for event in node:
                if isinstance(event, FunctionToolCallEvent):
                    # Tool is being called (complete call, not partial)
                elif isinstance(event, FunctionToolResultEvent):
                    # Tool has returned (complete result)
```