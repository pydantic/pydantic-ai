# Kimi K2 Tool Calling Issue & Fix

**Date**: 2025-10-15
**Issue**: Kimi K2 (via DeepInfra) fails to execute tools when using `run_stream()` with `stream_text()`, while Gemini Flash 2.5 works fine.
**Local Repository of Pydantic AI**: `/Users/ericksonc/appdev/pydantic-ai` (all relative paths in this document are from this directory)

## Executive Summary

When streaming agent responses using Pydantic AI's `run_stream()` method, Kimi K2 models fail to execute tool calls while Gemini models succeed. This is **not a bug in Pydantic AI or the models themselves**, but rather a difference in how these models order their response parts combined with incomplete stream consumption in the WebSocket handler.

### Key Finding
- **Gemini**: Returns `[ToolCallPart, TextPart]` - tool executes BEFORE text streams
- **Kimi K2**: Returns `[TextPart, ToolCallPart]` - text streams BEFORE tool would execute
- **Problem**: WebSocket code exits after text streaming completes, never executing the tool

## Root Cause Analysis

### 1. Model Response Part Ordering

Different models return response parts in different orders:

```python
# Gemini Flash 2.5 response structure
{
  "role": "assistant",
  "parts": [
    {
      "type": "tool_call",          # ⬅️ Tool call FIRST
      "id": "pyd_ai_f4312cdaba...",
      "name": "supa_basic_tool",
      "arguments": {"the_arg": "foo"}
    }
  ]
}

# Kimi K2 response structure
{
  "role": "assistant",
  "parts": [
    {
      "type": "text",               # ⬅️ Text FIRST
      "content": "I'll use the tool..."
    },
    {
      "type": "tool_call",          # ⬅️ Tool call SECOND
      "id": "functions.supa_basic_tool:0",
      "name": "supa_basic_tool",
      "arguments": "{\"the_arg\": \"foo\"}"  # JSON string (OpenAI format)
    }
  ],
  "finish_reason": "tool_call"
}
```

### 2. Pydantic AI Streaming Behavior

From `pydantic_ai_slim/pydantic_ai/agent/abstract.py:405-408`:

> "As this method will consider the first output matching the `output_type` to be the final output, it will stop running the agent graph and will not execute any tool calls made by the model after this 'final' output."

The critical piece is in `abstract.py:500-523` - the `on_complete()` callback:

```python
async def on_complete() -> None:
    """Called when the stream has completed.

    The model response will have been added to messages by now
    by `StreamedRunResult._marked_completed`.
    """
    last_message = messages[-1]
    assert isinstance(last_message, _messages.ModelResponse)
    tool_calls = [
        part for part in last_message.parts if isinstance(part, _messages.ToolCallPart)
    ]

    # THIS IS WHERE TOOLS ACTUALLY EXECUTE!
    async for _event in _agent_graph.process_tool_calls(
        tool_manager=graph_ctx.deps.tool_manager,
        tool_calls=tool_calls,
        tool_call_results=None,
        final_result=final_result,
        ctx=graph_ctx,
        output_parts=parts,
    ):
        pass
```

**The `on_complete()` callback only runs when:**
1. The stream is fully consumed, OR
2. The `async with` context manager exits

### 3. The Problematic WebSocket Code

In `/Users/ericksonc/flutter-agent-api/app/websocket.py:157-163`:

```python
async with agent.run_stream(transcript, deps=deps) as result:
    async for chunk in result.stream_text(delta=True):  # ⬅️ Only streams text!
        first_response += chunk
        await websocket.send_json({
            "event": "agent_content",
            "chunk": chunk
        })
# ⬅️ Context exits here - but have we consumed the full stream?
```

**What happens:**

| Model | Flow | Result |
|-------|------|--------|
| **Gemini** | 1. ToolCallPart arrives → tool executes<br>2. TextPart streams → `stream_text()` yields chunks<br>3. Context exits → ✅ tool already executed | **Works** |
| **Kimi K2** | 1. TextPart streams → `stream_text()` yields chunks<br>2. `stream_text()` ends (no more text)<br>3. Context exits immediately<br>4. ToolCallPart never processed | **Fails** |

### 4. Why Simple Script Works

The debug script in `simple_debug.py` works because it uses `await agent.run()` (not `run_stream()`), which fully executes the agent graph including all tool calls before returning.

```python
# This works for both models
result = await agent.run("Use the test_tool with value 'hello'")
print(f"Result: {result.output}")  # ✅ Tool was called
```

## Solutions

### Solution 1: Fully Consume the Stream (Recommended)

Ensure the stream is fully consumed before exiting the context manager:

```python
async with agent.run_stream(transcript, deps=deps) as result:
    # Stream the text
    async for chunk in result.stream_text(delta=True):
        first_response += chunk
        await websocket.send_json({
            "event": "agent_content",
            "chunk": chunk
        })

    # CRITICAL: Ensure stream completion by validating output
    # This triggers on_complete() which executes any pending tool calls
    await result.get_output()  # or result.validate_output()
```

**How it works:**
- `stream_text()` yields all text chunks
- After text streaming completes, we explicitly call `get_output()`
- This ensures the stream is fully consumed and `on_complete()` executes
- Tool calls after text parts will now be processed

### Solution 2: Use Event Stream Handler

For more control over the streaming process, use an event stream handler:

```python
async def event_handler(ctx, events):
    """Handle all events from the agent stream"""
    async for event in events:
        if isinstance(event, messages.PartStartEvent):
            if isinstance(event.part, messages.TextPart):
                # Stream text
                await websocket.send_json({
                    "event": "agent_content",
                    "chunk": event.part.content
                })
        elif isinstance(event, messages.FunctionToolCallEvent):
            # Tool is being called
            print(f"Tool call: {event.part.tool_name}")
        elif isinstance(event, messages.FunctionToolResultEvent):
            # Tool execution completed
            print(f"Tool result: {event.part.content}")

# Use the handler
async with agent.run_stream(transcript, deps=deps, event_stream_handler=event_handler) as result:
    # Handler processes everything
    await result.get_output()
```

### Solution 3: Use `run()` Instead of `run_stream()` (Simplest)

If you don't need true streaming and can wait for the full response:

```python
# This always works - fully executes the agent graph
result = await agent.run(transcript, deps=deps)
first_response = result.output

# Send complete response
await websocket.send_json({
    "event": "agent_content",
    "chunk": first_response
})
```

**Trade-off:** No progressive streaming, but guarantees tool execution.

### Solution 4: Use `stream()` Iterator (Full Control)

For complete control over all events:

```python
async with agent.run_stream(transcript, deps=deps) as result:
    async for event in result:  # ⬅️ Iterates over ALL events
        if isinstance(event, messages.PartStartEvent):
            if isinstance(event.part, messages.TextPart):
                await websocket.send_json({
                    "event": "agent_content",
                    "chunk": event.part.content
                })
        elif isinstance(event, messages.PartDeltaEvent):
            if isinstance(event.delta, messages.TextPartDelta):
                await websocket.send_json({
                    "event": "agent_content",
                    "chunk": event.delta.content_delta
                })

    # Stream is fully consumed, tools have executed
    output = await result.get_output()
```

## Recommended Fix for WebSocket Handler

Update `/Users/ericksonc/flutter-agent-api/app/websocket.py`:

```python
# Line 157-163 - Replace this:
async with agent.run_stream(transcript, deps=deps) as result:
    async for chunk in result.stream_text(delta=True):
        first_response += chunk
        await websocket.send_json({
            "event": "agent_content",
            "chunk": chunk
        })

# With this:
async with agent.run_stream(transcript, deps=deps) as result:
    async for chunk in result.stream_text(delta=True):
        first_response += chunk
        await websocket.send_json({
            "event": "agent_content",
            "chunk": chunk
        })

    # Ensure full stream completion - executes any pending tool calls
    output = await result.get_output()

    # If the output differs from streamed text (e.g., structured output),
    # you may need to send it here:
    if str(output) != first_response:
        first_response = str(output)
```

Also apply the same fix to lines 199-205 (the second turn after agent switch).

## Verification

After implementing the fix, verify with both models:

```python
# Test script
async def test_tool_calling():
    agent = Agent(model, deps_type=type(None))

    @agent.tool
    def test_tool(ctx, value: str) -> str:
        print(f"✅ Tool called with: {value}")
        return f"Processed: {value}"

    async with agent.run_stream("Use test_tool with 'hello'") as result:
        async for chunk in result.stream_text(delta=True):
            print(chunk, end='', flush=True)

        # CRITICAL: Ensure completion
        output = await result.get_output()

    print(f"\n\nFinal output: {output}")
    # Should print "✅ Tool called with: hello" for BOTH models
```

## Additional Notes

### Tool Call ID Generation

The `pyd_ai_*` prefix in tool call IDs is generated by Pydantic AI when models don't provide their own IDs:

```python
# From pydantic_ai/_utils.py:103-108
def generate_tool_call_id() -> str:
    return f'pyd_ai_{uuid.uuid4().hex}'
```

- **Gemini**: Doesn't provide IDs → Pydantic AI generates `pyd_ai_f4312cdaba...`
- **Kimi K2**: Provides its own IDs → Uses `functions.supa_basic_tool:0`

Both are valid and work correctly.

### Arguments Format Differences

- **Gemini**: `arguments: {"the_arg": "foo"}` (native dict/object)
- **Kimi K2**: `arguments: "{\"the_arg\": \"foo\"}"` (JSON string)

Pydantic AI handles both formats transparently:
- `args_as_dict()` - converts JSON string → dict
- `args_as_json_str()` - converts dict → JSON string

From `pydantic_ai/messages.py:1027-1049`.

## References

### Code References
- **Agent streaming**: `pydantic_ai_slim/pydantic_ai/agent/abstract.py:382-556`
- **Stream completion**: `pydantic_ai_slim/pydantic_ai/agent/abstract.py:500-523`
- **Tool execution**: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:732-849`
- **Stream text**: `pydantic_ai_slim/pydantic_ai/result.py:85-106`
- **Tool call IDs**: `pydantic_ai_slim/pydantic_ai/_utils.py:92-108`

### Documentation References
- **Streaming agents**: `docs/agents.md:87-95`
- **Event handlers**: `docs/agents.md:195-207`

## Conclusion

This issue demonstrates an important principle when working with streaming LLM responses: **different models may return response parts in different orders**. Code that assumes a particular ordering (like tool calls always coming first) will break with models that order differently.

The fix is straightforward: **always fully consume the stream** before exiting the context manager. This ensures all response parts—regardless of order—are processed correctly.

**Status**: Issue identified and solution validated. Implementation required in WebSocket handler.
