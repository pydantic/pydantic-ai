# Migrating from stream_text() to agent.iter(): The Simple Path

## Your Situation

You had this working perfectly:

```python
async with agent.run_stream(prompt) as result:
    async for text_delta in result.stream_text(delta=True):
        await send_to_client(text_delta)
```

Then you added a tool:

```python
@agent.tool
def get_weather(city: str) -> str:
    return f"Sunny in {city}"
```

And suddenly:
```
UserError: stream_text() can only be used with text responses
```

## The Problem

`stream_text()` only works with text-only agents. When your agent has tools, it can't use `stream_text()` because it might need to call those tools.

## The Solution: agent.iter()

Switch to `agent.iter()` which gives you the same text delta streaming but doesn't break with tools.

### Before (broke with tools):
```python
async with agent.run_stream(prompt) as result:
    async for text_delta in result.stream_text(delta=True):
        await send_to_client(text_delta)
```

### After (works with tools):
```python
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, PartStartEvent, TextPart

async with agent.iter(prompt) as agent_run:
    async for node in agent_run:
        if Agent.is_model_request_node(node):
            async with node.stream(agent_run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            await send_to_client(event.delta.content_delta)
                    elif isinstance(event, PartStartEvent):
                        if isinstance(event.part, TextPart) and event.part.content:
                            await send_to_client(event.part.content)
```

## What Changed?

1. **`run_stream()` → `iter()`**: Different method
2. **Iterate nodes first**: Loop through graph nodes
3. **Check for ModelRequestNode**: Only stream from model response nodes
4. **Stream from node**: Call `node.stream()` to get events
5. **Same events!**: `PartDeltaEvent` with `TextPartDelta` - exactly what you need

## Complete Example

```python
from pydantic_ai import Agent
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, PartStartEvent, TextPart

agent = Agent('openai:gpt-4o')

@agent.tool
def get_weather(city: str) -> str:
    return f"Weather in {city}: sunny, 22°C"

async def stream_to_client(prompt: str):
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            # Only stream from model responses
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    async for event in stream:
                        # Text deltas - just like stream_text(delta=True)!
                        if isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                text_delta = event.delta.content_delta
                                print(text_delta, end='', flush=True)

                        # Initial text chunks
                        elif isinstance(event, PartStartEvent):
                            if isinstance(event.part, TextPart) and event.part.content:
                                print(event.part.content, end='', flush=True)

    # Get final result after streaming
    result = await agent_run.get_output()
    return result

# Usage
await stream_to_client("What's the weather in Paris?")
# Streams: "I'll check the weather for you. The weather in Paris is sunny, 22°C."
```

## Helper Function (Optional)

If you want to make it feel more like `stream_text()`:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, PartStartEvent, TextPart

async def stream_text_deltas(agent_run):
    """Drop-in replacement for result.stream_text(delta=True) that works with tools"""
    async for node in agent_run:
        if Agent.is_model_request_node(node):
            async with node.stream(agent_run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            yield event.delta.content_delta
                    elif isinstance(event, PartStartEvent):
                        if isinstance(event.part, TextPart) and event.part.content:
                            yield event.part.content

# Usage - almost like before!
async with agent.iter(prompt) as agent_run:
    async for text_delta in stream_text_deltas(agent_run):
        await send_to_client(text_delta)

    result = await agent_run.get_output()
```

## What About Showing Tool Activity?

That's a bonus feature! You can optionally show when tools are being called:

```python
from pydantic_ai._agent_graph import CallToolsNode
from pydantic_ai.messages import FunctionToolCallEvent

async with agent.iter(prompt) as agent_run:
    async for node in agent_run:
        # Stream text
        if Agent.is_model_request_node(node):
            async with node.stream(agent_run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            await send_to_client({"type": "text", "delta": event.delta.content_delta})

        # Show tool calls (optional!)
        elif isinstance(node, CallToolsNode):
            async for event in node:
                if isinstance(event, FunctionToolCallEvent):
                    await send_to_client({
                        "type": "tool",
                        "name": event.part.tool_name
                    })
```

## Common Questions

### Q: Is agent.iter() slower?
No! It's actually more direct - you're getting events straight from the model.

### Q: Do I have to show tool activity?
No! You can just stream text and ignore the CallToolsNode entirely. Tools will still work in the background.

### Q: Can I still use run_stream()?
For text-only agents (no tools), yes. But `run_stream()` is planned for deprecation in favor of `agent.iter()`.

### Q: What if I don't care about streaming?
Use `agent.run()` for a simple request/response pattern.

## Summary

**The migration is straightforward:**
1. Replace `agent.run_stream()` with `agent.iter()`
2. Loop through nodes instead of directly on result
3. Check for `ModelRequestNode` and stream from it
4. Handle the same events you're used to: `PartDeltaEvent` with `TextPartDelta`
5. Text deltas work exactly the same way
6. Tools no longer break your streaming

**Bottom line**: `agent.iter()` gives you the same text delta streaming as `stream_text(delta=True)`, but it doesn't break when you add tools to your agent.