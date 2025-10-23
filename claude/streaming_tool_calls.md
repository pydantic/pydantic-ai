# Streaming Tool Calls and Arguments

## The Full Picture

When you stream with `agent.iter()`, you can see:
1. **Text deltas** - as the model generates text
2. **Tool call deltas** - as the model decides to call tools
3. **Tool argument deltas** - as the model generates the arguments (character by character for JSON, or key by key for dicts!)
4. **Tool execution** - when tools actually run
5. **Tool results** - when tools return

## Where Tool Calls Stream From

**Important**: Tool calls stream as part of the **model response**, not from CallToolsNode.

- **ModelRequestNode** → streams tool calls being generated
- **CallToolsNode** → streams tool execution and results

## Streaming Tool Arguments: Two Modes

The model can stream tool arguments in two ways:

### 1. JSON String Deltas (Most Common)
```python
# Arguments stream as JSON text, character by character
args_delta: str = '{"city"'
args_delta: str = ': "Par'
args_delta: str = 'is"}'
```

### 2. Dictionary Deltas (Some Models)
```python
# Arguments stream as structured updates, key by key
args_delta: dict = {"city": "Paris"}
args_delta: dict = {"country": "France"}
```

## Complete Example: Streaming Everything

```python
from pydantic_ai import Agent
from pydantic_ai._agent_graph import CallToolsNode
from pydantic_ai.messages import (
    PartDeltaEvent, TextPartDelta, ToolCallPartDelta,
    PartStartEvent, TextPart, ToolCallPart,
    FunctionToolCallEvent, FunctionToolResultEvent
)

agent = Agent('openai:gpt-4o')

@agent.tool
def get_weather(city: str, units: str = "celsius") -> str:
    return f"Weather in {city}: 22°{units[0].upper()}, sunny"

async def stream_everything(prompt: str):
    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:

            # ===== MODEL RESPONSE STREAMING =====
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    # Track partial tool calls
                    tool_calls_building = {}

                    async for event in stream:

                        # --- TEXT STREAMING ---
                        if isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                text_delta = event.delta.content_delta
                                print(f"[TEXT] {text_delta}", end='', flush=True)

                        elif isinstance(event, PartStartEvent):
                            if isinstance(event.part, TextPart) and event.part.content:
                                print(f"[TEXT] {event.part.content}", end='', flush=True)

                        # --- TOOL CALL STREAMING ---
                        # Tool call starting
                        if isinstance(event, PartStartEvent):
                            if isinstance(event.part, ToolCallPart):
                                tool_name = event.part.tool_name
                                print(f"\n[TOOL CALL START] {tool_name}")
                                tool_calls_building[event.index] = {
                                    "name": tool_name,
                                    "args": event.part.args or ""
                                }

                        # Tool call delta (arguments streaming!)
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, ToolCallPartDelta):
                                idx = event.index

                                # Tool name delta (rare)
                                if event.delta.tool_name_delta:
                                    print(f"[TOOL NAME+] {event.delta.tool_name_delta}")

                                # Arguments delta - THIS IS THE MAGIC!
                                if event.delta.args_delta:
                                    if isinstance(event.delta.args_delta, str):
                                        # JSON streaming character by character
                                        print(f"[ARGS+] {event.delta.args_delta}", end='', flush=True)
                                        if idx in tool_calls_building:
                                            tool_calls_building[idx]["args"] += event.delta.args_delta

                                    elif isinstance(event.delta.args_delta, dict):
                                        # Structured streaming key by key
                                        print(f"[ARGS+] {event.delta.args_delta}")
                                        if idx in tool_calls_building:
                                            if isinstance(tool_calls_building[idx]["args"], str):
                                                tool_calls_building[idx]["args"] = {}
                                            tool_calls_building[idx]["args"].update(event.delta.args_delta)

            # ===== TOOL EXECUTION STREAMING =====
            elif isinstance(node, CallToolsNode):
                async for event in node:

                    # Tool about to execute
                    if isinstance(event, FunctionToolCallEvent):
                        print(f"\n[EXECUTING] {event.part.tool_name}")
                        print(f"[ARGS] {event.part.args_as_dict()}")

                    # Tool finished executing
                    elif isinstance(event, FunctionToolResultEvent):
                        print(f"[RESULT] {event.result.content}")

    print("\n[DONE]")
    result = await agent_run.get_output()
    return result

# Usage
await stream_everything("What's the weather in Paris and London?")
```

## Output Example

```
[TEXT] I'll check
[TEXT]  the weather for
[TEXT]  you.

[TOOL CALL START] get_weather
[ARGS+] {"city"
[ARGS+] : "Paris"
[ARGS+] , "units"
[ARGS+] : "celsius"
[ARGS+] }

[EXECUTING] get_weather
[ARGS] {'city': 'Paris', 'units': 'celsius'}
[RESULT] Weather in Paris: 22°C, sunny

[TOOL CALL START] get_weather
[ARGS+] {"city"
[ARGS+] : "London"
[ARGS+] }

[EXECUTING] get_weather
[ARGS] {'city': 'London', 'units': 'celsius'}
[RESULT] Weather in London: 22°C, sunny

[TEXT] The weather
[TEXT]  in both cities is
[TEXT]  sunny at 22°C.
[DONE]
```

## Practical: Show Tool Status to Users

Here's a realistic example for a web app:

```python
async def stream_to_websocket(prompt: str, websocket):
    async with agent.iter(prompt) as agent_run:
        tool_args_accumulator = {}

        async for node in agent_run:
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as stream:
                    async for event in stream:

                        # Stream text
                        if isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                await websocket.send_json({
                                    "type": "text_delta",
                                    "content": event.delta.content_delta
                                })

                            # Stream tool arguments building up
                            elif isinstance(event.delta, ToolCallPartDelta):
                                if event.delta.args_delta:
                                    idx = event.index

                                    # Accumulate JSON arguments
                                    if isinstance(event.delta.args_delta, str):
                                        if idx not in tool_args_accumulator:
                                            tool_args_accumulator[idx] = ""
                                        tool_args_accumulator[idx] += event.delta.args_delta

                                        # Try to parse as we go (will fail until complete)
                                        try:
                                            import json
                                            parsed = json.loads(tool_args_accumulator[idx])
                                            await websocket.send_json({
                                                "type": "tool_args_preview",
                                                "args": parsed,
                                                "complete": False
                                            })
                                        except json.JSONDecodeError:
                                            pass  # Not complete yet

                        # Tool call complete
                        elif isinstance(event, PartStartEvent):
                            if isinstance(event.part, ToolCallPart):
                                await websocket.send_json({
                                    "type": "tool_call",
                                    "name": event.part.tool_name,
                                    "args": event.part.args_as_dict()
                                })

            # Tool execution
            elif isinstance(node, CallToolsNode):
                async for event in node:
                    if isinstance(event, FunctionToolCallEvent):
                        await websocket.send_json({
                            "type": "tool_executing",
                            "name": event.part.tool_name
                        })

                    elif isinstance(event, FunctionToolResultEvent):
                        await websocket.send_json({
                            "type": "tool_result",
                            "result": event.result.content
                        })
```

## Key Points

### 1. Tool Arguments Stream as JSON
Most models (OpenAI, Anthropic) stream tool arguments as JSON text. You see it being built character by character:
```python
'{"ci'
'ty": '
'"Paris'
'"}'
```

### 2. You Can Show Partial Arguments
As JSON accumulates, you can try parsing it and show preview to users:
```python
partial_json = ""
for chunk in arg_chunks:
    partial_json += chunk
    try:
        parsed = json.loads(partial_json)
        show_preview(parsed)  # Show what we have so far!
    except JSONDecodeError:
        pass  # Not complete yet, keep accumulating
```

### 3. Tool Execution Happens Later
- **Streaming phase**: Tool calls are *proposed* by the model
- **Execution phase**: Tools actually *run* in CallToolsNode

### 4. Two Separate Streams
```python
# Stream 1: Model proposing tool calls (in ModelRequestNode)
async with node.stream(agent_run.ctx) as stream:
    async for event in stream:
        # ToolCallPartDelta - arguments being generated

# Stream 2: Tools executing (in CallToolsNode)
async for event in node:
    # FunctionToolCallEvent - tool starting
    # FunctionToolResultEvent - tool finished
```

## Summary

**Yes, you can stream tool arguments!** And it's not manual - Pydantic AI gives you the events:

1. **`ToolCallPartDelta`** with `args_delta: str` - JSON arguments streaming in
2. **`PartStartEvent`** with `ToolCallPart` - complete tool call ready
3. **`FunctionToolCallEvent`** - tool about to execute
4. **`FunctionToolResultEvent`** - tool finished

The "manual" part is just deciding what to show users:
- Show raw JSON chunks?
- Parse and show preview?
- Wait for complete args?
- Show "Calling tool..." status?

The streaming itself is built-in!