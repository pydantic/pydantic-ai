# Accessing Tool Call and Return Parts in Real-Time

## Overview

In Pydantic AI, when you want to access `ToolCallPart` and `ToolReturnPart` data as soon as they're available (rather than waiting for the entire agent run to complete), you have several options depending on your use case.

## Method 1: Using `agent.iter()` with Node Iteration

The `agent.iter()` method provides access to agent execution events as they happen. This is the most flexible approach for real-time access to tool calls and returns.

```python
from pydantic_ai import Agent
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

agent = Agent('openai:gpt-4o')

async def stream_with_tool_events():
    async with agent.iter('Process this with tools') as agent_run:
        async for node in agent_run:
            # Check if this is a CallToolsNode that contains tool execution info
            if isinstance(node, CallToolsNode):
                # Iterate through the tool call events
                async for event in node:
                    if isinstance(event, FunctionToolCallEvent):
                        # Tool call is about to be executed
                        tool_call = event.part  # This is a ToolCallPart
                        print(f"Tool called: {tool_call.tool_name}")
                        print(f"Tool args: {tool_call.args}")
                        print(f"Call ID: {tool_call.tool_call_id}")

                        # Send this to your UI immediately
                        await send_to_ui({
                            'type': 'tool_call',
                            'tool_name': tool_call.tool_name,
                            'args': tool_call.args_as_dict(),
                            'call_id': tool_call.tool_call_id
                        })

                    elif isinstance(event, FunctionToolResultEvent):
                        # Tool has returned a result
                        result = event.result  # This is a ToolReturnPart or RetryPromptPart
                        if hasattr(result, 'content'):
                            print(f"Tool result: {result.content}")
                            print(f"Tool name: {result.tool_name}")
                            print(f"Call ID: {result.tool_call_id}")

                            # Send result to UI
                            await send_to_ui({
                                'type': 'tool_result',
                                'tool_name': result.tool_name,
                                'content': result.content,
                                'call_id': result.tool_call_id,
                                'metadata': result.metadata  # if you stored any metadata
                            })

        # Get final result after all streaming is done
        result = await agent_run.get_output()
        return result
```

## Method 2: Direct Access from Message History

If you're using `run_stream()`, you can access the message history as it builds up:

```python
async def stream_with_message_inspection():
    async with agent.run_stream('Process this') as stream:
        # Process streaming events
        async for event in stream:
            # Check current message for tool calls
            current_msg = stream.get()  # Get current ModelResponse

            for part in current_msg.parts:
                if isinstance(part, ToolCallPart):
                    # A tool is being called
                    print(f"Tool: {part.tool_name}, Args: {part.args}")
                    # Note: The result won't be in this message yet

        # After streaming, get the complete history
        result = await stream.get_output()

        # Access complete message history including tool returns
        for msg in stream.all_messages():
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):
                        print(f"Tool called: {part.tool_name}")
            elif isinstance(msg, ModelRequest):
                for part in msg.parts:
                    if isinstance(part, ToolReturnPart):
                        print(f"Tool returned: {part.content}")
```

## Method 3: Using AG-UI Protocol (For Web Applications)

If you're building a web UI, the AG-UI integration provides structured events:

```python
from pydantic_ai.ag_ui import run_ag_ui, RunAgentInput

async def handle_ag_ui_stream(agent, input_data):
    async for event_str in run_ag_ui(agent, input_data):
        # Events are already formatted for AG-UI protocol
        # They include tool call starts/ends and results
        yield event_str  # Stream to client via SSE/WebSocket
```

## Key Differences Between Approaches

1. **`agent.iter()`**:
   - Provides the most granular control
   - Access to `FunctionToolCallEvent` and `FunctionToolResultEvent`
   - Best for custom processing and non-web UIs
   - Events fire as soon as tool calls/returns happen

2. **Message History Inspection**:
   - Simpler but less real-time
   - Tool returns only available after they complete
   - Good for simpler use cases

3. **AG-UI Protocol**:
   - Best for web applications
   - Pre-formatted events for UI consumption
   - Handles state management and event formatting

## Important Notes

- Tool calls (`ToolCallPart`) appear in `ModelResponse` messages
- Tool returns (`ToolReturnPart`) appear in `ModelRequest` messages (as they're sent back to the model)
- The `tool_call_id` links a tool call to its corresponding return
- With `agent.iter()`, events are emitted as soon as they occur, not after the entire response is complete
- Tool metadata can be accessed via `ToolReturnPart.metadata` if you use `ToolReturn` objects

## Example: Complete Implementation with UI Updates

```python
from pydantic_ai import Agent
from pydantic_ai.tools import ToolReturn
from typing import Any

agent = Agent('openai:gpt-4o')

@agent.tool
async def process_data(data: str) -> ToolReturn:
    # Process the data
    result = f"Processed: {data}"

    # Return with metadata for UI
    return ToolReturn(
        return_value=result,
        content=result,
        metadata={'processing_time': 0.5, 'status': 'success'}
    )

async def run_with_live_updates(prompt: str, ui_callback):
    """Run agent with live UI updates for tool calls/returns"""

    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            if isinstance(node, CallToolsNode):
                async for event in node:
                    if isinstance(event, FunctionToolCallEvent):
                        # Notify UI of tool call
                        await ui_callback({
                            'event': 'tool_call_start',
                            'tool': event.part.tool_name,
                            'args': event.part.args_as_dict(),
                            'id': event.tool_call_id
                        })

                    elif isinstance(event, FunctionToolResultEvent):
                        result = event.result
                        # Notify UI of tool result
                        await ui_callback({
                            'event': 'tool_call_complete',
                            'tool': result.tool_name,
                            'result': result.content,
                            'id': event.tool_call_id,
                            'metadata': getattr(result, 'metadata', None)
                        })

        # Get final output
        final = await agent_run.get_output()
        await ui_callback({
            'event': 'complete',
            'result': final
        })
        return final
```

This approach gives you complete tool call/return data as soon as it's available, perfect for rendering real-time updates in your UI without waiting for the entire agent run to complete.