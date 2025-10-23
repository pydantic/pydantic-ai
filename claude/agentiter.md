# Agent.iter() Deep Dive

## Overview

`agent.iter()` is the most powerful and flexible method for running a Pydantic AI agent. It provides access to the internal agent graph execution as it happens, allowing you to observe and react to every step of the agent's processing in real-time.

## Key Characteristics

### 1. **Graph-Based Execution Model**
- Returns an `AgentRun` object that exposes the underlying graph execution
- Allows iteration through graph nodes as they execute
- Nodes include: `UserPromptNode`, `ModelRequestNode`, `CallToolsNode`, and `End`

### 2. **Real-Time Access**
- Provides immediate access to tool calls and returns as they occur
- No need to wait for the entire run to complete
- Perfect for building responsive UIs that show progress

### 3. **Async Context Manager**
- Used with `async with` syntax
- Returns an async iterator over graph nodes
- Provides `get_output()` method for final result

## Comparison with Other Run Methods

### agent.iter() vs agent.run()

| Aspect | agent.iter() | agent.run() |
|--------|-------------|------------|
| **Return Type** | AsyncIterator of nodes | Complete RunResult |
| **Access Pattern** | Real-time, node-by-node | Wait for completion |
| **Tool Visibility** | Immediate access to tool calls/returns | Only in final result |
| **Use Case** | Building UIs, debugging, monitoring | Simple request/response |
| **Complexity** | More complex, requires iteration | Simple, straightforward |

### agent.iter() vs agent.run_stream()

| Aspect | agent.iter() | agent.run_stream() |
|--------|-------------|-------------------|
| **Streaming Type** | Graph nodes with optional token streaming | Text chunks from model |
| **Granularity** | Every execution step + token streaming if desired | Model output tokens only |
| **Tool Access** | Full tool call/return events | Limited tool visibility |
| **Primary Use** | Complete execution monitoring + streaming | Streaming text to UI |
| **Deprecation Status** | Future replacement for run_stream | Planned deprecation (TODO at line 1097) |

## How agent.iter() Works

### Basic Flow

```python
async with agent.iter('user prompt') as agent_run:
    async for node in agent_run:
        # Process each node as it executes
        if isinstance(node, CallToolsNode):
            # Access tool execution events
            async for event in node:
                # Handle FunctionToolCallEvent, FunctionToolResultEvent
                pass

    # Get final result after iteration
    result = await agent_run.get_output()
```

### Node Types You'll Encounter

1. **UserPromptNode**: Initial user input and system prompts
2. **ModelRequestNode**: Request being sent to the LLM
3. **CallToolsNode**: Tool execution (contains tool call/return events)
4. **End**: Final node with the result

## Streaming Behavior

### Does it stream like run() or run_stream()?

**Answer: It's a superset that can do both!**

The key insight: **`agent.iter()` provides access to nodes, and nodes can be streamed for token-level output.**

### How Token Streaming Works with agent.iter()

When you encounter a `ModelRequestNode`, you can call `node.stream()` to get token-level streaming:

```python
async with agent.iter(prompt) as agent_run:
    async for node in agent_run:
        if Agent.is_model_request_node(node):
            # Stream tokens from this node!
            async with node.stream(agent_run.ctx) as stream:
                async for event in stream:
                    if isinstance(event, PartDeltaEvent):
                        if isinstance(event.delta, TextPartDelta):
                            # This is a streaming text token!
                            print(event.delta.content_delta, end='', flush=True)
```

### What You Get in Real-Time

1. **Graph Nodes**: As they execute (UserPromptNode, ModelRequestNode, etc.)
2. **Token Streaming**: Via `node.stream()` on ModelRequestNodes
3. **Tool Calls**: Immediately when the model decides to call a tool
4. **Tool Returns**: As soon as the tool completes execution
5. **Partial Arguments**: Tool arguments streamed as they're generated
6. **Final Results**: Notification when the model produces final output

## Practical Implementation for Tool Access

Based on `claude/tool_call_return.md`, here's the recommended approach:

```python
from pydantic_ai import Agent
from pydantic_ai._agent_graph import CallToolsNode
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

async def process_with_tool_monitoring(agent, prompt):
    tool_calls = []
    tool_results = []

    async with agent.iter(prompt) as agent_run:
        async for node in agent_run:
            if isinstance(node, CallToolsNode):
                async for event in node:
                    if isinstance(event, FunctionToolCallEvent):
                        # Tool is about to be called
                        tool_call = event.part
                        tool_calls.append({
                            'tool': tool_call.tool_name,
                            'args': tool_call.args,
                            'id': tool_call.tool_call_id
                        })
                        # Send to UI immediately

                    elif isinstance(event, FunctionToolResultEvent):
                        # Tool has returned
                        result = event.result
                        tool_results.append({
                            'tool': result.tool_name,
                            'content': result.content,
                            'id': result.tool_call_id
                        })
                        # Update UI with result

    final_result = await agent_run.get_output()
    return final_result, tool_calls, tool_results
```

## When to Use agent.iter()

### Use agent.iter() when you need:
- Real-time visibility into tool execution
- To build responsive UIs showing agent progress
- Debugging information about execution flow
- Custom processing of intermediate results
- Maximum control over the agent execution

### Don't use agent.iter() when:
- You just need the final result (use `run()`)
- You only care about streaming text output (use `run_stream()` for now)
- Simplicity is more important than control
- You're building a simple request/response system

## Key Advantages

1. **Granular Control**: Access to every step of execution
2. **Real-Time Updates**: No waiting for completion
3. **Tool Transparency**: Full visibility into tool calls and returns
4. **Debugging Power**: See exactly how the agent processes requests
5. **UI Building**: Perfect for progress indicators and live updates

## Important Notes

- **Deprecation Plan**: The TODO comment (line 1097-1098) indicates `run_stream()` will be deprecated once `iter()` adds a final result event
- **Why deprecate run_stream()?** Because `iter()` can do everything `run_stream()` does and more:
  - Access the same token streaming via `node.stream()`
  - Also get tool execution visibility
  - Plus full graph execution monitoring
- **The missing piece**: They need to add a `FinalResultEvent` to fully replace `run_stream()`
- Events fire as soon as they occur, not after buffering
- Tool call IDs link calls to their corresponding returns
- The method is async-only (no sync version exists)

## Summary

`agent.iter()` is the most powerful way to run a Pydantic AI agent, providing real-time access to the execution graph. **Crucially, it's not just about graph nodes - you can also stream tokens by calling `node.stream()` on ModelRequestNodes**, making it a true superset of `run_stream()` functionality.

The planned deprecation of `run_stream()` makes sense because `agent.iter()` can:
1. Do everything `run_stream()` does (token streaming)
2. Plus provide tool execution visibility
3. Plus expose the entire graph execution

The method essentially exposes the internal workings of the agent, making it ideal for scenarios where you need to:
- Stream text tokens to the UI (via `node.stream()`)
- Monitor and display agent progress
- Access tool execution in real-time
- Build sophisticated UI experiences
- Debug agent behavior
- Implement custom execution logic

**Bottom line**: `agent.iter()` isn't replacing token streaming - it's providing a unified API that includes token streaming as one of its capabilities.