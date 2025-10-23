# ModelMessage Flow Breakdown: How Tool Calls Create Request/Response Cycles

## Executive Summary

**Key Answer to Your Question**: Each tool call creates a **new request/response cycle**. The pattern is:
1. **ModelResponse** with tool calls →
2. **ModelRequest** with tool returns →
3. **ModelResponse** with more text/tool calls (repeat as needed)

It is **NOT** all within one giant ModelResponse. Each interaction with the model creates distinct ModelRequest and ModelResponse objects.

## The Core Pattern

### Single Tool Call Flow

```
agent.run("Roll a dice")
    ↓
1. ModelRequest([UserPromptPart("Roll a dice")])
    ↓
2. ModelResponse([ToolCallPart("roll_dice")])
    ↓
3. ModelRequest([ToolReturnPart("roll_dice", content=4)])
    ↓
4. ModelResponse([TextPart("You rolled a 4!")])
```

### Multiple Tool Calls in Parallel

When a model calls multiple tools **in the same response**, they're all part of ONE ModelResponse:

```
agent.run("Get info about Alice, Bob, Charlie, and Daisy")
    ↓
1. ModelRequest([UserPromptPart("Get info about...")])
    ↓
2. ModelResponse([
     TextPart("I'll retrieve information..."),
     ToolCallPart("retrieve_entity_info", args={"name": "Alice"}),
     ToolCallPart("retrieve_entity_info", args={"name": "Bob"}),
     ToolCallPart("retrieve_entity_info", args={"name": "Charlie"}),
     ToolCallPart("retrieve_entity_info", args={"name": "Daisy"})
   ])
    ↓
3. ModelRequest([
     ToolReturnPart("retrieve_entity_info", content="alice is bob's wife"),
     ToolReturnPart("retrieve_entity_info", content="bob is alice's husband"),
     ToolReturnPart("retrieve_entity_info", content="charlie is alice's son"),
     ToolReturnPart("retrieve_entity_info", content="daisy is bob's daughter")
   ])
    ↓
4. ModelResponse([TextPart("Daisy is the youngest...")])
```

## Key Insights

### 1. **ModelResponse Can Have Multiple Parts**

A single ModelResponse can contain:
- Multiple TextParts (concatenated together)
- Multiple ToolCallParts (executed in parallel)
- ThinkingParts (for reasoning)
- FileParts (for generated images/files)
- Mix of the above

### 2. **Tool Calls Always Create New Request Cycles**

When a ModelResponse contains ToolCallParts:
1. Pydantic AI executes all the tool functions
2. Creates a new ModelRequest with all ToolReturnParts
3. Sends this back to the model
4. Gets a new ModelResponse

### 3. **The Graph Execution Pattern**

The agent uses a graph-based execution with three main nodes:

```python
UserPromptNode → ModelRequestNode → CallToolsNode
                      ↑                    ↓
                      └────────────────────┘
```

- **UserPromptNode**: Handles initial user input
- **ModelRequestNode**: Makes a request to the model
- **CallToolsNode**: Processes model response, executes tools if present
  - If tool calls exist → creates new ModelRequestNode with tool returns
  - If final result → ends execution

## Real Example from Tests

Here's actual test output showing the message flow:

```python
# From test_agent.py:5411 - test_continue_conversation_that_ended_in_output_tool_call

# First run
result = agent.run_sync('Roll me a dice.')
messages = [
    ModelRequest([UserPromptPart('Roll me a dice.')]),
    ModelResponse([ToolCallPart('roll_dice')]),
    ModelRequest([ToolReturnPart('roll_dice', content=4)]),
    ModelResponse([ToolCallPart('final_result', args={'dice_roll': 4})]),
    ModelRequest([ToolReturnPart('final_result', content='Final result processed.')])
]
```

## The "Text + Tool Call" Scenario

To directly answer your original question about:
> "text part + tool call" => get tool response => another text part, another tool call => get tool response => final textpart"

This would create the following structure:

```python
[
    # Initial request
    ModelRequest([UserPromptPart("Do something complex")]),

    # First response with text AND tool call
    ModelResponse([
        TextPart("Let me help with that..."),
        ToolCallPart("tool1", args={...})
    ]),

    # Tool return
    ModelRequest([ToolReturnPart("tool1", content="result1")]),

    # Second response with more text and another tool
    ModelResponse([
        TextPart("Based on that result..."),
        ToolCallPart("tool2", args={...})
    ]),

    # Second tool return
    ModelRequest([ToolReturnPart("tool2", content="result2")]),

    # Final response
    ModelResponse([TextPart("Here's your final answer...")])
]
```

## Why Not Combine TextParts?

You asked about "combining TextParts into a single AssistantMessageAction". Here's why this is complex:

### Within a Single ModelResponse

If a ModelResponse has multiple TextParts (rare but possible):
```python
ModelResponse([
    TextPart("Hello"),
    TextPart(" world")  # Adjacent text parts
])
```
These SHOULD be combined when converting to ThreadProtocol's AssistantMessageAction.

### Across Multiple ModelResponses

TextParts from DIFFERENT ModelResponses should NOT be combined:
```python
ModelResponse([TextPart("First response")])
ModelRequest([ToolReturnPart(...)])
ModelResponse([TextPart("Second response")])  # Different cycle!
```

These represent different conversation turns and must remain separate AssistantMessageActions.

## Implications for ThreadProtocol

Based on this understanding:

### 1. **Each ModelResponse → Separate Actions**

```typescript
// ModelResponse with text + tool calls becomes:
[
  { action_type: "assistant_message", content: "Let me help..." },
  { action_type: "tool_call", tool_name: "tool1", ... }
]

// NOT combined into one action
```

### 2. **Tool Returns Create Clear Boundaries**

Tool returns always create a new "turn" in the conversation:
- Everything before the tool return is one logical group
- Everything after is a new logical group

### 3. **Preserve Request/Response Pairing**

The ThreadProtocol should preserve the request/response pairing implicit in the ModelMessage flow, even though it uses a flat action list.

## Code References

Key files that implement this flow:
- `pydantic_ai/_agent_graph.py:506-710` - CallToolsNode implementation
- `pydantic_ai/_agent_graph.py:641-673` - `_handle_tool_calls` method
- `pydantic_ai/_agent_graph.py:354-502` - ModelRequestNode implementation

The critical line that creates a new request after tool calls:
```python
# _agent_graph.py:670-672
self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
    _messages.ModelRequest(parts=output_parts, instructions=instructions)
)
```

## Summary

1. **Tool calls ALWAYS create new request/response cycles**
2. **Multiple tool calls in one response are executed in parallel**
3. **Each ModelResponse becomes its own set of ThreadProtocol actions**
4. **Don't combine TextParts across different ModelResponses**
5. **The flow is deterministic and follows a clear graph pattern**

This structure ensures proper conversation flow, maintains tool call boundaries, and preserves the logical grouping of related actions.