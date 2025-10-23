# Comprehensive Guide: Runs, API Calls, Messages, and Parts in Pydantic AI

## Overview of Granularity Hierarchy

Pydantic AI operates with a clear hierarchy of granularity, from largest to smallest:

1. **Run** (highest level) - A complete agent execution from start to finish
2. **Run Step** - Each LLM API call within a run
3. **Message** - Each request or response in the conversation
4. **Part** - Individual components within a message

## 1. Runs: The Top Level

### What is a Run?
- **Definition**: A single call to `agent.run()`, `agent.run_sync()`, or `agent.run_stream()`
- **Scope**: Encompasses the entire conversation from initial prompt to final result
- **Cardinality**: One run = One complete agent execution
- **State**: Maintained in `GraphAgentState` throughout the run

### Key Characteristics:
```python
# One run encompasses everything from start to finish
result = agent.run("What is 2+2?")  # This is ONE run

# Even if internally it makes multiple API calls due to tool usage:
result = agent.run("Search for X, then calculate Y")  # Still ONE run
```

### Run State Management:
Location: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:73-80`
```python
@dataclasses.dataclass
class GraphAgentState:
    message_history: list[_messages.ModelMessage]  # All messages in the run
    usage: _usage.Usage                            # Total usage for the run
    retries: int                                    # Retry count within the run
    run_step: int                                   # Current step number (API call count)
```

## 2. Run Steps: The API Call Level

### What is a Run Step?
- **Definition**: Each individual LLM API call within a run
- **Tracked by**: The `run_step` counter in `GraphAgentState`
- **Incremented**: Every time `ModelRequestNode` makes an API call
- **Cardinality**: One run can have MANY run steps

### When Run Steps Increment:
Location: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:373-374`
```python
# In ModelRequestNode._prepare_request():
ctx.state.run_step += 1  # Increments before each API call
```

### Examples of Multiple Run Steps in One Run:

#### Example 1: Tool Calls Requiring Multiple API Calls
```python
# User calls agent.run() once
result = agent.run("Calculate the sum of 5+3 and then multiply by 2")

# Internally, this might produce:
# Run Step 1: API call -> Model returns tool call for addition
# Run Step 2: API call -> Model returns tool call for multiplication
# Run Step 3: API call -> Model returns final answer
# Total: 1 Run, 3 Run Steps
```

#### Example 2: Validation Retry
```python
result = agent.run("Generate a valid email")

# If first response fails validation:
# Run Step 1: API call -> Invalid response
# Run Step 2: API call with retry prompt -> Valid response
# Total: 1 Run, 2 Run Steps
```

## 3. Messages: The Conversation Level

### What is a Message?
- **Definition**: A complete request to or response from the model
- **Types**: `ModelRequest` (to model) or `ModelResponse` (from model)
- **Cardinality**: Each run step produces exactly ONE request-response pair
- **Storage**: Accumulated in `message_history`

### Message-to-API Call Relationship:
**IMPORTANT: One API call = One ModelRequest + One ModelResponse**

```python
# Each API call follows this pattern:
1. Create ModelRequest
2. Send to LLM (API CALL HAPPENS HERE)
3. Receive ModelResponse
4. Append both to message_history
```

### Message Flow Example:
```python
# Single run with tool usage:
message_history = [
    ModelRequest([SystemPromptPart, UserPromptPart]),  # Step 1 request
    ModelResponse([ToolCallPart]),                     # Step 1 response
    ModelRequest([ToolReturnPart]),                    # Step 2 request
    ModelResponse([TextPart]),                         # Step 2 response
]
# Total: 2 API calls, 4 messages
```

## 4. Parts: The Component Level

### What is a Part?
- **Definition**: Individual components within a message
- **Request Parts**: SystemPromptPart, UserPromptPart, ToolReturnPart, RetryPromptPart
- **Response Parts**: TextPart, ToolCallPart, ThinkingPart
- **Cardinality**: One message can have MANY parts

### Part-to-Message Relationship:
**IMPORTANT: Parts are ALWAYS contained within messages**

```python
# A single ModelRequest can have multiple parts:
ModelRequest(parts=[
    SystemPromptPart("You are helpful"),
    UserPromptPart("What is 2+2?")
])

# A single ModelResponse can have multiple parts:
ModelResponse(parts=[
    TextPart("I'll calculate that"),
    ToolCallPart(tool_name="calculator", args={"a": 2, "b": 2})
])
```

## 5. Tool Calls: The Complexity Factor

### Parallel vs Sequential Tool Execution

Tool calls introduce complexity because they can be executed in different patterns, but they follow strict rules:

#### Pattern 1: Multiple Tools in One Response (Parallel)
```python
# Model returns multiple tool calls in ONE response
ModelResponse(parts=[
    ToolCallPart(tool_name="search", ...),
    ToolCallPart(tool_name="calculate", ...),
    ToolCallPart(tool_name="format", ...),
])

# These execute in PARALLEL (asyncio.wait)
# Result: ONE API call produced multiple tool calls
# Next: ONE ModelRequest with all ToolReturnParts
```

**Execution Flow:**
1. **Run Step 1**: API call → Model returns 3 tool calls
2. Tools execute in parallel (NOT additional API calls)
3. **Run Step 2**: API call with all tool returns → Final response

**Total: 2 API calls for 3 tool executions**

#### Pattern 2: Sequential Tool Calls
```python
# Model returns one tool call at a time
# Step 1:
ModelResponse(parts=[ToolCallPart(tool_name="search", ...)])
# Step 2:
ModelResponse(parts=[ToolCallPart(tool_name="calculate", ...)])
# Step 3:
ModelResponse(parts=[ToolCallPart(tool_name="format", ...)])
```

**Execution Flow:**
1. **Run Step 1**: API call → Model returns 1 tool call
2. Tool executes
3. **Run Step 2**: API call with tool return → Model returns another tool call
4. Tool executes
5. **Run Step 3**: API call with tool return → Model returns another tool call
6. Tool executes
7. **Run Step 4**: API call with tool return → Final response

**Total: 4 API calls for 3 tool executions**

### Tool Execution Code:
Location: `pydantic_ai_slim/pydantic_ai/_agent_graph.py:664-693`
```python
# All function tools in a single response execute in parallel:
if calls_to_run:
    tasks = [
        asyncio.create_task(_call_function_tool(tool_manager, call), name=call.tool_name)
        for call in calls_to_run
    ]

    pending = tasks
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        # Process completed tools as they finish
```

## 6. Cardinality Relationships

### Strict 1:1 Relationships:
- **1 API call = 1 ModelRequest + 1 ModelResponse** (always)
- **1 API call = 1 run_step increment** (always)

### Many-to-One Relationships:
- **Many run steps : 1 run** (multiple API calls per run)
- **Many messages : 1 run** (accumulated history)
- **Many parts : 1 message** (components within a message)
- **Many tool calls : 1 ModelResponse** (parallel execution)

### Variable Relationships:
- **Tool calls to API calls**: Depends on model behavior
  - Best case: N tools in 1 response = 2 API calls total
  - Worst case: N tools sequentially = N+1 API calls total

## 7. Practical Examples with Counts

### Example 1: Simple Question
```python
result = agent.run("What is the capital of France?")
```
- **Runs**: 1
- **Run Steps**: 1
- **Messages**: 2 (1 request, 1 response)
- **API Calls**: 1

### Example 2: Tool Usage (Parallel)
```python
result = agent.run("Search for weather in NYC and LA")
# Model returns both tool calls in one response
```
- **Runs**: 1
- **Run Steps**: 2
- **Messages**: 4 (2 requests, 2 responses)
- **API Calls**: 2
- **Tool Executions**: 2 (parallel)

### Example 3: Tool Usage (Sequential)
```python
result = agent.run("First search for X, then based on that search for Y")
# Model returns one tool call, then another based on result
```
- **Runs**: 1
- **Run Steps**: 3
- **Messages**: 6 (3 requests, 3 responses)
- **API Calls**: 3
- **Tool Executions**: 2 (sequential)

### Example 4: Validation Retry
```python
result = agent.run("Generate valid JSON")
# First attempt fails validation
```
- **Runs**: 1
- **Run Steps**: 2
- **Messages**: 4 (2 requests, 2 responses)
- **API Calls**: 2
- **Retries**: 1

## 8. Key Invariants

These relationships ALWAYS hold true:

1. **Every API call increments run_step by 1**
2. **Every API call adds exactly 2 messages to history** (request + response)
3. **Tool execution NEVER directly causes an API call** (only the model's response does)
4. **Parallel tool calls share the same run_step** (they're from one API response)
5. **Sequential tool calls have different run_steps** (each needs a new API call)

## 9. Disambiguation Summary

To directly answer the granularity questions:

### "What maps to a run?"
- **One `agent.run()` call = One run**, regardless of internal complexity
- A run encompasses the entire conversation from start to finish

### "What maps to an LLM API call?"
- **Each run_step = One API call**
- **Each ModelRequestNode execution = One API call**
- The `run_step` counter tracks exactly how many API calls have been made

### "What's in between?"
- **Messages**: The request/response pairs from each API call
- **Parts**: The components within messages (multiple parts per message)
- **Tool executions**: Can be parallel (same step) or sequential (different steps)

### "Which are 1:1 and which are many-to-one?"
- **1:1**: API call ↔ run_step, API call ↔ request/response pair
- **Many:1**: run_steps → run, messages → run, parts → message
- **Variable**: tool calls → API calls (depends on model behavior)

The key insight is that Pydantic AI clearly separates:
- **Logical operations** (runs, tool executions)
- **Network operations** (API calls, run_steps)
- **Data structures** (messages, parts)

This separation allows for efficient parallel tool execution while maintaining a clear, traceable conversation history.