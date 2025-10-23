# ModelMessage Flow in Pydantic AI - Complete Guide

## Executive Summary

In Pydantic AI, the agent execution follows a clear state machine pattern through three graph nodes:
1. **UserPromptNode** - Prepares the initial request
2. **ModelRequestNode** - Sends request to the model, receives ModelResponse
3. **CallToolsNode** - Processes tool calls, generates new ModelRequests with tool results

The key insight: **Each agent.run() creates multiple request/response cycles**, and **tool calls within a single ModelResponse create a new ModelRequest with tool results**.

---

## Message Types: The Core Building Blocks

### ModelRequest
A message sent FROM Pydantic AI TO the model. Contains:
- `parts`: List of `ModelRequestPart` objects
- `instructions`: Optional instructions for the model

**ModelRequestPart types:**
- `SystemPromptPart` - System prompt from developer
- `UserPromptPart` - User input/prompt
- `ToolReturnPart` - Result from a tool execution
- `RetryPromptPart` - Asking the model to retry (validation errors, etc.)

### ModelResponse
A message FROM the model back to Pydantic AI. Contains:
- `parts`: List of `ModelResponsePart` objects
- `usage`: Token usage info
- `model_name`, `timestamp`, `provider_name`: Metadata

**ModelResponsePart types:**
- `TextPart` - Plain text response
- `ToolCallPart` - A function/tool call to execute
- `ThinkingPart` - Model's thinking (when supported)
- `FilePart` - File/image response

**Key properties:**
- `response.tool_calls` - List of all `ToolCallPart` objects in the response
- `response.text` - Concatenated text from all `TextPart` objects

---

## The Agent Execution Graph

```
UserPromptNode
    ↓
ModelRequestNode ← Build ModelRequest → Send to Model → Get ModelResponse
    ↓
CallToolsNode ← Process ModelResponse
    ├─ If tool calls exist: Execute tools, create ModelRequest with ToolReturnPart
    ├─ If final output exists: End run
    └─ Loop back to ModelRequestNode if more tool calls to process
```

---

## Scenario 1: Simple Text Response (No Tool Calls)

```python
agent = Agent('openai:gpt-4o')

result = agent.run_sync('What is 2+2?')
```

### Message Flow:

```
MESSAGES HISTORY:
[
    ModelRequest(
        parts=[
            SystemPromptPart(...),
            UserPromptPart(content='What is 2+2?')
        ]
    ),
    ModelResponse(
        parts=[
            TextPart(content='2+2=4')
        ]
    )
]
```

**Key points:**
- ONE ModelRequest sent to model
- ONE ModelResponse received from model
- Response contains only TextPart, no tool calls
- Execution ends

---

## Scenario 2: Single Tool Call

```python
agent = Agent('openai:gpt-4o')

@agent.tool
def calculate(expression: str) -> str:
    """Calculate an expression"""
    return str(eval(expression))

result = agent.run_sync('What is 2+2?')
```

### Message Flow:

```
CYCLE 1: Send initial request
├─ ModelRequest(
│   parts=[UserPromptPart('What is 2+2?')]
│ )
└─ ModelResponse(
    parts=[
        ToolCallPart(
            tool_name='calculate',
            args={'expression': '2+2'},
            tool_call_id='call_123'
        )
    ]
  )

CYCLE 2: Send tool result, get final response
├─ ModelRequest(
│   parts=[
│       ToolReturnPart(
│           tool_name='calculate',
│           content='4',
│           tool_call_id='call_123'
│       )
│   ]
│ )
└─ ModelResponse(
    parts=[
        TextPart(content='The answer is 4')
    ]
  )

FINAL HISTORY:
[
    ModelRequest(...),      # Cycle 1 request
    ModelResponse(...),     # Cycle 1 response (has tool call)
    ModelRequest(...),      # Cycle 2 request (has tool result)
    ModelResponse(...)      # Cycle 2 response (final text)
]
```

**Key points:**
- Tool call in ModelResponse.parts creates new ModelRequest cycle
- ModelResponse contains ToolCallPart, not a final output
- ToolReturnPart wraps the tool result in ModelRequest
- Next ModelResponse contains final text output

---

## Scenario 3: Multiple Tool Calls (Parallel)

```python
agent = Agent('openai:gpt-4o')

@agent.tool
def get_price(fruit: str) -> float:
    """Get price of fruit"""
    return {'apple': 1.0, 'banana': 0.5}[fruit]

@agent.tool
def get_availability(fruit: str) -> bool:
    """Check if fruit is available"""
    return fruit != 'grape'

result = agent.run_sync('What are the prices and availability of apples and bananas?')
```

### Message Flow:

```
CYCLE 1: Model outputs multiple tool calls in one response
┌─ ModelRequest(
│   parts=[UserPromptPart('What are prices...?')]
│ )
│
└─ ModelResponse(
    parts=[
        ToolCallPart(tool_name='get_price', args={'fruit': 'apple'}, 
                     tool_call_id='call_1'),
        ToolCallPart(tool_name='get_availability', args={'fruit': 'apple'},
                     tool_call_id='call_2'),
        ToolCallPart(tool_name='get_price', args={'fruit': 'banana'},
                     tool_call_id='call_3'),
        ToolCallPart(tool_name='get_availability', args={'fruit': 'banana'},
                     tool_call_id='call_4'),
    ]
  )

IMPORTANT: All 4 tool calls are in ONE ModelResponse!
They are executed in PARALLEL (not sequentially).

CYCLE 2: ALL tool results collected in ONE ModelRequest
┌─ ModelRequest(
│   parts=[
│       ToolReturnPart(tool_name='get_price', content=1.0, 
│                      tool_call_id='call_1'),
│       ToolReturnPart(tool_name='get_availability', content=True, 
│                      tool_call_id='call_2'),
│       ToolReturnPart(tool_name='get_price', content=0.5, 
│                      tool_call_id='call_3'),
│       ToolReturnPart(tool_name='get_availability', content=True, 
│                      tool_call_id='call_4'),
│   ]
│ )
│
└─ ModelResponse(
    parts=[
        TextPart(content='Apple: $1.00 (available), Banana: $0.50 (available)')
    ]
  )

FINAL HISTORY:
[
    ModelRequest(...),      # Initial request
    ModelResponse(...),     # 4 tool calls in ONE response
    ModelRequest(...),      # 4 tool results in ONE request
    ModelResponse(...)      # Final text response
]
```

**Key points:**
- Multiple ToolCallParts can be in ONE ModelResponse
- Pydantic AI executes them in parallel (asyncio.wait)
- All results are collected in ONE ModelRequest
- Creates only ONE additional request/response cycle

---

## Scenario 4: Tool Call with Multiple Results and Errors (Complex)

Real-world example from Pydantic AI test suite:

```python
# Tool that returns data
@agent.tool_plain
def get_price(fruit: str) -> ToolReturn:
    if fruit in ['apple', 'pear']:
        return ToolReturn(
            return_value=10.0,
            content=f'The price of {fruit} is 10.0.',
            metadata={'fruit': fruit, 'price': 10.0},
        )
    else:
        raise ModelRetry(f'Unknown fruit: {fruit}')

# Tool that defers execution
@agent.tool_plain
def buy(fruit: str):
    raise CallDeferred

agent = Agent(FunctionModel(llm_func), output_type=[str, DeferredToolRequests])
result = agent.run_sync('What do apple, banana, pear and grape cost? Also buy me a pear.')
```

### Message Flow:

```
CYCLE 1: Model outputs 7 tool calls
ModelResponse(
    parts=[
        ToolCallPart('get_price', {'fruit': 'apple'}, id='get_price_apple'),
        ToolCallPart('get_price', {'fruit': 'banana'}, id='get_price_banana'),
        ToolCallPart('get_price', {'fruit': 'pear'}, id='get_price_pear'),
        ToolCallPart('get_price', {'fruit': 'grape'}, id='get_price_grape'),
        ToolCallPart('buy', {'fruit': 'apple'}, id='buy_apple'),
        ToolCallPart('buy', {'fruit': 'banana'}, id='buy_banana'),
        ToolCallPart('buy', {'fruit': 'pear'}, id='buy_pear'),
    ]
)

CYCLE 2: Tool execution results mixed
ModelRequest(
    parts=[
        ToolReturnPart('get_price', 10.0, id='get_price_apple', 
                      metadata={'fruit': 'apple', 'price': 10.0}),
        RetryPromptPart('Unknown fruit: banana', tool_name='get_price',
                       id='get_price_banana'),  # Error from ModelRetry
        ToolReturnPart('get_price', 10.0, id='get_price_pear',
                      metadata={'fruit': 'pear', 'price': 10.0}),
        RetryPromptPart('Unknown fruit: grape', tool_name='get_price',
                       id='get_price_grape'),   # Error from ModelRetry
        UserPromptPart('The price of apple is 10.0.'),  # From tool result.content
        UserPromptPart('The price of pear is 10.0.'),   # From tool result.content
        # buy_* tool calls are deferred, not returned here
    ]
)

RESULT: DeferredToolRequests(
    calls=[
        ToolCallPart('buy', {'fruit': 'apple'}, id='buy_apple'),
        ToolCallPart('buy', {'fruit': 'banana'}, id='buy_banana'),
        ToolCallPart('buy', {'fruit': 'pear'}, id='buy_pear'),
    ]
)
```

**Key points:**
- Tool execution can generate `RetryPromptPart` (validation/retry errors)
- Tool result `.content` field becomes `UserPromptPart` in next request
- Deferred tools don't appear in tool results - returned as `DeferredToolRequests`
- Multiple error types (success, retry, deferred) handled in ONE cycle

---

## Scenario 5: Text + Tool Call in Same Response

Some models (like Anthropic) may return text AND tool calls together:

```python
ModelResponse(
    parts=[
        TextPart(content='I will search for that information...'),
        ToolCallPart(tool_name='search', args={'query': 'topic'},
                    tool_call_id='call_123')
    ]
)
```

### Pydantic AI's Handling:

```python
# From _agent_graph.py, CallToolsNode._run_stream():
if tool_calls:
    # Tool calls are prioritized
    async for event in self._handle_tool_calls(ctx, tool_calls):
        yield event
    return  # Text is ignored in favor of tool calls
```

**Key points:**
- Tool calls are PRIORITIZED over text in same response
- Text before tool calls is essentially treated as "thinking"
- If tool calls exist, model won't immediately output the response

---

## How Tool Calls Create New Cycles: The Code Path

### Step 1: ModelResponse with tool calls received (in CallToolsNode._run_stream)

```python
# From _agent_graph.py line 604-607
if tool_calls:
    async for event in self._handle_tool_calls(ctx, tool_calls):
        yield event
    return  # Don't process text if tool calls exist
```

### Step 2: Tools are executed (in _call_tools)

```python
# From _agent_graph.py line 836-847
async for event in _call_tools(
    tool_manager=tool_manager,
    tool_calls=tool_calls,
    tool_call_results=calls_to_run_results,
    tracer=ctx.deps.tracer,
    usage=ctx.state.usage,
    usage_limits=ctx.deps.usage_limits,
    output_parts=output_parts,  # Collects ToolReturnPart objects
    output_deferred_calls=deferred_calls,
):
    yield event
```

### Step 3: New ModelRequest is created with tool results (in CallToolsNode._handle_tool_calls)

```python
# From _agent_graph.py line 668-672
if output_final_result:
    final_result = output_final_result[0]
    self._next_node = self._handle_final_result(ctx, final_result, output_parts)
else:
    instructions = await ctx.deps.get_instructions(run_context)
    self._next_node = ModelRequestNode[DepsT, NodeRunEndT](
        _messages.ModelRequest(parts=output_parts, instructions=instructions)
    )
```

### Step 4: Graph loops back to ModelRequestNode

The graph then calls `ModelRequestNode.run()` which:
1. Adds the new ModelRequest to message history
2. Sends it to the model
3. Returns CallToolsNode with the new ModelResponse

---

## Key Data Flow Patterns

### Pattern 1: ToolCallPart → ToolReturnPart Mapping

```
ModelResponse from model:
├─ ToolCallPart(tool_call_id='call_123', ...)

Tool execution result:
├─ Wraps result in ToolReturnPart(tool_call_id='call_123', ...)

Next ModelRequest:
└─ ToolReturnPart(tool_call_id='call_123', ...)  ← Uses same ID
```

This ID allows models to correlate which tool result corresponds to which tool call.

### Pattern 2: Error Handling in Tool Calls

```
Tool execution → Exception → Type checked:

1. ModelRetry exception
   └─ Creates RetryPromptPart in ModelRequest

2. CallDeferred exception
   └─ Tool call deferred, not included in results

3. UnexpectedModelBehavior exception
   └─ Increments retries, re-raises

4. Normal return value
   └─ Creates ToolReturnPart in ModelRequest
```

### Pattern 3: Tool Result Content Flow

```
Tool returns: ToolReturn(
    return_value=actual_data,
    content=user_friendly_text  ← This becomes UserPromptPart!
)

Result in next ModelRequest:
├─ ToolReturnPart(content=actual_data)
└─ UserPromptPart(content=user_friendly_text)
```

---

## Message History Structure: Complete Example

```python
# Actual structure from test_parallel_tool_return_with_deferred

messages = [
    # CYCLE 1: User prompt
    ModelRequest(
        parts=[
            UserPromptPart(
                content='What do an apple, a banana, a pear and a grape cost?'
            )
        ]
    ),
    
    # Model responds with 7 tool calls
    ModelResponse(
        parts=[
            ToolCallPart(tool_name='get_price', args={'fruit': 'apple'}, 
                        tool_call_id='get_price_apple'),
            ToolCallPart(tool_name='get_price', args={'fruit': 'banana'}, 
                        tool_call_id='get_price_banana'),
            # ... 5 more tool calls
        ]
    ),
    
    # CYCLE 2: All tool results and errors in one request
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='get_price',
                content=10.0,
                tool_call_id='get_price_apple',
                metadata={'fruit': 'apple', 'price': 10.0}
            ),
            RetryPromptPart(
                content='Unknown fruit: banana',
                tool_name='get_price',
                tool_call_id='get_price_banana'
            ),
            # ... more tool results
            UserPromptPart(content='The price of apple is 10.0.'),
            UserPromptPart(content='The price of pear is 10.0.'),
        ]
    ),
    
    # Final response
    ModelResponse(
        parts=[
            TextPart(content='Done!')
        ]
    ),
]
```

---

## Important Implementation Details

### 1. Message Merging (_clean_message_history)

Consecutive ModelRequests are merged:
```python
# Before cleaning:
[ModelRequest(parts=[...]), ModelRequest(parts=[...])]

# After cleaning:
[ModelRequest(parts=[..., ...])]  # Parts combined
```

This happens for internal efficiency but doesn't change user-facing behavior.

### 2. Tool Result Ordering

Tool results are appended in a deterministic order:
```python
# From _call_tools line 967-970
output_parts.extend([tool_parts_by_index[k] for k in sorted(tool_parts_by_index)])
output_parts.extend([user_parts_by_index[k] for k in sorted(user_parts_by_index)])
```

This ensures consistent ordering regardless of parallel execution.

### 3. Output Tool Calls (end_strategy)

Output tools can trigger final results:
```python
# 'early' strategy: stop after first output tool succeeds
# 'exhaustive' strategy: process all tool calls even after output tool succeeds

if final_result and ctx.deps.end_strategy == 'early':
    # Mark other tools as skipped
```

### 4. Deferred Tool Handling

Tools that raise `CallDeferred` are:
- NOT included in tool results
- Returned in `DeferredToolRequests` output type
- Returned to user for manual approval/handling

---

## Question Answers Recap

### Q1: If a model outputs "text + tool_call", does that create one ModelResponse with both parts?

**YES.** The ModelResponse can have multiple parts including both TextPart and ToolCallPart objects:
```python
ModelResponse(
    parts=[
        TextPart(content='I will help you...'),
        ToolCallPart(tool_name='search', ...)
    ]
)
```

However, if tool calls are present, the text is essentially ignored (not output to user).

### Q2: When the tool returns, does that create a new ModelRequest with the tool result?

**YES.** After executing tool calls from a ModelResponse, Pydantic AI creates a NEW ModelRequest containing ToolReturnPart objects wrapping the results.

### Q3: Does the model's next output (after seeing the tool result) create a new ModelResponse?

**YES.** The new ModelRequest is sent to the model, which generates a new ModelResponse. This creates a new request/response cycle.

### Q4: Or is everything within one giant ModelResponse?

**NO.** Multiple request/response cycles are created. Each tool call round-trip creates:
1. New ModelRequest (with ToolReturnPart)
2. New ModelResponse (from model processing results)

### Q5: Is a single agent.run() call just one request/response or multiple?

**MULTIPLE.** A typical agent.run() with tool calls involves:
1. Initial ModelRequest
2. ModelResponse (with tool calls)
3. New ModelRequest (with tool results)
4. Final ModelResponse
= 2 request/response cycles (minimum)

---

## Testing and Accessing Messages

### Capture all messages:
```python
result = agent.run_sync('prompt')
all_messages = result.all_messages()
```

### Capture only new messages (after history):
```python
result = agent.run_sync('prompt')
new_messages = result.new_messages()  # Only messages added by this run
```

### Capture during error:
```python
from pydantic_ai import capture_run_messages

with capture_run_messages() as messages:
    try:
        result = agent.run_sync('prompt')
    except Exception:
        print(messages)  # Messages up to the error
        raise
```

---

## References

- **Agent Graph Execution**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/_agent_graph.py`
- **Message Types**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/messages.py`
- **Test Examples**: `/Users/ericksonc/appdev/pydantic-ai/tests/test_tools.py` (test_parallel_tool_return_with_deferred)
- **Test Examples**: `/Users/ericksonc/appdev/pydantic-ai/tests/test_agent.py` (test_run_with_history_ending_on_model_response_with_tool_calls_and_no_user_prompt)

