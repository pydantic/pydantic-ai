# ModelMessage Flow in Pydantic AI: Understanding Tool Calls

## Quick Answer

**Does a single model output "text + tool_call" create one ModelResponse with both parts?**
- YES - A `ModelResponse` can contain multiple parts (text, tool calls, etc.) in a single response

**When tool returns, does that create a new ModelRequest with the tool result?**
- YES - Each tool execution creates a new `ModelRequest` with `ToolReturnPart` parts

**Does the model's next output create a new ModelResponse?**
- YES - Each model invocation creates a new `ModelResponse`

**So everything is NOT in one giant ModelResponse?**
- CORRECT - There's a cycle: each agent.run() involves multiple ModelRequest/ModelResponse pairs

---

## The Three-Node Graph Architecture

Pydantic AI uses a state machine with three nodes that cycle:

```
UserPromptNode 
    ↓
ModelRequestNode 
    ↓
CallToolsNode 
    ↓
(loops back to ModelRequestNode if tools need execution)
```

Located in: `/pydantic_ai_slim/pydantic_ai/_agent_graph.py`

---

## Detailed Message Flow Example

### Scenario: Agent with multiple tool calls

```python
agent = Agent('gpt-4o')

@agent.tool
def get_price(fruit: str) -> float:
    return 10.0

@agent.tool  
def get_quantity(fruit: str) -> int:
    return 5

result = agent.run_sync('What is the price and quantity of apples?')
```

### Message History (what you see in result.all_messages()):

```
[
    # Step 1: User sends prompt
    ModelRequest(
        parts=[
            UserPromptPart('What is the price and quantity of apples?')
        ]
    ),
    
    # Step 2: Model responds with TWO tool calls in ONE response
    ModelResponse(
        parts=[
            ToolCallPart(tool_name='get_price', args={'fruit': 'apple'}, tool_call_id='call_1'),
            ToolCallPart(tool_name='get_quantity', args={'fruit': 'apple'}, tool_call_id='call_2'),
        ]
    ),
    
    # Step 3: Tool results sent back as a NEW request
    ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name='get_price',
                content=10.0,
                tool_call_id='call_1'
            ),
            ToolReturnPart(
                tool_name='get_quantity',
                content=5,
                tool_call_id='call_2'
            ),
        ]
    ),
    
    # Step 4: Model processes results and returns final answer
    ModelResponse(
        parts=[
            TextPart('The price is $10.00 and quantity is 5.')
        ]
    ),
]
```

---

## Key Data Structures

### ModelMessage (Union type)
```python
ModelMessage = ModelRequest | ModelResponse
```

### ModelRequest (sent TO model)
```python
@dataclass
class ModelRequest:
    parts: Sequence[ModelRequestPart]
    instructions: str | None = None
    kind: Literal['request'] = 'request'

# ModelRequestPart can be:
#  - SystemPromptPart (instructions from developer)
#  - UserPromptPart (user input)
#  - ToolReturnPart (results from executed tools)
#  - RetryPromptPart (asking model to retry)
```

### ModelResponse (received FROM model)
```python
@dataclass
class ModelResponse:
    parts: Sequence[ModelResponsePart]
    usage: RequestUsage
    model_name: str | None
    timestamp: datetime
    kind: Literal['response'] = 'response'

# ModelResponsePart can be:
#  - TextPart (plain text)
#  - ToolCallPart (function to call)
#  - ThinkingPart (model's reasoning)
#  - FilePart (image/file output)
#  - BuiltinToolCallPart (built-in tool)
```

### ToolCallPart (in ModelResponse)
```python
@dataclass
class ToolCallPart:
    tool_name: str
    args: str | dict[str, Any] | None
    tool_call_id: str
    part_kind: Literal['tool-call'] = 'tool-call'
```

### ToolReturnPart (in ModelRequest)
```python
@dataclass  
class ToolReturnPart:
    tool_name: str
    content: Any  # The return value from the tool
    tool_call_id: str
    metadata: Any | None = None
    part_kind: Literal['tool-return'] = 'tool-return'
```

---

## The Three Nodes Explained

### 1. UserPromptNode
**Purpose**: Build the initial request with user input

**What it does**:
- Takes the user's prompt passed to `agent.run(prompt)`
- Adds system prompts from `@agent.system_prompt()` 
- Creates the first `ModelRequest` with `UserPromptPart`
- Passes control to `ModelRequestNode`

**Code location**: Lines 158-346 in `_agent_graph.py`

```python
class UserPromptNode(AgentNode[DepsT, NodeRunEndT]):
    user_prompt: str | Sequence[UserContent] | None
    
    async def run(self, ctx) -> ModelRequestNode | CallToolsNode:
        # Build initial ModelRequest with user prompt
        # Move to ModelRequestNode
```

---

### 2. ModelRequestNode
**Purpose**: Send request to model and get response

**What it does**:
- Appends the `ModelRequest` to message history
- Calls the model API
- Gets back a `ModelResponse` 
- Appends the response to message history
- Passes control to `CallToolsNode`

**Code location**: Lines 377-502 in `_agent_graph.py`

```python
class ModelRequestNode(AgentNode[DepsT, NodeRunEndT]):
    request: ModelRequest
    
    async def run(self, ctx) -> CallToolsNode:
        # Append request to history
        ctx.state.message_history.append(self.request)
        
        # Call model
        response = await ctx.deps.model.request(message_history, ...)
        
        # Append response to history
        ctx.state.message_history.append(response)
        
        # Move to CallToolsNode
        return CallToolsNode(response)
```

---

### 3. CallToolsNode
**Purpose**: Process model response and decide what to do next

**What it does**:
1. If response contains tool calls:
   - Extract tool calls
   - Execute tools in parallel
   - Create tool result parts
   - Build new `ModelRequest` with `ToolReturnPart`s
   - Return to `ModelRequestNode` with new request

2. If response contains text:
   - Extract text
   - Validate against output schema
   - Return final result (end of run)

**Code location**: Lines 505-710 in `_agent_graph.py`

```python
class CallToolsNode(AgentNode[DepsT, NodeRunEndT]):
    model_response: ModelResponse
    
    async def run(self, ctx) -> ModelRequestNode | End:
        # Process the response
        if model_response has tool calls:
            # Execute tools
            results = await execute_tools(tool_calls)
            
            # Create new request with results
            next_request = ModelRequest(
                parts=[ToolReturnPart(...) for each tool result]
            )
            
            # Loop back
            return ModelRequestNode(next_request)
        
        elif model_response has text:
            # Validate output schema
            # Return End (final result)
            return End(final_result)
```

---

## Real Test Example: Multiple Tool Calls

From `/tests/test_tools.py::test_parallel_tool_return_with_deferred`:

```python
def llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        # First call: model returns 7 tool calls at once
        return ModelResponse(
            parts=[
                ToolCallPart('get_price', {'fruit': 'apple'}, tool_call_id='get_price_apple'),
                ToolCallPart('get_price', {'fruit': 'banana'}, tool_call_id='get_price_banana'),
                ToolCallPart('get_price', {'fruit': 'pear'}, tool_call_id='get_price_pear'),
                ToolCallPart('get_price', {'fruit': 'grape'}, tool_call_id='get_price_grape'),
                ToolCallPart('buy', {'fruit': 'apple'}, tool_call_id='buy_apple'),
                ToolCallPart('buy', {'fruit': 'banana'}, tool_call_id='buy_banana'),
                ToolCallPart('buy', {'fruit': 'pear'}, tool_call_id='buy_pear'),
            ]
        )
    else:
        # Second call: model sees results and returns final text
        return ModelResponse(parts=[TextPart('Done!')])

result = agent.run_sync('...')
messages = result.all_messages()
```

**Message History**:

```
[0] ModelRequest - User asks question
    parts: [UserPromptPart(...)]

[1] ModelResponse - Model returns 7 tool calls (ONE response)
    parts: [
        ToolCallPart(get_price, apple),
        ToolCallPart(get_price, banana),
        ToolCallPart(get_price, pear),
        ToolCallPart(get_price, grape),
        ToolCallPart(buy, apple),
        ToolCallPart(buy, banana),
        ToolCallPart(buy, pear),
    ]

[2] ModelRequest - ALL tool results sent back (ONE request)
    parts: [
        ToolReturnPart(get_price_apple, 10.0),
        RetryPromptPart(get_price_banana, "Unknown"),     # tool failed
        ToolReturnPart(get_price_pear, 10.0),
        RetryPromptPart(get_price_grape, "Unknown"),      # tool failed
        UserPromptPart("The price of apple is 10.0."),    # from tool's content
        UserPromptPart("The price of pear is 10.0."),     # from tool's content
    ]

[3] ModelResponse - Model processes results and returns final answer
    parts: [TextPart("Done!")]
]
```

---

## Important Concepts

### Tool Execution is Parallel
All tool calls in a single `ModelResponse` are executed **in parallel** (unless configured otherwise).

```python
# In _agent_graph.py, lines 950-966
async def _call_tools(...):
    for call in tool_calls:
        yield FunctionToolCallEvent(call)  # Emit event for each
    
    # Run all tools concurrently
    tasks = [
        asyncio.create_task(_call_tool(tool_manager, call, ...))
        for call in tool_calls
    ]
    
    pending = tasks
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        # Handle completions...
```

### Tool Return Parts Include Content
When a tool returns via `ToolReturn(return_value=..., content=...)`:
- `return_value` goes in the `ToolReturnPart.content`
- `content` becomes additional `UserPromptPart`(s) in the request

```python
# Example from test:
get_price() returns ToolReturn(
    return_value=10.0,
    content='The price of apple is 10.0.'  # This becomes UserPromptPart!
)

# Results in ModelRequest parts:
ToolReturnPart(..., content=10.0),
UserPromptPart(content='The price of apple is 10.0.'),
```

### State Machine Cycles
A single `agent.run()` call can result in many cycles:

```
UserPromptNode
    → ModelRequestNode  (request 1)
    → CallToolsNode     (response 1 with tool calls)
    → ModelRequestNode  (request 2 with tool results)
    → CallToolsNode     (response 2 with tool calls or final text)
    → ModelRequestNode  (request 3 if needed)
    → CallToolsNode     (response 3)
    → ... (until no more tool calls or final output)
    → End
```

Each cycle adds a ModelRequest and ModelResponse to the message history.

---

## State Management

### GraphAgentState
Maintains state across the entire run:

```python
@dataclass
class GraphAgentState:
    message_history: list[ModelMessage]  # All messages exchanged
    usage: RunUsage                       # Token counts
    retries: int                          # Validation retry counter
    run_step: int                         # Current step number
```

### Message History Cleaning
Before sending to model, consecutive `ModelRequest`s are merged:

```python
def _clean_message_history(messages):
    # Merge consecutive ModelRequest parts
    # This simplifies what the model sees while preserving history
```

---

## Streaming Support

The agent supports streaming via `ModelRequestNode.stream()`:

```python
async with agent.run_stream(prompt) as stream:
    async for event in stream:
        if isinstance(event, AgentStreamEvent):
            # PartStartEvent - new part started
            # PartDeltaEvent - part updated with delta
            # FunctionToolCallEvent - tool call started
            # FunctionToolResultEvent - tool result received
            # FinalResultEvent - final output matches schema
```

Each stream event provides real-time updates as the model generates content.

---

## Common Patterns

### Pattern 1: Simple Tool Call
```
User: "What's the weather?"

Message History:
1. ModelRequest: [UserPromptPart("What's the weather?")]
2. ModelResponse: [ToolCallPart("get_weather", ...)]
3. ModelRequest: [ToolReturnPart("get_weather", "Sunny")]
4. ModelResponse: [TextPart("The weather is sunny")]
```

### Pattern 2: Multiple Parallel Tool Calls
```
User: "What's the weather and time?"

Message History:
1. ModelRequest: [UserPromptPart(...)]
2. ModelResponse: [
     ToolCallPart("get_weather", ...),
     ToolCallPart("get_time", ...)
   ]  # Both in ONE response
3. ModelRequest: [
     ToolReturnPart("get_weather", "Sunny"),
     ToolReturnPart("get_time", "3 PM")
   ]  # Both in ONE request
4. ModelResponse: [TextPart("...")]
```

### Pattern 3: Tool Validates, Retries
```
User: "Calculate 2 + 2"

Message History:
1. ModelRequest: [UserPromptPart(...)]
2. ModelResponse: [ToolCallPart("calculate", {"expr": "invalid"})]
3. ModelRequest: [RetryPromptPart("Invalid expression", ...)]
4. ModelResponse: [ToolCallPart("calculate", {"expr": "2+2"})]
5. ModelRequest: [ToolReturnPart("calculate", 4)]
6. ModelResponse: [TextPart("The answer is 4")]
```

### Pattern 4: Deferred Tool Calls
```
User: "Buy me an apple"

Message History:
1. ModelRequest: [UserPromptPart(...)]
2. ModelResponse: [ToolCallPart("buy", ...)]
3. ModelRequest: []  # Tool raises CallDeferred
4. Returns DeferredToolRequests (output type) to user
   User approves: buy_apple = True
5. ModelRequest: [ToolReturnPart("buy", ...)]
6. ModelResponse: [TextPart("Order placed")]
```

---

## Key Takeaways

1. **ModelResponse = Single API call from model**
   - Can contain multiple tool calls
   - Can contain text + tool calls
   - Creates ONE entry in message history

2. **ModelRequest = Response to model**
   - Contains tool results
   - Sent as new request to model
   - Creates ONE entry in message history

3. **Tool execution is PARALLEL**
   - All tool calls in one response run concurrently
   - Results collected and sent back in one request

4. **Multiple cycles possible**
   - If model returns tool calls, new cycle begins
   - Continues until final output or no more tools

5. **Message history is linear**
   - Alternates: Request, Response, Request, Response...
   - Preserves full conversation for model context
   - Can be inspected via `result.all_messages()`

---

## Code References

All code is in `/pydantic_ai_slim/pydantic_ai/`:

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| UserPromptNode | _agent_graph.py | 158-346 | Build initial request |
| ModelRequestNode | _agent_graph.py | 377-502 | Send to model |
| CallToolsNode | _agent_graph.py | 505-710 | Process response |
| process_tool_calls | _agent_graph.py | 732-882 | Tool execution logic |
| _call_tools | _agent_graph.py | 884-974 | Parallel tool runner |
| ModelMessage types | messages.py | 1-1350+ | All message definitions |
| ModelRequest | messages.py | 898-919 | Request structure |
| ModelResponse | messages.py | 1096-1333 | Response structure |
| ToolCallPart | messages.py | 1063-1071 | Tool call in response |
| ToolReturnPart | messages.py | 783-790 | Tool result in request |

