# Tool Call Flow Diagrams

## High-Level Agent Run Cycle

```
agent.run(prompt)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ UserPromptNode                                              │
│ - Takes user prompt                                         │
│ - Adds system prompts                                       │
│ - Creates ModelRequest with UserPromptPart                  │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│ ModelRequestNode                                            │
│ - Appends request to message_history                        │
│ - Calls model API                                           │
│ - Receives ModelResponse                                    │
│ - Appends response to message_history                       │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
        ┌──────────────────────┐
        │ CallToolsNode        │
        │ Inspect response     │
        └──────┬───────────────┘
               ↓
      ┌────────────────────┐
      │ Response contains  │
      │ tool calls?        │
      └────┬───────────┬───┘
           │           │
           YES         NO
           ↓           ↓
    ┌──────────┐  ┌──────────────┐
    │ Execute  │  │ Extract text │
    │ tools    │  │ Validate     │
    │ parallel │  │ Return End   │
    └────┬─────┘  │ (final result)
         ↓        └──────────────┘
    Create new ModelRequest
    with ToolReturnPart(s)
         ↓
    Loop back to ModelRequestNode
```

---

## Message Sequence: Multiple Tool Calls

```
Agent State                     Message History
────────────────────────────────────────────────────────────────

User submits prompt
   ↓
[UserPromptNode]
   ├→ builds ModelRequest
   │  with UserPromptPart
   └→ passes to ModelRequestNode
          ↓
     [0] ModelRequest
         parts: [UserPromptPart("...")]
            ↓
        [ModelRequestNode]
        ├→ appends [0] to history
        ├→ calls model API
        ├→ receives ModelResponse
        │  parts: [
        │    ToolCallPart("get_price", ...),
        │    ToolCallPart("get_qty", ...)
        │  ]
        ├→ appends [1] to history
        └→ passes to CallToolsNode
                ↓
           [1] ModelResponse
               parts: [
                 ToolCallPart("get_price", id=1),
                 ToolCallPart("get_qty", id=2)
               ]
                ↓
           [CallToolsNode]
           ├→ extracts tool calls
           ├→ executes tools in parallel:
           │  - get_price() → 10.0
           │  - get_qty()   → 5
           ├→ creates ToolReturnPart(s)
           ├→ creates new ModelRequest [2]
           └→ loops back to ModelRequestNode
                    ↓
              [2] ModelRequest
                  parts: [
                    ToolReturnPart(
                      tool_name="get_price",
                      content=10.0,
                      tool_call_id=1
                    ),
                    ToolReturnPart(
                      tool_name="get_qty",
                      content=5,
                      tool_call_id=2
                    )
                  ]
                    ↓
              [ModelRequestNode] (second cycle)
              ├→ appends [2] to history
              ├→ calls model API
              ├→ receives ModelResponse
              │  parts: [TextPart("Price: $10, Qty: 5")]
              ├→ appends [3] to history
              └→ passes to CallToolsNode
                      ↓
                 [3] ModelResponse
                     parts: [
                       TextPart("Price: $10, Qty: 5")
                     ]
                      ↓
                 [CallToolsNode]
                 ├→ extracts text
                 ├→ validates against output schema
                 ├→ returns final result
                 └→ End

Final message_history = [0, 1, 2, 3]
```

---

## Detailed Tool Execution Flow

```
CallToolsNode receives ModelResponse with tool calls
    ↓
Extract ToolCallPart(s) from response.parts
    ↓
Group by tool kind:
├─ output tools (return final result)
├─ function tools (execute)
├─ external tools (deferred)
└─ unapproved tools (need approval)
    ↓
[process_tool_calls] in _agent_graph.py
├─ Handle output tools
├─ Handle function tools (if no final result yet)
└─ Handle deferred/unapproved tools
    ↓
[_call_tools] function
├─ For each tool call:
│  ├─ Create ToolCallPart event
│  └─ Create asyncio.Task
│
├─ Wait for all tasks (FIRST_COMPLETED)
│  ├─ On completion:
│  │  ├─ If success: create ToolReturnPart
│  │  ├─ If retry: create RetryPromptPart
│  │  └─ Emit FunctionToolResultEvent
│  │
│  └─ Continue until all done
│
└─ Append result parts in order:
   ├─ Tool return parts first
   └─ User prompt parts (if any)
    ↓
Create new ModelRequest with result parts
    ↓
Return to ModelRequestNode
```

---

## Tool Execution Parallelism

```
ModelResponse from Model
├─ ToolCallPart(tool_1, id=1)
├─ ToolCallPart(tool_2, id=2)
└─ ToolCallPart(tool_3, id=3)

    ↓ [_call_tools]

┌─────────────────────────────────────┐
│ Async Execution                     │
├─────────────────────────────────────┤
│                                     │
│  Task 1: tool_1()  ✓ returns "A"    │
│  Task 2: tool_2()  ✓ returns "B"    │
│  Task 3: tool_3()  ✓ returns "C"    │
│                                     │
│  All run CONCURRENTLY, not sequentially!
│                                     │
└─────────────────────────────────────┘

    ↓ Collect Results

ModelRequest
├─ ToolReturnPart(tool_1, content="A", id=1)
├─ ToolReturnPart(tool_2, content="B", id=2)
└─ ToolReturnPart(tool_3, content="C", id=3)

    ↓ Same order as original calls
```

---

## Message Structure: Simple Example

```
ModelResponse from model with tool calls:

ModelResponse(
    parts=[
        ToolCallPart(
            tool_name='calculate',
            args='{"expr": "2+2"}',
            tool_call_id='call_1'
        )
    ],
    usage=RequestUsage(input_tokens=50, output_tokens=10),
    model_name='gpt-4',
    timestamp=datetime.now()
)

    ↓ [CallToolsNode executes tool]

Tool execution:
- tool_name: 'calculate'
- args: {"expr": "2+2"}
- returns: 4

    ↓ [Create response]

ModelRequest(
    parts=[
        ToolReturnPart(
            tool_name='calculate',
            content=4,  # ← The return value
            tool_call_id='call_1'
        )
    ]
)
```

---

## ToolReturn with Content Example

```
Tool definition:
@agent.tool
def get_price(fruit: str) -> ToolReturn:
    price = 10.0
    return ToolReturn(
        return_value=price,
        content=f'The {fruit} costs ${price}'  # Extra info for model
    )

    ↓ [Tool executed with args: {"fruit": "apple"}]

    ↓ [Tool execution creates]

ModelRequest parts:
├─ ToolReturnPart(
│  tool_name='get_price',
│  content=10.0,  # ← return_value
│  tool_call_id='call_1',
│  metadata={...}
│ )
│
└─ UserPromptPart(
   content='The apple costs $10'  # ← content field
  )

    ↓ [Model sees both parts]

Model now knows:
- Tool returned 10.0 (technical result)
- User context: "The apple costs $10" (human-readable)
```

---

## State Machine Transitions

```
┌──────────────────┐
│ UserPromptNode   │
│                  │
│ input: prompt    │
│ output:          │
│ ModelRequestNode │
└────────┬─────────┘
         ↓
┌──────────────────────────────┐
│ ModelRequestNode             │
│                              │
│ input: ModelRequest          │
│ output:                      │
│ - CallToolsNode (response)   │
│                              │
│ side effects:                │
│ - append request to history  │
│ - call model API             │
│ - append response to history │
└────────┬─────────────────────┘
         ↓
      ┌──────────────────────────────────────┐
      │ CallToolsNode                        │
      │                                      │
      │ input: ModelResponse                 │
      │                                      │
      │ ┌─ Has tool calls?                  │
      │ │  YES:                             │
      │ │  - Execute tools                  │
      │ │  - Create ModelRequest            │
      │ │  - return ModelRequestNode ────┐  │
      │ │                                 │  │
      │ │  NO:                            │  │
      │ │  - Extract text                 │  │
      │ │  - Validate output              │  │
      │ │  - return End (final result) ─┐ │  │
      │ └                                │ │  │
      └────────────────┬─────────────────┼─┘
                       │                 │
                ┌──────▼──────┐    ┌─────▼────┐
                │ Loop back   │    │ End      │
                │ (if tools)  │    │ (return) │
                └──────▲──────┘    └──────────┘
                       │
                       └─ if has tool calls
```

---

## Message Flow Timeline

```
Time    Agent State          Message History          API Calls
────────────────────────────────────────────────────────────────

T0      user calls run()

T1      UserPromptNode       [0] ← ModelRequest        
        creates request      

T2      ModelRequestNode     [0] appended              call model
        sends to model       [1] ← ModelResponse       ← returns

T3      CallToolsNode        (response analyzed)
        finds tool calls     

T4                           tools execute parallel
                             Tool 1: 100ms ✓
                             Tool 2: 80ms ✓

T5      CallToolsNode        [2] ← ModelRequest
        creates request      (with ToolReturnPart)

T6      ModelRequestNode     [2] appended              call model
        sends to model       [3] ← ModelResponse       ← returns

T7      CallToolsNode        no tool calls
        processes text       validates output

T8      End                  final result returned

Timeline Shows:
- Linear message history growth: [0], [1], [2], [3]
- Parallel tool execution (not sequential)
- Multiple request/response cycles
```

---

## Graph Execution Context

```
┌─────────────────────────────────────────────────────────┐
│ Graph[                                                  │
│   State=GraphAgentState,                                │
│   Deps=GraphAgentDeps,                                  │
│   End=FinalResult                                       │
│ ]                                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ GraphAgentState (maintained across run):                │
│ ├─ message_history: list[ModelMessage]                  │
│ │  └─ grows: [], [req], [req, resp], [req, resp, req], │
│ ├─ usage: RunUsage (token counts)                       │
│ ├─ retries: int (validation failures)                   │
│ └─ run_step: int (current iteration)                    │
│                                                         │
│ GraphAgentDeps (dependencies):                          │
│ ├─ user_deps: DepsT (your custom deps)                  │
│ ├─ model: Model (LLM provider)                          │
│ ├─ tool_manager: ToolManager (your tools)               │
│ ├─ output_schema: OutputSchema (validation)             │
│ └─ ... (other config)                                   │
│                                                         │
│ Nodes:                                                  │
│ ├─ UserPromptNode[DepsT, OutputT]                       │
│ ├─ ModelRequestNode[DepsT, OutputT]                     │
│ └─ CallToolsNode[DepsT, OutputT]                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Tool Kind Hierarchy

```
All tool calls are categorized:

ToolCall
├─ output tools
│  └─ Purpose: return final result (end run)
│
├─ function tools
│  └─ Purpose: execute and continue
│
├─ external tools
│  └─ Purpose: defer for external approval
│
└─ unapproved tools
   └─ Purpose: require approval before execution
```

---

## Message Part Relationships

```
ModelRequest ────┬─ SystemPromptPart
                 │  └─ System instructions
                 │
                 ├─ UserPromptPart
                 │  └─ User or tool content
                 │
                 ├─ ToolReturnPart
                 │  └─ Result from executed tool
                 │
                 └─ RetryPromptPart
                    └─ Retry due to error

ModelResponse ───┬─ TextPart
                 │  └─ Text content
                 │
                 ├─ ToolCallPart
                 │  └─ Tool to call
                 │
                 ├─ ThinkingPart
                 │  └─ Model reasoning
                 │
                 ├─ FilePart
                 │  └─ Image/file output
                 │
                 └─ BuiltinToolCallPart
                    └─ Built-in tool (deprecated)
```

