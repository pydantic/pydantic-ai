# Pydantic AI Streaming: Complete Delta and Event Types Reference

## Overview
Pydantic AI provides a comprehensive system of delta and event types for streaming responses. These types handle incremental updates (deltas) for various response parts and events that occur during streaming operations.

## Delta Types (ModelResponsePartDelta)

Delta types represent incremental changes to parts of a model response during streaming. They are used in `PartDeltaEvent` with `ModelResponsePartDelta` as the discriminator.

### 1. TextPartDelta
**Location**: `pydantic_ai/messages.py` (lines 1345-1372)
**Discriminator**: `part_delta_kind: Literal['text']`

Represents incremental text content to append to an existing `TextPart`.

**Fields**:
- `content_delta: str` - The incremental text content to add

**Methods**:
- `apply(part: ModelResponsePart) -> TextPart` - Applies the delta to a TextPart

**Example Usage**:
```python
delta = TextPartDelta(content_delta="hello ")
# Can be applied multiple times to build up a response
```

---

### 2. ThinkingPartDelta
**Location**: `pydantic_ai/messages.py` (lines 1376-1434)
**Discriminator**: `part_delta_kind: Literal['thinking']`

Represents incremental changes to a thinking part (used by models that support extended thinking/reasoning).

**Fields**:
- `content_delta: str | None = None` - Incremental thinking content to add
- `signature_delta: str | None = None` - Optional signature delta (never treated as a true delta, replaces None)
- `provider_name: str | None = None` - Provider name (signatures only sent back to same provider)

**Methods**:
- `apply(part: ModelResponsePart | ThinkingPartDelta) -> ThinkingPart | ThinkingPartDelta` - Applies to a part or another delta

**Supported By**:
- Anthropic (uses `signature` field)
- Bedrock (uses `signature` field)
- Google (uses `thought_signature` field)
- OpenAI (uses `encrypted_content` field)

**Note**: Can be applied to another delta to merge incremental updates.

---

### 3. ToolCallPartDelta
**Location**: `pydantic_ai/messages.py` (lines 1437-1557)
**Discriminator**: `part_delta_kind: Literal['tool_call']`

Represents incremental changes to a tool call (tool name, arguments, or tool call ID).

**Fields**:
- `tool_name_delta: str | None = None` - Incremental text to add to the tool name
- `args_delta: str | dict[str, Any] | None = None` - Incremental arguments (JSON string or dict)
- `tool_call_id: str | None = None` - Tool call ID (not treated as delta, can replace None)

**Methods**:
- `as_part() -> ToolCallPart | None` - Converts delta to a full ToolCallPart if possible
- `apply(part: ModelResponsePart | ToolCallPartDelta) -> ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta` - Applies to part or delta
- `_apply_to_part(part: ToolCallPart | BuiltinToolCallPart)` - Internal helper
- `_apply_to_delta(delta: ToolCallPartDelta)` - Internal helper to merge deltas

**Complex Behavior**:
- Can handle both JSON string args and dict args
- Merges dict arguments with `{**existing, **new}`
- Appends JSON strings `existing + new`
- Raises `UnexpectedModelBehavior` if mixing JSON and dict arg types
- Can convert accumulated delta to a full ToolCallPart once tool_name_delta is set

---

## Streaming Event Types

### ModelResponseStreamEvent (Base Category)
**Location**: `pydantic_ai/messages.py` (lines 1616-1619)
**Type**: `Annotated[PartStartEvent | PartDeltaEvent | FinalResultEvent, pydantic.Discriminator('event_kind')]`

Discriminator field: `event_kind`

#### 1. PartStartEvent
**Location**: `pydantic_ai/messages.py` (lines 1566-1583)
**Event Kind**: `'part_start'`

Indicates a new part has started. If multiple `PartStartEvent`s are received with the same index, the new one should fully replace the old one.

**Fields**:
- `index: int` - Index of the part within response parts list
- `part: ModelResponsePart` - The newly started part

**Part Types**:
- `TextPart` - Plain text response
- `ThinkingPart` - Extended thinking/reasoning
- `ToolCallPart` - Tool call invocation
- `BuiltinToolCallPart` - Built-in tool call
- `BuiltinToolReturnPart` - Built-in tool result
- `FilePart` - Binary file response

---

#### 2. PartDeltaEvent
**Location**: `pydantic_ai/messages.py` (lines 1586-1599)
**Event Kind**: `'part_delta'`

Indicates a delta update for an existing part.

**Fields**:
- `index: int` - Index of the part within response parts list
- `delta: ModelResponsePartDelta` - Delta to apply (TextPartDelta | ThinkingPartDelta | ToolCallPartDelta)

**Workflow**:
1. Receive PartStartEvent with index and initial part
2. Receive multiple PartDeltaEvents with deltas for that index
3. Each delta is applied to the accumulated part state

---

#### 3. FinalResultEvent
**Location**: `pydantic_ai/messages.py` (lines 1602-1613)
**Event Kind**: `'final_result'`

Indicates that the response matches the output schema and will produce a result.

**Fields**:
- `tool_name: str | None` - Name of output tool called (None if result from text)
- `tool_call_id: str | None` - Tool call ID if any
- `event_kind: Literal['final_result']` - Discriminator

**Usage**:
- Signals end of model response streaming
- Can appear multiple times if multiple tool calls present
- Used to distinguish text output from structured output

---

### HandleResponseEvent (Tool Response Category)
**Location**: `pydantic_ai/messages.py` (lines 1703-1710)
**Type**: `Annotated[FunctionToolCallEvent | FunctionToolResultEvent | BuiltinToolCallEvent | BuiltinToolResultEvent, pydantic.Discriminator('event_kind')]`

Discriminator field: `event_kind`

#### 1. FunctionToolCallEvent
**Location**: `pydantic_ai/messages.py` (lines 1622-1645)
**Event Kind**: `'function_tool_call'`

Indicates the start of a call to a function tool (user-defined tool).

**Fields**:
- `part: ToolCallPart` - The tool call to make
- `event_kind: Literal['function_tool_call']`

**Properties**:
- `tool_call_id: str` - Mirrors `part.tool_call_id` for matching to results

**Deprecated**: `call_id` property (use `tool_call_id` instead)

---

#### 2. FunctionToolResultEvent
**Location**: `pydantic_ai/messages.py` (lines 1648-1668)
**Event Kind**: `'function_tool_result'`

Indicates the result of a function tool call.

**Fields**:
- `result: ToolReturnPart | RetryPromptPart` - Tool result or retry prompt
- `content: str | Sequence[UserContent] | None = None` - Content sent to model after result
- `event_kind: Literal['function_tool_result']`

**Properties**:
- `tool_call_id: str` - Matches to original call

**Result Types**:
- `ToolReturnPart` - Successful tool execution result
- `RetryPromptPart` - Validation error or retry request

---

#### 3. BuiltinToolCallEvent (Deprecated)
**Location**: `pydantic_ai/messages.py` (lines 1671-1684)
**Event Kind**: `'builtin_tool_call'`

**Status**: DEPRECATED - Use `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolCallPart` instead

Represents a built-in tool call (e.g., web search).

**Fields**:
- `part: BuiltinToolCallPart` - The built-in tool call
- `event_kind: Literal['builtin_tool_call']`

---

#### 4. BuiltinToolResultEvent (Deprecated)
**Location**: `pydantic_ai/messages.py` (lines 1687-1700)
**Event Kind**: `'builtin_tool_result'`

**Status**: DEPRECATED - Use `PartStartEvent` and `PartDeltaEvent` with `BuiltinToolReturnPart` instead

Represents the result of a built-in tool call.

**Fields**:
- `result: BuiltinToolReturnPart` - Built-in tool result
- `event_kind: Literal['builtin_tool_result']`

---

### Agent-Level Event

#### AgentRunResultEvent
**Location**: `pydantic_ai/run.py` (lines 371-383)
**Event Kind**: `'agent_run_result'`

Indicates the agent run has ended with a final result.

**Fields**:
- `result: AgentRunResult[OutputDataT]` - Final result with output and metadata
- `event_kind: Literal['agent_run_result']`

---

## Combined Event Type Hierarchy

```
AgentStreamEvent
├── ModelResponseStreamEvent
│   ├── PartStartEvent (event_kind='part_start')
│   ├── PartDeltaEvent (event_kind='part_delta')
│   │   └── delta: ModelResponsePartDelta
│   │       ├── TextPartDelta
│   │       ├── ThinkingPartDelta
│   │       └── ToolCallPartDelta
│   └── FinalResultEvent (event_kind='final_result')
├── HandleResponseEvent
│   ├── FunctionToolCallEvent (event_kind='function_tool_call')
│   ├── FunctionToolResultEvent (event_kind='function_tool_result')
│   ├── BuiltinToolCallEvent (event_kind='builtin_tool_call', DEPRECATED)
│   └── BuiltinToolResultEvent (event_kind='builtin_tool_result', DEPRECATED)
└── AgentRunResultEvent (event_kind='agent_run_result')
```

---

## Internal/Model-Specific Delta Types

### DeltaToolCall
**Location**: `pydantic_ai/models/function.py` (lines 222-237)

Used internally by `FunctionModel` to describe streaming structured responses.

**Fields**:
- `name: str | None = None` - Incremental tool name
- `json_args: str | None = None` - Incremental JSON arguments
- `tool_call_id: str | None = None` - Tool call ID

**Used by**: `FunctionModel._get_event_iterator()` to convert to public delta types

---

### DeltaThinkingPart
**Location**: `pydantic_ai/models/function.py` (lines 241-250)

Used internally by `FunctionModel` for streaming thinking responses.

**Fields**:
- `content: str | None = None` - Incremental thinking content
- `signature: str | None = None` - Incremental signature

**Used by**: `FunctionModel._get_event_iterator()` for thinking part deltas

---

## Type Aliases

**Location**: `pydantic_ai/models/function.py`

- `DeltaToolCalls: TypeAlias = dict[int, DeltaToolCall]` (line 253)
  - Mapping of tool call indices to incremental changes
  
- `DeltaThinkingCalls: TypeAlias = dict[int, DeltaThinkingPart]` (line 256)
  - Mapping of thinking part indices to incremental changes

- `ModelResponsePartDelta` (lines 1560-1562)
  - Union of: `TextPartDelta | ThinkingPartDelta | ToolCallPartDelta`

- `ModelResponseStreamEvent` (lines 1616-1619)
  - Union of: `PartStartEvent | PartDeltaEvent | FinalResultEvent`

- `HandleResponseEvent` (lines 1703-1709)
  - Union of: `FunctionToolCallEvent | FunctionToolResultEvent | BuiltinToolCallEvent | BuiltinToolResultEvent`

- `AgentStreamEvent` (line 1712)
  - Union of: `ModelResponseStreamEvent | HandleResponseEvent`

---

## Streaming Workflow

### Typical Model Response Streaming Sequence:

1. **PartStartEvent** - New part begins (e.g., TextPart with empty content)
2. **PartDeltaEvent** (multiple) - Incremental content arrives
   - Each delta applies to the accumulated part
   - `TextPartDelta` appends to text
   - `ToolCallPartDelta` accumulates tool name and args
   - `ThinkingPartDelta` appends thinking content
3. **FinalResultEvent** - Response complete and validated

### Tool Calling Sequence:

1. **PartStartEvent** - ToolCallPart starts
2. **PartDeltaEvent** (multiple) - Tool name and arguments stream in
3. **FunctionToolCallEvent** - Tool ready to be called
4. **FunctionToolResultEvent** - Tool execution result or retry prompt
5. (Optional) **PartStartEvent** - Next response part begins

### Complete Agent Run:

1. Model request handling → ModelResponseStreamEvent events
2. Tool execution → HandleResponseEvent events
3. **AgentRunResultEvent** - Final result with complete output

---

## Key Design Patterns

### Delta Accumulation
Deltas are designed to be accumulated:
```python
part = TextPart(content='')
delta1 = TextPartDelta(content_delta='Hello ')
part = delta1.apply(part)  # TextPart(content='Hello ')
delta2 = TextPartDelta(content_delta='world')
part = delta2.apply(part)  # TextPart(content='Hello world')
```

### Delta-to-Delta Merging
Tool call deltas can be merged:
```python
delta1 = ToolCallPartDelta(tool_name_delta='tool_')
delta2 = ToolCallPartDelta(tool_name_delta='name')
merged = delta2.apply(delta1)  # tool_name_delta='tool_name'
```

### Partial Validation
Streaming allows validation on incomplete data:
- Partial JSON in ToolCallPartDelta
- Incremental text that can be validated
- Thinking content accumulated separately

### Index-Based Part Tracking
Events use `index` to track which part is being updated:
- PartStartEvent(index=0, part=...)
- PartDeltaEvent(index=0, delta=...)
- Multiple parts can be in flight simultaneously

---

## Event Stream Usage

### Via stream_responses()
```python
async with agent.run_stream('prompt') as result:
    async for response, is_last in result.stream_responses(debounce_by=None):
        # Receives accumulated ModelResponse objects
        pass
```

### Via event_stream_handler
```python
async def handler(ctx, stream):
    async for event in stream:
        if isinstance(event, PartStartEvent):
            print(f"Part {event.index} started: {event.part}")
        elif isinstance(event, PartDeltaEvent):
            print(f"Part {event.index} delta: {event.delta}")
        elif isinstance(event, FunctionToolCallEvent):
            print(f"Calling tool: {event.part.tool_name}")
```

### Via iter() with node.stream()
```python
async with agent.iter('prompt') as run:
    async for node in run:
        if agent.is_model_request_node(node):
            async with node.stream(run.ctx) as stream:
                async for event in stream:
                    # Streaming events available here
                    pass
```

---

## Discriminator Values Summary

| Class | Discriminator Field | Value |
|-------|-------------------|-------|
| TextPartDelta | part_delta_kind | 'text' |
| ThinkingPartDelta | part_delta_kind | 'thinking' |
| ToolCallPartDelta | part_delta_kind | 'tool_call' |
| PartStartEvent | event_kind | 'part_start' |
| PartDeltaEvent | event_kind | 'part_delta' |
| FinalResultEvent | event_kind | 'final_result' |
| FunctionToolCallEvent | event_kind | 'function_tool_call' |
| FunctionToolResultEvent | event_kind | 'function_tool_result' |
| BuiltinToolCallEvent | event_kind | 'builtin_tool_call' |
| BuiltinToolResultEvent | event_kind | 'builtin_tool_result' |
| AgentRunResultEvent | event_kind | 'agent_run_result' |

