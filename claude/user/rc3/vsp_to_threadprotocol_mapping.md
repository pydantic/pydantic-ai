# VSP → ThreadProtocol → Pydantic AI Mapping

## Overview

This document defines the canonical mapping between:
1. **Vercel AI SDK Streaming Protocol (VSP)** - Real-time streaming to client
2. **ThreadProtocol JSON** - Persisted canonical format
3. **Pydantic AI** - Server-side agent message structures

## The Four Layers

```
┌─────────────────────────────────────────────────────────┐
│ 1. Vercel AI SDK Streaming Protocol (VSP)              │
│    - SSE-based streaming format                         │
│    - start/delta/end patterns for all content types     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. ThreadProtocol-Specific Streaming Events            │
│    - data-tp-* (canonical multi-agent events)          │
│    - data-app-* (application custom events)            │
│    - data-sys-* (system/telemetry - usage, timing)     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. ThreadProtocol JSON (Persisted Format)              │
│    - Turns, Messages, Parts, SystemMessages            │
│    - Canonical representation                           │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. Pydantic AI ModelMessages & Events                  │
│    - PartStartEvent, PartDeltaEvent                     │
│    - ModelRequest, ModelResponse                        │
│    - Parts: UserPromptPart, TextPart, ToolCallPart...  │
└─────────────────────────────────────────────────────────┘
```

## Layer 1: VSP Standard Events

### Message Lifecycle

```typescript
// Message start
data: {"type":"start","messageId":"msg_123"}

// ... content streaming (text, tools, etc.) ...

// Message finish
data: {"type":"finish"}

// Stream termination
data: [DONE]
```

### Text Content (Start/Delta/End Pattern)

#### VSP Stream

```typescript
data: {"type":"text-start","id":"text_001"}
data: {"type":"text-delta","id":"text_001","delta":"Hello"}
data: {"type":"text-delta","id":"text_001","delta":" world"}
data: {"type":"text-end","id":"text_001"}
```

#### Maps to Pydantic AI Events

```python
# PartStartEvent
PartStartEvent(
    index=0,
    part=TextPart(content='')
)

# PartDeltaEvent (multiple)
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Hello'))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))

# Accumulated part
TextPart(content='Hello world')
```

#### Maps to ThreadProtocol JSON

```json
{
  "part_kind": "text",
  "content": "Hello world"
}
```

### Reasoning/Thinking Content (Start/Delta/End Pattern)

#### VSP Stream

```typescript
data: {"type":"reasoning-start","id":"reasoning_001"}
data: {"type":"reasoning-delta","id":"reasoning_001","delta":"Let me think..."}
data: {"type":"reasoning-delta","id":"reasoning_001","delta":" step by step"}
data: {"type":"reasoning-end","id":"reasoning_001"}
```

#### Maps to Pydantic AI Events

```python
# PartStartEvent
PartStartEvent(
    index=0,
    part=ThinkingPart(content='', provider_name='openai')
)

# PartDeltaEvent (multiple)
PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta='Let me think...'))
PartDeltaEvent(index=0, delta=ThinkingPartDelta(content_delta=' step by step'))

# Accumulated part
ThinkingPart(content='Let me think... step by step', provider_name='openai')
```

#### Maps to ThreadProtocol JSON

```json
{
  "part_kind": "thinking",
  "content": "Let me think... step by step",
  "provider_name": "openai"
}
```

**Note**: VSP's `reasoning-*` events map directly to Pydantic AI's `ThinkingPart`. We don't need custom `data-tp-thinking` events.

### Tool Calls (Start/Delta/Available Pattern)

#### VSP Stream

```typescript
data: {"type":"tool-input-start","toolCallId":"call_001","toolName":"get_weather"}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":"{\"city\""}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":":\"Paris\""}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":"}"}
data: {"type":"tool-input-available","toolCallId":"call_001","toolName":"get_weather","input":{"city":"Paris"}}
```

#### Maps to Pydantic AI Events

```python
# PartStartEvent (when tool-input-start received)
PartStartEvent(
    index=0,
    part=ToolCallPart(
        tool_call_id='call_001',
        tool_name='get_weather',
        args={}  # Empty initially
    )
)

# PartDeltaEvent (multiple, from tool-input-delta)
PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"city"'))
PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta=':"Paris"'))
PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='}'))

# FunctionToolCallEvent (when tool-input-available received)
FunctionToolCallEvent(
    part=ToolCallPart(
        tool_call_id='call_001',
        tool_name='get_weather',
        args={'city': 'Paris'}
    )
)
```

#### Maps to ThreadProtocol JSON

```json
{
  "part_kind": "tool-call",
  "tool_call_id": "call_001",
  "tool_name": "get_weather",
  "args": {"city": "Paris"}
}
```

**Important**: We use `tool-input-delta` events to progressively show tool arguments being generated, just like we stream text deltas.

### Tool Results

#### VSP Stream

```typescript
data: {"type":"tool-output-available","toolCallId":"call_001","output":{"temp":"72F","conditions":"sunny"}}
```

#### Maps to Pydantic AI Events

```python
# FunctionToolResultEvent
FunctionToolResultEvent(
    result=ToolReturnPart(
        tool_call_id='call_001',
        tool_name='get_weather',
        content={'temp': '72F', 'conditions': 'sunny'}
    )
)
```

#### Maps to ThreadProtocol JSON

```json
{
  "part_kind": "tool-return",
  "tool_call_id": "call_001",
  "tool_name": "get_weather",
  "status": "success",
  "content": {"temp": "72F", "conditions": "sunny"}
}
```

### Steps (Cycle Boundaries)

#### VSP Stream

```typescript
data: {"type":"start-step"}
// ... text, tool calls, etc. ...
data: {"type":"finish-step"}

data: {"type":"start-step"}
// ... next cycle ...
data: {"type":"finish-step"}
```

**Purpose**: `start-step` and `finish-step` mark the boundaries of individual LLM API calls. This is critical for understanding when a ModelRequest/ModelResponse cycle is complete.

**Maps to**: The boundary between ModelResponse → (tool execution) → ModelRequest in Pydantic AI's message_history.

### Tool Errors (Undocumented but Implemented)

⚠️ **Important**: These events exist in the VSP implementation code but are NOT documented in the official stream protocol docs. They are defined in `ui-message-chunks.ts` and used throughout the SDK.

#### Tool Input Validation Error

When tool arguments fail validation (e.g., invalid JSON, schema mismatch):

```typescript
// VSP stream (undocumented)
data: {"type":"tool-input-error","toolCallId":"call_001","toolName":"get_weather","input":{"city":123},"errorText":"Expected string, got number","providerExecuted":false,"dynamic":false}
```

#### Tool Output/Execution Error

When tool execution fails or produces an error:

```typescript
// VSP stream (undocumented)
data: {"type":"tool-output-error","toolCallId":"call_001","errorText":"API request failed: 503 Service Unavailable","providerExecuted":true,"dynamic":false}
```

#### Maps to Pydantic AI

Tool errors trigger the multi-step pattern where the error is sent back to the LLM:

```python
# Tool validation error creates a RetryPromptPart
RetryPromptPart(
    tool_call_id='call_001',
    tool_name='get_weather',
    content=[
        ValidationError(message='Expected string, got number')
    ]
)

# This appears in the NEXT ModelRequest sent to the LLM
ModelRequest(parts=[
    ToolReturnPart(
        tool_call_id='call_001',
        tool_name='get_weather',
        content='Validation failed: Expected string, got number'
        # Or as a RetryPromptPart depending on error severity
    )
])
```

#### Maps to ThreadProtocol JSON

```json
{
  "part_kind": "tool-return",
  "tool_call_id": "call_001",
  "tool_name": "get_weather",
  "status": "error",
  "content": "Expected string, got number"
}
```

Or for validation errors that should trigger retry:

```json
{
  "part_kind": "retry-prompt",
  "tool_call_id": "call_001",
  "tool_name": "get_weather",
  "content": [
    {
      "type": "validation-error",
      "message": "Expected string, got number"
    }
  ]
}
```

**Key Insight**: VSP sends tool errors as stream events (`tool-input-error`, `tool-output-error`), but they're added to the conversation history as `tool-result` messages for the next LLM step. The SDK doesn't auto-retry - it lets the LLM see the error and decide whether to retry.

### System Errors

The native VSP `error` event handles all infrastructure and system-level errors.

#### VSP Stream

```typescript
data: {"type":"error","errorText":"Rate limit exceeded"}
```

#### Maps to ThreadProtocol

```json
{
  "message_type": "system",
  "event_type": "error",
  "event_data": {
    "error": "Rate limit exceeded",
    "timestamp": "2025-01-20T10:00:00Z"
  }
}
```

**Use cases for `error` event:**
- Provider API failures (authentication, network, rate limits)
- Gateway errors (Vercel AI Gateway issues)
- Mid-stream failures (connection interruption)
- Multi-step errors (subsequent LLM call fails)
- Unexpected exceptions during processing

**Not used for:**
- Tool validation errors (use `tool-input-error`)
- Tool execution errors (use `tool-output-error`)

**Security note**: Error messages should be masked in production to prevent leaking sensitive information (stack traces, connection strings, etc.).

### Source References (Not Yet Implemented)

#### VSP Stream

```typescript
data: {"type":"source-url","sourceId":"https://example.com","url":"https://example.com"}
data: {"type":"source-document","sourceId":"doc_001","mediaType":"file","title":"Research Paper"}
```

**Current Status**: VSP natively supports these source reference events, but:
- Pydantic AI has no equivalent parts in ModelMessage
- ThreadProtocol doesn't yet define how to handle them
- No implementation exists to bridge this gap

**Roadmap**: Future versions may handle source references as:
- Custom SystemMessages for metadata tracking
- Extended part types if needed for semantic content
- Pass-through preservation for VSP compatibility

For now, these VSP events would be ignored or logged but not preserved in ThreadProtocol.

### File Parts

#### VSP Stream

```typescript
data: {"type":"file","url":"https://example.com/image.png","mediaType":"image/png"}
```

#### Maps to ThreadProtocol JSON

```json
{
  "part_kind": "file",
  "content": {
    "content_type": "image/png",
    "url": "https://example.com/image.png"
  }
}
```

#### Maps to Pydantic AI

```python
FilePart(
    content=BinaryImage(
        content_type='image/png',
        url='https://example.com/image.png'
    )
)
```

## Layer 2: ThreadProtocol-Specific Streaming Events

These use VSP's `data-*` custom event pattern.

### Canonical Multi-Agent Events (data-tp-*)

#### Thread Spawn

```typescript
// VSP stream
data: {"type":"data-tp-thread_spawn","data":{"spawned_thread_id":"thread-456","timestamp":"2025-01-20T10:00:00Z"}}

// ThreadProtocol JSON
{
  "message_type": "system",
  "event_type": "data-tp-thread_spawn",
  "event_data": {
    "spawned_thread_id": "thread-456",
    "timestamp": "2025-01-20T10:00:00Z"
  }
}
```

**Pydantic AI**: Not mapped (thread relationship metadata).


### System Telemetry Events (data-sys-*)

These events are **optional** and may be excluded from canonical hashing.

#### Usage/Token Counts

```typescript
// VSP stream
data: {"type":"data-sys-usage","data":{"input_tokens":100,"output_tokens":50,"total_tokens":150}}

// ThreadProtocol JSON (optional, may be excluded from canonical hash)
{
  "message_type": "system",
  "event_type": "data-sys-usage",
  "event_data": {
    "input_tokens": 100,
    "output_tokens": 50,
    "total_tokens": 150
  }
}
```

**Note**: Usage is runtime metadata about the request, not part of the canonical conversation content. It's stored in ThreadProtocol's `usage` field on messages, but streamed as `data-sys-*` for real-time observability.

#### Latency/Performance

```typescript
// VSP stream
data: {"type":"data-sys-latency","data":{"model_latency_ms":1234,"total_latency_ms":1500}}

// ThreadProtocol JSON (optional)
{
  "message_type": "system",
  "event_type": "data-sys-latency",
  "event_data": {
    "model_latency_ms": 1234,
    "total_latency_ms": 1500
  }
}
```

#### Cache Hits

```typescript
// VSP stream
data: {"type":"data-sys-cache_hit","data":{"cache_key":"request-hash-abc123","latency_ms":5}}
```

### Application Custom Events (data-app-*)

Application-specific orchestration events. Preserved but not interpreted by ThreadProtocol.

#### User Feedback

```typescript
// VSP stream
data: {"type":"data-app-user_feedback","data":{"rating":5,"comment":"Very helpful!"}}

// ThreadProtocol JSON
{
  "message_type": "system",
  "event_type": "data-app-user_feedback",
  "event_data": {
    "rating": 5,
    "comment": "Very helpful!"
  }
}
```
## Complete Example: Multi-Step Weather Request

### 1. VSP Stream (What Client Receives)

```typescript
// Message starts
data: {"type":"start","messageId":"msg_001"}

// Step 1 starts (initial response)
data: {"type":"start-step"}

// Text response
data: {"type":"text-start","id":"text_001"}
data: {"type":"text-delta","id":"text_001","delta":"I'll"}
data: {"type":"text-delta","id":"text_001","delta":" check"}
data: {"type":"text-delta","id":"text_001","delta":" the"}
data: {"type":"text-delta","id":"text_001","delta":" weather"}
data: {"type":"text-delta","id":"text_001","delta":"."}
data: {"type":"text-end","id":"text_001"}

// Tool call streaming
data: {"type":"tool-input-start","toolCallId":"call_001","toolName":"get_weather"}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":"{\"city\""}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":":\"Paris\""}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":"}"}
data: {"type":"tool-input-available","toolCallId":"call_001","toolName":"get_weather","input":{"city":"Paris"}}

// Step 1 finishes
data: {"type":"finish-step"}

// Tool executes (server-side, not streamed via VSP)
// Tool result
data: {"type":"tool-output-available","toolCallId":"call_001","output":{"temp":"72F","conditions":"sunny"}}

// Usage telemetry (optional)
data: {"type":"data-sys-usage","data":{"input_tokens":50,"output_tokens":20,"total_tokens":70}}

// Step 2 starts (final response)
data: {"type":"start-step"}

// Final text response
data: {"type":"text-start","id":"text_002"}
data: {"type":"text-delta","id":"text_002","delta":"The"}
data: {"type":"text-delta","id":"text_002","delta":" weather"}
data: {"type":"text-delta","id":"text_002","delta":" in"}
data: {"type":"text-delta","id":"text_002","delta":" Paris"}
data: {"type":"text-delta","id":"text_002","delta":" is"}
data: {"type":"text-delta","id":"text_002","delta":" currently"}
data: {"type":"text-delta","id":"text_002","delta":" 72°F"}
data: {"type":"text-delta","id":"text_002","delta":" and"}
data: {"type":"text-delta","id":"text_002","delta":" sunny"}
data: {"type":"text-delta","id":"text_002","delta":"."}
data: {"type":"text-end","id":"text_002"}

// Step 2 finishes
data: {"type":"finish-step"}

// Usage telemetry (optional)
data: {"type":"data-sys-usage","data":{"input_tokens":80,"output_tokens":15,"total_tokens":95}}

// Message finishes
data: {"type":"finish"}

// Stream terminates
data: [DONE]
```

### 2. Pydantic AI Events (Server-Side)

```python
# Step 1: Initial ModelResponse
PartStartEvent(index=0, part=TextPart(content=''))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='I'll'))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' check'))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' the'))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' weather'))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='.'))

PartStartEvent(index=1, part=ToolCallPart(tool_call_id='call_001', tool_name='get_weather', args={}))
PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='{"city"'))
PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta=':"Paris"'))
PartDeltaEvent(index=1, delta=ToolCallPartDelta(args_delta='}'))

FunctionToolCallEvent(
    part=ToolCallPart(
        tool_call_id='call_001',
        tool_name='get_weather',
        args={'city': 'Paris'}
    )
)

# Tool execution
FunctionToolResultEvent(
    result=ToolReturnPart(
        tool_call_id='call_001',
        tool_name='get_weather',
        content={'temp': '72F', 'conditions': 'sunny'}
    )
)

# Step 2: Final ModelResponse
PartStartEvent(index=0, part=TextPart(content=''))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='The'))
PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' weather'))
# ... more deltas ...

# Final result
FinalResultEvent(tool_name=None, tool_call_id=None)
AgentRunResultEvent(result=...)
```

### 3. ThreadProtocol JSON (Persisted)

```json
{
  "version": "0.0.4",
  "thread_id": "thread-123",
  "turns": [
    {
      "turn_type": "user",
      "submitted_at": "2025-01-20T10:00:00Z",
      "parts": [
        {
          "part_kind": "user-prompt",
          "content": "What's the weather in Paris?"
        }
      ]
    },
    {
      "turn_type": "agent",
      "agent_id": "agent-001",
      "started_at": "2025-01-20T10:00:01Z",
      "completed_at": "2025-01-20T10:00:05Z",
      "completion_status": "complete",
      "messages": [
        {
          "message_type": "request",
          "timestamp": "2025-01-20T10:00:01Z",
          "parts": [
            {
              "part_kind": "user-prompt",
              "content": "What's the weather in Paris?"
            }
          ]
        },
        {
          "message_type": "response",
          "timestamp": "2025-01-20T10:00:02Z",
          "parts": [
            {
              "part_kind": "text",
              "content": "I'll check the weather."
            },
            {
              "part_kind": "tool-call",
              "tool_call_id": "call_001",
              "tool_name": "get_weather",
              "args": {"city": "Paris"}
            }
          ],
          "finish_reason": "tool_calls",
          "usage": {
            "input_tokens": 50,
            "output_tokens": 20,
            "total_tokens": 70
          }
        },
        {
          "message_type": "request",
          "timestamp": "2025-01-20T10:00:03Z",
          "parts": [
            {
              "part_kind": "tool-return",
              "tool_call_id": "call_001",
              "tool_name": "get_weather",
              "status": "success",
              "content": {"temp": "72F", "conditions": "sunny"}
            }
          ]
        },
        {
          "message_type": "response",
          "timestamp": "2025-01-20T10:00:04Z",
          "parts": [
            {
              "part_kind": "text",
              "content": "The weather in Paris is currently 72°F and sunny."
            }
          ],
          "finish_reason": "stop",
          "usage": {
            "input_tokens": 80,
            "output_tokens": 15,
            "total_tokens": 95
          }
        }
      ],
      "total_usage": {
        "input_tokens": 130,
        "output_tokens": 35,
        "total_tokens": 165
      }
    }
  ]
}
```

## Summary Tables

### VSP → Pydantic AI Event Mapping

| VSP Event Type | Pydantic AI Event | Notes |
|----------------|-------------------|-------|
| `text-start` | `PartStartEvent(part=TextPart)` | Begins text accumulation |
| `text-delta` | `PartDeltaEvent(delta=TextPartDelta)` | Incremental text |
| `text-end` | (implicit) | Text complete |
| `reasoning-start` | `PartStartEvent(part=ThinkingPart)` | Begins thinking |
| `reasoning-delta` | `PartDeltaEvent(delta=ThinkingPartDelta)` | Incremental thinking |
| `reasoning-end` | (implicit) | Thinking complete |
| `tool-input-start` | `PartStartEvent(part=ToolCallPart)` | Tool call begins |
| `tool-input-delta` | `PartDeltaEvent(delta=ToolCallPartDelta)` | Args streaming |
| `tool-input-available` | `FunctionToolCallEvent` | Tool ready to execute |
| `tool-input-error` ⚠️ | (stream event only) | Undocumented; validation failed |
| `tool-output-available` | `FunctionToolResultEvent` | Tool result (success) |
| `tool-output-error` ⚠️ | `FunctionToolResultEvent` (error content) | Undocumented; execution failed |
| `start-step` | (cycle boundary) | New ModelRequest/Response |
| `finish-step` | (cycle boundary) | Cycle complete |
| `finish` | `FinalResultEvent`, `AgentRunResultEvent` | Message complete |
| `error` | SystemMessage with `event_type: "error"` | Infrastructure/system errors |

⚠️ = Implemented in SDK code but not in official documentation

### VSP → ThreadProtocol Part Kind Mapping

| VSP Pattern | ThreadProtocol part_kind | Notes |
|-------------|-------------------------|-------|
| `text-start/delta/end` | `text` | Accumulated from deltas |
| `reasoning-start/delta/end` | `thinking` | Extended thinking content |
| `tool-input-start/delta/available` | `tool-call` | Accumulated args from deltas |
| `tool-output-available` | `tool-return` (status: success) | Tool execution result |
| `file` | `file` | Binary content reference |
| User input | `user-prompt` | Not streamed, submitted complete |

### ThreadProtocol Custom Events

| Event Name | Tier | Required | Streamed As | Maps to PAI |
|------------|------|----------|-------------|-------------|
| `data-tp-thread_spawn` | Canonical | Yes | VSP data event | No |
| `data-sys-usage` | Telemetry | No | VSP data event | No (metadata) |
| `data-sys-latency` | Telemetry | No | VSP data event | No (metadata) |
| `data-sys-cache_hit` | Telemetry | No | VSP data event | No (metadata) |
| `data-app-*` | Application | No | VSP data event | No |

## Key Design Principles

1. **Use VSP patterns verbatim**: Don't reinvent start/delta/end for text, thinking, or tool calls
2. **Steps = Cycles**: `start-step`/`finish-step` mark ModelRequest/ModelResponse boundaries
3. **Reasoning = Thinking**: VSP's `reasoning-*` maps directly to PAI's `ThinkingPart`
4. **Stream tool deltas**: Use `tool-input-delta` to show progressive tool argument generation
5. **Usage is telemetry**: `data-sys-usage` events are optional metadata, not canonical content
6. **Multi-agent via data-tp-***: Only add custom events for orchestration PAI doesn't handle
7. **Preserve application extensions**: `data-app-*` for custom domain events

## Error Handling

### VSP Error Event Types (Documented and Undocumented)

VSP has three types of error events:

1. **`tool-input-error`** (⚠️ undocumented) - Tool argument validation failed
2. **`tool-output-error`** (⚠️ undocumented) - Tool execution failed
3. **`error`** (documented) - System-level errors (network, gateway, etc.)

### How Tool Errors Work

When a tool call fails validation or execution:

```typescript
// VSP stream sequence
data: {"type":"tool-input-start","toolCallId":"call_001","toolName":"get_weather"}
data: {"type":"tool-input-delta","toolCallId":"call_001","inputTextDelta":"..."}
data: {"type":"tool-input-available","toolCallId":"call_001","toolName":"get_weather","input":{...}}

// Validation fails
data: {"type":"tool-input-error","toolCallId":"call_001","toolName":"get_weather","errorText":"Invalid input"}

// Error is finalized
data: {"type":"tool-output-error","toolCallId":"call_001","errorText":"Validation failed: Invalid input"}

// Step finishes
data: {"type":"finish-step"}

// SDK continues to next step (if maxSteps allows)
// The error is added to conversation history
// LLM sees the error and can retry with corrected args
data: {"type":"start-step"}
// ... LLM's retry attempt or alternative approach ...
```

### Pydantic AI Mapping

Tool errors in Pydantic AI use:

- `RetryPromptPart` - For validation errors that should trigger retry logic

Both are added to the conversation history and sent to the LLM in the next step.

### Implementation Notes

⚠️ **Important**: Since `tool-input-error` and `tool-output-error` are undocumented, you must reference:
- TypeScript schema: `ui-message-chunks.ts` (lines 57-80)
- Implementation code: `stream-text.ts` (lines 1891-1955)
- Multi-step logic: `stream-text.ts` (lines 1469-1503)

These show the actual event format and behavior, even though the markdown docs don't mention them.
