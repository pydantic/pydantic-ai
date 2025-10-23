# ThreadProtocol v0.0.4 Amendments

## Executive Summary

ThreadProtocol v0.0.3 enforced "complete turns only" - discarding entire AgentTurns if interrupted. This was too coarse-grained and threw away valuable completed work (tool calls, tool returns, etc.).

**v0.0.4 key change**: Track complete **request/response cycles**, not complete **turns**.

A "complete cycle" = `ModelRequest → ModelResponse → ToolReturns` (if response had tool calls)

An interrupted AgentTurn can be preserved if it contains at least one complete cycle. This ensures semantic integrity - we never save tool calls without their corresponding returns.

## ThreadProtocol Philosophy: "What Happened" Not "How/Why"

ThreadProtocol records **declarative facts** about what occurred in a conversation. It explicitly does NOT specify orchestration logic, routing decisions, or execution models.

### What ThreadProtocol Records (Canonical)

- **Fact**: Thread X was spawned at timestamp T
- **Fact**: Agent A completed a turn, then Agent B started a turn
- **Fact**: Tool call C was made, then tool return D was received
- **Fact**: User interrupted the response at timestamp T

### What ThreadProtocol Does NOT Specify (Application Layer)

- **NOT specified**: Did the spawned thread block the parent? Run in parallel? Execute conditionally?
- **NOT specified**: Why was control handed from Agent A to Agent B? What routing logic decided this?
- **NOT specified**: How was the tool executed? What triggered the tool call?
- **NOT specified**: How should the client handle the interruption? Should it retry? Resume?

### The Boundary

```typescript
// ✅ CANONICAL (part of ThreadProtocol)
{
  "message_type": "system",
  "event_type": "data-tp-thread_spawn",
  "event_data": {
    "spawned_thread_id": "thread-456",
    "timestamp": "2025-01-20T10:00:00Z"
  }
}

// ❌ NOT CANONICAL (application-specific orchestration)
{
  "message_type": "system",
  "event_type": "data-app-spawn_config",
  "event_data": {
    "execution_mode": "blocking",         // How it's executed
    "priority": "high",                   // Scheduling hint
    "timeout_ms": 5000                    // Implementation detail
  }
}
```

```typescript
// ✅ CANONICAL (part of ThreadProtocol)
{
  "turn_type": "agent",
  "agent_id": "agent-B",
  "started_at": "2025-01-20T10:05:00Z",
  // Agent B took a turn (the fact)
}

// ❌ NOT CANONICAL (application-specific routing)
{
  "message_type": "system",
  "event_type": "data-app-routing_decision",
  "event_data": {
    "previous_agent": "agent-A",
    "selected_agent": "agent-B",
    "routing_strategy": "round-robin",    // Why this agent was chosen
    "confidence_score": 0.95               // Decision metadata
  }
}
```

### Implications for v0.0.4

The `interruption` field records **that** an interruption happened (the fact), not **how** the client should handle it:

```typescript
// ✅ CANONICAL - Records the fact
{
  "completion_status": "interrupted",
  "interruption": {
    "reason": "user_cancelled",
    "interrupted_at": "2025-01-20T10:00:15Z"
  }
}

// The client decides what to do with this fact:
// - Retry the request?
// - Resume from last complete cycle?
// - Show error message?
// - Discard and move on?
// All of these are application-layer decisions, not ThreadProtocol concerns.
```

This separation enables:
- **Portable threads**: The same ThreadProtocol can be used by different orchestrators
- **Replay/analysis**: Reconstruct what happened without needing to understand why
- **Multiple interpretations**: Different clients can handle the same facts differently

## Namespace Taxonomy: What's Canonical vs. Custom

Following the Vercel AI SDK streaming protocol (VSP), all custom events are sent as `data` stream parts with specific naming conventions. ThreadProtocol uses prefixes to distinguish between canonical protocol events, system/telemetry metadata, and application-specific extensions.

### Tier 1: ThreadProtocol Canonical (`data-tp-*`)

**Part of ThreadProtocol specification. Required for semantic integrity and round-tripping.**

These are the normative event types defined by ThreadProtocol:

```typescript
// Normative SystemMessage event_type values
"data-tp-agent_handoff"    // Control passed between agents
"data-tp-thread_spawn"     // Child thread created
"data-tp-thread_merge"     // Thread incorporated into this thread
"data-tp-thread_end"       // Thread terminated
"data-tp-error"            // Error condition

// These MUST be interpreted consistently by all conforming implementations
// They describe facts about multi-agent orchestration that happened
```

All normative message part kinds are also canonical:
- `"text"`, `"thinking"`, `"tool-call"`, `"tool-return"`, `"retry-prompt"`, `"user-prompt"`, `"file"`

### Tier 2: System/Runtime Metadata (`data-sys-*`)

**Implementation telemetry and debugging information. NOT required for round-tripping.**

These events MAY be excluded from canonical hashing and thread diffing:

```typescript
// Examples (these are NOT defined by ThreadProtocol)
"data-sys-cache_hit"             // Was response cached?
"data-sys-retry_count"           // How many retries occurred?
"data-sys-latency_ms"            // Request latency
"data-sys-token_estimate"        // Estimated token usage
"data-sys-model_temperature"     // Model parameters used
"data-sys-rate_limit_remaining"  // API quota info
```

**Use case**: Debugging, monitoring, observability. Can be stripped before persistence or when diffing threads for semantic equivalence.

### Tier 3: Application Custom (`data-app-*`)

**Application-specific events and orchestration logic. Preserved but not interpreted by ThreadProtocol.**

These events are preserved during serialization but have no normative meaning:

```typescript
// Examples (these are NOT defined by ThreadProtocol)
"data-app-user_feedback"         // User rated response
"data-app-routing_decision"      // Why agent B was selected
"data-app-spawn_config"          // How spawned thread should execute
"data-app-context_switch"        // Application mode change
"data-app-embeddings"            // Vector embeddings for this turn
"data-app-analytics_event"       // Analytics tracking
"data-app-custom_validation"     // Application-specific validation
```

**Use case**: Recording application-specific orchestration decisions, analytics, or domain events. Different applications using ThreadProtocol will have completely different `data-app-*` events.

### Namespace Conventions

All events use the `data-*` prefix to align with VSP streaming protocol:

```typescript
// Pattern matching
/^data-tp-/      → ThreadProtocol canonical (Tier 1)
/^data-sys-/     → System/runtime metadata (Tier 2)
/^data-app-/     → Application custom (Tier 3)

// Unrecognized data-* patterns default to Tier 3 (application custom)
```

### VSP Streaming Format

When streaming to the client via Vercel AI SDK:

```typescript
// VSP data stream part for ThreadProtocol event
{
  type: 'data',
  name: 'data-tp-thread_spawn',
  value: {
    spawned_thread_id: 'thread-456',
    timestamp: '2025-01-20T10:00:00Z'
  }
}

// VSP data stream part for application custom event
{
  type: 'data',
  name: 'data-app-spawn_config',
  value: {
    execution_mode: 'blocking',
    timeout_ms: 5000
  }
}
```

### Canonicalization Rules

When hashing or comparing threads for semantic equivalence:

1. **Include**: All `data-tp-*` events, normative message parts, and turns
2. **Include**: All `data-app-*` events (preserve but don't interpret)
3. **Optional**: `data-sys-*` events may be excluded (implementation-defined)

### Example: Recording a Spawned Thread

In ThreadProtocol JSON (persisted format):

```typescript
// Canonical fact (REQUIRED by ThreadProtocol)
{
  "message_type": "system",
  "event_type": "data-tp-thread_spawn",
  "timestamp": "2025-01-20T10:00:00Z",
  "event_data": {
    "spawned_thread_id": "thread-456"
  }
}

// Application orchestration (OPTIONAL, application-defined)
{
  "message_type": "system",
  "event_type": "data-app-spawn_execution_config",
  "timestamp": "2025-01-20T10:00:00Z",
  "event_data": {
    "execution_mode": "blocking",
    "timeout_ms": 5000,
    "priority": "high"
  }
}

// System telemetry (OPTIONAL, may be excluded from canonical hash)
{
  "message_type": "system",
  "event_type": "data-sys-spawn_latency",
  "timestamp": "2025-01-20T10:00:01Z",
  "event_data": {
    "spawn_duration_ms": 123
  }
}
```

All three may appear in the same thread, but only the first is required for semantic integrity.

### How This Maps to VSP Streaming

During real-time streaming, these SystemMessages are sent as VSP `data` stream parts:

```typescript
// Server sends to client via VSP
streamData('data-tp-thread_spawn', {
  spawned_thread_id: 'thread-456',
  timestamp: '2025-01-20T10:00:00Z'
});

streamData('data-app-spawn_execution_config', {
  execution_mode: 'blocking',
  timeout_ms: 5000,
  priority: 'high'
});

streamData('data-sys-spawn_latency', {
  spawn_duration_ms: 123
});

// Client receives VSP events and assembles them into ThreadProtocol SystemMessages
```

The client reconstructs the ThreadProtocol JSON format by converting VSP `data` events into `SystemMessage` objects with the corresponding `event_type` and `event_data`.

## Motivation: The Interruption Problem

When a user interrupts streaming (e.g., hitting escape in Claude Code), what should be preserved?

### The Hierarchy

```
AgentTurn (= single agent.run() call)
└── ModelRequest/ModelResponse (= single LLM API call = Vercel AI SDK "step")
    └── Parts (TextPart, ToolCallPart, ThinkingPart, etc.)
        └── Assembled from deltas during streaming
```

### Example Interruption Scenario

```python
# User submits: "What's the weather in Paris and Berlin?"
ModelRequest(parts=[UserPromptPart("What's the weather...")])

# Agent responds with tool call
ModelResponse(parts=[
    TextPart("Let me check the weather for both cities."),
    ToolCallPart(tool_name="get_weather", args={"city": "Paris"}),
    ToolCallPart(tool_name="get_weather", args={"city": "Berlin"})
])  # ✅ COMPLETE

# Tools execute
ModelRequest(parts=[
    ToolReturnPart(tool_name="get_weather", content={"temp": "72F"}),  # Paris
    ToolReturnPart(tool_name="get_weather", content={"temp": "68F"})   # Berlin
])  # ✅ COMPLETE

# Agent starts final response
ModelResponse(parts=[
    # Streaming: "Based on the weather data, Paris is currently 72°F and..."
    # ❌ USER HITS CANCEL MID-STREAM
])  # ❌ INCOMPLETE - has no complete parts
```

**v0.0.3 behavior**: Discard entire AgentTurn (lose all 3 complete messages!)
**v0.0.4 behavior**: Keep the 2 complete ModelResponses and 1 complete ModelRequest, mark turn as interrupted

## The Graph Execution Cycle

### Critical Discovery: Tool Calls Are Declarative

**Tool calls do NOT execute during ModelResponse streaming.** They execute AFTER the ModelResponse completes.

From `pydantic_ai/_agent_graph.py:579-607`:

```python
# The execution flow:
class CallToolsNode:
    async def _run_stream(self, ctx):
        # Lines 579-596: Iterate through the COMPLETED ModelResponse
        for part in self.model_response.parts:
            if isinstance(part, ToolCallPart):
                tool_calls.append(part)  # Collect all tool calls

        # Line 605: NOW execute the tools (in parallel)
        if tool_calls:
            async for event in self._handle_tool_calls(ctx, tool_calls):
                yield event
```

### The Complete Cycle

```
ModelRequestNode
  ├─> Line 447: Append ModelRequest to message_history
  ├─> Stream ModelResponse from LLM
  └─> Line 495: Append ModelResponse to message_history
      └─> Return CallToolsNode

CallToolsNode
  ├─> Line 741: "Process tool calls in parallel"
  ├─> Execute all tools via asyncio.gather
  ├─> Collect ToolReturnParts
  └─> Create new ModelRequest with ToolReturnParts
      └─> Return new ModelRequestNode

[Cycle repeats until no more tool calls]
```

**Atomic unit**: `ModelRequest → ModelResponse → ToolReturns`

If interrupted after ModelResponse streams but before CallToolsNode completes, the tool calls were **declared but never executed**.

### Why This Matters

```python
# BAD: Including tool calls without returns
message_history = [
    ModelRequest(parts=[UserPromptPart("Get weather for Paris and Berlin")]),
    ModelResponse(parts=[
        ToolCallPart(tool_name="get_weather", args={"city": "Paris"}),
        ToolCallPart(tool_name="get_weather", args={"city": "Berlin"})
    ])  # These tools NEVER RAN
]

# If you pass this to next agent.run():
# Agent thinks: "I called these tools, where are the returns?"
# This is semantically broken!
```

```python
# GOOD: Complete cycle only
message_history = [
    ModelRequest(parts=[UserPromptPart("Get weather for Paris and Berlin")]),
    ModelResponse(parts=[ToolCallPart(...), ToolCallPart(...)]),
    ModelRequest(parts=[ToolReturnPart(...), ToolReturnPart(...)])
]

# Agent sees: "I called tools, got results, can continue"
# Semantically valid!
```

## The Core Invariant

### v0.0.4 Rule

> **Only complete request/response cycles appear in ThreadProtocol.**
>
> - A **complete part** = fully formed (not a delta)
> - A **complete ModelResponse** = finished streaming AND (has no tool calls OR all tool calls executed)
> - A **complete cycle** = ModelRequest + ModelResponse + (optionally) ToolReturns
> - An **interrupted turn** = has completion_status: "interrupted", but contains all complete cycles

### What Pydantic AI Considers "Complete"

From `pydantic_ai/_parts_manager.py:62-68`:

```python
def get_parts(self) -> list[ModelResponsePart]:
    """Return only model response parts that are complete (i.e., not ToolCallPartDelta's).

    Returns:
        A list of ModelResponsePart objects. ToolCallPartDelta objects are excluded.
    """
    return [p for p in self._parts if not isinstance(p, ToolCallPartDelta)]
```

**Key insight**: Pydantic AI maintains a two-tier system:
- **Incomplete parts**: `ToolCallPartDelta` - still accumulating from streaming deltas
- **Complete parts**: `TextPart`, `ToolCallPart`, `ThinkingPart`, etc. - fully formed

When you call `streamed_response.get()`, it automatically:
- ✅ Includes all complete parts
- ❌ Excludes any incomplete `ToolCallPartDelta`s

### Include/Exclude Logic

The decision for each ModelResponse:

```typescript
function shouldIncludeModelResponse(
  response: ModelResponse,
  nextMessage: ModelMessage | null
): boolean {
  // First: Does it have any complete parts?
  if (response.parts.length === 0) {
    return false;  // No complete parts
  }

  // Second: Does it have tool calls?
  const hasToolCalls = response.parts.some(p => p.part_kind === "tool-call");
  if (!hasToolCalls) {
    return true;  // Text/thinking only - semantically complete
  }

  // Third: Check if next message has the corresponding returns
  if (!nextMessage || nextMessage.kind !== "request") {
    return false;  // Tool calls without returns = incomplete
  }

  const hasReturns = nextMessage.parts.some(
    p => p.part_kind === "tool-return"
  );
  return hasReturns;  // Include only if we have the returns
}
```

## How Pydantic AI Handles Cancellation

### The Streaming Lifecycle

From `pydantic_ai/_agent_graph.py:399-431`:

```python
class ModelRequestNode:
    async def stream(self, ctx):
        # Prepare the request
        model_settings, params, history, run_ctx = await self._prepare_request(ctx)

        # Line 447: Request is added to message_history immediately
        ctx.state.message_history.append(self.request)

        # Open streaming context
        async with ctx.deps.model.request_stream(...) as streamed_response:
            # Create AgentStream for user consumption
            agent_stream = AgentStream(
                _raw_stream_response=streamed_response,
                ...
            )
            yield agent_stream

            # Lines 424-425: Consume any remaining stream events
            async for _ in agent_stream:
                pass

        # Line 427: Get the complete ModelResponse
        model_response = streamed_response.get()

        # Line 429: Finalize and add to history
        self._finish_handling(ctx, model_response)

    def _finish_handling(self, ctx, response):
        # Update usage
        ctx.state.usage.incr(response.usage)

        # Line 495: Append response to message_history
        ctx.state.message_history.append(response)

        return CallToolsNode(response)
```

### What Happens on Cancellation

**When `CancelledError` is raised during streaming:**

1. **Lines 427-429 DON'T execute** - the response is NOT automatically added to `message_history`
2. **BUT**: The `streamed_response` object still exists in the current scope
3. You can manually call `streamed_response.get()` to retrieve the partial response with complete parts only

### The State After Cancellation

```python
# After CancelledError during streaming:
agent_run.result                        # None (run didn't complete normally)
agent_run.ctx.state.message_history     # Has all complete messages BEFORE interruption
# Note: Does NOT include the interrupted ModelResponse

# But you still have access to:
streamed_response.get()                 # The interrupted ModelResponse with complete parts
```

### Semantic Completeness Rules

A ModelResponse is **semantically complete** if:

1. **It has at least one complete part** (no ToolCallPartDeltas), AND
2. **Either:**
   - It contains NO tool calls (text/thinking only), OR
   - ALL its tool calls have executed and returned results

A ModelResponse is **semantically incomplete** if:
- It has tool calls but those tools never executed
- Even if the response finished streaming, tool calls without returns = broken state

### Examples

```python
# ✅ SEMANTICALLY COMPLETE: Text only
[
    ModelRequest(parts=[UserPromptPart("Explain quantum computing")]),
    ModelResponse(parts=[TextPart("Quantum computing uses...")]),
]

# ✅ SEMANTICALLY COMPLETE: Tool calls with returns
[
    ModelRequest(parts=[UserPromptPart("Get weather for Paris")]),
    ModelResponse(parts=[ToolCallPart(tool_name="get_weather", args={...})]),
    ModelRequest(parts=[ToolReturnPart(tool_name="get_weather", content={...})]),
]

# ❌ SEMANTICALLY INCOMPLETE: Tool calls without returns
[
    ModelRequest(parts=[UserPromptPart("Get weather for Paris")]),
    ModelResponse(parts=[ToolCallPart(tool_name="get_weather", args={...})]),
    # Missing: ToolReturnParts
]
# Agent would be confused: "I called the tool, where's the result?"

# ✅ SEMANTICALLY COMPLETE: Multiple cycles
[
    ModelRequest(parts=[UserPromptPart("Compare Paris and Berlin weather")]),
    ModelResponse(parts=[ToolCallPart(...), ToolCallPart(...)]),  # Call tools
    ModelRequest(parts=[ToolReturnPart(...), ToolReturnPart(...)]),  # Got returns
    ModelResponse(parts=[TextPart("Paris is warmer than Berlin")]),  # Final response
]
```

### The Include/Exclude Decision

When building ThreadProtocol from an interrupted run:

```typescript
function shouldIncludeInThreadProtocol(
  message: ModelMessage,
  nextMessage: ModelMessage | null
): boolean {
  if (message.kind === "request") {
    return true;  // Requests are always atomic
  }

  if (message.kind === "response") {
    // Does it have tool calls?
    const hasToolCalls = message.parts.some(p => p.part_kind === "tool-call");

    if (!hasToolCalls) {
      return true;  // Text/thinking only - semantically complete
    }

    // Has tool calls - check if next message has the returns
    if (!nextMessage || nextMessage.kind !== "request") {
      return false;  // No returns = incomplete cycle
    }

    // Check that next request has ToolReturnParts
    const hasReturns = nextMessage.parts.some(
      p => p.part_kind === "tool-return"
    );
    return hasReturns;  // Include only if we have the returns
  }
}
```

## Implementation Pattern for ThreadProtocol

### Building an Interrupted Turn

```python
from asyncio import CancelledError

async with agent.iter(prompt, message_history=prior_history) as agent_run:
    try:
        async for node in agent_run:
            if Agent.is_model_request_node(node):
                async with node.stream(agent_run.ctx) as stream_handler:
                    async for text in stream_handler.stream_text():
                        send_to_client(text)

                        if check_cancellation_signal():
                            raise CancelledError()

    except CancelledError:
        # Get messages that are in message_history
        all_messages = agent_run.ctx.state.message_history

        # Filter to only semantically complete messages
        complete_messages = filter_semantically_complete(all_messages)

        # Build ThreadProtocol turn
        interrupted_turn = {
            "turn_type": "agent",
            "agent_id": agent_id,
            "started_at": started_at,
            "completion_status": "interrupted",
            "interruption": {
                "reason": "user_cancelled",
                "interrupted_at": datetime.now().isoformat(),
            },
            "messages": convert_to_thread_messages(complete_messages)
        }

        save_thread(interrupted_turn)


def filter_semantically_complete(messages: list[ModelMessage]) -> list[ModelMessage]:
    """
    Filter to only semantically complete messages.

    A ModelResponse with tool calls is only included if the next message
    contains the corresponding ToolReturnParts.
    """
    result = []

    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            result.append(msg)

        elif isinstance(msg, ModelResponse):
            has_tool_calls = any(
                isinstance(part, ToolCallPart)
                for part in msg.parts
            )

            if not has_tool_calls:
                # Text/thinking only - include
                result.append(msg)
            else:
                # Has tool calls - check if next message has returns
                next_msg = messages[i + 1] if i + 1 < len(messages) else None

                if next_msg and isinstance(next_msg, ModelRequest):
                    has_returns = any(
                        isinstance(part, ToolReturnPart)
                        for part in next_msg.parts
                    )
                    if has_returns:
                        result.append(msg)  # Complete cycle
                # else: incomplete cycle, exclude

    return result
```

## v0.0.4 Schema Changes

### Updated AgentTurn

```typescript
interface AgentTurn {
  turn_type: "agent";
  agent_id: string;
  started_at: string;      // ISO 8601

  // All COMPLETE messages from this agent.run()
  // A message is complete if it has at least one complete part
  messages: (ThreadMessage | SystemMessage)[];

  // NEW: Completion status
  completion_status: "complete" | "interrupted";

  // NEW: Present when completion_status === "interrupted"
  interruption?: {
    reason: InterruptionReason;
    interrupted_at: string;  // ISO 8601
  };

  // Optional: only present if completion_status === "complete"
  completed_at?: string;

  total_usage?: Usage;
}

type InterruptionReason =
  | "user_cancelled"       // User explicitly stopped (e.g., hit escape)
  | "timeout"              // No events received within threshold
  | "network_failure"      // Connection lost
  | "safety_halt"          // Content filter triggered
  | "error"                // Model or system error
  | string;                // Extensible
```

### Updated Invariant

```typescript
// v0.0.3 invariant (too strict):
// "Only complete turns appear in ThreadProtocol"

// v0.0.4 invariant:
// "Only complete request/response cycles appear in ThreadProtocol"
//
// Where a complete cycle is:
// - ModelRequest (always atomic)
// - ModelResponse with complete parts only (no ToolCallPartDeltas)
// - IF ModelResponse contains tool calls, THEN:
//   - ModelRequest with corresponding ToolReturnParts MUST follow
//   - All tool calls must have executed successfully
// - IF ModelResponse has NO tool calls (text/thinking only), THEN:
//   - Cycle is complete after ModelResponse streams
//
// An AgentTurn may be interrupted, but must contain only complete cycles
```

## Client-Side Handling

### No Special Prepending Needed

Unlike Claude Code's approach (prepending "[Request interrupted by user]" to the next message), ThreadProtocol uses structured metadata:

```json
{
  "turns": [
    {
      "turn_type": "user",
      "submitted_at": "2025-01-20T10:00:00Z",
      "parts": [{"part_kind": "user-prompt", "content": "What's the weather?"}]
    },
    {
      "turn_type": "agent",
      "agent_id": "agent_001",
      "started_at": "2025-01-20T10:00:01Z",
      "completion_status": "interrupted",
      "interruption": {
        "reason": "user_cancelled",
        "interrupted_at": "2025-01-20T10:00:15Z"
      },
      "messages": [
        {
          "message_type": "response",
          "timestamp": "2025-01-20T10:00:02Z",
          "parts": [
            {"part_kind": "tool-call", "tool_name": "get_weather", "args": {...}}
          ]
        },
        {
          "message_type": "request",
          "timestamp": "2025-01-20T10:00:10Z",
          "parts": [
            {"part_kind": "tool-return", "tool_name": "get_weather", "content": {...}}
          ]
        }
        // Note: The partial final response (no complete parts) is NOT included
      ]
    },
    {
      "turn_type": "user",
      "submitted_at": "2025-01-20T10:00:20Z",
      "parts": [{"part_kind": "user-prompt", "content": "Try Berlin instead"}]
      // No special prefix needed - interruption is documented above
    }
  ]
}
```

### Reconstructing for Next agent.run()

```python
def reconstruct_message_history(thread: Thread) -> list[ModelMessage]:
    """Convert ThreadProtocol turns back to Pydantic AI message history."""
    messages = []

    for turn in thread.turns:
        if turn.turn_type == "user":
            messages.append(ModelRequest(
                parts=turn.parts,
                timestamp=turn.submitted_at
            ))
        elif turn.turn_type == "agent":
            # Include all messages regardless of completion_status
            # The interruption metadata is for human/logging purposes
            for msg in turn.messages:
                messages.append(convert_from_thread_message(msg))

    return messages
```

## Benefits Over v0.0.3

### Example: Multi-Tool Scenario

```
User: "Compare weather in Paris, Berlin, and Tokyo"

Agent executes:
✅ ModelResponse: tool calls for all 3 cities
✅ ModelRequest: tool returns for all 3 cities
❌ ModelResponse: Started saying "Based on the data..." [USER CANCELS]

v0.0.3: Discard everything, retry all 3 tool calls
v0.0.4: Keep tool calls + returns, only retry final response
```

**Savings**:
- 3 expensive tool executions preserved
- Token usage reduced (don't re-send tool results)
- User sees "I already got the data, just tell me the answer"

### Multi-Agent Handoff

```
Agent A:
✅ Completes analysis
✅ Returns structured data
❌ Started generating explanation [INTERRUPTED]

v0.0.3: Lose the analysis, Agent B has no context
v0.0.4: Agent B gets the analysis data, can continue
```

## Migration from v0.0.3

### Detection

```typescript
function getThreadVersion(thread: Thread): string {
  return thread.version;  // "0.0.3" or "0.0.4"
}

function hasInterruptions(thread: Thread): boolean {
  return thread.turns.some(t =>
    t.turn_type === "agent" && t.completion_status === "interrupted"
  );
}
```

### Backward Compatibility

v0.0.3 threads can be read as v0.0.4 by adding default fields:

```typescript
function upgradeToV4(v3_turn: AgentTurn_v3): AgentTurn_v4 {
  return {
    ...v3_turn,
    completion_status: "complete",  // All v0.0.3 turns were complete
    // interruption field omitted
  };
}
```

v0.0.4 threads can be downgraded by filtering out interrupted turns:

```typescript
function downgradeToV3(v4_turn: AgentTurn_v4): AgentTurn_v3 | null {
  if (v4_turn.completion_status === "interrupted") {
    return null;  // Discard interrupted turns for v0.0.3 compatibility
  }
  return {
    ...v4_turn,
    // Remove v0.0.4-specific fields
    completed_at: v4_turn.completed_at,
  };
}
```

## Open Questions

1. **Nested interruptions**: If an agent.run() calls another agent, and both are interrupted, each AgentTurn tracks its own interruption independently

2. **Usage accounting**: Interrupted responses still consumed tokens and should be included in total_usage

3. **Partial tool call arguments**: If streaming tool call args and user cancels mid-args, the ToolCallPartDelta is excluded by `get_parts()` (as designed)

## References

### Pydantic AI Code Paths

- `pydantic_ai/_parts_manager.py:62-68` - `get_parts()` filtering logic (excludes ToolCallPartDeltas)
- `pydantic_ai/_agent_graph.py:399-431` - `ModelRequestNode.stream()` lifecycle
- `pydantic_ai/_agent_graph.py:447` - ModelRequest added to message_history
- `pydantic_ai/_agent_graph.py:495` - ModelResponse added to message_history
- `pydantic_ai/_agent_graph.py:427-429` - Response finalization (skipped on cancel)
- `pydantic_ai/_agent_graph.py:505-673` - `CallToolsNode` processes tool calls
- `pydantic_ai/_agent_graph.py:579-607` - Tool calls collected from completed response
- `pydantic_ai/_agent_graph.py:641-672` - `_handle_tool_calls()` executes tools
- `pydantic_ai/_agent_graph.py:732-746` - `process_tool_calls()` runs tools in parallel
- `pydantic_ai/models/__init__.py:558-569` - `StreamedResponse.get()` method
- `pydantic_ai/run.py:123-138` - `AgentRun.result` property

### Key Insights from Discussion

1. LLM APIs allow consecutive user messages (no "alternation violation")
2. Interruption is a client-side concern, not a protocol concern
3. Pydantic AI already implements "complete parts only" filtering
4. CancelledError doesn't prevent access to partial state
5. **Tool calls are declarative, not imperative** - they execute AFTER ModelResponse completes
6. The atomic unit is the complete cycle: Request → Response → (optionally) ToolReturns
7. Including tool calls without returns creates semantically invalid message history
8. ThreadProtocol should mirror what actually happened, not what was intended

## Conclusion

v0.0.4's "complete request/response cycles" approach:
- **Maintains semantic integrity**: Never saves tool calls without their returns
- **Preserves maximum useful work**: Keeps all complete cycles before interruption
- **Aligns with Pydantic AI's execution model**: Respects the graph node boundaries
- **Provides structured metadata**: `interruption` object instead of text prepending
- **Enables better resumption**: Agent sees valid history, can continue from last complete cycle
- **Maintains deterministic bijection**: Server and client share identical understanding of state

### The Key Trade-off

v0.0.4 is **more conservative** than initially proposed:
- We DON'T save: ModelResponses with tool calls if tools didn't execute
- We DO save: ModelResponses with text/thinking only
- We DO save: Complete cycles where tools executed and returned

This ensures the message history is always **semantically valid** - something the LLM can meaningfully continue from.
