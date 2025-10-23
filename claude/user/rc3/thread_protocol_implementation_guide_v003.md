# ThreadProtocol v0.0.3 Implementation Guide

## Overview

This guide covers implementation concerns for ThreadProtocol v0.0.3 that are not part of the core specification, particularly around streaming, interruption handling, and maintaining protocol consistency between server and client.

## Key Principle

**ThreadProtocol contains only complete turns. Streaming state and partial turns are implementation concerns, not protocol concerns.**

## Working with Extensions

ThreadProtocol provides clear extension points while protecting canonical semantics.

### Custom System Events

Use `data-*` events for application-specific signals:

```typescript
// Record a routing decision
{
  message_type: "system",
  timestamp: "2025-01-20T10:00:00Z",
  event_type: "data-routing-decision",
  event_data: {
    selected_agent: "specialist_42",
    confidence: 0.87,
    alternatives: ["generalist_01", "specialist_19"]
  }
}

// Record embeddings generation
{
  message_type: "system",
  timestamp: "2025-01-20T10:00:05Z",
  event_type: "data-embeddings-generated",
  event_data: {
    chunks: 15,
    model: "text-embedding-3-small",
    vector_db_ids: ["vec_001", "vec_002", "..."]
  }
}

// Track parallel thread spawning
{
  message_type: "system",
  timestamp: "2025-01-20T10:00:10Z",
  event_type: "data-thread-spawned",
  event_data: {
    child_thread_id: "550e8400-e29b-41d4-a716-446655440099",
    execution_mode: "parallel",
    timeout_ms: 30000
  }
}
```

### Custom Message Parts

When you need parts beyond the normative set:

```typescript
// Not in normative spec, so use custom: namespace
{
  part_kind: "custom:planning-step",
  step_id: "plan_42",
  action: "research",
  dependencies: ["plan_40", "plan_41"],
  estimated_cost: 1500  // tokens
}

// Vector database interaction
{
  part_kind: "custom:vector-search",
  query_vector: [...],
  top_k: 10,
  similarity_threshold: 0.8,
  result_ids: ["doc_1", "doc_2", "doc_3"]
}
```

### Implementation Metadata

Use `meta:*` for data that shouldn't affect canonical hashing:

```typescript
{
  message_type: "system",
  timestamp: "2025-01-20T10:00:00Z",
  event_type: "meta:performance",
  event_data: {
    ttfb: 150,  // ms
    cache_hit: true,
    region: "us-east-1",
    model_latency: 245
  }
}

// Retry metadata
{
  message_type: "system",
  timestamp: "2025-01-20T10:00:05Z",
  event_type: "meta:retry-info",
  event_data: {
    attempt: 2,
    max_attempts: 3,
    backoff_ms: 1000,
    reason: "rate_limit"
  }
}
```

### Handling Unknown Extensions

Implementations must handle unrecognized content gracefully:

```typescript
function processMessage(msg: ThreadMessage) {
  for (const part of msg.parts) {
    switch (part.part_kind) {
      case "text":
      case "thinking":
      case "tool-call":
      case "tool-return":
      case "retry-prompt":
      case "user-prompt":
      case "file":
        // Normative handling - process as defined
        handleNormativePart(part);
        break;

      default:
        // Unknown/custom part - preserve but don't process
        if (part.part_kind.startsWith("custom:")) {
          console.debug("Skipping custom part:", part.part_kind);
        } else if (part.part_kind.startsWith("meta:")) {
          console.debug("Skipping metadata part:", part.part_kind);
        } else {
          console.warn("Unknown part kind (preserving):", part.part_kind);
        }
        // Store unchanged for round-trip fidelity
        preserveUnknownPart(part);
        break;
    }
  }
}

function processSystemMessage(msg: SystemMessage) {
  // Check if it's a normative event type
  const normativeEvents = [
    "agent.handoff",
    "thread.spawn",
    "thread.merge",
    "thread.end",
    "error"
  ];

  if (normativeEvents.includes(msg.event_type)) {
    handleNormativeEvent(msg);
  } else if (msg.event_type.startsWith("data-")) {
    // Custom domain event - application decides how to handle
    handleCustomEvent(msg);
  } else if (msg.event_type.startsWith("meta:")) {
    // Metadata - typically for logging/observability
    logMetadata(msg);
  } else {
    console.warn("Unknown event type (preserving):", msg.event_type);
    preserveUnknownEvent(msg);
  }
}
```

### Best Practices for Extensions

1. **Always use namespaces**: `data-*`, `custom:*`, or `meta:*`
2. **Document your extensions**: Maintain a registry of custom event/part types
3. **Don't override normative types**: Never repurpose `text`, `tool-call`, etc.
4. **Preserve unknown content**: Enable forward compatibility
5. **Keep extensions JSON-serializable**: Avoid circular references or functions

## Streaming Architecture

### Two-Layer Model

```
┌─────────────────────────────────────────────┐
│            Application Layer                 │
│  (Handles streaming, partial state, UI)      │
├─────────────────────────────────────────────┤
│           ThreadProtocol Layer               │
│    (Canonical, complete turns only)          │
└─────────────────────────────────────────────┘
```

### Server-Side Streaming State

The server maintains two representations during streaming:

1. **Streaming State** (in-memory only):
```typescript
interface StreamingTurn {
  turn_type: "agent";
  agent_id: string;
  started_at: string;

  // Accumulating parts as events arrive
  partial_messages: ThreadMessage[];
  current_text_buffer?: string;
  current_tool_calls: Map<string, Partial<ToolCallPart>>;

  // Streaming metadata
  stream_id: string;
  last_event_at: string;
  completion_status: "streaming" | "complete" | "interrupted";
  interruption_reason?: string;
}
```

2. **Canonical ThreadProtocol** (persistent):
- Only contains AgentTurns where `completion_status === "complete"`
- Updated atomically when streaming completes

### Client-Side Streaming State

The client similarly maintains:

1. **Active Streaming State**:
```typescript
interface ClientStreamingState {
  pending_turn?: {
    parts: MessagePart[];
    incomplete_parts: Map<string, IncompleteStreamPart>;
    last_event_received: Date;
    stream_status: "active" | "stalled" | "ended";
  };

  // Canonical history (matches server exactly)
  thread: Thread;
}

interface IncompleteStreamPart {
  type: "text" | "thinking" | "tool-call";
  id: string;
  accumulated_content: string;
  started_at: Date;
  has_end_event: boolean;
}
```

2. **Canonical ThreadProtocol**:
- Identical to server's ThreadProtocol for all complete turns

## Handling Streaming Events

### Event Processing Pipeline

```typescript
// Server-side (from Pydantic AI events)
function processModelEvent(event: ModelEvent, streamingTurn: StreamingTurn) {
  switch (event.type) {
    case "text":
      streamingTurn.current_text_buffer += event.content;
      break;

    case "tool_call":
      streamingTurn.current_tool_calls.set(event.id, {
        tool_name: event.name,
        tool_call_id: event.id,
        args: event.args
      });
      break;

    case "completion":
      // Finalize and move to canonical ThreadProtocol
      const completeTurn = finalizeStreamingTurn(streamingTurn);
      thread.turns.push(completeTurn);
      break;
  }
}

// Client-side (from SSE stream)
function processStreamEvent(event: StreamEvent, clientState: ClientStreamingState) {
  if (!clientState.pending_turn) {
    clientState.pending_turn = initializePendingTurn();
  }

  switch (event.type) {
    case "text-start":
      clientState.pending_turn.incomplete_parts.set(event.id, {
        type: "text",
        id: event.id,
        accumulated_content: "",
        started_at: new Date(),
        has_end_event: false
      });
      break;

    case "text-delta":
      const part = clientState.pending_turn.incomplete_parts.get(event.id);
      if (part) {
        part.accumulated_content += event.delta;
      }
      break;

    case "text-end":
      const completePart = clientState.pending_turn.incomplete_parts.get(event.id);
      if (completePart) {
        completePart.has_end_event = true;
        // Convert to canonical TextPart
        clientState.pending_turn.parts.push({
          part_kind: "text",
          content: completePart.accumulated_content,
          id: event.id
        });
        clientState.pending_turn.incomplete_parts.delete(event.id);
      }
      break;

    case "finish":
      // Turn is complete - add to canonical ThreadProtocol
      const completeTurn = finalizePendingTurn(clientState.pending_turn);
      clientState.thread.turns.push(completeTurn);
      clientState.pending_turn = undefined;
      break;
  }
}
```

## Interruption Handling

### Types of Interruptions

1. **User Cancellation**: User explicitly stops generation
2. **Network Failure**: Connection lost during streaming
3. **Timeout**: No events received within threshold
4. **Safety Halt**: Content filter or safety system triggered
5. **Error**: Model or system error during generation

### Detection Strategies

#### Server-Side Detection

```typescript
class ServerStreamMonitor {
  detectInterruption(streamingTurn: StreamingTurn): InterruptionType | null {
    // User cancellation (from API request)
    if (this.cancellationRequested(streamingTurn.stream_id)) {
      return "user_cancelled";
    }

    // Timeout (no events for X seconds)
    const timeSinceLastEvent = Date.now() - Date.parse(streamingTurn.last_event_at);
    if (timeSinceLastEvent > STREAM_TIMEOUT_MS) {
      return "timeout";
    }

    // Model-reported completion/error
    if (streamingTurn.model_finish_reason === "safety_filter") {
      return "safety_halt";
    }

    return null;
  }
}
```

#### Client-Side Detection

```typescript
class ClientStreamMonitor {
  detectInterruption(pendingTurn: PendingTurn): InterruptionType | null {
    // Network failure (EventSource closed unexpectedly)
    if (this.eventSource.readyState === EventSource.CLOSED && !pendingTurn.has_finish) {
      return "network_failure";
    }

    // Timeout (no events received)
    const timeSinceLastEvent = Date.now() - pendingTurn.last_event_received.getTime();
    if (timeSinceLastEvent > CLIENT_TIMEOUT_MS) {
      return "timeout";
    }

    // Incomplete parts (start without end)
    const incompleteParts = Array.from(pendingTurn.incomplete_parts.values())
      .filter(p => !p.has_end_event);
    if (incompleteParts.length > 0 && this.streamEnded) {
      return "incomplete_stream";
    }

    return null;
  }
}
```

### Recovery Strategies

#### Strategy 1: Discard Partial Turn (Recommended)

```typescript
function handleInterruption(clientState: ClientStreamingState, interruption: InterruptionType) {
  // Log the interruption for debugging
  console.warn(`Stream interrupted: ${interruption}`, clientState.pending_turn);

  // Discard the partial turn entirely
  clientState.pending_turn = undefined;

  // UI shows the last complete turn
  // ThreadProtocol remains unchanged (already has only complete turns)

  // Optionally notify user
  showNotification(`Generation was interrupted (${interruption}). You can retry from here.`);
}
```

#### Strategy 2: Save Partial Content as System Message

```typescript
function handleInterruptionWithPartialSave(
  thread: Thread,
  partialTurn: StreamingTurn,
  interruption: InterruptionType
) {
  // Add a system message recording the interruption
  const lastCompleteTurn = thread.turns[thread.turns.length - 1];

  if (lastCompleteTurn.turn_type === "agent") {
    lastCompleteTurn.messages.push({
      message_type: "system",
      timestamp: new Date().toISOString(),
      event_type: "stream.interrupted",
      event_data: {
        reason: interruption,
        partial_content: extractPartialContent(partialTurn),
        stream_id: partialTurn.stream_id
      }
    });
  }
}
```

#### Strategy 3: Convert to Error Turn (Not Recommended)

This violates the "complete turns only" principle but shown for completeness:

```typescript
// DON'T DO THIS - violates ThreadProtocol principles
function handleInterruptionAsErrorTurn(thread: Thread, partialTurn: StreamingTurn) {
  // Create a "synthetic" complete turn with error
  thread.turns.push({
    turn_type: "agent",
    agent_id: partialTurn.agent_id,
    started_at: partialTurn.started_at,
    completed_at: new Date().toISOString(),
    messages: [
      ...partialTurn.partial_messages,
      {
        message_type: "system",
        timestamp: new Date().toISOString(),
        event_type: "error",
        event_data: {
          error: "Stream interrupted",
          partial: true  // Marks this as synthetic
        }
      }
    ]
  });
}
```

## Synchronization Guarantees

### Ensuring Client-Server Consistency

Both client and server follow these rules to ensure identical ThreadProtocol:

1. **Atomic Turn Addition**: Turns are added to ThreadProtocol only when complete
2. **Deterministic Part Construction**: Same events → same parts
3. **Timestamp Alignment**: Use event timestamps, not local time
4. **No Partial State in Protocol**: Incomplete parts never enter ThreadProtocol

### Verification Mechanism

```typescript
// Periodic consistency check
async function verifyConsistency(
  clientThread: Thread,
  serverEndpoint: string
): Promise<boolean> {
  const serverThread = await fetch(`${serverEndpoint}/thread/${clientThread.thread_id}`);

  // Compare canonical representations
  const clientHash = hashThread(clientThread);
  const serverHash = hashThread(serverThread);

  if (clientHash !== serverHash) {
    console.error("Thread divergence detected", {
      client: clientThread,
      server: serverThread,
      lastMatchingTurn: findLastMatchingTurn(clientThread, serverThread)
    });
    return false;
  }

  return true;
}

function hashThread(thread: Thread): string {
  // Hash only complete turns, ignore any streaming state
  const canonical = {
    version: thread.version,
    turns: thread.turns.filter(t =>
      t.turn_type === "user" ||
      (t.turn_type === "agent" && !isStreamingTurn(t))
    )
  };

  return sha256(JSON.stringify(canonical));
}
```

## Timeout Configuration

### Recommended Timeouts

```typescript
const TIMEOUTS = {
  // Server-side
  SERVER_STREAM_TIMEOUT_MS: 30000,        // 30s - for slow model responses
  SERVER_IDLE_TIMEOUT_MS: 5000,           // 5s - between events

  // Client-side
  CLIENT_STREAM_TIMEOUT_MS: 35000,        // 35s - slightly higher than server
  CLIENT_RECONNECT_DELAY_MS: 1000,        // 1s - before reconnection attempt
  CLIENT_MAX_RECONNECT_ATTEMPTS: 3,       // Maximum reconnection attempts

  // Heartbeat (keep-alive)
  HEARTBEAT_INTERVAL_MS: 15000,           // 15s - SSE ping events
};
```

### Heartbeat Implementation

```typescript
// Server sends periodic ping events
class StreamHeartbeat {
  private interval: NodeJS.Timer;

  start(stream: ServerStream) {
    this.interval = setInterval(() => {
      stream.write("event: ping\ndata: \n\n");
    }, TIMEOUTS.HEARTBEAT_INTERVAL_MS);
  }

  stop() {
    clearInterval(this.interval);
  }
}

// Client monitors heartbeat
class ClientHeartbeatMonitor {
  private lastPing = Date.now();

  handleEvent(event: MessageEvent) {
    if (event.type === "ping") {
      this.lastPing = Date.now();
      return;
    }
    // Handle other events...
  }

  isAlive(): boolean {
    return Date.now() - this.lastPing < TIMEOUTS.HEARTBEAT_INTERVAL_MS * 2;
  }
}
```

## Client Metadata Handling

### Server-Side Processing

```typescript
function processUserTurn(userTurn: UserTurn): AgentRoutingDecision {
  const metadata = userTurn.client_metadata || {};

  // Parse routing hints
  const routing: AgentRoutingDecision = {
    preferred_agent: metadata["agent.preferred"],
    mode: metadata["ui:mode"],
    session_id: metadata["session.id"],
    max_turns: metadata["limits.max_turns"],

    // Custom application logic
    features: extractFeatureFlags(metadata, "app/")
  };

  // Validate namespace usage
  for (const key of Object.keys(metadata)) {
    if (!hasNamespace(key)) {
      console.warn(`Unnamespaced metadata key: ${key}`);
    }
  }

  return routing;
}

function hasNamespace(key: string): boolean {
  // Check for common separators
  return /[:\.\/_-]/.test(key);
}
```

### Client Metadata Best Practices

1. **Always use namespaces** to prevent collisions
2. **Document your namespaces** in application code
3. **Don't send sensitive data** in client_metadata
4. **Keep values JSON-serializable**
5. **Use metadata for routing**, not conversation content

## ContentRef Implementation

### Server-Side Storage

```typescript
class ContentStorage {
  async storeToolReturn(
    toolReturn: any,
    sizeThreshold: number = 100 * 1024  // 100KB
  ): Promise<ToolReturnPart> {
    const serialized = JSON.stringify(toolReturn);

    // Use inline for small content
    if (serialized.length < sizeThreshold) {
      return {
        part_kind: "tool-return",
        tool_name: "...",
        tool_call_id: "...",
        status: "success",
        content: toolReturn
      };
    }

    // Store externally for large content
    const uri = await this.uploadToS3(serialized);
    const hash = sha256(serialized);

    return {
      part_kind: "tool-return",
      tool_name: "...",
      tool_call_id: "...",
      status: "success",
      content_ref: {
        uri,
        size_bytes: serialized.length,
        hash,
        media_type: "application/json"
      },
      metadata: {
        preview: generatePreview(toolReturn)
      }
    };
  }
}
```

### Client-Side Resolution

```typescript
class ContentResolver {
  async resolveToolReturn(part: ToolReturnPart): Promise<any> {
    // Use inline content if available
    if (part.content !== undefined) {
      return part.content;
    }

    // Fetch external content
    if (part.content_ref) {
      const content = await this.fetchContent(part.content_ref);

      // Verify integrity
      if (part.content_ref.hash) {
        const actualHash = sha256(content);
        if (actualHash !== part.content_ref.hash) {
          throw new Error("Content integrity check failed");
        }
      }

      return JSON.parse(content);
    }

    return null;
  }

  private async fetchContent(ref: ContentRef): Promise<string> {
    const url = this.resolveUri(ref.uri);
    const response = await fetch(url);
    return response.text();
  }
}
```

## Error Handling Patterns

### Graceful Degradation

```typescript
class StreamErrorHandler {
  handle(error: StreamError): StreamRecoveryAction {
    switch (error.severity) {
      case "fatal":
        // Can't recover - show error to user
        return {
          action: "abort",
          message: "Generation failed. Please try again.",
          preserveHistory: true
        };

      case "recoverable":
        // Can retry
        return {
          action: "retry",
          delay: 1000,
          maxAttempts: 3,
          preserveHistory: true
        };

      case "partial":
        // Some content received
        return {
          action: "use_partial",
          message: "Generation incomplete. Partial response shown.",
          preserveHistory: false  // Don't save partial turn
        };
    }
  }
}
```

## Testing Considerations

### Simulating Interruptions

```typescript
class StreamInterruptionSimulator {
  // Simulate various interruption scenarios for testing

  async simulateTimeout(afterMs: number) {
    await delay(afterMs);
    this.closeStream();
  }

  async simulateNetworkError(afterEvents: number) {
    this.eventCount = 0;
    this.onEvent = () => {
      if (++this.eventCount >= afterEvents) {
        throw new Error("Network error");
      }
    };
  }

  async simulateIncompleteStream() {
    // Send start events but no end events
    this.sendEvent({ type: "text-start", id: "text_1" });
    this.sendEvent({ type: "text-delta", id: "text_1", delta: "Hello" });
    // Missing text-end
    this.closeStream();
  }
}
```

### Consistency Testing

```typescript
describe("ThreadProtocol Consistency", () => {
  it("should generate identical protocol from server and client events", async () => {
    // Server-side events (Pydantic AI)
    const serverEvents = [
      { type: "model_response", parts: [{ part_kind: "text", content: "Hello" }] },
      { type: "tool_call", tool_name: "search", args: { q: "weather" } },
      { type: "tool_return", content: { results: ["sunny"] } },
      { type: "model_response", parts: [{ part_kind: "text", content: "It's sunny" }] },
      { type: "completion" }
    ];

    // Client-side events (SSE stream)
    const clientEvents = [
      { type: "text-start", id: "1" },
      { type: "text-delta", id: "1", delta: "Hello" },
      { type: "text-end", id: "1" },
      { type: "tool-input-available", toolCallId: "tc1", toolName: "search", input: { q: "weather" } },
      { type: "tool-output-available", toolCallId: "tc1", output: { results: ["sunny"] } },
      { type: "text-start", id: "2" },
      { type: "text-delta", id: "2", delta: "It's sunny" },
      { type: "text-end", id: "2" },
      { type: "finish" }
    ];

    const serverThread = processServerEvents(serverEvents);
    const clientThread = processClientEvents(clientEvents);

    expect(serverThread).toEqual(clientThread);
  });

  it("should handle interruption identically", async () => {
    // Both should discard partial turn
    const partialServerEvents = [
      { type: "model_response", parts: [{ part_kind: "text", content: "Hello" }] }
      // Missing completion
    ];

    const partialClientEvents = [
      { type: "text-start", id: "1" },
      { type: "text-delta", id: "1", delta: "Hello" }
      // Missing text-end and finish
    ];

    const serverThread = processServerEvents(partialServerEvents);
    const clientThread = processClientEvents(partialClientEvents);

    // Both should have no turns (partial turn discarded)
    expect(serverThread.turns).toHaveLength(0);
    expect(clientThread.turns).toHaveLength(0);
  });
});
```

## Summary

This implementation guide provides patterns and strategies for:

1. **Maintaining streaming state** separate from canonical ThreadProtocol
2. **Detecting interruptions** on both server and client
3. **Recovering gracefully** from incomplete streams
4. **Ensuring consistency** between server and client representations
5. **Handling client metadata** with proper namespacing
6. **Managing external content** via ContentRef
7. **Testing** for consistency and interruption scenarios

The key principle throughout: **ThreadProtocol contains only complete turns**, while streaming concerns are handled at the application layer.