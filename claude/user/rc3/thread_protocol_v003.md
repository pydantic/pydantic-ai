# ThreadProtocol Specification v0.0.3

## Overview

ThreadProtocol is a canonical JSON representation for multi-agent conversation threads that:
- Faithfully serializes Pydantic AI ModelMessages with minimal transformation
- Adds multi-agent attribution and turn boundaries
- Supports system-level events outside Pydantic AI's scope
- Enables deterministic round-trip conversion between server (Pydantic AI) and client (Vercel AI SDK)
- **New in v0.0.3**: Enforces complete-turn semantics, adds client metadata, content references, and thread relationships

## Core Design Principles

1. **Follow Pydantic AI's Lead**: Reuse their proven abstractions for LLM primitives
2. **Minimal Transformation**: Store what actually happened, not an interpretation
3. **Multi-Agent Native**: Agent attribution without text manipulation
4. **Natural Ordering**: Use array position and timestamps, no artificial numbering
5. **Extensible for System Events**: Clean separation between LLM and system concerns
6. **Complete Turns Only**: Only fully-streamed turns appear in the canonical representation
7. **Deterministic Bijection**: Server and client generate identical ThreadProtocol from their respective event streams

## Conformance and Extensibility

### Normative Core

The following elements are normatively defined by ThreadProtocol and MUST be interpreted consistently by all conforming implementations:

**Turn Structure:**
- `UserTurn`, `AgentTurn` types and their required fields
- Turn lifecycle semantics (complete turns only)

**Message Part Kinds (Normative):**
- `text` - Text content from model
- `thinking` - Model reasoning (extended thinking)
- `tool-call` - Tool invocation request
- `tool-return` - Tool execution result
- `retry-prompt` - Validation error retry
- `user-prompt` - User input
- `file` - Binary/media content

**System Event Types (Normative):**
- `agent.handoff` - Control passed between agents
- `thread.spawn` - Thread created from this thread
- `thread.merge` - Thread incorporated into this thread
- `thread.end` - Thread terminated
- `error` - Error condition

Implementations MUST NOT redefine or repurpose these symbols. Canonical hashing depends on their deterministic meaning.

### Extensible Surface

All other event types and metadata are extensible. Developers MAY introduce custom content within these namespaces:

**Extension Namespaces:**
- `data-*` - Domain-specific events (mirrors Vercel AI SDK pattern)
  - Example: `data-embeddings`, `data-user-score`, `data-function-trace`
- `custom:*` - Application-defined part kinds
  - Example: `custom:planning-graph`, `custom:vector-store-call`
- `meta:*` - Implementation metadata (MAY be excluded from canonicalization)
  - Example: `meta:cache-hit`, `meta:retry-count`

**Extension Rules:**
1. Unrecognized event types MUST be preserved during serialization
2. Implementations MUST NOT reject threads containing valid extensions
3. Canonical hashing includes all parts unchanged (except `meta:*` if implementation excludes it)
4. Custom extensions do not alter normative turn/part semantics

### Philosophy

ThreadProtocol records **what happened** in a conversation, not **why or when** it should happen.

The protocol assumes—but does not require—multi-agent capability. Scheduling, concurrency, orchestration, and agent routing are implementation-defined concerns that MAY be expressed through extension events (e.g., `data-routing-decision`) but are not part of this specification.

Thread relationships (`spawned_from`, `merged_from`, `referenced`) describe linkage, not execution model. Whether spawned threads run in parallel, block the parent, or execute conditionally is an implementation detail.

## Schema Structure

### Thread (Root Object)

```typescript
interface Thread {
  version: "0.0.3";          // Protocol version
  thread_id: string;         // UUID
  created_at: string;        // ISO 8601
  updated_at: string;        // ISO 8601

  // Human-readable metadata
  title?: string;
  metadata?: Record<string, any>;

  // Agent registry
  agents: Record<string, AgentConfig>;

  // The conversation as a sequence of turns
  turns: Turn[];

  // NEW in v0.0.3: Thread relationships (outgoing links only)
  relationships?: ThreadRelationships;
}
```

### Agent Configuration

```typescript
interface AgentConfig {
  agent_id: string;          // UUID - globally unique
  agent_name: string;        // Human-readable name
  model_name?: string;       // e.g., "gpt-4", "claude-3-opus"
  provider_name?: string;    // e.g., "openai", "anthropic"
  created_at: string;        // ISO 8601

  // System prompt and tools configured elsewhere
  config_ref?: string;       // External reference
}
```

### Turns

```typescript
type Turn = UserTurn | AgentTurn;

interface UserTurn {
  turn_type: "user";
  submitted_at: string;      // ISO 8601 - when user submitted

  // What the user actually submitted
  // This effectively serves as the initial ModelRequest
  parts: MessagePart[];      // Usually just UserPromptPart

  // NEW in v0.0.3: Structured directives from the client
  // Keys MUST contain a namespace separator (: . / _ -)
  // Examples: "ui:mode", "agent.routing", "app/feature", "session_id"
  client_metadata?: Record<string, any>;
}

interface AgentTurn {
  turn_type: "agent";
  agent_id: string;          // Which agent's turn
  started_at: string;        // ISO 8601 - agent.run() start
  completed_at: string;      // ISO 8601 - agent.run() end

  // All messages from this agent.run()
  messages: (ThreadMessage | SystemMessage)[];

  // Aggregated usage for the entire turn
  total_usage?: Usage;

  // IMPORTANT: AgentTurns in ThreadProtocol are ALWAYS complete
  // Partial/interrupted turns do not appear in the canonical representation
}
```

### ThreadMessage (Maps to Pydantic AI ModelMessage)

```typescript
interface ThreadMessage {
  message_type: "request" | "response";
  timestamp: string;         // ISO 8601

  // Nearly direct mapping to Pydantic AI's parts
  parts: MessagePart[];

  // Multi-agent attribution (not in PAI)
  agent_id: string;          // Which agent generated this

  // For responses (from PAI's ModelResponse)
  model_name?: string;
  provider_name?: string;
  provider_response_id?: string;
  usage?: Usage;
  finish_reason?: FinishReason;
}
```

### Message Parts (Direct from Pydantic AI)

```typescript
// These match Pydantic AI's part types
// NOTE: SystemPromptPart is NOT stored in ThreadProtocol
type MessagePart =
  | UserPromptPart
  | TextPart
  | ThinkingPart
  | ToolCallPart
  | ToolReturnPart
  | RetryPromptPart
  | FilePart;

interface TextPart {
  part_kind: "text";
  content: string;
  id?: string;
}

interface ThinkingPart {
  part_kind: "thinking";
  content?: string;          // Visible thinking text
  signature?: string;        // Opaque provider reference
  provider_name: string;     // Critical for round-tripping
  thinking_id?: string;
}

interface ToolCallPart {
  part_kind: "tool-call";
  tool_name: string;
  tool_call_id: string;
  args: any;                 // Tool arguments
}

interface ToolReturnPart {
  part_kind: "tool-return";
  tool_name: string;
  tool_call_id: string;      // Correlates with call
  status: "success" | "error" | "validation_error";
  content?: any;             // Return value or error (inline)

  // NEW in v0.0.3: Reference for large/external content
  content_ref?: ContentRef;

  metadata?: any;            // Additional context
}

interface RetryPromptPart {
  part_kind: "retry-prompt";
  content: string | ValidationError[];
  tool_name?: string;
  tool_call_id?: string;
}

interface FilePart {
  part_kind: "file";
  content: BinaryContent;    // Image or other file
  id?: string;
}

interface UserPromptPart {
  part_kind: "user-prompt";
  content: string | UserContent[];
}
```

### Content Reference (NEW in v0.0.3)

```typescript
// For referencing large or external content instead of inlining
interface ContentRef {
  uri: string;              // e.g. "s3://bucket/key", "https://..."
  size_bytes?: number;      // Size of referenced content
  hash?: string;            // SHA-256 of content for verification
  media_type?: string;      // MIME type
}
```

### SystemMessage (Outside Pydantic AI's Scope)

```typescript
interface SystemMessage {
  message_type: "system";
  timestamp: string;

  // Extensible event system
  event_type: SystemEventType;
  event_data: any;           // Event-specific payload

  // Multi-agent context
  source_agent?: string;     // Which agent triggered this
  target_agents?: string[];  // Which agents are affected
}

// Examples of system events (extensible)
type SystemEventType =
  | "agent.handoff"          // Control passed between agents
  | "tool.grant"             // Dynamic tool permission
  | "tool.revoke"
  | "state.update"           // Application state change
  | "ui.update"              // UI-specific event
  | "context.switch"         // Context or mode change
  | string;                  // Extensible
```

### Thread Relationships (NEW in v0.0.3)

```typescript
// Outgoing relationships only (append-only friendly)
// Incoming relationships (e.g., "parent thread") are part of turn 0 configuration
interface ThreadRelationships {
  links?: ThreadLink[];      // Outgoing references
}

interface ThreadLink {
  thread_id: string;         // Target thread UUID
  relation: "spawned_from" | "merged_from" | "referenced" | string;
  metadata?: any;            // Relationship-specific data
}
```

### Supporting Types

```typescript
interface Usage {
  input_tokens?: number;
  output_tokens?: number;
  thinking_tokens?: number;
  total_tokens?: number;
}

type FinishReason = "stop" | "length" | "content_filter" | "tool_call" | "error";

// Multi-modal content types (matching PAI's structure exactly)
type UserContent = string | MultiModalContent;
type MultiModalContent = ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent;

interface ImageUrl {
  kind: "image-url";
  url: string;
  identifier: string;         // For tool references
  force_download?: boolean;   // Whether to download vs pass URL
  vendor_metadata?: any;      // Provider-specific metadata
  media_type?: string;        // Inferred from URL if not provided
}

interface AudioUrl {
  kind: "audio-url";
  url: string;
  identifier: string;
  force_download?: boolean;
  vendor_metadata?: any;
  media_type?: string;
}

interface VideoUrl {
  kind: "video-url";
  url: string;
  identifier: string;
  force_download?: boolean;
  vendor_metadata?: any;
  media_type?: string;
}

interface DocumentUrl {
  kind: "document-url";
  url: string;
  identifier: string;
  force_download?: boolean;
  vendor_metadata?: any;
  media_type?: string;
}

interface BinaryContent {
  kind: "binary";
  data: string;               // base64 encoded
  media_type: string;         // MIME type
  identifier: string;         // For tool references
  vendor_metadata?: any;      // Provider-specific metadata
}

interface Attachment {
  name: string;
  url?: string;
  data?: string;             // base64
  media_type: string;
  size_bytes?: number;
}
```

## Important: Complete Turns Only

ThreadProtocol v0.0.3 enforces a critical invariant:

**Only complete turns appear in the canonical Thread representation.**

This means:
- If a turn starts streaming but doesn't complete, it does NOT appear in ThreadProtocol
- Partial turns exist only in application memory during streaming
- The canonical Thread always represents a valid, complete conversation history
- Both server and client can deterministically generate identical ThreadProtocol from complete turns

When reconstructing the conversation:
- The last turn in ThreadProtocol is always complete
- Interrupted/partial turns are an implementation concern (see Implementation Guide)
- Recovery from interruption means continuing from the last complete turn

## Event Stream to ThreadProtocol Bijection

### Vercel AI SDK Stream Events → ThreadProtocol Parts

The following mapping ensures deterministic ThreadProtocol generation from SSE events:

| Stream Event Type | ThreadProtocol Part | Notes |
|------------------|-------------------|--------|
| `message-start` | New AgentTurn begins | Store message ID |
| `text-start` + `text-delta` + `text-end` | TextPart | Concatenate deltas |
| `reasoning-start` + `reasoning-delta` + `reasoning-end` | ThinkingPart | Include provider_name |
| `tool-input-start` + `tool-input-delta` + `tool-input-available` | ToolCallPart | Parse final args |
| `tool-output-available` | ToolReturnPart | Include status |
| `source-url`, `source-document` | SystemMessage (event_type: "source.*") | Preserve metadata |
| `file` | FilePart | With BinaryContent |
| `data-*` | SystemMessage (event_type: custom) | Extensible |
| `error` | SystemMessage | Depends on context |
| `finish-step` | Complete current message | Add to turn |
| `finish` | Complete AgentTurn | Turn is now canonical |

### Critical Rules

1. **Incomplete Sequences**: If a `*-start` event has no corresponding `*-end`, the entire part is discarded
2. **Turn Completion**: Only when `finish` is received does the turn become canonical
3. **Atomic Turns**: Either all parts of a turn are included, or none are
4. **Timestamp Alignment**: Use event timestamps when available, otherwise use receipt time

## UserTurn Client Metadata

### Namespace Requirements

The `client_metadata` field in UserTurn MUST use namespaced keys to prevent collisions and clarify ownership.

**Valid namespace separators**: `:` `.` `/` `_` `-`

**Examples**:
```json
{
  "ui:mode": "creative",
  "agent.routing": "specialist",
  "app/feature": "enabled",
  "session_id": "abc123",
  "model-config": { "temperature": 0.7 }
}
```

**Validation**: Keys without a namespace separator SHOULD generate a warning (but not error) to encourage good practices.

### Common Namespaces (Recommendations)

- `ui:*` - UI state and preferences
- `agent.*` - Agent routing and configuration
- `model.*` - Model-specific parameters
- `session.*` - Session management
- `app/*` - Application-specific features

## ContentRef Usage

### When to Use ContentRef

Use `content_ref` instead of inline `content` when:
- Tool return data exceeds 100KB
- Content is already stored externally (S3, CDN)
- Content needs to be shared across multiple threads
- Binary data that shouldn't be base64 encoded in JSON

### URI Schemes

Supported URI schemes:
- `https://` - Public HTTP(S) URLs
- `s3://` - Amazon S3 (bucket/key format)
- `gs://` - Google Cloud Storage
- `azure://` - Azure Blob Storage
- `file://` - Local filesystem (development only)
- Custom schemes allowed with documentation

### Example

```json
{
  "part_kind": "tool-return",
  "tool_name": "generate_report",
  "tool_call_id": "call_123",
  "status": "success",
  "content_ref": {
    "uri": "s3://reports-bucket/2025-01/report-456.pdf",
    "size_bytes": 2048576,
    "hash": "sha256:abcd1234...",
    "media_type": "application/pdf"
  },
  "metadata": {
    "preview": "Q1 2025 Financial Report"
  }
}
```

## Thread Relationships

### Design Philosophy

- **Outgoing Only**: Threads store only their outgoing relationships
- **Append-Only**: New relationships can be added without modifying existing data
- **Query for Incoming**: To find children/references, query across all threads

### Relationship Types

- `spawned_from`: This thread was created as a branch/fork
- `merged_from`: This thread incorporates another thread
- `referenced`: This thread references another for context
- Custom types allowed (use clear naming)

### Example

```json
{
  "relationships": {
    "links": [
      {
        "thread_id": "550e8400-e29b-41d4-a716-446655440001",
        "relation": "spawned_from",
        "metadata": {
          "branch_point": "turn_5",
          "reason": "exploring alternative approach"
        }
      },
      {
        "thread_id": "550e8400-e29b-41d4-a716-446655440002",
        "relation": "referenced",
        "metadata": {
          "context": "prior analysis"
        }
      }
    ]
  }
}
```

## Transformation Rules

### Pydantic AI → ThreadProtocol

```typescript
// Single agent.run() → Single AgentTurn
function fromAgentRun(messages: ModelMessage[], agentId: string): AgentTurn {
  return {
    turn_type: "agent",
    agent_id: agentId,
    started_at: messages[0].timestamp,
    completed_at: messages[messages.length - 1].timestamp,
    messages: messages.map(msg => ({
      message_type: msg.kind === "request" ? "request" : "response",
      timestamp: msg.timestamp,
      parts: transformParts(msg.parts), // Handle ContentRef
      agent_id: agentId,
      // Copy response fields if present
      ...(msg.kind === "response" && {
        model_name: msg.model_name,
        provider_name: msg.provider_name,
        usage: msg.usage,
        finish_reason: msg.finish_reason
      })
    })),
    total_usage: aggregateUsage(messages)
  };
}
```

### ThreadProtocol → Pydantic AI ModelMessages

```typescript
// When reconstructing for agent.run(), UserTurn becomes ModelRequest
function toModelMessages(turns: Turn[], viewingAgentId: string): ModelMessage[] {
  const messages: ModelMessage[] = [];

  for (const turn of turns) {
    if (turn.turn_type === "user") {
      // UserTurn → ModelRequest (this is the initial request)
      messages.push({
        kind: "request",
        parts: turn.parts,  // UserPromptPart typically
        timestamp: turn.submitted_at
        // Note: client_metadata is NOT passed to Pydantic AI
        // It's for application-level routing/configuration
      });
    } else {
      // Add agent's messages, handling multi-agent attribution
      messages.push(...processAgentTurn(turn, viewingAgentId));
    }
  }

  return messages;
}
```

## Validation Rules

1. All timestamps must be valid ISO 8601
2. `tool_call_id` must match between ToolCallPart and ToolReturnPart
3. `agent_id` must reference a key in the agents registry
4. Turn timestamps must not overlap (completed_at < next started_at)
5. Messages within a turn must be chronologically ordered
6. **NEW**: UserTurn.client_metadata keys MUST contain a namespace separator
7. **NEW**: ContentRef.uri MUST be a valid URI with recognized scheme
8. **NEW**: ThreadLink.thread_id MUST be a valid UUID

## Summary of v0.0.3 Features

1. **Complete Turns Only**: Explicit rule that only complete turns appear in ThreadProtocol
2. **UserTurn.client_metadata**: Namespaced key-value pairs for client directives
3. **ContentRef**: External content references for large tool returns
4. **Thread Relationships**: Outgoing links to other threads
5. **Bijection Rules**: Explicit mapping from stream events to ThreadProtocol parts
6. **Namespace Enforcement**: Client metadata keys must use namespacing

These features maintain the core philosophy while providing the foundation for robust multi-agent conversation management.