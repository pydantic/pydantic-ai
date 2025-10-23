# ThreadProtocol Specification v2.0.0

## Overview

ThreadProtocol is a canonical JSON representation for multi-agent conversation threads that:
- Faithfully serializes Pydantic AI ModelMessages with minimal transformation
- Adds multi-agent attribution and turn boundaries
- Supports system-level events outside Pydantic AI's scope
- Enables deterministic round-trip conversion between server (Pydantic AI) and client (Vercel AI SDK)

## Core Design Principles

1. **Follow Pydantic AI's Lead**: Reuse their proven abstractions for LLM primitives
2. **Minimal Transformation**: Store what actually happened, not an interpretation
3. **Multi-Agent Native**: Agent attribution without text manipulation
4. **Natural Ordering**: Use array position and timestamps, no artificial numbering
5. **Extensible for System Events**: Clean separation between LLM and system concerns

## Schema Structure

### Thread (Root Object)

```typescript
interface Thread {
  version: "2.0.0";          // Protocol version
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

  // Note: When converting to PAI ModelMessages, UserTurn becomes:
  // ModelRequest(parts=<these parts>, timestamp=submitted_at)
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
// NOTE: SystemPromptPart is NOT stored in ThreadProtocol (see above)
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
  content: any;              // Return value or error
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

## Important: What ThreadProtocol Does NOT Store

### SystemPromptPart Handling

ThreadProtocol deliberately does **NOT** store SystemPromptPart inline. Here's why:

**What SystemPromptPart contains in PAI:**
- Agent identity/persona ("You are Fred, an expert in...")
- Access to resources (files, documents, APIs)
- Instructions for multi-agent interaction
- Tool definitions and constraints
- Behavioral guidelines

**How ThreadProtocol handles this:**
- `agent_id` references the agent's full configuration
- `config_ref` points to external configuration storage
- The multi-agent system (not PAI) manages these prompts
- SystemPrompts are configuration, not conversation events

This separation means:
- Agent configurations can evolve without changing the thread
- Prompts aren't redundantly stored in every conversation
- The system controlling agents owns prompt management
- ThreadProtocol remains a pure event log, not a config dump

When reconstructing ModelMessages for PAI:
```typescript
// The system adds SystemPromptPart based on agent_id
const systemPrompt = await loadAgentConfig(agentId).systemPrompt;
messages.unshift({
  kind: "request",
  parts: [{ part_kind: "system-prompt", content: systemPrompt }]
});
```

## Important: UserTurn = Initial ModelRequest

A key insight of ThreadProtocol v2: **UserTurn effectively IS the ModelRequest that starts the conversation**.

In Pydantic AI's flow:
1. User provides input
2. Creates ModelRequest with UserPromptPart
3. Sends to model
4. Gets ModelResponse

In ThreadProtocol:
1. UserTurn stores the user input (with UserPromptPart)
2. AgentTurn stores only the ModelResponse and subsequent cycles
3. No duplication of the initial request

This means:
- Each piece of data is stored exactly once
- UserTurn → ModelRequest during reconstruction
- The full PAI message flow can be perfectly reconstructed

## Important: Pending Tool States Are Implicit

ThreadProtocol does **NOT** have explicit "pending", "awaiting_approval", or "deferred" fields. The state is implicit in the message structure:

**How to Detect Pending States:**
```typescript
// A tool call is pending if it has no corresponding return
function getPendingToolCalls(turn: AgentTurn): ToolCallPart[] {
  const calls = new Map<string, ToolCallPart>();
  const returns = new Set<string>();

  for (const msg of turn.messages) {
    for (const part of msg.parts) {
      if (part.part_kind === 'tool-call') {
        calls.set(part.tool_call_id, part);
      } else if (part.part_kind === 'tool-return') {
        returns.add(part.tool_call_id);
      }
    }
  }

  // Return calls without returns
  return Array.from(calls.entries())
    .filter(([id]) => !returns.has(id))
    .map(([_, call]) => call);
}
```

**State Transitions:**
1. **Tool Called** → ToolCallPart exists (state: pending)
2. **Tool Approved/Executed** → ToolReturnPart with content (state: completed)
3. **Tool Denied** → ToolReturnPart with error message (state: rejected)

**Why Implicit?**
- The message history itself encodes the state
- No synchronization issues between state fields and actual messages
- Follows PAI's approach: the messages ARE the state
- "Pending" is temporary - exists only between agent runs

When tools require approval or external execution, PAI handles this through `DeferredToolRequests` during the run. But in the stored ThreadProtocol, we just see:
- First run ends with ToolCallParts
- Next run starts with ToolReturnParts
- The gap between represents the "pending" period

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
      parts: msg.parts,  // Direct mapping
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
      });
    } else {
      // Add agent's messages, handling multi-agent attribution
      messages.push(...processAgentTurn(turn, viewingAgentId));
    }
  }

  return messages;
}

// For a specific agent to consume
function processAgentTurn(turn: AgentTurn, viewingAgentId: string): ModelMessage[] {
  return turn.messages
    .filter(msg => msg.message_type !== "system")  // Skip system messages
    .map(msg => {
      const parts = msg.agent_id !== viewingAgentId
        ? prefixAgentNameToParts(msg.parts, msg.agent_id)
        : msg.parts;

      return {
        kind: msg.message_type === "request" ? "request" : "response",
        parts: parts,
        timestamp: msg.timestamp,
        // ... other fields
      };
    });
}

// Helper: Add agent prefix to text parts only
function prefixAgentNameToParts(parts: MessagePart[], agentId: string): MessagePart[] {
  const agentName = lookupAgentName(agentId);
  return parts.map(part => {
    if (part.part_kind === "text") {
      return { ...part, content: `{agent:${agentName}}: ${part.content}` };
    }
    return part;
  });
}
```

### Vercel AI SDK Stream → ThreadProtocol

```typescript
// Stream events map to parts or system messages
function fromStreamEvent(event: StreamEvent): MessagePart | SystemMessage | null {
  switch(event.type) {
    case "text-delta":
      return { part_kind: "text", content: event.delta };

    case "thinking-delta":
      return { part_kind: "thinking", content: event.delta };

    case "tool-input-available":
      return {
        part_kind: "tool-call",
        tool_name: event.toolName,
        tool_call_id: event.toolCallId,
        args: event.input
      };

    case "tool-output-available":
      return {
        part_kind: "tool-return",
        tool_name: event.toolName,
        tool_call_id: event.toolCallId,
        status: "success",
        content: event.output
      };

    case "data-*":  // Custom system event
      return {
        message_type: "system",
        timestamp: new Date().toISOString(),
        event_type: event.type.substring(5),  // Remove "data-"
        event_data: event.data
      };

    default:
      return null;
  }
}
```

## Example: Message Enrichment

When the system transforms or enriches a user's message:

```json
{
  "turns": [
    {
      "turn_type": "user",
      "submitted_at": "2025-01-15T10:00:00Z",
      "parts": [
        {
          "part_kind": "user-prompt",
          "content": "weather tokyo"
        }
      ]
    },
    {
      "turn_type": "agent",
      "agent_id": "agent_001",
      "started_at": "2025-01-15T10:00:01Z",
      "completed_at": "2025-01-15T10:00:05Z",
      "messages": [
        {
          "message_type": "system",
          "timestamp": "2025-01-15T10:00:01Z",
          "event_type": "prompt.enriched",
          "event_data": {
            "original": "weather tokyo",
            "enriched": "Please provide current weather information for Tokyo, Japan, including temperature, conditions, and humidity."
          }
        },
        {
          "message_type": "response",
          "timestamp": "2025-01-15T10:00:02Z",
          "agent_id": "agent_001",
          "parts": [
            {
              "part_kind": "text",
              "content": "Let me check the weather in Tokyo for you..."
            }
          ]
        }
      ]
    }
  ]
}
```

This way we record:
1. What the user actually said (once, in UserTurn)
2. Any transformation that occurred (as SystemMessage)
3. What the agent produced (in its responses)

## Example Thread

```json
{
  "version": "2.0.0",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:05:00Z",
  "title": "Multi-agent weather discussion",

  "agents": {
    "agent_001": {
      "agent_id": "agent_001",
      "agent_name": "Weather Assistant",
      "model_name": "gpt-4",
      "provider_name": "openai",
      "created_at": "2025-01-15T10:00:00Z"
    },
    "agent_002": {
      "agent_id": "agent_002",
      "agent_name": "Travel Planner",
      "model_name": "claude-3-opus",
      "provider_name": "anthropic",
      "created_at": "2025-01-15T10:00:00Z"
    }
  },

  "turns": [
    {
      "turn_type": "user",
      "submitted_at": "2025-01-15T10:00:00Z",
      "parts": [
        {
          "part_kind": "user-prompt",
          "content": "What's the weather like in Tokyo?"
        }
      ]
    },
    {
      "turn_type": "agent",
      "agent_id": "agent_001",
      "started_at": "2025-01-15T10:00:01Z",
      "completed_at": "2025-01-15T10:00:05Z",
      "messages": [
        {
          "message_type": "response",
          "timestamp": "2025-01-15T10:00:02Z",
          "agent_id": "agent_001",
          "model_name": "gpt-4",
          "parts": [
            {
              "part_kind": "text",
              "content": "Let me check the current weather in Tokyo."
            },
            {
              "part_kind": "tool-call",
              "tool_name": "get_weather",
              "tool_call_id": "call_001",
              "args": {"city": "Tokyo", "units": "celsius"}
            }
          ]
        },
        {
          "message_type": "request",
          "timestamp": "2025-01-15T10:00:03Z",
          "agent_id": "agent_001",
          "parts": [
            {
              "part_kind": "tool-return",
              "tool_name": "get_weather",
              "tool_call_id": "call_001",
              "status": "success",
              "content": {
                "temperature": 18,
                "conditions": "partly cloudy"
              }
            }
          ]
        },
        {
          "message_type": "response",
          "timestamp": "2025-01-15T10:00:04Z",
          "agent_id": "agent_001",
          "model_name": "gpt-4",
          "parts": [
            {
              "part_kind": "text",
              "content": "The weather in Tokyo is currently 18°C and partly cloudy. Travel Planner, what do you think?"
            }
          ]
        },
        {
          "message_type": "system",
          "timestamp": "2025-01-15T10:00:05Z",
          "event_type": "agent.handoff",
          "event_data": {
            "from": "agent_001",
            "to": "agent_002",
            "reason": "explicit_mention"
          },
          "source_agent": "agent_001",
          "target_agents": ["agent_002"]
        }
      ],
      "total_usage": {
        "input_tokens": 120,
        "output_tokens": 45,
        "total_tokens": 165
      }
    },
    {
      "turn_type": "agent",
      "agent_id": "agent_002",
      "started_at": "2025-01-15T10:00:05Z",
      "completed_at": "2025-01-15T10:00:08Z",
      "messages": [
        {
          "message_type": "request",
          "timestamp": "2025-01-15T10:00:05Z",
          "agent_id": "agent_002",
          "parts": [
            {
              "part_kind": "user-prompt",
              "content": "Based on the weather, what activities would you recommend?"
            }
          ]
        },
        {
          "message_type": "response",
          "timestamp": "2025-01-15T10:00:06Z",
          "agent_id": "agent_002",
          "model_name": "claude-3-opus",
          "provider_name": "anthropic",
          "parts": [
            {
              "part_kind": "thinking",
              "content": "The weather is mild and partly cloudy...",
              "provider_name": "anthropic"
            },
            {
              "part_kind": "text",
              "content": "Perfect weather for sightseeing! I'd recommend visiting temples and parks."
            }
          ]
        }
      ],
      "total_usage": {
        "input_tokens": 150,
        "output_tokens": 35,
        "thinking_tokens": 20,
        "total_tokens": 205
      }
    }
  ]
}
```

## Key Design Decisions

### 1. **Turn-Based Organization**
- One `agent.run()` = one `AgentTurn`
- Natural boundaries for multi-agent conversations
- Clear attribution and usage tracking
- **UserTurn IS the initial ModelRequest** - contains the UserPromptPart that starts the conversation
- No duplication: user input recorded once in UserTurn, not repeated in AgentTurn

### 2. **Minimal Transformation of PAI Types**
- MessagePart types match PAI exactly
- No consolidation of TextParts
- Preserve all provider-specific fields

### 3. **Agent Attribution Without Text Manipulation**
- `agent_id` field on messages (not in text)
- Text prefixing only during transformation to ModelMessage
- Clean storage, flexible presentation

### 4. **SystemMessage for Extensibility**
- Handles events outside PAI's scope
- Maps to Vercel's `data-*` events
- System can evolve independently

### 5. **Natural Ordering**
- Array position provides order
- Timestamps provide temporal context
- No redundant sequence numbers

### 6. **Configuration vs Conversation**
- SystemPrompts are configuration (referenced via agent_id)
- ThreadProtocol stores conversation events, not agent configs
- System owns prompt management, not the protocol
- Clean separation of concerns

## Migration and Compatibility

- **Version per thread**: Entire thread uses one protocol version
- **Backward compatibility**: Parsers should handle older versions
- **Forward compatibility**: Unknown fields should be preserved
- **Lossless round-trip**: PAI → ThreadProtocol → PAI preserves all data

## Validation Rules

1. All timestamps must be valid ISO 8601
2. `tool_call_id` must match between ToolCallPart and ToolReturnPart
3. `agent_id` must reference a key in the agents registry
4. Turn timestamps must not overlap (completed_at < next started_at)
5. Messages within a turn must be chronologically ordered

## Future Considerations

- **Branching**: `parent_thread_id` for conversation branches
- **Streaming**: Partial turn representation during streaming
- **Compression**: Reference deduplication for large threads
- **Search**: Indexing strategy for message retrieval