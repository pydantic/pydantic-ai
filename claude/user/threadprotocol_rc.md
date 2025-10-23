# ThreadProtocol Specification v1.0.0

## Overview

ThreadProtocol is a canonical JSON representation for multi-agent conversation threads that:
- Serves as a deterministic serialization format for both Pydantic AI ModelMessages and Vercel AI SDK streams
- Clearly separates universal LLM primitives from system-specific extensions
- Supports multi-agent conversations with proper attribution
- Enables thread branching and parallelism

## Core Design Principles

1. **Universal vs Custom**: Clear separation between LLM primitives and system extensions
2. **Deterministic**: Both server (Pydantic AI) and client (Vercel AI SDK) generate identical JSON
3. **Recording vs Interpretation**: ThreadProtocol records what happened; transformation layers handle presentation
4. **Versioned**: Protocol version for future evolution
5. **KISS**: Keep it simple, add complexity only when needed

## Schema Structure

### Thread (Root Object)

```typescript
interface Thread {
  // Metadata
  version: "1.0.0";  // Protocol version
  thread_id: string; // UUID
  parent_thread_id?: string; // UUID for branching/parallelism
  
  // Timestamps
  created_at: string; // ISO 8601
  updated_at: string; // ISO 8601
  
  // Human-readable info
  title: string;
  metadata?: Record<string, any>; // Optional extension point
  
  // Agent registry
  agents: Record<string, AgentConfig>;
  
  // The actual conversation
  actions: Action[];
}
```

### Agent Configuration

```typescript
interface AgentConfig {
  agent_id: string;        // UUID - globally unique
  agent_identifier: string; // e.g., "fred_v1" - unique within thread
  agent_name: string;      // e.g., "Fred" - human-readable
  created_at: string;      // ISO 8601
  
  // Extension point for agent configuration
  // (system prompt, tools, etc. - referenced but not included)
  config_ref?: string;     // External reference to full config
}
```

### Action (Discriminated Union)

```typescript
type Action = CoreAction | SystemAction;

interface BaseAction {
  action_id?: string;     // Optional UUID
  timestamp: string;      // ISO 8601
  sequence: number;       // Sequential order within thread
}

// Core actions map to universal LLM primitives
type CoreAction =
  | UserMessageAction
  | AssistantMessageAction
  | ThinkingAction
  | ToolCallAction
  | ToolReturnAction;

// System actions map to custom system events  
type SystemAction = {
  action_type: `system.${string}`; // e.g., "system.tool_grant"
  data: any;
} & BaseAction;
```

### Core Actions (Universal LLM Primitives)

```typescript
interface UserMessageAction extends BaseAction {
  action_type: "user_message";
  content: MessageContent;
  attachments?: Attachment[];
}

interface AssistantMessageAction extends BaseAction {
  action_type: "assistant_message";
  agent_id: string;        // UUID reference to agents registry
  content: MessageContent;
  finish_reason?: "stop" | "tool_call" | "length" | "content_filter";
  usage?: {
    input_tokens: number;
    output_tokens: number;
    total_tokens?: number;
  };
}

interface ThinkingAction extends BaseAction {
  action_type: "thinking";
  agent_id: string;        // Which agent is thinking

  // Core thinking data - matching Pydantic AI's ThinkingPart
  content?: string;        // Visible thinking text (when available)
  signature?: string;      // Opaque provider reference (for hidden thinking)

  // Essential metadata
  provider_name: string;   // e.g., "anthropic", "openai" - required for signature round-tripping
  thinking_id?: string;    // Provider-specific ID if available

  // Usage tracking
  usage?: {
    thinking_tokens?: number;  // If provider reports them separately
  };
}

interface ToolCallAction extends BaseAction {
  action_type: "tool_call";
  agent_id: string;        // Which agent called the tool
  tool_name: string;
  tool_call_id: string;    // Correlates with return
  args: any;               // Tool arguments (usually object)
}

interface ToolReturnAction extends BaseAction {
  action_type: "tool_return";
  tool_call_id: string;    // Correlates with call
  tool_name: string;       // For convenience/validation
  status: "success" | "error" | "validation_error";
  content: any;            // Return value or error details
}
```

### Content Types

```typescript
type MessageContent = string | ContentPart[];

interface ContentPart {
  type: "text" | "image" | "file" | "audio" | "video";
  // Type-specific fields...
}

interface TextContent extends ContentPart {
  type: "text";
  text: string;
}

interface ImageContent extends ContentPart {
  type: "image";
  image_url?: string;
  image_base64?: string;
  media_type?: string;
}

interface Attachment {
  name: string;
  url?: string;
  data?: string; // base64
  media_type: string;
  size_bytes?: number;
}
```

### System Actions (Custom Extensions)

System actions use the `system.*` namespace and can evolve independently:

```typescript
// Examples of system actions
interface ToolGrantAction extends BaseAction {
  action_type: "system.tool_grant";
  data: {
    agent_id: string;
    tool_name: string;
    granted_by?: string; // user or another agent
    expires_at?: string;
  };
}

interface AgentJoinAction extends BaseAction {
  action_type: "system.agent_join";
  data: {
    agent_id: string;
    invited_by?: string;
  };
}

interface ConfigUpdateAction extends BaseAction {
  action_type: "system.config_update";
  data: {
    agent_id?: string;
    changes: Record<string, any>;
  };
}
```

## Transformation Rules

### Pydantic AI → ThreadProtocol

1. **ModelRequest parts** map to actions:
   - `UserPromptPart` → `UserMessageAction`
   - `SystemPromptPart` → Stored in agent config, not in actions
   - `ToolReturnPart` → `ToolReturnAction`
   - `RetryPromptPart` → `ToolReturnAction` with status="error"

2. **ModelResponse parts** map to actions:
   - `TextPart` → `AssistantMessageAction`
   - `ThinkingPart` → `ThinkingAction` (preserves content, signature, provider_name)
   - `ToolCallPart` → `ToolCallAction`
   - Multiple text parts in single response → Combine into single `AssistantMessageAction`
   - Thinking parts remain separate actions to preserve timing and provider metadata

### Vercel AI SDK Stream → ThreadProtocol

1. **Core stream events**:
   - `text-start/delta/end` → Accumulate into `AssistantMessageAction`
   - `tool-input-*` → `ToolCallAction`
   - `tool-output-available` → `ToolReturnAction`
   - `error` → `ToolReturnAction` with status="error"

2. **Custom stream events**:
   - `data-*` → `system.*` actions

### ThreadProtocol → ModelMessages

When generating ModelMessages for a specific agent:

1. **Filter actions** relevant to that agent
2. **Transform agent messages**:
   - Own messages: Use content as-is
   - Other agents' messages: Prepend `{agent:Name}: ` to text
3. **Handle tool visibility**:
   - Own tool calls: Include normally
   - Others' tool calls: Transform based on policy (hide/summarize/show)
4. **Skip system actions** (don't become ModelMessages)

## Example Thread

```json
{
  "version": "1.0.0",
  "thread_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-01-15T10:00:00Z",
  "updated_at": "2025-01-15T10:05:00Z",
  "title": "Multi-agent discussion about weather",
  
  "agents": {
    "agent_001": {
      "agent_id": "agent_001",
      "agent_identifier": "weather_assistant_v1",
      "agent_name": "Weather Assistant",
      "created_at": "2025-01-15T10:00:00Z"
    },
    "agent_002": {
      "agent_id": "agent_002", 
      "agent_identifier": "travel_planner_v1",
      "agent_name": "Travel Planner",
      "created_at": "2025-01-15T10:00:00Z"
    }
  },
  
  "actions": [
    {
      "action_type": "user_message",
      "timestamp": "2025-01-15T10:00:00Z",
      "sequence": 1,
      "content": "What's the weather like in Tokyo?"
    },
    {
      "action_type": "assistant_message",
      "timestamp": "2025-01-15T10:00:01Z",
      "sequence": 2,
      "agent_id": "agent_001",
      "content": "Let me check the current weather in Tokyo for you."
    },
    {
      "action_type": "tool_call",
      "timestamp": "2025-01-15T10:00:02Z",
      "sequence": 3,
      "agent_id": "agent_001",
      "tool_name": "get_weather",
      "tool_call_id": "call_001",
      "args": {"city": "Tokyo", "units": "celsius"}
    },
    {
      "action_type": "tool_return",
      "timestamp": "2025-01-15T10:00:03Z",
      "sequence": 4,
      "tool_call_id": "call_001",
      "tool_name": "get_weather",
      "status": "success",
      "content": {
        "temperature": 18,
        "conditions": "partly cloudy",
        "humidity": 65
      }
    },
    {
      "action_type": "assistant_message",
      "timestamp": "2025-01-15T10:00:04Z",
      "sequence": 5,
      "agent_id": "agent_001",
      "content": "The weather in Tokyo is currently 18°C and partly cloudy with 65% humidity."
    },
    {
      "action_type": "system.agent_join",
      "timestamp": "2025-01-15T10:00:05Z",
      "sequence": 6,
      "data": {
        "agent_id": "agent_002",
        "invited_by": "user"
      }
    },
    {
      "action_type": "assistant_message",
      "timestamp": "2025-01-15T10:00:06Z",
      "sequence": 7,
      "agent_id": "agent_002",
      "content": "Great weather for sightseeing! Would you like recommendations for outdoor activities in Tokyo?"
    }
  ]
}
```

## Key Design Decisions

1. **Flat action list** instead of nested structure for simplicity
2. **Agent registry** at thread level to avoid repetition
3. **Clear action_type naming**: `snake_case` for core, `system.*` for extensions
4. **Tool calls and returns** as separate actions (matches both Pydantic AI and Vercel patterns)
5. **Sequence numbers** for explicit ordering alongside timestamps
6. **Optional action_id** for future correlation needs

## Migration and Extensibility

- **Version field** enables future schema evolution
- **System actions** can be added without changing core schema
- **metadata fields** provide extension points
- **Clear boundaries** between universal and custom components

## Validation Rules

1. `sequence` must be unique and sequential
2. `tool_call_id` must match between ToolCallAction and ToolReturnAction
3. `agent_id` must reference a key in the agents registry
4. `action_type` must follow naming conventions (core vs system)
5. Timestamps should be chronologically consistent with sequence numbers