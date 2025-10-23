# Pydantic AI ModelMessage Documentation Index

## Overview
Complete documentation of the Pydantic AI ModelMessage schema and architecture. This documentation provides comprehensive understanding of how messages flow through the Pydantic AI system.

**Source Code Location**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/messages.py`

---

## Documentation Files

### 1. MODELMESSAGE_ARCHITECTURE.md (20 KB, 637 lines)
**Comprehensive technical reference for the entire ModelMessage system**

Contains:
- Core message types and hierarchy
- Request messages (ModelRequest and all request parts)
- Response messages (ModelResponse and all response parts)
- Multi-modal content support (FileUrl, ImageUrl, AudioUrl, VideoUrl, DocumentUrl, BinaryContent)
- Streaming support (deltas, events, streaming architecture)
- Serialization and deserialization patterns
- Agent integration details
- OpenTelemetry integration
- Design patterns and best practices

**Best for**: Understanding the complete schema, architectural decisions, and how everything fits together

### 2. MODELMESSAGE_DIAGRAM.md (14 KB, 462 lines)
**Visual diagrams and ASCII trees showing message structure and flow**

Contains:
- Overall message hierarchy tree
- Request message flow diagram
- Response message flow diagram
- Multi-modal content type hierarchy
- Tool call and response lifecycle
- Streaming architecture with deltas
- Discriminator field reference matrix
- Agent execution loop with messages
- JSON serialization format example
- Streaming event sequence example
- Type conversion matrix
- Provider-specific extensions

**Best for**: Quick visual reference, understanding relationships between types, debugging message structures

### 3. MODELMESSAGE_EXAMPLES.md (16 KB, 671 lines)
**Practical code examples and usage patterns**

Contains:
- Basic message creation (requests and responses)
- JSON serialization and deserialization
- Working with tool calls and tool returns
- Multi-modal content examples (images, documents, audio, video)
- Streaming event processing
- Agent integration patterns
- Advanced scenarios (thinking, built-in tools, file responses)
- Complete end-to-end example
- Key takeaways

**Best for**: Learning by example, implementing message handling, common patterns

---

## Quick Navigation

### I need to understand...

**The overall structure:**
- Start with MODELMESSAGE_DIAGRAM.md (Section 1 - Overall Message Hierarchy)
- Then read MODELMESSAGE_ARCHITECTURE.md (Core Message Types section)

**How tool calls work:**
- MODELMESSAGE_DIAGRAM.md (Section 5 - Tool Call and Response Lifecycle)
- MODELMESSAGE_EXAMPLES.md (Section 3 - Working with Tool Calls)

**Multi-modal content:**
- MODELMESSAGE_ARCHITECTURE.md (Multi-Modal Content Support section)
- MODELMESSAGE_EXAMPLES.md (Section 4 - Multi-Modal Content)

**Streaming responses:**
- MODELMESSAGE_ARCHITECTURE.md (Streaming Support section)
- MODELMESSAGE_DIAGRAM.md (Section 6 - Streaming Architecture)
- MODELMESSAGE_EXAMPLES.md (Section 5 - Streaming Events)

**JSON serialization:**
- MODELMESSAGE_ARCHITECTURE.md (Serialization & Deserialization section)
- MODELMESSAGE_DIAGRAM.md (Section 9 - Serialization Format Example)
- MODELMESSAGE_EXAMPLES.md (Section 2 - Serialization/Deserialization)

**Agent message flow:**
- MODELMESSAGE_ARCHITECTURE.md (Agent Integration section)
- MODELMESSAGE_DIAGRAM.md (Section 8 - Agent Execution Loop)

**Provider-specific behavior:**
- MODELMESSAGE_ARCHITECTURE.md (Design Patterns section - Vendor-Specific Extensibility)
- MODELMESSAGE_DIAGRAM.md (Section 12 - Provider-Specific Extensions)

---

## Message Type Reference

### Request Parts (`part_kind` discriminator)
```
'system-prompt'      → SystemPromptPart
'user-prompt'        → UserPromptPart
'tool-return'        → ToolReturnPart
'builtin-tool-return' → BuiltinToolReturnPart
'retry-prompt'       → RetryPromptPart
```

### Response Parts (`part_kind` discriminator)
```
'text'               → TextPart
'thinking'           → ThinkingPart
'file'               → FilePart
'tool-call'          → ToolCallPart
'builtin-tool-call'  → BuiltinToolCallPart
'builtin-tool-return' → BuiltinToolReturnPart
```

### Top-Level Messages (`kind` discriminator)
```
'request'            → ModelRequest
'response'           → ModelResponse
```

### Multi-Modal Content (`kind` discriminator)
```
'image-url'          → ImageUrl
'audio-url'          → AudioUrl
'video-url'          → VideoUrl
'document-url'       → DocumentUrl
'binary'             → BinaryContent
```

### Streaming Events (`event_kind` discriminator)
```
'part_start'         → PartStartEvent
'part_delta'         → PartDeltaEvent
'final_result'       → FinalResultEvent
'function_tool_call' → FunctionToolCallEvent
'function_tool_result' → FunctionToolResultEvent
```

---

## Key Concepts

### 1. Discriminated Unions
All polymorphic message types use Pydantic's discriminated unions with explicit discriminator fields for type safety and clear serialization.

### 2. Message History
Agent maintains complete message history as `list[ModelMessage]` containing both requests and responses in chronological order.

### 3. Tool Call Lifecycle
```
Model Response (ToolCallPart)
    ↓
Agent executes tool
    ↓
Agent creates ToolReturnPart
    ↓
Agent sends new ModelRequest with ToolReturnPart
    ↓
Model responds with next content
```

### 4. Streaming with Deltas
Streaming uses incremental delta updates that can be applied to parts with `.apply()` methods, enabling memory-efficient real-time processing.

### 5. Vendor-Agnostic Protocol
Single unified message format works across all LLM providers (OpenAI, Anthropic, Google, etc.) with provider-specific fields for extended features.

### 6. Backward Compatibility
System handles schema evolution through alias validation, supporting old field names (e.g., `vendor_details` → `provider_details`).

---

## Common Tasks

### Serialize messages to JSON
```python
from pydantic_ai import ModelMessagesTypeAdapter

json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
with open('messages.json', 'wb') as f:
    f.write(json_bytes)
```
See: MODELMESSAGE_EXAMPLES.md (Section 2)

### Deserialize messages from JSON
```python
from pydantic_ai import ModelMessagesTypeAdapter

with open('messages.json', 'rb') as f:
    messages = ModelMessagesTypeAdapter.validate_json(f.read())
```
See: MODELMESSAGE_EXAMPLES.md (Section 2)

### Process tool calls from response
```python
for tool_call in response.tool_calls:
    args = tool_call.args_as_dict()
    result = execute_tool(tool_call.tool_name, **args)
```
See: MODELMESSAGE_EXAMPLES.md (Section 3)

### Add multi-modal content to request
```python
from pydantic_ai import UserPromptPart, ImageUrl

request = ModelRequest(
    parts=[
        UserPromptPart(content=["Analyze this:", ImageUrl(url="...")])
    ]
)
```
See: MODELMESSAGE_EXAMPLES.md (Section 4)

### Process streaming events
```python
for event in stream:
    if isinstance(event, PartDeltaEvent):
        parts[event.index] = event.delta.apply(parts[event.index])
```
See: MODELMESSAGE_EXAMPLES.md (Section 5)

---

## Implementation Notes

### Tool Arguments Storage
Tool arguments in `ToolCallPart` can be:
- **JSON string**: `'{"key": "value"}'` - from streaming/some models
- **Python dict**: `{"key": "value"}` - from direct API calls

Use `args_as_dict()` and `args_as_json_str()` for conversion.

### Tool Return Content
`ToolReturnPart.content` accepts:
- **String**: Sent to model as-is
- **Dict**: Sent to model as JSON
- **Other types**: Automatically JSON serialized

Use `model_response_str()` or `model_response_object()` for serialization.

### Provider-Specific Fields
- `ThinkingPart.signature` - Only sent back to same provider (signature format varies by provider)
- `BuiltinToolCallPart`/`BuiltinToolReturnPart` - Provider-specific tools, only sent to that provider
- `FileUrl.vendor_metadata` - Provider-specific configuration (e.g., image detail for OpenAI)
- `FileUrl.force_download` - Whether to download or send URL directly

### Binary Data Handling
Binary data is automatically:
- **Encoded**: Base64 on serialization
- **Decoded**: Base64 on deserialization

No manual encoding/decoding needed.

---

## Testing & Validation

The system includes comprehensive test coverage:
- Unit tests: `/Users/ericksonc/appdev/pydantic-ai/tests/test_messages.py`
- Serialization tests: Round-trip validation
- Backward compatibility: Old message formats still deserialize

---

## Additional Resources

- **Source Code**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/messages.py`
- **Test File**: `/Users/ericksonc/appdev/pydantic-ai/tests/test_messages.py`
- **Parts Manager**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/_parts_manager.py`
- **Agent Graph**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/_agent_graph.py`

---

## Summary

The ModelMessage system is the foundation of Pydantic AI's vendor-agnostic agent framework. Understanding its structure, serialization, and message flow is essential for:

1. **Implementing custom message handlers**
2. **Storing and loading agent conversations**
3. **Debugging agent behavior**
4. **Extending agents with custom logic**
5. **Building monitoring and observability solutions**

This documentation provides the complete reference needed to work with messages at any level of the system.
