# Vercel AI SDK vs Pydantic AI: Message Schema Comparison

## Executive Summary

Both Vercel AI SDK and Pydantic AI provide message schemas for LLM interactions, but with different architectural approaches. Vercel uses a TypeScript-based union type system with fine-grained content parts, while Pydantic AI uses Python dataclasses with a dual-message system (Request/Response). Vercel's schema is more granular and provider-agnostic, while Pydantic AI's is more structured around agent workflows.

## Core Message Types

### Vercel AI SDK
- **Union Type**: `ModelMessage = SystemModelMessage | UserModelMessage | AssistantModelMessage | ToolModelMessage`
- **4 distinct roles**: system, user, assistant, tool
- Each message type has:
  - `role`: Literal type identifier
  - `content`: Type-specific content
  - `providerOptions`: Optional vendor metadata

### Pydantic AI
- **Dual System**:
  - `ModelRequest`: Messages TO the model (system prompts, user prompts, tool returns, retry prompts)
  - `ModelResponse`: Messages FROM the model (text, tool calls, thinking)
- **Discriminated Union**: `ModelMessage = ModelRequest | ModelResponse`
- Uses `part_kind` and `kind` discriminators for type identification

## Message Structure Comparison

### System Messages

| Aspect | Vercel | Pydantic AI |
|--------|--------|-------------|
| Type | `SystemModelMessage` | `SystemPromptPart` (within ModelRequest) |
| Content | Simple string | String with timestamp and optional dynamic_ref |
| Metadata | `providerOptions` | Timestamp, dynamic reference |
| Philosophy | Standalone message | Part of a request message |

### User Messages

| Aspect | Vercel | Pydantic AI |
|--------|--------|-------------|
| Type | `UserModelMessage` | `UserPromptPart` (within ModelRequest) |
| Content | String or array of parts | String or sequence of UserContent |
| Multi-modal | TextPart, ImagePart, FilePart | ImageUrl, AudioUrl, DocumentUrl, VideoUrl, BinaryContent |
| Metadata | `providerOptions` | Timestamp, vendor_metadata on individual content |

### Assistant Messages

| Aspect | Vercel | Pydantic AI |
|--------|--------|-------------|
| Type | `AssistantModelMessage` | `ModelResponse` with parts |
| Content Structure | Array of content parts | List of ModelResponsePart |
| Content Types | Text, File, Reasoning, ToolCall, ToolResult | TextPart, ToolCallPart, ThinkingPart |
| Unique Features | ReasoningPart (o1-style) | ThinkingPart with signature, usage tracking |

### Tool Messages

| Aspect | Vercel | Pydantic AI |
|--------|--------|-------------|
| Type | `ToolModelMessage` | `ToolReturnPart` and `RetryPromptPart` |
| Purpose | Tool results only | Tool results AND retry requests |
| Content | Array of ToolResultPart | Any type with model_response methods |
| Error Handling | In content | Separate RetryPromptPart type |

## Content Parts Deep Dive

### Multi-modal Content

**Vercel:**
- `ImagePart`: image data/URL + mediaType
- `FilePart`: data/URL + filename + mediaType
- No dedicated audio/video types (use FilePart)

**Pydantic AI:**
- Separate URL classes: `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`
- `BinaryContent`: Unified binary handler with format detection
- Rich media type enums and format mappings
- YouTube detection for videos
- `force_download` flag for URL handling

### Tool Interactions

**Vercel:**
- `ToolCallPart`: toolCallId, toolName, input (unknown type)
- `ToolResultPart`: toolCallId, toolName, output
- `providerExecuted` flag for provider-side execution

**Pydantic AI:**
- `ToolCallPart`: tool_name, args (string or dict), tool_call_id
- `ToolReturnPart`: tool_name, content (Any), tool_call_id, metadata
- `ToolReturn` class for structured returns with metadata
- `RetryPromptPart` for validation failures

### Thinking/Reasoning

**Vercel:**
- `ReasoningPart`: Simple text field for reasoning content
- Part of assistant message content array

**Pydantic AI:**
- `ThinkingPart`: content + id + signature (Anthropic-specific)
- Separate part type in model response
- Has dedicated delta type for streaming

## Streaming Support

### Vercel
- Not explicitly shown in the schema files provided
- Content parts structure suggests stream compatibility

### Pydantic AI
- Comprehensive streaming with delta types:
  - `TextPartDelta`, `ThinkingPartDelta`, `ToolCallPartDelta`
  - `ModelResponseStreamEvent`: PartStartEvent, PartDeltaEvent
  - `FinalResultEvent` for result signaling
- Apply methods for incremental updates

## Vendor/Provider Support

### Vercel
- `providerOptions` on every message/part type
- Provider-agnostic design
- Metadata passed through to providers

### Pydantic AI
- `vendor_metadata` on specific content types
- `vendor_details` and `vendor_id` on ModelResponse
- Provider-specific implementations in separate modules
- Explicit Google/Gemini video metadata support

## Key Architectural Differences

### 1. Message Flow Philosophy
- **Vercel**: Flat message array with role-based types
- **Pydantic AI**: Request/Response paradigm with nested parts

### 2. Type System
- **Vercel**: TypeScript unions with discriminated types
- **Pydantic AI**: Python dataclasses with Pydantic validation

### 3. Content Granularity
- **Vercel**: Fine-grained content parts within messages
- **Pydantic AI**: Coarser message parts with rich content types

### 4. Error Handling
- **Vercel**: Implicit in tool results
- **Pydantic AI**: Explicit RetryPromptPart with validation details

### 5. Metadata Strategy
- **Vercel**: Uniform providerOptions everywhere
- **Pydantic AI**: Type-specific metadata fields

## Alignment Points

1. **Role-based messaging**: Both use role concepts (system, user, assistant, tool)
2. **Multi-modal support**: Both handle images, files, and other media
3. **Tool calling**: Both support tool invocation and results
4. **Provider flexibility**: Both allow vendor-specific extensions
5. **Content arrays**: Both use arrays/lists for multi-part content
6. **Discriminated unions**: Both use type discriminators

## Key Differences

1. **Message grouping**: Vercel uses flat messages; Pydantic AI groups into requests/responses
2. **Streaming model**: Pydantic AI has explicit delta types; Vercel's approach unclear
3. **Media handling**: Pydantic AI has richer media type system with format detection
4. **Error recovery**: Pydantic AI has dedicated retry mechanism
5. **Usage tracking**: Pydantic AI includes usage in responses
6. **Thinking support**: Both have it but Pydantic AI's is more elaborate
7. **Validation**: Pydantic AI leverages Pydantic validation; Vercel uses TypeScript types

## Interoperability Considerations

### Adapting Vercel to Pydantic AI
1. Map Vercel's flat messages to Request/Response structure
2. Convert role-based messages to appropriate parts
3. Transform content arrays to Pydantic AI's content sequences
4. Handle providerOptions → vendor_metadata mapping

### Adapting Pydantic AI to Vercel
1. Flatten Request/Response into role-based messages
2. Convert parts to appropriate message types
3. Map rich media types to simpler FilePart
4. Extract vendor_metadata to providerOptions

## Recommendations

1. **For unified interface**: Create adapter layer that can translate between schemas
2. **For streaming**: Pydantic AI's delta system is more complete
3. **For multi-modal**: Pydantic AI offers richer type system
4. **For simplicity**: Vercel's flat structure is easier to understand
5. **For validation**: Pydantic AI's integration with Pydantic provides stronger guarantees

## Conclusion

While both schemas serve similar purposes, they reflect different design philosophies. Vercel prioritizes simplicity and provider-agnosticism with a flat, role-based structure. Pydantic AI emphasizes workflow structure and rich typing with its request/response paradigm and comprehensive media handling. The choice between them depends on whether you prioritize simplicity (Vercel) or feature richness and type safety (Pydantic AI).