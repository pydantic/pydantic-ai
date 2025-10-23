# Pydantic AI ModelMessage Architecture - Comprehensive Guide

## Overview

The `ModelMessage` system is the core communication protocol in Pydantic AI that enables vendor-agnostic message passing between the agent framework and external language models. It provides a unified, discriminated union-based schema for representing all communication in agent-to-model and model-to-agent interactions.

**Key Location**: `/Users/ericksonc/appdev/pydantic-ai/pydantic_ai_slim/pydantic_ai/messages.py`

---

## Core Message Types

### 1. ModelMessage (Top-Level Type)
```python
ModelMessage = Annotated[ModelRequest | ModelResponse, pydantic.Discriminator('kind')]
```

The root type that represents any message sent to or returned by a model. It's a discriminated union using the `kind` field:
- `ModelRequest` (kind='request') - Sent TO the model
- `ModelResponse` (kind='response') - Received FROM the model

**Serialization Support**:
```python
ModelMessagesTypeAdapter = pydantic.TypeAdapter(
    list[ModelMessage], 
    config=pydantic.ConfigDict(
        defer_build=True, 
        ser_json_bytes='base64', 
        val_json_bytes='base64'
    )
)
```
This TypeAdapter handles JSON serialization/deserialization of message lists with automatic base64 encoding for binary data.

---

## Request Messages

### ModelRequest
Represents a request sent from Pydantic AI to the model.

**Structure**:
```python
@dataclass(repr=False)
class ModelRequest:
    parts: Sequence[ModelRequestPart]  # List of request parts
    instructions: str | None = None     # Optional model instructions
    kind: Literal['request'] = 'request'
```

**Factory Method**:
```python
ModelRequest.user_text_prompt(
    user_prompt: str, 
    *, 
    instructions: str | None = None
) -> ModelRequest
```

### ModelRequestPart (Discriminated Union)
```python
ModelRequestPart = Annotated[
    SystemPromptPart | UserPromptPart | ToolReturnPart | RetryPromptPart,
    pydantic.Discriminator('part_kind')
]
```

#### SystemPromptPart
**Purpose**: System context/guidance written by application developers

**Fields**:
- `content: str` - The prompt text
- `timestamp: datetime` - When the prompt was created
- `dynamic_ref: str | None` - Reference to dynamic system prompt function (if applicable)
- `part_kind: Literal['system-prompt'] = 'system-prompt'` - Discriminator

**Use Case**: Initial system instructions, behavioral guidelines, output format requirements

#### UserPromptPart
**Purpose**: User input from end users

**Fields**:
- `content: str | Sequence[UserContent]` - Can be text or multi-modal content
- `timestamp: datetime` - When the prompt was created
- `part_kind: Literal['user-prompt'] = 'user-prompt'` - Discriminator

**MultiModal Support**:
```python
UserContent: TypeAlias = str | MultiModalContent

MultiModalContent = (
    ImageUrl | AudioUrl | DocumentUrl | VideoUrl | BinaryContent
)
```

#### ToolReturnPart
**Purpose**: Result from executing a tool call

**Base Class - BaseToolReturnPart**:
```python
@dataclass(repr=False)
class BaseToolReturnPart:
    tool_name: str                  # Name of the tool that was called
    content: Any                    # The return value
    tool_call_id: str = field(default_factory=_generate_tool_call_id)
    metadata: Any = None            # App-level metadata (not sent to LLM)
    timestamp: datetime = field(default_factory=_now_utc)
```

**Subclasses**:

1. **ToolReturnPart** - Regular tool returns
   - `part_kind: Literal['tool-return'] = 'tool-return'`

2. **BuiltinToolReturnPart** - Built-in tool returns (provider-specific)
   - `provider_name: str | None = None` - Only sent to same provider
   - `part_kind: Literal['builtin-tool-return'] = 'builtin-tool-return'`

**Key Methods**:
- `model_response_str() -> str` - JSON serialization for model consumption
- `model_response_object() -> dict[str, Any]` - Dict form (wraps non-dict types)
- `has_content() -> bool` - Check if return has content

#### RetryPromptPart
**Purpose**: Signal to model to retry due to validation errors or tool failures

**Fields**:
- `content: list[pydantic_core.ErrorDetails] | str` - Error details or message
- `tool_name: str | None = None` - Which tool (if any) failed
- `tool_call_id: str` - Associated tool call ID
- `timestamp: datetime` - When retry was triggered
- `part_kind: Literal['retry-prompt'] = 'retry-prompt'` - Discriminator

**Triggers**:
- Pydantic validation failure on tool arguments
- Tool raised `ModelRetry` exception
- Tool not found
- Plain text returned when structured response expected
- Pydantic validation failure on structured response
- Output validator raised `ModelRetry` exception

**Key Methods**:
- `model_response() -> str` - Human-readable retry message

---

## Response Messages

### ModelResponse
Represents a response from the model back to Pydantic AI.

**Structure**:
```python
@dataclass(repr=False)
class ModelResponse:
    parts: Sequence[ModelResponsePart]  # Response parts
    usage: RequestUsage = field(default_factory=RequestUsage)
    model_name: str | None = None       # Model identifier
    timestamp: datetime = field(default_factory=_now_utc)
    kind: Literal['response'] = 'response'
    provider_name: str | None = None    # LLM provider (e.g., 'openai')
    provider_details: dict[str, Any] | None = None  # Provider-specific data
    provider_response_id: str | None = None  # Provider request ID
    finish_reason: FinishReason | None = None  # Normalized finish reason
```

**FinishReason Type**:
```python
FinishReason: TypeAlias = Literal[
    'stop',              # Normal completion
    'length',            # Max tokens reached
    'content_filter',    # Content policy violation
    'tool_call',         # Model called a tool
    'error',             # Error occurred
]
```

### ModelResponsePart (Discriminated Union)
```python
ModelResponsePart = Annotated[
    TextPart | ToolCallPart | BuiltinToolCallPart | 
    BuiltinToolReturnPart | ThinkingPart | FilePart,
    pydantic.Discriminator('part_kind'),
]
```

#### TextPart
**Purpose**: Plain text response from model

**Fields**:
- `content: str` - Text content
- `id: str | None = None` - Optional part identifier
- `part_kind: Literal['text'] = 'text'` - Discriminator

**Methods**:
- `has_content() -> bool` - Non-empty check

#### ThinkingPart
**Purpose**: Model's reasoning/thinking process (extended thinking)

**Fields**:
- `content: str` - Thinking content
- `id: str | None = None` - Optional identifier
- `signature: str | None = None` - Encrypted thinking signature
- `provider_name: str | None = None` - Provider that generated it
- `part_kind: Literal['thinking'] = 'thinking'` - Discriminator

**Provider Support**:
- Anthropic: `signature` field
- Bedrock: `signature` field
- Google: `thought_signature` field
- OpenAI: `encrypted_content` field

**Note**: Signatures are provider-specific and only sent back to the same provider.

#### FilePart
**Purpose**: File/image/media response from model

**Fields**:
- `content: Annotated[BinaryContent, pydantic.AfterValidator(BinaryImage.narrow_type)]`
- `id: str | None = None` - Optional identifier
- `provider_name: str | None = None` - Provider that generated it
- `part_kind: Literal['file'] = 'file'` - Discriminator

**Supported Content Types**: Binary images, audio, video, documents

#### BaseToolCallPart (Abstract)
**Purpose**: Represents tool invocations from the model

**Base Fields**:
```python
@dataclass(repr=False)
class BaseToolCallPart:
    tool_name: str                                    # Tool to invoke
    args: str | dict[str, Any] | None = None        # Arguments (JSON or dict)
    tool_call_id: str = field(default_factory=_generate_tool_call_id)
```

**Key Methods**:
- `args_as_dict() -> dict[str, Any]` - Convert to dict form
- `args_as_json_str() -> str` - Convert to JSON string form
- `has_content() -> bool` - Check if args have content

**Subclasses**:

1. **ToolCallPart** - Regular tool call
   - `part_kind: Literal['tool-call'] = 'tool-call'`
   - Standard tool invocation

2. **BuiltinToolCallPart** - Built-in tool call
   - `provider_name: str | None = None` - Provider-specific
   - `part_kind: Literal['builtin-tool-call'] = 'builtin-tool-call'`
   - Only sent back to the same provider

---

## Multi-Modal Content Support

### FileUrl (Abstract Base)
Base class for URL-based files with consistent interface.

**Common Fields**:
```python
@dataclass(init=False, repr=False)
class FileUrl(ABC):
    url: str                                          # File URL
    identifier: str                                   # Unique ID for LLM reference
    force_download: bool = False                      # Download vs send URL
    vendor_metadata: dict[str, Any] | None = None    # Provider-specific metadata
    media_type: str                                   # Computed property
```

**Identifier Usage**: The model can reference files in tool calls using the identifier, allowing tools to look up files from message history.

### Concrete FileUrl Types

#### ImageUrl
```python
@dataclass(init=False, repr=False)
class ImageUrl(FileUrl):
    kind: Literal['image-url'] = 'image-url'
```

**Supported Media Types**: `image/jpeg`, `image/png`, `image/gif`, `image/webp`

**Vendor Metadata**: `detail` setting for OpenAI/Google (for image quality)

#### AudioUrl
```python
@dataclass(init=False, repr=False)
class AudioUrl(FileUrl):
    kind: Literal['audio-url'] = 'audio-url'
```

**Supported Media Types**: `audio/wav`, `audio/mpeg`, `audio/ogg`, `audio/flac`, `audio/aiff`, `audio/aac`

#### VideoUrl
```python
@dataclass(init=False, repr=False)
class VideoUrl(FileUrl):
    kind: Literal['video-url'] = 'video-url'
```

**Supported Media Types**: `video/x-matroska`, `video/quicktime`, `video/mp4`, `video/webm`, `video/x-flv`, `video/mpeg`, `video/x-ms-wmv`, `video/3gpp`

**YouTube Support**: Automatically detects YouTube URLs and infers `video/mp4` media type.

**Vendor Metadata**: `video_metadata` for Google Gemini video processing customization

#### DocumentUrl
```python
@dataclass(init=False, repr=False)
class DocumentUrl(FileUrl):
    kind: Literal['document-url'] = 'document-url'
```

**Supported Media Types**: `application/pdf`, `text/plain`, `text/csv`, `text/markdown`, `text/html`, `text/x-asciidoc`, `application/rtf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document` (docx), `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (xlsx), `application/vnd.ms-excel` (xls)

### BinaryContent
Direct binary data (not URL-based).

```python
@dataclass(init=False, repr=False)
class BinaryContent:
    data: bytes                                       # Raw binary data
    media_type: AudioMediaType | ImageMediaType |   # MIME type
                DocumentMediaType | str
    identifier: str                                  # Unique ID
    vendor_metadata: dict[str, Any] | None = None   # Provider metadata
    kind: Literal['binary'] = 'binary'
```

**Type Detection Methods**:
- `is_audio -> bool`
- `is_image -> bool`
- `is_video -> bool`
- `is_document -> bool`

**Conversion**:
- `from_data_uri(data_uri: str) -> BinaryContent` - Parse data URI
- `data_uri -> str` - Convert to data URI format (base64 encoded)

### BinaryImage
Specialized binary content guaranteed to be an image.

```python
class BinaryImage(BinaryContent):
    # Automatically narrows BinaryContent when media_type starts with 'image/'
```

---

## Streaming Support

### ModelResponsePartDelta
Partial updates for streaming responses.

```python
ModelResponsePartDelta = Annotated[
    TextPartDelta | ThinkingPartDelta | ToolCallPartDelta,
    pydantic.Discriminator('part_delta_kind')
]
```

#### TextPartDelta
```python
@dataclass(repr=False)
class TextPartDelta:
    content_delta: str                              # Text to append
    part_delta_kind: Literal['text'] = 'text'
    
    def apply(self, part: ModelResponsePart) -> TextPart:
        """Apply delta to existing TextPart"""
```

#### ThinkingPartDelta
```python
@dataclass(repr=False, kw_only=True)
class ThinkingPartDelta:
    content_delta: str | None = None               # Thinking to append
    signature_delta: str | None = None             # Signature update
    provider_name: str | None = None               # Provider for signature
    part_delta_kind: Literal['thinking'] = 'thinking'
    
    def apply(
        self, 
        part: ModelResponsePart | ThinkingPartDelta
    ) -> ThinkingPart | ThinkingPartDelta:
        """Apply delta to ThinkingPart or another ThinkingPartDelta"""
```

#### ToolCallPartDelta
```python
@dataclass(repr=False, kw_only=True)
class ToolCallPartDelta:
    tool_name_delta: str | None = None              # Append to tool name
    args_delta: str | dict[str, Any] | None = None # Append/merge args
    tool_call_id: str | None = None                 # Set tool call ID
    part_delta_kind: Literal['tool_call'] = 'tool_call'
    
    def apply(
        self, 
        part: ModelResponsePart | ToolCallPartDelta
    ) -> ToolCallPart | BuiltinToolCallPart | ToolCallPartDelta:
        """Apply delta and potentially upgrade to complete ToolCallPart"""
```

### Streaming Events

#### ModelResponseStreamEvent
```python
ModelResponseStreamEvent = Annotated[
    PartStartEvent | PartDeltaEvent | FinalResultEvent,
    pydantic.Discriminator('event_kind')
]
```

**Event Types**:

1. **PartStartEvent** - New part created
   - `index: int` - Position in parts list
   - `part: ModelResponsePart` - The new part

2. **PartDeltaEvent** - Existing part updated
   - `index: int` - Position in parts list
   - `delta: ModelResponsePartDelta` - Update to apply

3. **FinalResultEvent** - Response complete and matches output schema
   - `tool_name: str | None` - Output tool name (if any)
   - `tool_call_id: str | None` - Associated tool call ID

#### AgentStreamEvent
```python
AgentStreamEvent = Annotated[
    ModelResponseStreamEvent | HandleResponseEvent,
    pydantic.Discriminator('event_kind')
]
```

Includes handling events:

1. **FunctionToolCallEvent** - User function tool invoked
   - `part: ToolCallPart`
   - Properties: `tool_call_id`

2. **FunctionToolResultEvent** - User function tool result
   - `result: ToolReturnPart | RetryPromptPart`
   - `content: str | Sequence[UserContent] | None` - Content sent to model

---

## Serialization & Deserialization

### JSON Serialization Pattern
```python
# Serialize to JSON
json_bytes = ModelMessagesTypeAdapter.dump_json(messages)
json_str = json_bytes.decode()

# Serialize to Python dict
dict_data = ModelMessagesTypeAdapter.dump_python(
    messages, 
    mode='json'  # Use JSON-compatible types
)

# Deserialize from JSON
messages = ModelMessagesTypeAdapter.validate_python(json_data)
```

### Binary Data Handling
Binary data is automatically base64 encoded/decoded during serialization:
```python
config=pydantic.ConfigDict(
    ser_json_bytes='base64',   # Encode on serialization
    val_json_bytes='base64'    # Decode on validation
)
```

### Discriminator Fields Used
The system uses Pydantic's discriminated unions with specific fields:

**Request Parts**: `part_kind` field
- `'system-prompt'` → SystemPromptPart
- `'user-prompt'` → UserPromptPart
- `'tool-return'` → ToolReturnPart
- `'builtin-tool-return'` → BuiltinToolReturnPart
- `'retry-prompt'` → RetryPromptPart

**Response Parts**: `part_kind` field
- `'text'` → TextPart
- `'thinking'` → ThinkingPart
- `'file'` → FilePart
- `'tool-call'` → ToolCallPart
- `'builtin-tool-call'` → BuiltinToolCallPart
- `'builtin-tool-return'` → BuiltinToolReturnPart

**Top-Level Messages**: `kind` field
- `'request'` → ModelRequest
- `'response'` → ModelResponse

**Deltas**: `part_delta_kind` field
- `'text'` → TextPartDelta
- `'thinking'` → ThinkingPartDelta
- `'tool_call'` → ToolCallPartDelta

**Stream Events**: `event_kind` field
- `'part_start'` → PartStartEvent
- `'part_delta'` → PartDeltaEvent
- `'final_result'` → FinalResultEvent
- `'function_tool_call'` → FunctionToolCallEvent
- `'function_tool_result'` → FunctionToolResultEvent

---

## Agent Integration

### Message History
The agent maintains a complete message history:
```python
@dataclass
class GraphAgentState:
    message_history: list[ModelMessage] = dataclasses.field(default_factory=list)
    # ... other state fields
```

### Message Flow in Agent Execution
1. **UserPromptNode**: Creates `ModelRequest` with `UserPromptPart`
2. **ModelRequestNode**: Sends request to model, gets `ModelResponse`
3. **CallToolsNode**: 
   - Extracts `ToolCallPart`s from response
   - Executes tools
   - Creates `ToolReturnPart` messages
   - Adds `RetryPromptPart` if validation fails
4. **Loop**: Creates new `ModelRequest` with all parts

### History Processing
Optional history processors can transform messages before sending to model:
```python
_HistoryProcessorSync = Callable[
    [list[ModelMessage]], 
    list[ModelMessage]
]

_HistoryProcessorAsync = Callable[
    [list[ModelMessage]], 
    Awaitable[list[ModelMessage]]
]
```

---

## OpenTelemetry Integration

Each message part implements OpenTelemetry event generation:

```python
def otel_event(self, settings: InstrumentationSettings) -> Event:
    """Generate OpenTelemetry event"""

def otel_message_parts(
    self, 
    settings: InstrumentationSettings
) -> list[_otel_messages.MessagePart]:
    """Generate OpenTelemetry message parts"""
```

This enables observability and tracing throughout the message lifecycle.

---

## ToolReturn Special Case

### ToolReturn Class
Allows tools to return both structured data AND custom content for the model:

```python
@dataclass(repr=False)
class ToolReturn:
    return_value: Any                              # Actual tool return value
    content: str | Sequence[UserContent] | None = None  # Content to model
    metadata: Any = None                          # App-level metadata
    kind: Literal['tool-return'] = 'tool-return'
```

**Use Case**: Tool returns computation result but sends formatted/multi-modal content to model for further processing.

---

## Key Design Patterns

### 1. Discriminated Unions with pydantic.Discriminator
All polymorphic message types use discriminated unions with discriminator fields for type safety and clear serialization.

### 2. Vendor-Specific Extensibility
- `provider_name` field allows provider-specific behavior (thinking signatures, builtin tools)
- `vendor_metadata` for provider-specific configuration
- Provider details stored for post-processing

### 3. Backward Compatibility
The system handles schema evolution through alias validation:
```python
provider_details: Annotated[
    dict[str, Any] | None,
    pydantic.Field(
        validation_alias=pydantic.AliasChoices(
            'provider_details', 
            'vendor_details'  # Legacy name
        )
    ),
]
```

### 4. Streaming as Deltas
Streaming uses delta updates that can be applied to parts incrementally, enabling memory-efficient processing.

### 5. Base Classes for Shared Behavior
- `BaseToolReturnPart` - Shared tool return logic
- `BaseToolCallPart` - Shared tool call logic
- `FileUrl` - Shared URL-based file logic

---

## Summary

The ModelMessage architecture provides:

- **Vendor-Agnostic Protocol**: Single unified schema across all LLM providers
- **Rich Type System**: Discriminated unions for type safety and clarity
- **Multi-Modal Support**: Images, audio, video, documents, and binary data
- **Streaming First**: Delta-based updates for efficient real-time processing
- **Extensibility**: Provider-specific fields and metadata
- **Serialization**: JSON-ready with automatic base64 encoding for binary data
- **Observability**: OpenTelemetry integration throughout
- **History Tracking**: Complete agent conversation history
- **Retry Mechanism**: Automatic retry handling with error details

This design enables Pydantic AI to provide a clean, type-safe abstraction over diverse LLM APIs while supporting advanced features like streaming, multi-modal content, and extended thinking.
