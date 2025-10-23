# Pydantic AI ModelMessage Architecture - Visual Diagrams

## 1. Overall Message Hierarchy

```
ModelMessage (Discriminator: 'kind')
├── ModelRequest (kind='request')
│   └── parts: Sequence[ModelRequestPart]
│       ├── SystemPromptPart (part_kind='system-prompt')
│       ├── UserPromptPart (part_kind='user-prompt')
│       ├── ToolReturnPart (part_kind='tool-return')
│       ├── BuiltinToolReturnPart (part_kind='builtin-tool-return')
│       └── RetryPromptPart (part_kind='retry-prompt')
│
└── ModelResponse (kind='response')
    ├── parts: Sequence[ModelResponsePart]
    │   ├── TextPart (part_kind='text')
    │   ├── ThinkingPart (part_kind='thinking')
    │   ├── FilePart (part_kind='file')
    │   ├── ToolCallPart (part_kind='tool-call')
    │   ├── BuiltinToolCallPart (part_kind='builtin-tool-call')
    │   └── BuiltinToolReturnPart (part_kind='builtin-tool-return')
    ├── usage: RequestUsage
    ├── model_name: str | None
    ├── provider_name: str | None
    ├── provider_details: dict[str, Any] | None
    ├── provider_response_id: str | None
    └── finish_reason: FinishReason
```

---

## 2. Request Message Flow

```
Agent → ModelRequest
    │
    ├─ SystemPromptPart
    │  └─ content: str
    │     timestamp: datetime
    │     dynamic_ref: str | None
    │
    ├─ UserPromptPart
    │  ├─ content: str | Sequence[UserContent]
    │  │  ├─ str (text)
    │  │  └─ MultiModalContent
    │  │     ├─ ImageUrl
    │  │     ├─ AudioUrl
    │  │     ├─ VideoUrl
    │  │     ├─ DocumentUrl
    │  │     └─ BinaryContent
    │  └─ timestamp: datetime
    │
    ├─ ToolReturnPart
    │  ├─ tool_name: str
    │  ├─ content: Any
    │  ├─ tool_call_id: str
    │  └─ metadata: Any
    │
    └─ RetryPromptPart
       ├─ content: list[ErrorDetails] | str
       ├─ tool_name: str | None
       └─ tool_call_id: str
```

---

## 3. Response Message Flow

```
Model → ModelResponse
    │
    └─ parts: Sequence[ModelResponsePart]
       │
       ├─ TextPart
       │  ├─ content: str
       │  └─ id: str | None
       │
       ├─ ThinkingPart
       │  ├─ content: str
       │  ├─ signature: str | None (provider-specific)
       │  └─ provider_name: str | None
       │
       ├─ FilePart
       │  ├─ content: BinaryContent | BinaryImage
       │  ├─ id: str | None
       │  └─ provider_name: str | None
       │
       ├─ ToolCallPart
       │  ├─ tool_name: str
       │  ├─ args: str | dict[str, Any] | None
       │  └─ tool_call_id: str
       │
       ├─ BuiltinToolCallPart
       │  ├─ tool_name: str
       │  ├─ args: str | dict[str, Any] | None
       │  ├─ tool_call_id: str
       │  └─ provider_name: str | None
       │
       └─ BuiltinToolReturnPart
          ├─ tool_name: str
          ├─ content: Any
          ├─ tool_call_id: str
          └─ provider_name: str | None
```

---

## 4. Multi-Modal Content Support

```
FileUrl (Abstract)
├── url: str
├── identifier: str (for LLM reference)
├── force_download: bool
├── vendor_metadata: dict[str, Any] | None
└── media_type: str (computed)
    │
    ├─ ImageUrl (kind='image-url')
    │  └─ media_types: jpeg, png, gif, webp
    │
    ├─ AudioUrl (kind='audio-url')
    │  └─ media_types: wav, mpeg, ogg, flac, aiff, aac
    │
    ├─ VideoUrl (kind='video-url')
    │  ├─ media_types: matroska, quicktime, mp4, webm, flv, mpeg, wmv, 3gpp
    │  └─ YouTube detection: youtu.be, youtube.com
    │
    └─ DocumentUrl (kind='document-url')
       └─ media_types: pdf, txt, csv, docx, xlsx, html, md, xls

BinaryContent (kind='binary')
├── data: bytes
├── media_type: str
├── identifier: str
└── vendor_metadata: dict[str, Any] | None
    │
    ├─ is_audio: bool
    ├─ is_image: bool
    ├─ is_video: bool
    └─ is_document: bool
        │
        └─ BinaryImage (specialized)
           └─ Guaranteed media_type starts with 'image/'
```

---

## 5. Tool Call and Response Lifecycle

```
Model Response                Tool Execution              Next Request
───────────────              ──────────────              ────────────

ModelResponse
  └─ TextPart
  └─ ToolCallPart ──────────→ Tool Execution ──────────→ ModelRequest
     ├─ tool_name             ├─ tool.func()               └─ ToolReturnPart
     ├─ args                  ├─ Validation                   ├─ tool_name
     └─ tool_call_id          └─ Execution                    ├─ content
                                                              └─ tool_call_id


If Validation Fails:
      │
      ├─ Pydantic Error ──────→ ValidationError ────────→ ModelRequest
      │                                                   └─ RetryPromptPart
      │                                                      ├─ content: ErrorDetails
      │                                                      └─ tool_call_id
      │
      └─ Tool Error ──────────→ ModelRetry/Exception ──→ ModelRequest
                                                         └─ RetryPromptPart
```

---

## 6. Streaming Architecture with Deltas

```
Streaming Response                  Agent Processing
──────────────────                  ─────────────────

Model sends:                         Agent receives:
  ├─ PartStartEvent ──────────→ TextPartDelta
  │  └─ TextPart                  ├─ content_delta: "Hello"
  │     content: ""               └─ apply() → TextPart
  │                                   content: "Hello"
  │
  ├─ PartDeltaEvent ──────────→ TextPartDelta
  │  └─ TextPartDelta             ├─ content_delta: " world"
  │     content_delta: " world"    └─ apply() → TextPart
  │                                   content: "Hello world"
  │
  ├─ PartStartEvent ──────────→ ToolCallPartDelta
  │  └─ ToolCallPart              ├─ tool_name_delta: "get_"
  │     tool_name: ""             └─ apply() → ToolCallPartDelta
  │
  ├─ PartDeltaEvent ──────────→ ToolCallPartDelta
  │  └─ ToolCallPartDelta         ├─ tool_name_delta: "weather"
  │     tool_name_delta: "weather"│ └─ apply() → ToolCallPart
  │                                   tool_name: "get_weather"
  │
  └─ FinalResultEvent ────────→ Result complete
```

---

## 7. Discriminator Fields Reference

```
ModelMessage
├── kind
│   ├─ 'request' → ModelRequest
│   └─ 'response' → ModelResponse

ModelRequestPart
├── part_kind
│   ├─ 'system-prompt' → SystemPromptPart
│   ├─ 'user-prompt' → UserPromptPart
│   ├─ 'tool-return' → ToolReturnPart
│   ├─ 'builtin-tool-return' → BuiltinToolReturnPart
│   └─ 'retry-prompt' → RetryPromptPart

ModelResponsePart
├── part_kind
│   ├─ 'text' → TextPart
│   ├─ 'thinking' → ThinkingPart
│   ├─ 'file' → FilePart
│   ├─ 'tool-call' → ToolCallPart
│   ├─ 'builtin-tool-call' → BuiltinToolCallPart
│   └─ 'builtin-tool-return' → BuiltinToolReturnPart

ModelResponsePartDelta
├── part_delta_kind
│   ├─ 'text' → TextPartDelta
│   ├─ 'thinking' → ThinkingPartDelta
│   └─ 'tool_call' → ToolCallPartDelta

ModelResponseStreamEvent / HandleResponseEvent
├── event_kind
│   ├─ 'part_start' → PartStartEvent
│   ├─ 'part_delta' → PartDeltaEvent
│   ├─ 'final_result' → FinalResultEvent
│   ├─ 'function_tool_call' → FunctionToolCallEvent
│   └─ 'function_tool_result' → FunctionToolResultEvent

MultiModalContent
├── kind
│   ├─ 'image-url' → ImageUrl
│   ├─ 'audio-url' → AudioUrl
│   ├─ 'video-url' → VideoUrl
│   ├─ 'document-url' → DocumentUrl
│   └─ 'binary' → BinaryContent
```

---

## 8. Agent Execution Loop with Messages

```
┌─────────────────────────────────────────────────────────────┐
│ GraphAgentState                                             │
│ ├─ message_history: list[ModelMessage]                     │
│ ├─ user_prompt: str                                        │
│ └─ ... other state                                         │
└─────────────────────────────────────────────────────────────┘
           │
           ├─ UserPromptNode
           │  └─ Creates: ModelRequest
           │     ├─ SystemPromptPart (if configured)
           │     └─ UserPromptPart (from user_prompt)
           │
           ├─ MessageAdded to history
           │  └─ message_history.append(request)
           │
           ├─ ModelRequestNode
           │  ├─ Sends: ModelRequest → Model
           │  └─ Receives: ModelResponse
           │
           ├─ MessageAdded to history
           │  └─ message_history.append(response)
           │
           ├─ CallToolsNode
           │  ├─ Extract: ToolCallPart[] from response
           │  ├─ Execute: tool_name(args) for each call
           │  └─ Create: ToolReturnPart[] or RetryPromptPart[]
           │
           ├─ MessagesAdded to history
           │  ├─ message_history.append(ToolReturnPart)
           │  └─ (or RetryPromptPart if validation fails)
           │
           └─ Loop Back to ModelRequestNode
              └─ Creates: ModelRequest with all history parts
```

---

## 9. Serialization Format Example

```json
[
  {
    "kind": "request",
    "parts": [
      {
        "part_kind": "system-prompt",
        "content": "You are a helpful assistant.",
        "timestamp": "2025-01-15T10:30:00Z",
        "dynamic_ref": null
      },
      {
        "part_kind": "user-prompt",
        "content": "What is 2 + 2?",
        "timestamp": "2025-01-15T10:30:01Z"
      }
    ],
    "instructions": null
  },
  {
    "kind": "response",
    "parts": [
      {
        "part_kind": "text",
        "content": "2 + 2 = 4",
        "id": null
      },
      {
        "part_kind": "tool-call",
        "tool_name": "calculator",
        "args": "{\"operation\": \"add\", \"a\": 2, \"b\": 2}",
        "tool_call_id": "call_123"
      }
    ],
    "usage": {
      "input_tokens": 20,
      "output_tokens": 15,
      "details": {}
    },
    "model_name": "gpt-4o",
    "timestamp": "2025-01-15T10:30:02Z",
    "provider_name": "openai",
    "provider_details": {"finish_reason": "tool_calls"},
    "provider_response_id": "chatcmpl-ABC123",
    "finish_reason": "tool_call"
  }
]
```

---

## 10. Streaming Event Sequence Example

```
Streaming Response to Agent
────────────────────────────

Time 1: PartStartEvent
  event_kind: 'part_start'
  index: 0
  part: TextPart(content: '')

Time 2: PartDeltaEvent
  event_kind: 'part_delta'
  index: 0
  delta: TextPartDelta(content_delta: 'Hello')

Time 3: PartDeltaEvent
  event_kind: 'part_delta'
  index: 0
  delta: TextPartDelta(content_delta: ' ')

Time 4: PartDeltaEvent
  event_kind: 'part_delta'
  index: 0
  delta: TextPartDelta(content_delta: 'world')

Time 5: PartStartEvent
  event_kind: 'part_start'
  index: 1
  part: ToolCallPart(tool_name: 'search', args: '{}', tool_call_id: 'call_1')

Time 6: FinalResultEvent
  event_kind: 'final_result'
  tool_name: 'search'
  tool_call_id: 'call_1'

Reconstructed ModelResponse:
  parts: [
    TextPart(content: 'Hello world'),
    ToolCallPart(tool_name: 'search', args: '{}', tool_call_id: 'call_1')
  ]
```

---

## 11. Type Conversion Matrix

```
Tool Arguments Conversion:
─────────────────────────

ToolCallPart.args: str | dict[str, Any] | None
    │
    ├─ String Format:
    │  args: '{"key": "value"}'
    │  args_as_dict() → {"key": "value"}
    │  args_as_json_str() → '{"key": "value"}'
    │
    └─ Dict Format:
       args: {"key": "value"}
       args_as_dict() → {"key": "value"}
       args_as_json_str() → '{"key": "value"}'


Tool Return Content Conversion:
──────────────────────────────

ToolReturnPart.content: Any
    │
    ├─ If str:
    │  model_response_str() → content
    │  model_response_object() → {'return_value': content}
    │
    ├─ If dict:
    │  model_response_str() → JSON string
    │  model_response_object() → content (as-is)
    │
    └─ If other:
       model_response_str() → JSON serialized
       model_response_object() → {'return_value': serialized}
```

---

## 12. Provider-Specific Extensions

```
Provider-Specific Behavior:
──────────────────────────

ThinkingPart.signature (provider-specific)
├─ Anthropic: signature field directly
├─ Bedrock: signature field
├─ Google: thought_signature field
├─ OpenAI: encrypted_content field
└─ Rule: Only sent back to same provider

BuiltinToolCallPart / BuiltinToolReturnPart
├─ provider_name: str | None
└─ Rule: Only sent back to same provider (model can't process)

FileUrl.vendor_metadata
├─ OpenAI/Google: ImageUrl.vendor_metadata['detail']
├─ Google: VideoUrl.vendor_metadata['video_metadata']
└─ Other providers: ignored

FileUrl.force_download
├─ OpenAI: If True, download and send bytes
├─ Google: If True, download and send bytes
└─ Other providers: Behavior varies
```

