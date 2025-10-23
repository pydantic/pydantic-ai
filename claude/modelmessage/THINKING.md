# Pydantic AI Thinking/Reasoning Implementation Analysis

This document provides a comprehensive analysis of how Pydantic AI handles thinking/reasoning across different LLM providers, based on research of both documentation and implementation code.

## Executive Summary

Pydantic AI handles thinking through a flexible `ThinkingPart` message component that supports both:
- **Visible thinking**: Full thinking text available (Anthropic, Groq, Mistral)
- **Hidden thinking**: Only signatures/summaries available (OpenAI o1, Gemini 2.0 Flash Thinking)
- **Tag-based thinking**: Text with `<think>` tags converted to ThinkingPart (OpenAI Chat, Hugging Face)

The implementation cleverly uses a combination of `content` (visible text) and `signature` (opaque reference) fields to handle all provider variations in a unified way.

## Core Data Structures

### ThinkingPart (messages.py:944-979)

```python
@dataclass(repr=False)
class ThinkingPart:
    """A thinking response from a model."""

    content: str
    """The thinking content of the response."""

    id: str | None = None
    """The identifier of the thinking part."""

    signature: str | None = None
    """The signature of the thinking.

    Supported by:
    * Anthropic (corresponds to the `signature` field)
    * Bedrock (corresponds to the `signature` field)
    * Google (corresponds to the `thought_signature` field)
    * OpenAI (corresponds to the `encrypted_content` field)
    """

    provider_name: str | None = None
    """The name of the provider that generated the response.

    Signatures are only sent back to the same provider.
    """

    part_kind: Literal['thinking'] = 'thinking'
```

### Key Design Decisions

1. **Dual representation**: Both `content` and `signature` can exist, allowing:
   - Full thinking text when available
   - Opaque signatures for hidden thinking
   - Both for providers that offer them

2. **Provider binding**: The `provider_name` field ensures signatures only round-trip to the same provider

3. **Unified abstraction**: All provider variations map to this single structure

## Provider Implementation Matrix

| Provider | Configuration | Visible Content | Signature | Streaming Support | Notes |
|----------|--------------|-----------------|-----------|-------------------|-------|
| **Anthropic** | `anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024}` | ✅ Full text | ✅ signature field | ✅ Delta events | Most complete implementation |
| **OpenAI Responses** | `openai_reasoning_effort='low'` | ❌ Hidden | ✅ encrypted_content | ✅ Event-based | Signature arrives after content |
| **OpenAI Chat** | Auto-detect `<think>` tags | ✅ Full text | ❌ | ❌ | Simple tag parsing |
| **Google Gemini** | `google_thinking_config={'include_thoughts': True}` | ❌ Hidden | ✅ thought_signature | ✅ In content parts | Base64 encoded signature |
| **Groq** | `groq_reasoning_format='parsed'` | ✅ Full text | ❌ | ✅ Simple deltas | Three formats: raw/hidden/parsed |
| **Mistral** | Auto-enabled for magistral | ✅ Full text | ❌ | ✅ Simple deltas | No config needed |
| **Cohere** | Auto-enabled for command-a-reasoning | ✅ Full text | ❌ | ✅ Simple deltas | No config needed |

## Implementation Patterns

### 1. Anthropic (Most Complete)

```python
# From anthropic.py
if isinstance(item, BetaThinkingBlock):
    # Full content + signature
    yield self._thinking_part(item, delta=False, pending_parts=pending_parts)
elif isinstance(item, BetaRedactedThinkingBlock):
    # Signature only (no content)
    yield ThinkingPart(
        content='',
        id=item.id,
        signature=item.signature,
        provider_name=self.name()
    )
```

### 2. OpenAI Responses (Hidden Thinking)

```python
# From openai.py
case 'response.reasoning':
    # Only summary available during streaming
    if event.type == 'content.start':
        if content := event.content:
            summary = content.get('reasoning_summary')
            # Store for later

case 'response.done':
    # Signature arrives at the end
    for item in response.content:
        if item.type == 'reasoning':
            signature = item.encrypted_content
```

### 3. Google Gemini (Signature with Hidden Content)

```python
# From google.py
if hasattr(part, 'thought') and part.thought:
    # Has thinking but content might be hidden
    thinking_signature = getattr(part, 'thought_signature', None)
    if thinking_signature:
        # Base64 encoded signature
        signature = base64.b64encode(thinking_signature).decode('ascii')
```

### 4. Tag-Based Detection (OpenAI Chat, Hugging Face)

```python
# From profile.py
def detect_thinking_parts(text: str, thinking_tags: tuple) -> list:
    start_tag, end_tag = thinking_tags
    # Extract text between tags and convert to ThinkingPart
    if start_tag in text:
        # Parse and create ThinkingPart
```

## Streaming Architecture

### ThinkingPartDelta

Used for incremental streaming of thinking content:

```python
class ThinkingPartDelta(TypedDict):
    type: Literal['thinking-part-delta']
    content_delta: NotRequired[str]
    signature_delta: NotRequired[str]
    provider_name: NotRequired[str]
```

### Streaming Coordination

Different providers stream thinking at different points:
- **Anthropic**: Thinking streams first, then response
- **OpenAI o1**: Summary first, signature at the end
- **Google**: Mixed with content parts

The system uses `vendor_part_id` mapping to accumulate deltas:

```python
# Track which vendor part maps to which thinking part
vendor_part_id_to_genai_part_id: dict[str, str] = {}
```

## Key Insights for ThreadProtocol

Based on this analysis, here are the key considerations for ThreadProtocol:

### 1. **Thinking Representation Should Be Flexible**

The ThinkingPart structure works well because it can represent:
- Full visible thinking (`content` only)
- Hidden thinking with reference (`signature` only)
- Both (Anthropic case)
- Neither (empty/redacted)

### 2. **Provider Binding Is Critical**

The `provider_name` field is essential - signatures are opaque and provider-specific. They must only round-trip to the same provider.

### 3. **Timing Matters for Some Providers**

- **Anthropic**: Can have interleaved thinking (multiple ThinkingParts)
- **OpenAI/Gemini**: Always upfront only
- **Others**: Usually upfront but flexible

### 4. **Streaming Complexity**

Different providers stream thinking differently:
- Some stream content incrementally (Anthropic)
- Some provide summary upfront, signature later (OpenAI)
- Some don't stream thinking at all

### 5. **Three Distinct Use Cases**

1. **Development/Debugging**: Want full thinking text
2. **Production with Audit**: Want signatures for verification
3. **Production without Audit**: Don't need thinking at all

## Recommendations for ThreadProtocol

Given Pydantic AI's approach, I recommend:

### 1. **Adopt Similar Dual Structure**

```typescript
interface ThinkingAction extends BaseAction {
  action_type: "thinking";
  agent_id: string;

  // Core fields (one or both may be present)
  content?: string;          // Visible thinking text
  signature?: string;        // Opaque provider reference

  // Metadata
  provider_name: string;     // Critical for round-tripping
  thinking_id?: string;      // Provider's ID if available

  // Positioning hint for reconstruction
  position?: "before_message" | "during_message" | "after_tool";
}
```

### 2. **Keep Thinking Separate from AssistantMessage**

Unlike embedding in AssistantMessage, separate ThinkingActions allow:
- Precise timing preservation
- Easier filtering for different audiences
- Cleaner streaming implementation

### 3. **Support Both Patterns**

- **Upfront thinking** (OpenAI/Gemini): Single ThinkingAction before AssistantMessage
- **Interleaved thinking** (Anthropic): Multiple ThinkingActions at various points

### 4. **Include Usage Tracking**

```typescript
interface ThinkingAction {
  // ... other fields ...
  usage?: {
    thinking_tokens?: number;  // If provider reports them
  };
}
```

### 5. **Transform Rules**

- **To ModelMessages**: Include ThinkingPart with signature if same provider
- **To Vercel Stream**: Map to reasoning-start/delta/end events
- **To User Display**: Optionally hide or collapse based on preference

## Conclusion

Pydantic AI's thinking implementation is sophisticated and handles the wide variety of provider behaviors well. The key insight is that thinking must be treated as a first-class concept with its own representation, not just embedded text. The dual content/signature approach elegantly handles both visible and hidden thinking scenarios.

For ThreadProtocol, adopting a similar approach with separate ThinkingAction events will provide the flexibility needed to handle all current providers and likely future variations.

## References

### Documentation
- `/docs/thinking.md` - User-facing thinking documentation
- `/docs/api/messages.md` - Message structure documentation
- `/docs/agents.md` - Model settings configuration

### Key Implementation Files
- `/pydantic_ai_slim/pydantic_ai/messages.py:944-979` - ThinkingPart definition
- `/pydantic_ai_slim/pydantic_ai/models/anthropic.py` - Complete thinking implementation
- `/pydantic_ai_slim/pydantic_ai/models/openai.py` - Both Chat and Responses variants
- `/pydantic_ai_slim/pydantic_ai/models/google.py` - Signature-based thinking
- `/pydantic_ai_slim/pydantic_ai/profiles.py` - Tag detection logic