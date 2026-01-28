# Spec: Multimodal Tool Results (PR #3826 Rewrite)

## Problem
Tools returning multimodal content get split into separate user parts → breaks Bedrock, suboptimal elsewhere.

## Solution
Keep multimodal in `ToolReturnPart`, let each provider handle based on API capabilities.

---

## Provider Support Matrix

| Provider | Image | Document | Audio | Video |
|----------|-------|----------|-------|-------|
| **Bedrock** | Native | Native | ❌ Error | Native |
| **Anthropic** | Native | Native | Fallback | Fallback |
| **Google** | Native | Native | Native | Native |
| **OpenAI Chat** | Fallback | Fallback | Fallback | ❌ Error |
| **OpenAI Responses** | Native | Native | Native | ❌ Error |
| **xAI** | Fallback | Fallback | ❌ Error | ❌ Error |
| **Groq** | Fallback | ❌ Error | ❌ Error | ❌ Error |
| **OpenRouter** | (inherits OpenAI Chat) | | | |

- **Native** = in tool_result directly
- **Fallback** = `"See file <id>"` in tool_result + separate user msg with `"This is file <id>:"`
- **❌ Error** = raises error (same as user prompts)

---

## API Changes (`messages.py`)

```python
class BaseToolReturnPart:
    content: MultiModalContent | list[MultiModalContent | Any] | Any  # Updated type

    def _split_content(self) -> tuple[list[MultiModalContent], list[Any]]:
        '''Split into (files, other). Private, called by both properties.'''

    @property
    def files(self) -> list[MultiModalContent]: ...

    @property
    def content_excluding_files(self) -> list[str | dict | ...]: ...
```

Add deprecation warning to `ToolReturn.content` field.

---

## Fallback Behavior Pattern

When files can't go native:
1. Tool result: `"See file {identifier}"`
2. User message: `"This is file {identifier}:"` + file
3. **Order**: ALL tool returns first, THEN all user parts

---

## Test Strategy: Parametrized VCR Matrix

### Dimensions
1. **Provider**: `anthropic`, `bedrock`, `google`, `openai-chat`, `openai-responses`, `xai`, `groq`
2. **File Type**: `image`, `document`, `audio`, `video`
3. **Return Style**: `direct` (return file), `tool_return_content` (use `ToolReturn.content`)

### Smart Test Harness

```python
from typing import Literal

Expectation = Literal['native', 'fallback', 'error']

SUPPORT_MATRIX: dict[tuple[str, str], Expectation] = {
    ('bedrock', 'image'): 'native',
    ('bedrock', 'document'): 'native',
    ('bedrock', 'audio'): 'error',
    ('bedrock', 'video'): 'native',
    ('anthropic', 'image'): 'native',
    ('anthropic', 'document'): 'native',
    ('anthropic', 'audio'): 'fallback',
    ('anthropic', 'video'): 'fallback',
    # ... etc
}

def get_expectation(provider: str, file_type: str) -> Expectation:
    return SUPPORT_MATRIX[(provider, file_type)]

def assert_multimodal_result(
    result: AgentRunResult,
    expectation: Expectation,
    file_identifier: str
):
    messages = result.all_messages()

    match expectation:
        case 'error':
            pass  # Test uses pytest.raises, won't reach here
        case 'native':
            # Assert: file content IN tool_return part
            # Assert: NO separate user message with file
            assert_file_in_tool_return(messages, file_identifier)
            assert_no_separate_user_file(messages)
        case 'fallback':
            # Assert: 'See file <id>' in tool_return
            # Assert: separate user msg with 'This is file <id>:' + file
            # Assert: tool returns come BEFORE user parts
            assert_identifier_in_tool_return(messages, file_identifier)
            assert_separate_user_file(messages, file_identifier)
            assert_correct_message_order(messages)
```

### Additional Test Cases (beyond matrix)
- **Mixed content ordering**: `[text, image, dict]` must preserve order
- **Multiple files**: Multiple files in single return

---

## Backwards Compatibility

### Existing multimodal tests to re-run after changes
Must verify no regressions in existing multimodal functionality:

- `tests/models/test_bedrock.py` - multimodal user prompts
- `tests/models/test_openai.py` - multimodal user prompts
- `tests/models/test_openai_responses.py` - multimodal
- `tests/models/test_xai.py` - multimodal
- `tests/models/test_groq.py` - image support
- `tests/models/test_openrouter.py` - inherits OpenAI
- `tests/models/test_gemini_vertex.py` - multimodal
- `tests/models/test_cohere.py` - multimodal
- `tests/models/test_mistral.py` - multimodal
- `tests/test_messages.py` - message types

### Models that inherit/wrap (no direct changes needed)
- **OpenRouter** → inherits `OpenAIChatModel`
- **Gemini** → uses same infra as Google

---

## Files to Modify

1. `messages.py` - properties, type annotation, deprecation
2. `_agent_graph.py` - stop splitting
3. `anthropic.py` - hybrid (image/doc native, audio/video fallback)
4. `bedrock.py` - native (image/doc/video), error (audio)
5. `google.py` - all native
6. `openai.py` - chat (all fallback), responses (native except video)
7. `xai.py` - fallback (image/doc), error (audio/video)
8. `groq.py` - fallback (image), error (doc/audio/video)
9. `tests/models/test_multimodal_tool_returns_matrix.py` - parametrized VCR tests

---

## Key Pitfalls
1. Don't reuse `_map_user_prompt()` for tool results - capabilities differ
2. Preserve order in mixed content - iterate directly, don't group
3. Tool returns before user parts - ordering matters for APIs
4. AudioUrl on Bedrock → error, not silent None
5. Run ALL existing multimodal tests after changes to catch regressions
