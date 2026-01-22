# Implementation Analysis: Reasoning Content Handling (#3867)

This document outlines the analysis, cautions, and tradeoffs for implementing "Send back reasoning content via field if it's received in field" in Pydantic-AI.

## Background
Currently, when Pydantic-AI receives reasoning/thinking content from a model (e.g., OpenAI o1's `reasoning_content` or Anthropic's `thinking`), it maps this to a `ThinkingPart`. However, when sending these messages back to the model in subsequent turns, the default behavior for many providers (especially OpenAI-compatible ones) is to wrap the thinking content in XML-like tags (e.g., `<think>...</think>`) and append it to the main `content` string.

Issue #3867 argues that if the content was originally received in a dedicated field, it should be sent back in a dedicated field by default.

## Cautions

### 1. Provider-Specific Field Names
OpenAI uses `reasoning_content`, while other "compatible" providers use `reasoning` (Ollama/OpenRouter) or other custom fields. We must ensure we use the *correct* field name when sending it back.
- **Risk**: Sending an unknown field to a strict API will cause a 400 Bad Request.

### 2. State Mapping in `ThinkingPart`
The `ThinkingPart` currently uses the `id` field to store the name of the provider field it was received from (in `openai.py`). We must rely on this or a more explicit mechanism to know *which* field to use for the return trip.

### 3. Model Switching
If a conversation starts with a model that uses `reasoning_content` and then switches to one that uses `<think>` tags, the `ThinkingPart` needs to be transformed correctly. 
- **Decision**: The model implementation should be responsible for deciding how to represent a `ThinkingPart` received from "any" provider.

### 4. Streaming Dfiferences
Streaming responses handle thinking via `ThinkingPartDelta`. We must ensure that the logic for "field vs tags" is consistent between synchronous runs and streaming runs.

## Tradeoffs

### Approach A: Automatic Field Detection (Implicit)
If `ThinkingPart.id` matches a known reasoning field for the current provider, default to sending it back via that field.
- **Pros**: Zero-config for users. Just works "out of the box".
- **Cons**: Might be surprising if users expect the tags. Potential for field name collisions.

### Approach B: New Profile Option `include_method='auto'` (Explicit)
Introduce a new state for `openai_chat_send_back_thinking_parts` called `'auto'`, which is now the new default.
- **Pros**: Explicit, manageable, and provides a clear fallback path.
- **Cons**: Adds another layer of configuration complexity.

### Approach C: Stick to `ThinkingPart.id` + `provider_name`
Only send back via field if `provider_name` matches the current provider AND `id` is a known field.
- **Pros**: Safest. Prevents sending OpenAI-specific fields to Groq, etc.
- **Cons**: Might miss opportunities to use fields when switching between compatible providers (e.g. from Ollama to vLLM).

## Final Plan Proposal

1.  **Refine `ThinkingPart` handling in `OpenAIChatModel`**:
    *   Update `_map_response_thinking_part` in `openai.py` to prioritize using a field if the `ThinkingPart` has an `id` that matches a known reasoning field, *or* if the profile explicitly specifies a field.
2.  **Update `OpenAIModelProfile` defaults**:
    *   Consider changing the default of `openai_chat_send_back_thinking_parts` to a value that allows "preferring fields if available".
3.  **Cross-Check Anthropic/Bedrock**:
    *   Verify that they already handle this safely (preliminary check says yes, but we should confirm via tests).

## Potential Failure Points

*   **Test Failures**: Existing tests might expect `<think>` tags in the resulting message params. These will need to be updated or the new behavior must be opt-in if we fear breaking changes.
*   **API Rejections**: If a provider returns `reasoning_content` but then *refuses* it in the assistant message for the next turn. (OpenAI o1-preview had some quirks like this early on).
*   **VCR Cassettes**: Many tests use recorded responses. Changing the request format will invalidate some cassettes or require "manual" mocking updates.

## Long-term Considerations
As more providers converge on a standard (likely `reasoning_content`), our implementation should naturally favor that. We should avoid hardcoding too many provider-specific quirks and instead rely on the `ModelProfile` system.

## Detailed Code Rationale

### 1. Dictionary-based Thinking Storage (`dict[str, list[str]]`)
In `OpenAIChatModel._MapModelResponseContext`, we moved from a simple `thinkings: list[str]` to a mapping.
- **Why**: Different providers return thinking in different fields (e.g., `reasoning`, `reasoning_content`). If we merge them into a single list, we lose the information of *which* field they belonged to.
- **Decision vs Alternative**: I considered adding a `target_field` attribute to `ThinkingPart`, but that would require a larger API change across the entire framework. Using the existing `id` field and a dictionary in the model-specific mapper is much cleaner and preserves the mapping without changing the public `ThinkingPart` interface significantly.

### 2. Multi-Field Extraction Policy
In `OpenAIChatModel._map_response_thinking_parts`, the previous code would `return` as soon as it found *any* reasoning field.
- **What Changed**: We now use a `seen_fields` set and continue iterating through all possible fields.
- **Why**: Some aggregator providers (like OpenRouter) or local setups (Ollama) might technically return multiple metadata fields. By capturing all of them as separate `ThinkingPart`s, we ensure no information is lost and everything can be round-tripped.

### 3. The `item.id != 'content'` Safety Check
- **Context**: When Pydantic-AI parses thinking from XML tags (e.g. `<think>`), it often assigns `'content'` as the source ID.
- **Why**: If we blindly used `item.id` as a field name in the OpenAI payload, we would end up sending something like `{"role": "assistant", "content": "actual content", "content": "thinking content"}`. This is invalid JSON or would cause the API to reject the request. We explicitly filter out `'content'` to ensure thinking always falls back to the configured `openai_chat_thinking_field` (usually `reasoning_content`) when it originates from tags.

### 4. Intentional ID Standardizing for Other Providers
- **What**: Added `id='thinking'` to Anthropic/Mistral and `id='reasoning'` to Groq.
- **Why**: This prevents "Field Drift". If you start a chat with Anthropic and then switch the model to an OpenAI-compatible one (like vLLM) mid-conversation, the `ThinkingPart` received from Anthropic will now have an ID that the next provider can potentially use to route it into a dedicated field rather than merging it into text tags.

## Status: Implemented & Stabilized 

The changes have been implemented across OpenAI, Anthropic, Mistral, and Groq providers. Verification tests have been added to the main test suite:
- `test_openai_reasoning_roundtrip_multiple_fields`
- `test_openai_reasoning_roundtrip_custom_field_id`
- `test_openai_reasoning_streaming_roundtrip`
in `tests/models/test_openai.py`.

## CI Failures and Regressions

The introduction of default `id` values for `ThinkingPart` across different providers (Anthropic, Mistral, Groq) caused existing tests to fail because they expected `id=None`.

### Reasons for Failure
1.  **Assertion Errors**: Existing snapshots and manual assertions in `tests/models/test_anthropic.py`, `tests/models/test_groq.py`, and `tests/models/test_mistral.py` were strictly checking for `ThinkingPart(...)` without the `id` field.
2.  **Linting**: Custom tests added to `test_openai.py` might have violated formatting rules or required `inline-snapshot` updates.
3.  **PR Labels**: The CI required a "feature" label which was missing.

### Fixes Applied
1.  **Test Updates**: Updated all failing assertions in the respective provider test suites to expect `id='thinking'` (Anthropic/Mistral) or `id='reasoning'` (Groq).
2.  **Linting Fix**: Re-formatted test cases in `test_openai.py` to match the project's standards.
3.  **Labeling**: Applied the "feature" label to the PR.
