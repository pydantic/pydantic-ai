# Plan: Unified Thinking Follow-ups (All 5 Items)

## Context

PR #4829 merged the initial thinking fixes. Douwe explicitly deferred 5 items to follow-up PRs:
1. **Cohere** `_translate_thinking()` (or remove profile flags)
2. **Mistral** `_translate_thinking()` (or remove profile flags)
3. **Profile flags** for Meta, Qwen, MoonshotAI, Harmony, ZAI
4. **`thinking_mode` property** on `ModelProfile`
5. **Rename Google `google_supports_thinking_level`** field

## Research Findings

### Cohere SDK (`cohere==5.20.6`)
- `AsyncClientV2.chat()` accepts `thinking: Thinking | None = OMIT`
- `Thinking(type: Literal['enabled', 'disabled'], token_budget: int | None = None)`
- Binary on/off + optional token budget. No effort levels.
- Response already processes `ThinkingPart` (lines 207-208).
- **No streaming** — only `request()`, no `request_stream()`. Only need to modify `_chat()`.
- Profile: `cohere_model_profile()` already sets `supports_thinking=True, thinking_always_enabled=True` for models with `'reasoning'` in the name.

### Mistral SDK (`mistralai==1.9.11`)
- `chat.complete_async()` and `chat.stream_async()` both accept `prompt_mode: MistralPromptMode | UNSET`
- `MistralPromptMode = Literal["reasoning"] | UnrecognizedStr` — single value, not an enum of levels.
- Binary: set `prompt_mode='reasoning'` to enable reasoning mode; omit to use default.
- Response already processes `ThinkingPart` via `MistralThinkChunk` (lines 542-561).
- **Has streaming** — need to modify `_completions_create()` (line 234) AND `_stream_completions_create()` (lines 271, 294, 313) — 4 call sites.
- Profile: `mistral_model_profile()` already sets `supports_thinking=True, thinking_always_enabled=True` for `magistral*` models.

### Existing `_translate_thinking` pattern (from Groq — simplest reference)
```python
# models/groq.py:246-260
def _translate_thinking(self, model_settings, model_request_parameters):
    if fmt := model_settings.get('groq_reasoning_format'):  # provider-specific wins
        return fmt
    thinking = model_request_parameters.thinking
    if thinking is False:
        return 'hidden'
    if thinking is not None:
        return 'parsed'
    return NOT_GIVEN
```
All providers follow the same shape:
1. Check provider-specific setting first (it wins)
2. Read `model_request_parameters.thinking` (already resolved by base `prepare_request()`)
3. Map to provider's native format
4. Return OMIT/NOT_GIVEN/None when not applicable

---

## PR A: Cohere + Mistral `_translate_thinking()`

### A1. Cohere — `pydantic_ai_slim/pydantic_ai/models/cohere.py`

**Add `_translate_thinking()` method** on `CohereModel` (after `_chat`, around line 199):
```python
def _translate_thinking(
    self,
    model_settings: CohereModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> Thinking | type[OMIT]:
    """Get thinking config, falling back to unified thinking."""
    if cohere_thinking := model_settings.get('cohere_thinking'):
        return cohere_thinking
    thinking = model_request_parameters.thinking
    if thinking is None:
        return OMIT
    if thinking is False:
        return Thinking(type='disabled')
    return Thinking(type='enabled')
```

**Modify `_chat()` call** (line 183-194) — add `thinking=self._translate_thinking(model_settings, model_request_parameters)`:
```python
return await self.client.chat(
    model=self._model_name,
    messages=cohere_messages,
    tools=tools or OMIT,
    thinking=self._translate_thinking(model_settings, model_request_parameters),
    # ... rest unchanged
)
```

**Add import:** `from cohere import Thinking` (add to the existing import block, line 39-58).

**Note on `OMIT`:** Cohere SDK uses `OMIT` (from `cohere.v2.client`) as its sentinel for "not provided", similar to how Groq/OpenAI use `NOT_GIVEN`. Already imported at line 58.

### A2. Cohere settings — `pydantic_ai_slim/pydantic_ai/settings.py`

No changes needed yet. The `cohere_thinking` provider-specific setting goes in `CohereModelSettings` (see below).

### A3. Cohere model settings — `CohereModelSettings` in `cohere.py`

Add to the existing empty `CohereModelSettings` class:
```python
class CohereModelSettings(ModelSettings, total=False):
    """Settings used for a Cohere model request."""
    cohere_thinking: Thinking
    """Cohere-specific thinking configuration. Takes precedence over unified `thinking`."""
```

### A4. Mistral — `pydantic_ai_slim/pydantic_ai/models/mistral.py`

**Add `_translate_thinking()` method** on `MistralModel`:
```python
def _translate_thinking(
    self,
    model_settings: MistralModelSettings,
    model_request_parameters: ModelRequestParameters,
) -> MistralOptionalNullable[MistralPromptMode]:
    """Get prompt_mode, falling back to unified thinking."""
    if prompt_mode := model_settings.get('mistral_prompt_mode'):
        return prompt_mode
    thinking = model_request_parameters.thinking
    if thinking is None or thinking is False:
        return UNSET
    return 'reasoning'
```

**Key design note:** Mistral uses `UNSET` (already imported at line 60) as its "not provided" sentinel, analogous to Groq's `NOT_GIVEN` and Cohere's `OMIT`.

**Modify ALL 4 chat call sites** to add `prompt_mode=self._translate_thinking(model_settings, model_request_parameters)`:

1. `_completions_create()` (line 234) — add `prompt_mode=...` param
2. `_stream_completions_create()` with function tools (line 271) — add `prompt_mode=...`
3. `_stream_completions_create()` with output tools / JSON mode (line 294) — add `prompt_mode=...`
4. `_stream_completions_create()` plain (line 313) — add `prompt_mode=...`

**Important:** The plain streaming path (line 313) currently only passes `model`, `messages`, `stream`, and `http_headers`. It needs to also receive `model_settings` and `model_request_parameters` to call `_translate_thinking()`. Currently the method signature does accept them but this call site doesn't pass `prompt_mode`. We just add the kwarg.

**Add import:** `MistralPromptMode` needs to be imported. Check if it's already available from `mistralai.models` — it's defined in `mistralai.models.mistralpromptmode`. Add to the SDK import block.

### A5. Mistral model settings — `MistralModelSettings` in `mistral.py`

Add to the existing empty `MistralModelSettings` class:
```python
class MistralModelSettings(ModelSettings, total=False):
    """Settings used for a Mistral model request."""
    mistral_prompt_mode: Literal['reasoning']
    """Mistral-specific prompt mode for reasoning. Takes precedence over unified `thinking`."""
```

Using `Literal['reasoning']` rather than `MistralPromptMode` to avoid exposing the SDK's `UnrecognizedStr` union in our public API. Matches pydantic-ai's convention of using simple Literal types for provider-specific enums.

### A6. Tests — `tests/test_thinking.py`

Add two new test classes following the existing pattern (e.g., `TestGroqThinkingTranslation`):

**`TestCohereThinkingTranslation`:**
- `test_thinking_true` → `Thinking(type='enabled')`
- `test_thinking_false` → `Thinking(type='disabled')`
- `test_effort_levels_map_to_enabled` → all of `'low'`/`'medium'`/`'high'` → `Thinking(type='enabled')` (no effort granularity)
- `test_provider_specific_wins` → `cohere_thinking` takes precedence
- `test_thinking_none` → returns `OMIT`

**`TestMistralThinkingTranslation`:**
- `test_thinking_true` → `'reasoning'`
- `test_thinking_false` → `UNSET`
- `test_effort_levels_map_to_reasoning` → all → `'reasoning'` (no granularity)
- `test_provider_specific_wins` → `mistral_prompt_mode` takes precedence
- `test_thinking_none` → `UNSET`

### A7. Docs — `docs/thinking.md`

Update the provider mapping table to reflect that Cohere and Mistral now respond to the unified `thinking` setting. Add notes that both are binary (no effort granularity).

---

## PR B: Profile flags for profile-only providers

### B1. Meta — `pydantic_ai_slim/pydantic_ai/profiles/meta.py`

```python
def meta_model_profile(model_name: str) -> ModelProfile | None:
    is_reasoning = 'reasoning' in model_name.lower()
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        supports_thinking=is_reasoning,
        thinking_always_enabled=is_reasoning,
    )
```

Llama-4 reasoning variants (e.g., `llama-4-scout-reasoning`, `llama-4-maverick-reasoning`) include "reasoning" in the name. Tag-based thinking (`<think>` tags) is already handled by `thinking_tags` default on `ModelProfile`.

### B2. Qwen — `pydantic_ai_slim/pydantic_ai/profiles/qwen.py`

Add thinking flags to QwQ models. QwQ uses `<think>` tags, already the default on `ModelProfile.thinking_tags`:
```python
def qwen_model_profile(model_name: str) -> ModelProfile | None:
    is_qwq = 'qwq' in model_name.lower()
    if model_name.startswith('qwen-3-coder'):
        return OpenAIModelProfile(...)  # unchanged
    if _QWEN_3_5_RE.search(model_name):
        return ModelProfile(...)  # unchanged
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_qwq,
        thinking_always_enabled=is_qwq,
    )
```

### B3. MoonshotAI — `pydantic_ai_slim/pydantic_ai/profiles/moonshotai.py`

```python
def moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    is_thinking = 'thinking' in model_name.lower()
    return ModelProfile(
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_thinking,
    )
```

### B4. Harmony — `pydantic_ai_slim/pydantic_ai/profiles/harmony.py`

Need research: verify `gpt-oss-120b` actually supports thinking and what its mechanism is. If it does:
```python
def harmony_model_profile(model_name: str) -> ModelProfile | None:
    profile = openai_model_profile(model_name)
    is_reasoning = 'gpt-oss' in model_name and '120b' in model_name
    return OpenAIModelProfile(
        openai_supports_tool_choice_required=False,
        ignore_streamed_leading_whitespace=True,
        supports_thinking=is_reasoning,
    ).update(profile)
```

### B5. ZAI — `pydantic_ai_slim/pydantic_ai/profiles/zai.py`

Need research: verify `zai-glm-4.6` supports thinking. If it does:
```python
def zai_model_profile(model_name: str) -> ModelProfile | None:
    is_reasoning = 'zai-glm' in model_name
    if is_reasoning:
        return ModelProfile(supports_thinking=True)
    return None
```

### B6. Tests

Add `TestProfileThinkingDetection` cases for each new provider in `tests/test_thinking.py`:
- Meta: `llama-4-scout-reasoning` → `supports_thinking=True`, plain llama → `False`
- Qwen: `qwq-32b-preview` → `supports_thinking=True`, plain qwen → `False`
- MoonshotAI: `kimi-thinking-preview` → `True`, plain moonshot → `False`
- Harmony: `gpt-oss-120b` → `True`
- ZAI: `zai-glm-4.6` → `True`, other → `None`

---

## PR C: API hygiene (thinking_mode + Google field rename)

### C1. `thinking_mode` property — `pydantic_ai_slim/pydantic_ai/profiles/__init__.py`

Add derived property on `ModelProfile` (after line 73):
```python
@property
def thinking_mode(self) -> Literal['unsupported', 'optional', 'always_on']:
    """Derived thinking mode from supports_thinking and thinking_always_enabled."""
    if not self.supports_thinking and not self.thinking_always_enabled:
        return 'unsupported'
    return 'always_on' if self.thinking_always_enabled else 'optional'
```

**Optionally** refactor `prepare_request()` in `models/__init__.py` (line 781) to use it:
```python
# Before:
if self.profile.supports_thinking or self.profile.thinking_always_enabled:
    if not (thinking_value is False and self.profile.thinking_always_enabled):
# After:
if (mode := self.profile.thinking_mode) != 'unsupported':
    if not (thinking_value is False and mode == 'always_on'):
```

Douwe was lukewarm on the refactor part ("benefit's not super clear") — we should add the property but keep the refactor minimal. Only update `prepare_request()` if it makes things cleaner.

### C2. Google field rename — `pydantic_ai_slim/pydantic_ai/profiles/google.py`

Rename `google_supports_thinking_level: bool` → `google_thinking_api: Literal['budget', 'level'] | None`.

**Backward compatibility concern:** Douwe said "I don't think this is worth doing, especially as it'd be backward incompatible" about a full replacement. Approach:
1. Add new field `google_thinking_api: Literal['budget', 'level'] | None = None`
2. Deprecate old field but keep it working during transition
3. In `__post_init__`, if old field is set but new isn't, populate new from old

**Files:**
- `pydantic_ai_slim/pydantic_ai/profiles/google.py` — add field, update factory
- `pydantic_ai_slim/pydantic_ai/models/google.py` — update usages of `google_supports_thinking_level` to `google_thinking_api`

### C3. Tests

- Test `thinking_mode` property returns correct values for each combination
- Test Google `_translate_thinking` still works with new field name

---

## Files to modify (complete list)

### PR A (Cohere + Mistral)
- `pydantic_ai_slim/pydantic_ai/models/cohere.py` — add `_translate_thinking()`, modify `_chat()`, add import, add settings
- `pydantic_ai_slim/pydantic_ai/models/mistral.py` — add `_translate_thinking()`, modify 4 call sites, add import, add settings
- `tests/test_thinking.py` — add `TestCohereThinkingTranslation`, `TestMistralThinkingTranslation`
- `docs/thinking.md` — update provider tables

### PR B (Profile flags)
- `pydantic_ai_slim/pydantic_ai/profiles/meta.py`
- `pydantic_ai_slim/pydantic_ai/profiles/qwen.py`
- `pydantic_ai_slim/pydantic_ai/profiles/moonshotai.py`
- `pydantic_ai_slim/pydantic_ai/profiles/harmony.py`
- `pydantic_ai_slim/pydantic_ai/profiles/zai.py`
- `tests/test_thinking.py` — add profile detection tests

### PR C (API hygiene)
- `pydantic_ai_slim/pydantic_ai/profiles/__init__.py` — add `thinking_mode` property
- `pydantic_ai_slim/pydantic_ai/profiles/google.py` — add `google_thinking_api` field
- `pydantic_ai_slim/pydantic_ai/models/google.py` — update references
- `pydantic_ai_slim/pydantic_ai/models/__init__.py` — optionally refactor `prepare_request()`
- `tests/test_thinking.py` — add property + field tests

---

## Verification

```bash
# Run thinking tests
pytest tests/test_thinking.py -v

# Run affected model tests
pytest tests/models/test_cohere.py tests/models/test_mistral.py -v

# Type checking
make typecheck

# Lint
make lint

# Full test suite
make test
```
