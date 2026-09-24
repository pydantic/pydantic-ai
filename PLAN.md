# Plan: Unified Thinking Follow-ups

> **Parent PRs**: #4640 (capabilities), #4812 (unified thinking), #4829 (thinking fixes)
> **Date**: 2026-04-02
> **Branch**: `thinking-cap-context`

---

## Background

The unified `thinking` setting landed across PRs #4640, #4812, and #4829. The design is simple: users set `thinking=True` or `thinking='high'` in `ModelSettings`, and each model's `_translate_thinking()` method converts that into the provider's native API format. The base `Model.prepare_request()` resolves the setting, strips it from `model_settings`, and places it on `model_request_parameters.thinking` for downstream consumption.

This works well for Anthropic, OpenAI, Google, Groq, Bedrock, xAI, OpenRouter, and Cerebras — all of which have `_translate_thinking()` implementations. But during review, Douwe identified several gaps that were explicitly deferred to follow-up PRs. This plan covers all five of them.

---

## 1. Cohere: Wire up `_translate_thinking()`

**Problem:** The Cohere profile sets `supports_thinking=True` and `thinking_always_enabled=True` for reasoning models, and the response handler already processes `ThinkingPart`. But the *request* side never reads `model_request_parameters.thinking` — the unified setting is silently ignored.

**Cohere's API:** The SDK (`cohere==5.20.6`) exposes `thinking: Thinking | None` on `AsyncClientV2.chat()`, where `Thinking` is:

```python
Thinking(type: Literal['enabled', 'disabled'], token_budget: int | None = None)
```

This is a simple binary toggle with an optional token budget. There are no effort levels — `'low'`, `'medium'`, and `'high'` all map to the same `Thinking(type='enabled')`. Users who want fine-grained `token_budget` control can use a new `cohere_thinking` provider-specific setting.

**Why this is straightforward:** Cohere has no streaming support in pydantic-ai yet, so there's only one call site to modify (`_chat()`). The SDK sentinel is `OMIT`, already imported.

**Implementation:**

Add `_translate_thinking()` to `CohereModel`, following the same pattern as Groq (the simplest existing reference):

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

Then pass it into the `client.chat()` call alongside the existing parameters. Add `cohere_thinking: Thinking` to `CohereModelSettings` for users who need token budget control.

| Unified | Cohere native |
|---------|--------------|
| `True` / any effort | `Thinking(type='enabled')` |
| `False` | `Thinking(type='disabled')` |
| omitted | `OMIT` (provider default) |

**Files:** `models/cohere.py`, `tests/test_thinking.py`, `docs/thinking.md`

---

## 2. Mistral: Wire up `_translate_thinking()`

**Problem:** Same as Cohere — the profile flags are set for Magistral models, and `ThinkingPart` is processed in responses via `MistralThinkChunk`, but the request side never passes the thinking parameter.

**Mistral's API:** The SDK (`mistralai==1.9.11`) uses a different mechanism than most providers. Instead of a thinking-specific parameter, it uses `prompt_mode: MistralPromptMode`:

```python
MistralPromptMode = Literal["reasoning"] | UnrecognizedStr
```

Setting `prompt_mode='reasoning'` enables the reasoning system prompt for Magistral models. Omitting it (via `UNSET`) uses the default behavior. Like Cohere, this is purely binary — there are no effort levels.

**What makes Mistral trickier:** Unlike Cohere's single call site, Mistral has **four** places where chat calls are made:

1. `_completions_create()` — non-streaming
2. `_stream_completions_create()` — streaming with function tools
3. `_stream_completions_create()` — streaming with output tools / JSON mode
4. `_stream_completions_create()` — plain streaming (currently only passes `model`, `messages`, `stream`, `http_headers`)

All four need `prompt_mode=self._translate_thinking(...)` added. The SDK sentinel is `UNSET`, already imported.

**Implementation:**

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

For the provider-specific setting in `MistralModelSettings`, we use `Literal['reasoning']` rather than `MistralPromptMode` to avoid exposing the SDK's `UnrecognizedStr` union in our public API. This matches pydantic-ai's convention of using simple Literal types for provider-specific enums (see `GroqModelSettings.groq_reasoning_format`).

| Unified | Mistral native |
|---------|---------------|
| `True` / any effort | `prompt_mode='reasoning'` |
| `False` (always-on) | `UNSET` (silently ignored — Magistral can't disable) |
| omitted | `UNSET` (provider default) |

**Files:** `models/mistral.py`, `tests/test_thinking.py`, `docs/thinking.md`

---

## 3. Profile flags for profile-only providers

**Problem:** Several model families support thinking but their profile functions don't set `supports_thinking` or `thinking_always_enabled`. This means the unified `thinking` setting is silently dropped even when the model actually supports it — users get no feedback that their setting was ignored.

These are "profile-only" providers because they're accessed through hosting providers like OpenRouter, Ollama, Fireworks, or Together, which already have model implementations. The profile is the only touch point pydantic-ai has.

**Proposed changes:**

### Meta (Llama-4)

Llama-4 reasoning variants include `"reasoning"` in the model name (e.g., `llama-4-scout-reasoning`). They use `<think>` tags for thinking output, which is already the default on `ModelProfile.thinking_tags`. These models always think — there's no API parameter to disable it.

```python
def meta_model_profile(model_name: str) -> ModelProfile | None:
    is_reasoning = 'reasoning' in model_name.lower()
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        supports_thinking=is_reasoning,
        thinking_always_enabled=is_reasoning,
    )
```

### Qwen (QwQ)

QwQ models (e.g., `qwq-32b-preview`) are Qwen's reasoning-focused family. They also use `<think>` tags and always think. The existing `qwen_model_profile()` already handles multiple model families — QwQ detection needs to be added to the default return path.

### MoonshotAI (Kimi Thinking)

Kimi thinking models include `"thinking"` in the name (e.g., `kimi-thinking-preview`). Unlike Meta/Qwen, these are optional-thinking (not always-on), so only `supports_thinking=True` is needed.

### Harmony and ZAI

These need additional research before implementation. The original plan references `gpt-oss-120b` and `zai-glm-4.6` as thinking-capable, but we should verify this against current API documentation before setting flags. If the research doesn't clearly confirm thinking support, we skip them rather than risk setting incorrect profile flags.

**Files:** `profiles/meta.py`, `profiles/qwen.py`, `profiles/moonshotai.py`, possibly `profiles/harmony.py` and `profiles/zai.py`, `tests/test_thinking.py`

---

## 4. `thinking_mode` property on `ModelProfile`

**Problem:** The base `prepare_request()` currently checks two booleans inline:

```python
if self.profile.supports_thinking or self.profile.thinking_always_enabled:
    if not (thinking_value is False and self.profile.thinking_always_enabled):
```

This works but is a bit clunky. The two boolean flags (`supports_thinking`, `thinking_always_enabled`) represent a three-state enum that would be clearer as a derived property.

**Proposal:**

```python
@property
def thinking_mode(self) -> Literal['unsupported', 'optional', 'always_on']:
    """Derived thinking mode from supports_thinking and thinking_always_enabled."""
    if not self.supports_thinking and not self.thinking_always_enabled:
        return 'unsupported'
    return 'always_on' if self.thinking_always_enabled else 'optional'
```

This is primarily a readability improvement. Douwe agreed it was worth adding but was lukewarm on refactoring `prepare_request()` to use it ("benefit's not super clear"), so the initial scope should be limited to adding the property and using it only where it clearly improves readability.

**Files:** `profiles/__init__.py`, optionally `models/__init__.py`, `tests/test_thinking.py`

---

## 5. Rename Google `google_supports_thinking_level`

**Problem:** The `GoogleModelProfile` field `google_supports_thinking_level: bool` is a boolean that distinguishes between Gemini 2.5 (budget-based) and Gemini 3+ (level-based) thinking APIs. A `Literal['budget', 'level'] | None` would be more expressive and self-documenting.

**Caution:** Douwe explicitly flagged backward compatibility concerns here ("I don't think this is worth doing, especially as it'd be backward incompatible"). Since `GoogleModelProfile` is a public dataclass, users may be constructing instances with `google_supports_thinking_level=True`.

**Approach:** Add the new `google_thinking_api: Literal['budget', 'level'] | None` field alongside the old one. Deprecate the old field but keep it working — in `__post_init__`, if the old field is set and the new one isn't, bridge them. This lets us migrate without breaking existing code.

**Files:** `profiles/google.py`, `models/google.py`, `tests/test_thinking.py`

---

## PR Structure

These five items can be split into 2-3 PRs:

- **PR A** (Cohere + Mistral): Items 1-2. These are the most impactful — they close the "silently ignored" gap for two providers. Tightly scoped and independently testable.
- **PR B** (Profile flags): Item 3. Independent of the model-level changes. Requires research for some providers.
- **PR C** (API hygiene): Items 4-5. Lower priority, purely internal improvements. Can be combined or split further.

---

## Testing

Each item follows the existing test patterns in `tests/test_thinking.py`. The test classes instantiate the model with a specific profile, call `_translate_thinking()` directly, and assert the returned provider-native value.

For Cohere and Mistral, we also need to verify the values actually reach the SDK call. The existing test infrastructure for other providers (e.g., `TestGroqThinkingTranslation`) shows the pattern — construct model, prepare request, call the private method, check the output.

Profile flag tests use the profile factory functions directly (e.g., `meta_model_profile('llama-4-scout-reasoning').supports_thinking` should be `True`).
