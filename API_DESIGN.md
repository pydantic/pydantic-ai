# Unified Thinking Settings — Final API Design

> **PR**: #3894 · **Issue**: #3756 · **Branch**: `unified-thinking-settings`
> **Status**: Revised design incorporating maintainer review feedback

---

## 1. Design Philosophy

Three principles drive this design, derived from maintainer feedback and pydantic-ai's existing patterns:

### Portability First

Users should be able to write `model_settings={'thinking': True, 'thinking_effort': 'high'}` and have it work across *all* models. Models that don't support thinking silently ignore it. Models that don't support effort control just enable thinking at their default depth. This makes it trivial to swap models without changing settings.

> *"We prefer documenting lack of support + dropping silently over warnings or errors, because we want it to be very easy to use the same agent with different models."* — Douwe

### Effort, Not Budgets

All three major providers are converging on categorical effort levels:

- **Anthropic**: `output_config.effort` with `low | medium | high | max` is the new primary control. `budget_tokens` is deprecated and will be removed in future model releases. `type: "adaptive"` (recommended for Opus 4.6) paired with effort levels is the path forward.
- **Google (Gemini 3)**: `thinking_level: Literal["minimal", "low", "medium", "high"]` replaces `thinking_budget`. The budget parameter is still accepted on Gemini 3 for backward compatibility but results in "suboptimal performance" — Google explicitly recommends migrating to `thinking_level`.
- **OpenAI**: `reasoning_effort` has been the sole control from the start, expanding from `low | medium | high` to include `none`, `minimal`, and `xhigh`.

Token budgets are a legacy mechanism. The unified API should reflect where the ecosystem is heading, not where it's been. Users who need precise token budgets can use provider-specific settings.

### Minimal Surface Area

Only add unified fields that are meaningfully supported by 2+ providers. Single-provider features stay in provider-specific settings. This keeps the API small, learnable, and honest about what's truly cross-provider.

---

## 2. API Surface

### ModelSettings (two flat fields)

```python
class ModelSettings(TypedDict, total=False):
    # ... existing fields ...

    thinking: bool
    """Enable or disable thinking/reasoning.

    - `True`: Enable thinking. Provider picks the best mode automatically
      (adaptive for Anthropic Opus 4.6, enabled for older Anthropic,
      default-on for OpenAI o-series, etc.)
    - `False`: Disable thinking. Silently ignored on models where
      thinking cannot be disabled (o-series, DeepSeek R1, etc.)
    - Omitted: Use provider default behavior.

    When `thinking` is `False`, `thinking_effort` is ignored.

    Provider-specific settings (e.g. `anthropic_thinking`, `openai_reasoning_effort`)
    always take precedence over this unified field.

    Supported by:

    * Anthropic (Claude 3.7+)
    * Gemini (2.5+)
    * OpenAI (o-series, GPT-5+)
    * Bedrock (Claude models)
    * OpenRouter
    * Groq (reasoning models)
    * Cerebras (GLM, GPT-OSS)
    * Mistral (Magistral models — always-on)
    * Cohere (Command A Reasoning)
    * xAI (Grok 3 Mini, Grok 4)
    """

    thinking_effort: Literal['low', 'medium', 'high']
    """Control the depth of thinking/reasoning.

    - `'low'`: Minimal thinking, faster responses, lower cost
    - `'medium'`: Balanced thinking depth (typical default)
    - `'high'`: Deep thinking, most thorough analysis

    Setting `thinking_effort` without `thinking` implicitly enables thinking.
    Silently ignored on models that don't support effort control.

    Provider-specific effort levels (OpenAI's `xhigh`/`minimal`,
    Anthropic's `max`) are available through provider-specific settings.

    Supported by:

    * Anthropic (Opus 4.5+ via `output_config.effort`)
    * Gemini 3 (via `thinking_level`)
    * OpenAI (o-series, GPT-5+ via `reasoning_effort`)
    * OpenRouter (via `reasoning.effort`)
    * xAI (Grok 3 Mini only, via `reasoning_effort`)
    """
```

### What's NOT in the unified API (and why)

| Feature | Why excluded | Where it lives |
|---------|-------------|----------------|
| `budget_tokens` | Deprecated by Anthropic; Gemini 3 recommends levels over budgets; OpenAI never supported it. Only 2 legacy providers use it. | `anthropic_thinking`, `google_thinking_config` |
| `include_in_response` | Not universal — Anthropic always returns thinking; OpenAI has no hide toggle. Google's `include_thoughts` is specific to their response format. | `google_thinking_config`, `groq_reasoning_format` |
| `summary` | OpenAI-only feature (`reasoning_summary`). No second provider supports it. | `openai_reasoning` dict |
| `reasoning_format` | Output formatting concern (Groq/Cerebras), not thinking control. | `groq_reasoning_format`, Cerebras `extra_body` |
| `max` / `xhigh` / `minimal` / `none` effort | Provider-specific granularity beyond the common `low/medium/high` set. | Provider-specific settings |
| `token_budget` | Cohere-only budget control. No effort levels available. | `cohere_thinking` (future provider-specific field) |
| `prompt_mode` | Mistral-only. Controls system prompt, not thinking behavior. | `mistral_prompt_mode` (future provider-specific field) |
| `encrypted_content` | xAI ZDR privacy feature, grok-4 only. | `xai_include_encrypted_content` |
| `xai_reasoning_effort` (2 levels) | Only `"low"` / `"high"`, only grok-3-mini. Too narrow for unified API. | `xai_reasoning_effort` (future provider-specific field) |

---

## 3. Precedence Rules

```
1. Provider-specific settings ALWAYS win.
   If `anthropic_thinking` is set, `thinking` and `thinking_effort` are ignored for Anthropic.

2. `thinking=False` makes `thinking_effort` irrelevant.
   Thinking is off; there's no effort to control.

3. `thinking_effort` set without `thinking` → implicitly enables thinking.
   `{'thinking_effort': 'high'}` is equivalent to `{'thinking': True, 'thinking_effort': 'high'}`.

4. `thinking=True` without `thinking_effort` → provider default depth.
   Enable thinking but let the provider choose its own default effort/budget.
```

---

## 4. Error Handling Policy

**No errors. No warnings. Silent drop.**

The unified `thinking` and `thinking_effort` fields are "best effort" — they work where they can, and are silently dropped where they can't. This follows the existing pattern for fields like `max_tokens` and `temperature`, which are also not supported by every model but don't raise errors.

| Scenario | Behavior |
|----------|----------|
| `thinking=True` on non-thinking model | Silently ignored |
| `thinking=False` on always-on model (o-series, R1) | Silently ignored; model continues to think |
| `thinking_effort='high'` on model without effort control | Thinking enabled (if supported), effort ignored |
| `thinking_effort='high'` on non-thinking model | Silently ignored entirely |

**Rationale**: The docstrings clearly document which providers support each setting. Users who need guaranteed behavior should use provider-specific settings. The unified API is about convenience and portability, not strictness.

**Contrast with provider-specific settings**: Provider-specific settings (e.g., `anthropic_thinking`, `openai_reasoning_effort`) MAY raise errors for invalid values, because the user has explicitly chosen to use a provider-specific feature and expects it to work.

---

## 5. Profile Fields

### Keep in `ModelProfile`

```python
@dataclass
class ModelProfile:
    # ... existing fields ...

    supports_thinking: bool = False
    """Whether the model supports thinking/reasoning capabilities."""

    thinking_always_enabled: bool = False
    """Whether thinking is always on and cannot be disabled
    (e.g., o-series, GPT-5, DeepSeek R1)."""
```

Two boolean flags. That's it.

### Remove from `ModelProfile`

| Field | Why remove | Where it goes |
|-------|-----------|---------------|
| `supports_thinking_level` | Google-specific implementation detail (Gemini 3 vs 2.5 distinction) | `GoogleModel` internal logic, keyed off model name |
| `default_thinking_budget` | Provider-family-specific constant (e.g., 4096 for Anthropic) | Constant in `AnthropicModel`, `GoogleModel`, etc. |
| `effort_to_budget_map` | Provider-family-specific constant, not model-specific. Same map for all Anthropic thinking models, same for all Gemini 2.5 models. | Constant dict in each model class |

**Rationale (from review)**: These mappings are model-*family*-specific, not model-specific. They don't vary between e.g. `claude-opus-4-6` and `claude-sonnet-4-5` — they're constants of the Anthropic provider. Putting them in `ModelProfile` (which is per-model) is misleading and creates unnecessary coupling between profile detection and provider implementation.

### Profile detection patterns

Use `in` instead of `startswith` for model name checks, since some providers add prefixes:

```python
# Good: handles "anthropic.claude-opus-4-6", "claude-opus-4-6", etc.
supports_thinking = any(name in model_name for name in _THINKING_MODELS)

# Bad: breaks when provider adds prefix
supports_thinking = model_name.startswith(_THINKING_MODELS)
```

---

## 6. Resolution Architecture

### Centralized normalization (thinking.py)

```python
@dataclass
class ResolvedThinkingConfig:
    """Normalized thinking configuration after input parsing.

    No validation against profile — that's the model class's job.
    """
    enabled: bool          # True or False
    effort: Literal['low', 'medium', 'high'] | None = None


def resolve_thinking_config(
    model_settings: ModelSettings,
) -> ResolvedThinkingConfig | None:
    """Normalize unified thinking settings into a canonical form.

    Returns None if no thinking settings are specified.
    Does NOT validate against model capabilities — that happens in each model class.
    """
    thinking = model_settings.get('thinking')
    effort = model_settings.get('thinking_effort')

    # Nothing set → no unified thinking config
    if thinking is None and effort is None:
        return None

    # thinking=False → disabled (effort ignored per precedence rule 2)
    if thinking is False:
        return ResolvedThinkingConfig(enabled=False)

    # thinking=True or effort set (implicit enable, precedence rule 3)
    return ResolvedThinkingConfig(
        enabled=True,
        effort=effort,
    )
```

Key changes from current implementation:
- **No profile parameter** — normalization doesn't validate against capabilities
- **No UserError** — invalid combos are the model class's problem to silently handle
- **No budget_tokens, include_in_response, summary** — removed from unified API
- **Reads flat fields** from `ModelSettings`, not a nested `ThinkingConfig` dict

### Per-provider translation (in model classes)

Each model class checks profile capabilities and translates to native format. The model code should be a simple pass-through, not a complex resolver.

```
ModelSettings (user input)
    ├── thinking: bool
    └── thinking_effort: 'low' | 'medium' | 'high'
         │
         ▼
resolve_thinking_config() — pure normalization
    └── ResolvedThinkingConfig(enabled, effort)
         │
         ▼
Model._resolve_thinking_config() — per-provider
    ├── Checks own profile capabilities (silent drop)
    ├── Applies provider-specific defaults and constants
    └── Returns native API format
```

---

## 7. Provider Mapping Table

### Anthropic

```python
# Constants in AnthropicModel (not in profile)
_ANTHROPIC_EFFORT_TO_BUDGET: dict[str, int] = {
    'low': 1024, 'medium': 4096, 'high': 16384,
}
_DEFAULT_THINKING_BUDGET = 4096
```

| Input | Opus 4.6 (adaptive-capable) | Older thinking models |
|-------|----------------------------|-----------------------|
| `thinking=True` | `type: "adaptive"` | `type: "enabled", budget_tokens: 4096` |
| `thinking=False` | `type: "disabled"` | `type: "disabled"` |
| `effort='low'` | `type: "adaptive"` + `output_config.effort: "low"` | `type: "enabled", budget_tokens: 1024` |
| `effort='medium'` | `type: "adaptive"` + `output_config.effort: "medium"` | `type: "enabled", budget_tokens: 4096` |
| `effort='high'` | `type: "adaptive"` + `output_config.effort: "high"` | `type: "enabled", budget_tokens: 16384` |

**Note**: For Opus 4.6, `type: "adaptive"` is recommended over `type: "enabled"` with `budget_tokens`. The adaptive mode uses effort levels rather than token counts. For older models (Sonnet 4.5, etc.), effort maps to budget since they don't support effort-based control.

**Distinguishing adaptive-capable models**: This is an internal concern of `AnthropicModel`. It can check for known adaptive models (e.g., Opus 4.6+) and fall back to budget-based for older thinking models. This does NOT need a profile flag — it's provider implementation detail.

### Google

```python
# Constants in GoogleModel
_GOOGLE_EFFORT_TO_BUDGET: dict[str, int] = {
    'low': 1024, 'medium': 8192, 'high': 32768,
}
_EFFORT_TO_LEVEL = {
    'low': ThinkingLevel.LOW,
    'medium': ThinkingLevel.MEDIUM,
    'high': ThinkingLevel.HIGH,
}
```

| Input | Gemini 3 (level-based) | Gemini 2.5 (budget-based) |
|-------|----------------------|--------------------------|
| `thinking=True` | No-op (default on) | No-op (default on) |
| `thinking=False` | `thinking_level: "minimal"` (Flash) / silent ignore (Pro — can't disable) | `thinking_budget: 0` (Flash) / silent ignore (Pro) |
| `effort='low'` | `thinking_level: "low"` | `thinking_budget: 1024` |
| `effort='medium'` | `thinking_level: "medium"` | `thinking_budget: 8192` |
| `effort='high'` | `thinking_level: "high"` | `thinking_budget: 32768` |

**Key**: Gemini 3 Pro and Gemini 2.5 Pro CANNOT disable thinking — even `minimal` on Gemini 3 may still think for complex tasks. When `thinking=False` hits a Pro model, we silently ignore rather than error.

**Distinguishing Gemini 3 vs 2.5**: `GoogleModel` already knows the model name. The check for `'gemini-3' in model_name` determines whether to use level-based or budget-based API. This is internal to `GoogleModel`, not a profile concern.

### OpenAI

| Input | o-series (always-on) | GPT-5 (always-on) | GPT-5.1+ (opt-in) |
|-------|---------------------|-------------------|-------------------|
| `thinking=True` | No-op (always thinks) | No-op (always thinks) | `reasoning_effort: "medium"` |
| `thinking=False` | Silent ignore (can't disable) | Silent ignore | No-op (default off) |
| `effort='low'` | `reasoning_effort: "low"` | `reasoning_effort: "low"` | `reasoning_effort: "low"` |
| `effort='medium'` | `reasoning_effort: "medium"` | `reasoning_effort: "medium"` | `reasoning_effort: "medium"` |
| `effort='high'` | `reasoning_effort: "high"` | `reasoning_effort: "high"` | `reasoning_effort: "high"` |

**Direct 1:1 mapping** — OpenAI's `reasoning_effort` values align exactly with our `thinking_effort` values.

### Bedrock (Claude models)

Forwards Anthropic-format thinking config via `additionalModelRequestFields`:

```python
additional_fields['thinking'] = {
    'type': 'adaptive',  # or 'enabled'/'disabled'
}
```

- Maps identically to Anthropic (since Bedrock routes to Claude)
- Uses `snake_case` matching Anthropic's API (not Bedrock's usual camelCase)
- For non-Claude models on Bedrock: silently ignore `thinking` settings unless we confirm they support thinking with different request fields
- DeepSeek R1 on Bedrock: always-on, no config needed

### OpenRouter

Maps to OpenRouter's `reasoning` object, letting OpenRouter handle per-provider translation:

| Input | OpenRouter mapping |
|-------|--------------------|
| `thinking=True` | `reasoning.enabled: true` |
| `thinking=False` | `reasoning.effort: "none"` |
| `effort='low'` | `reasoning.effort: "low"` |
| `effort='medium'` | `reasoning.effort: "medium"` |
| `effort='high'` | `reasoning.effort: "high"` |

**Key insight**: OpenRouter has already solved the unified-to-provider translation problem. Rather than re-implementing their effort-to-budget ratios, we pass our effort level through and let OpenRouter handle the mapping to whatever underlying provider is being used.

### Groq

| Input | Behavior |
|-------|----------|
| `thinking=True` | `reasoning_format: "parsed"` |
| `thinking=False` | `reasoning_format: "hidden"` (note: still bills for reasoning tokens) |
| `effort` | Silently ignored (Groq SDK v0.25 lacks `reasoning_effort`; future upgrade may enable this) |

**Note**: Groq's `reasoning_format: "hidden"` doesn't truly disable thinking — it only suppresses output while still billing for reasoning tokens. Only Qwen 3's `reasoning_effort='none'` truly disables. This is a provider limitation we document but don't try to paper over.

### Cerebras

| Input | GLM models | GPT-OSS models | Qwen3 |
|-------|-----------|----------------|-------|
| `thinking=True` | `disable_reasoning: false` | No-op (default) | No-op (always reasons) |
| `thinking=False` | `disable_reasoning: true` | Lowest effort | Silent ignore (can only `hidden`) |
| `effort` | Silent ignore (no effort control) | `reasoning_effort` passthrough | Silent ignore |

### Mistral (Magistral)

| Input | Mapping |
|-------|---------|
| `thinking=True` | No-op (Magistral always thinks) |
| `thinking=False` | Silently ignored (thinking cannot be disabled) |
| `effort` | Silently ignored (no effort control in Mistral API) |

**Note**: Magistral is the simplest provider — zero API parameters for thinking control. The model always produces `ThinkChunk` output. `prompt_mode: "reasoning"` controls the system prompt, not thinking behavior, and stays provider-specific. Most similar to DeepSeek R1.

### Cohere (Command A Reasoning)

| Input | Mapping |
|-------|---------|
| `thinking=True` | `Thinking(type='enabled')` or no-op (default enabled) |
| `thinking=False` | `Thinking(type='disabled')` |
| `effort` | Silently ignored (Cohere has no effort/level control) |

**Note**: Cohere's `token_budget` stays provider-specific since the unified API is effort-only. Users who need budget control can use `cohere_thinking: Thinking(type='enabled', token_budget=N)` directly. No temperature or tool constraints documented (unique among providers).

### xAI / Grok

| Input | grok-3-mini | grok-4 / reasoning variants | non-reasoning variants |
|-------|------------|----------------------------|----------------------|
| `thinking=True` | No-op | No-op (always on) | Silently ignored |
| `thinking=False` | Silently ignored | Silently ignored | No-op (already off) |
| `effort='low'` | `reasoning_effort: "low"` | Silently ignored | Silently ignored |
| `effort='medium'` | `reasoning_effort: "low"` (downmap) | Silently ignored | Silently ignored |
| `effort='high'` | `reasoning_effort: "high"` | Silently ignored | Silently ignored |

**Note**: xAI uses a unique model-variant pattern — `grok-4-fast-reasoning` vs `grok-4-fast-non-reasoning` are separate model names, not a parameter toggle. Only `grok-3-mini` has `reasoning_effort` with 2 levels (`"low"` / `"high"`). Our 3-level `low/medium/high` maps conservatively: `low`+`medium` → `"low"`, `high` → `"high"`. `encrypted_content` stays in `XaiModelSettings` (ZDR privacy feature, grok-4 only).

---

## 8. Terminology

Use **"thinking"** consistently in all user-facing text (settings, docstrings, error messages, docs). Reserve provider-specific terms only when referring to the provider's actual API:

| Context | Use | Don't use |
|---------|-----|-----------|
| User-facing field name | `thinking`, `thinking_effort` | `reasoning`, `reasoning_effort` |
| Docstring explanation | "thinking/reasoning" (first mention), then "thinking" | Just "reasoning" |
| Provider-specific field | `openai_reasoning_effort` (matches their API) | `openai_thinking_effort` |
| Internal variable | `resolved_thinking`, `thinking_config` | `reasoning_config` |

---

## 9. Implementation Checklist

### Core Changes
- [ ] Flatten `thinking: bool | ThinkingConfig` → `thinking: bool` in `ModelSettings`
- [ ] Add `thinking_effort: Literal['low', 'medium', 'high']` as separate field in `ModelSettings`
- [ ] Remove `ThinkingConfig` TypedDict entirely
- [ ] Simplify `resolve_thinking_config()` — pure normalization, no profile validation, no errors
- [ ] Simplify `ResolvedThinkingConfig` — only `enabled: bool` + `effort: str | None`
- [ ] Remove `supports_thinking_level`, `default_thinking_budget`, `effort_to_budget_map` from `ModelProfile`
- [ ] Keep `supports_thinking` and `thinking_always_enabled` in `ModelProfile`
- [ ] Update exports in `__init__.py` (remove `ThinkingConfig`)

### Provider Implementations
- [ ] Anthropic: effort-to-budget constant, adaptive vs enabled logic, silent drop
- [ ] Google: level vs budget logic (internal), effort-to-level/budget constants, silent drop for Pro disable
- [ ] OpenAI: 1:1 effort mapping, silent drop for always-on disable
- [ ] Bedrock: forward Anthropic-format config, guard against non-Claude models
- [ ] OpenRouter: map to `reasoning` object, let OpenRouter handle translation
- [ ] Groq: enable/disable via `reasoning_format`, silent drop for effort
- [ ] Cerebras: model-family dispatch, silent drop for unsupported features
- [ ] Mistral: no-op (always-on, no config params to send)
- [ ] Cohere: map to `cohere.Thinking(type=..., token_budget=...)`, silent drop for effort
- [ ] xAI: `reasoning_effort` for grok-3-mini (low+medium→low, high→high), silent drop for others

### Profile Updates
- [ ] All profile files: use `in` instead of `startswith` for model name checks
- [ ] Remove `effort_to_budget_map` and `default_thinking_budget` from all profiles
- [ ] Remove `supports_thinking_level` from Google profiles
- [ ] Keep `supports_thinking` and `thinking_always_enabled` accurate per model
- [ ] `profiles/mistral.py`: Add `thinking_always_enabled=True` for Magistral
- [ ] `profiles/cohere.py`: Add `supports_thinking=True` for `command-a-reasoning`
- [ ] `profiles/grok.py`: Add `supports_thinking`, `thinking_always_enabled` based on model variant

### Tests
- [ ] Remove all `UserError` assertion tests for unsupported thinking combos
- [ ] Add tests verifying silent drop behavior (settings ignored, no error)
- [ ] Per-provider tests: verify correct native params generated from unified settings
- [ ] Precedence tests: provider-specific settings override unified
- [ ] Cross-model portability test: same settings work across all providers without error

### Documentation
- [ ] Docstrings: list supported providers with `Supported by:` pattern (matching existing fields)
- [ ] Provider docs: note which thinking features are unified vs provider-specific
- [ ] Remove PLAN.md before merge (working document, not part of the codebase)

---

## 10. Open Questions

### Should `thinking_effort` alone imply `thinking=True`?

**Recommendation: Yes.** Setting `thinking_effort='high'` without `thinking=True` should implicitly enable thinking. This is more ergonomic — users shouldn't have to write both fields when the intent is obvious. Documented in precedence rule 3.

### Should Groq SDK be upgraded for `reasoning_effort` support?

The current SDK (v0.25) lacks `reasoning_effort` (added in v0.28+). Options:
1. **Upgrade SDK** — enables effort support for GPT-OSS models on Groq
2. **Use `extra_body`** — works without SDK upgrade but loses type safety
3. **Skip for now** — enable/disable only, document the limitation

**Recommendation**: Upgrade if straightforward; otherwise use `extra_body` for now and upgrade later.

### Should `thinking=False` on always-on models warn?

The design says silent drop. But `thinking=False` on o-series means thinking *stays on*, which could surprise users expecting cost savings. Counter-argument: if users need guaranteed behavior, they should use provider-specific settings.

**Recommendation: Silent drop.** Consistent with the portability-first philosophy. Document the behavior clearly in the `thinking` field docstring ("Silently ignored on models where thinking cannot be disabled").

### Does Bedrock need thinking support for non-Claude models?

Currently, Bedrock only handles Claude-format thinking config. If other Bedrock-hosted models (e.g., Llama, Mistral) gain thinking support, they may need different request field formats.

**Recommendation**: Guard with a model-family check in `BedrockModel._resolve_thinking_config()`. Only inject thinking config for known Claude models. Silently ignore for others. Expand as needed.
