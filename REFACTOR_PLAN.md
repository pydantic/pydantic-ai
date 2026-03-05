# Thinking Resolution Architecture Refactor

> **Context**: DouweM's review on PR #3894 raised concerns that each model's `_resolve_thinking_config` carries too much knowledge about the thinking abstraction. This plan proposes a refactoring modeled on how pydantic-ai already handles its most successful cross-cutting abstraction: **output mode**.

---

## 1. The Established Patterns

Before proposing changes, here's how pydantic-ai handles existing cross-cutting concerns. These patterns reveal clear maintainer preferences:

### Output mode — the gold standard (profile-driven, base-class-resolved)

```
Profile declares:     supports_json_schema_output, default_structured_output_mode
Base class resolves:  'auto' → concrete mode, clears irrelevant fields, validates
Result stored in:     ModelRequestParameters.output_mode (resolved before models see it)
Models translate:     Read params.output_mode at request time → native API format
```

**No model overrides `prepare_request()` for output mode.** The base class does 100% of resolution. Models just read the resolved value and translate.

### Builtin tools — validation-only centralization

```
Profile declares:     supported_builtin_tools (frozenset of tool types)
Model declares:       supported_builtin_tools() classmethod (what's implemented)
Base class validates: Intersection of both, raises UserError for unsupported
Models translate:     _add_builtin_tools() maps to native API format at request time
```

**Base class validates. Models translate.** No resolution hooks needed.

### Basic settings (max_tokens, temperature, etc.) — direct pass-through

```
Declared in:          ModelSettings TypedDict
Merged in:            Model.prepare_request() via merge_model_settings()
Consumed:             Each model reads model_settings.get('field') at request time
Translation:          Inline name mapping (e.g., stop_sequences → stop)
```

**No resolution layer.** Direct read-through with inline name translation.

### Current thinking implementation — the outlier

```
Declared in:          ModelSettings (thinking, thinking_effort)
Normalized in:        Each model calls resolve_thinking_config() itself
Injected in:          Each model's prepare_request() override (9x boilerplate)
Consumed:             model_settings.get('provider_thinking_field') at request time
Strip pattern:        OpenAI subclasses strip parent's injection (Cerebras, OpenRouter)
```

**Every model overrides `prepare_request()` just for thinking.** This violates the patterns above — the base class should resolve, not each model.

---

## 2. The Refactoring: Follow the Output Mode Pattern

### Core change: Add `resolved_thinking` to `ModelRequestParameters`

```python
@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    # ... existing fields ...
    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    # ...

    # NEW: Resolved thinking config (follows output_mode pattern)
    resolved_thinking: ResolvedThinkingConfig | None = None
```

### Base class resolves thinking in `prepare_request()`

```python
# models/__init__.py — Model.prepare_request()

def prepare_request(self, model_settings, model_request_parameters):
    model_settings = merge_model_settings(self.settings, model_settings)
    params = self.customize_request_parameters(model_request_parameters)

    # ... existing output mode resolution ...

    # NEW: Resolve unified thinking settings (mirrors output mode resolution)
    if model_settings:
        resolved_thinking = resolve_thinking_config(model_settings, self.profile)
        if resolved_thinking is not None:
            params = replace(params, resolved_thinking=resolved_thinking)

    return model_settings, params
```

This mirrors how output mode resolution works:
- Profile capabilities (`supports_thinking`, `thinking_always_enabled`) drive the guards
- The canonical result is stored in `ModelRequestParameters`
- Models receive an already-resolved config — they never call `resolve_thinking_config()` themselves

### Models translate at request time — no `prepare_request()` overrides needed for thinking

Each model reads `model_request_parameters.resolved_thinking` where it currently reads the provider-specific field, with provider-specific settings taking precedence:

```python
# Anthropic — in _messages_create():
# BEFORE:
#   thinking=model_settings.get('anthropic_thinking', OMIT),
# AFTER:
if 'anthropic_thinking' in model_settings:
    thinking_config = model_settings['anthropic_thinking']
elif model_request_parameters.resolved_thinking is not None:
    thinking_config = self._translate_thinking(model_request_parameters.resolved_thinking)
else:
    thinking_config = OMIT
```

The `_translate_thinking()` method is a pure translator — no resolution, no profile checks:

```python
def _translate_thinking(self, resolved: ResolvedThinkingConfig) -> BetaThinkingConfigParam:
    """Translate canonical thinking config to Anthropic native format."""
    if not resolved.enabled:
        return {'type': 'disabled'}
    if AnthropicModelProfile.from_profile(self.profile).anthropic_supports_adaptive_thinking:
        return {'type': 'adaptive'}
    budget = EFFORT_TO_BUDGET.get(resolved.effort, DEFAULT_THINKING_BUDGET) if resolved.effort else DEFAULT_THINKING_BUDGET
    return {'type': 'enabled', 'budget_tokens': budget}
```

---

## 3. Per-Provider Changes

### Simple providers — `prepare_request()` override eliminated entirely

| Provider | Current | After |
|---|---|---|
| **Groq** | Overrides `prepare_request()` only for thinking | Override **removed** |
| **Cohere** | Overrides `prepare_request()` only for thinking | Override **removed** |
| **xAI** | Overrides `prepare_request()` only for thinking | Override **removed** |

**Groq example (at request time):**

```python
# In _chat_completions_create():
if 'groq_reasoning_format' in model_settings:
    reasoning_format = model_settings['groq_reasoning_format']
elif model_request_parameters.resolved_thinking is not None:
    reasoning_format = self._translate_thinking(model_request_parameters.resolved_thinking)
else:
    reasoning_format = NOT_GIVEN
```

### OpenAI inheritance chain — strip pattern eliminated

This is the biggest structural win. Currently:
1. `OpenAIChatModel.prepare_request()` resolves thinking → injects `openai_reasoning_effort`
2. `CerebrasModel.prepare_request()` **strips** `openai_reasoning_effort`, injects own
3. `OpenRouterModel.prepare_request()` **strips** `openai_reasoning_effort`, injects own

After the refactor:
1. Base `Model.prepare_request()` resolves thinking → `params.resolved_thinking`
2. `OpenAIChatModel._completions_create()` reads `params.resolved_thinking` at request time
3. `CerebrasModel` and `OpenRouterModel` read `params.resolved_thinking` at request time
4. **No injection, no stripping, no hidden parent dependency**

### Anthropic — conflict check uses `resolved_thinking` directly

Anthropic's `prepare_request()` currently checks thinking-vs-output-tools conflict by reading `merged_settings.get('anthropic_thinking')`. After the refactor:

```python
# AnthropicModel.prepare_request():
merged_settings, params = super().prepare_request(model_settings, model_request_parameters)
merged_settings = cast(AnthropicModelSettings, merged_settings or {})

# Check thinking-vs-output conflict using resolved config directly
thinking_enabled = False
if 'anthropic_thinking' in merged_settings:
    thinking_enabled = merged_settings['anthropic_thinking'].get('type') in ('enabled', 'adaptive')
elif params.resolved_thinking is not None:
    thinking_enabled = params.resolved_thinking.enabled

if thinking_enabled and params.output_tools and not params.allow_text_output:
    suggested = 'NativeOutput' if self.profile.supports_json_schema_output else 'PromptedOutput'
    raise UserError(f'Anthropic does not support thinking and output tools...')

return merged_settings, params
```

Anthropic's `_build_output_config` reads effort from `params.resolved_thinking.effort` instead of a forwarded `anthropic_effort` field — no more effort forwarding.

### OpenRouter — overrides resolution for no-profile-guard behavior

OpenRouter intentionally skips profile guards. It overrides `prepare_request()` to re-resolve without profile:

```python
# OpenRouterModel.prepare_request():
merged_settings, params = super().prepare_request(model_settings, model_request_parameters)
merged_settings = cast(OpenRouterModelSettings, merged_settings or {})

# Strip parent's openai_reasoning_effort (still needed for OpenAI parent)
merged_settings.pop('openai_reasoning_effort', None)

# Override resolved_thinking: bypass profile guards, let OpenRouter handle capabilities
if params.resolved_thinking is None and (model_settings or {}):
    resolved = resolve_thinking_config(merged_settings)  # No profile
    if resolved is not None:
        params = replace(params, resolved_thinking=resolved)

new_settings = _openrouter_settings_to_openai_settings(merged_settings)
return new_settings, params
```

### Bedrock — reads at request time with model-family dispatch

```python
# BedrockConverseModel — at request time:
if model_request_parameters.resolved_thinking is not None:
    key, config = self._translate_bedrock_thinking(model_request_parameters.resolved_thinking)
    if key and config:
        additional[key] = config
```

---

## 4. Why This Pattern Works

### Follows the output mode precedent exactly

| Aspect | Output mode | Thinking (proposed) |
|---|---|---|
| Profile capabilities | `supports_json_schema_output` | `supports_thinking`, `thinking_always_enabled` |
| Resolved in | Base `Model.prepare_request()` | Base `Model.prepare_request()` |
| Stored in | `ModelRequestParameters.output_mode` | `ModelRequestParameters.resolved_thinking` |
| Models call resolver? | No | No |
| Models override `prepare_request` for it? | No | No (except Anthropic for conflict check, OpenRouter for no-profile) |
| Translation | Inline at request time | `_translate_thinking()` at request time |

### Eliminates all three structural problems

1. **9x boilerplate in `prepare_request()`** — gone. Base class resolves once.
2. **Strip-and-reinject in OpenAI chain** — gone. No injection into `model_settings`.
3. **Each model calling `resolve_thinking_config()` itself** — gone. Base class calls it.

### What each model knows (before vs. after)

**Before**: Each model knows how to resolve unified settings, when to apply profile guards, how to check provider-specific precedence, AND how to translate to native format.

**After**: Each model knows how to translate `ResolvedThinkingConfig` to native format, and which provider-specific field takes precedence. That's it.

---

## 5. What This Does NOT Change

- `resolve_thinking_config()` in `thinking.py` — stays as the centralized normalizer
- `ResolvedThinkingConfig` dataclass — stays as the canonical form
- Profile fields (`supports_thinking`, `thinking_always_enabled`) — stay on `ModelProfile`
- Provider-specific settings (`anthropic_thinking`, `openai_reasoning_effort`, etc.) — still exist and take precedence
- Silent-drop semantics — unchanged
- Provider-specific profile extensions — unchanged

---

## 6. Naming Summary

| Component | Name | Location |
|---|---|---|
| Centralized normalizer | `resolve_thinking_config()` | `thinking.py` (unchanged) |
| Canonical representation | `ResolvedThinkingConfig` | `thinking.py` (unchanged) |
| Resolved result on params | `resolved_thinking` | `ModelRequestParameters` (new field) |
| Per-model translator | `_translate_thinking()` | Each model class (renamed from `_resolve_thinking_config`) |

---

## 7. Discoverability: Adding a New Thinking-Capable Provider

Under this pattern, a new provider developer:

1. Sets `supports_thinking=True` in their profile — the base class will then resolve thinking settings
2. Checks `model_request_parameters.resolved_thinking` at request time — translates to native format
3. If they forget step 2, thinking settings are silently ignored (correct behavior — same as forgetting to read `output_mode`)
4. `models/AGENTS.md` documents the pattern with a checklist

This is analogous to how new providers handle output mode: set the profile capabilities, then read the resolved mode at request time.

---

## 8. Migration Checklist

### Phase 1: Infrastructure
- [ ] Add `resolved_thinking: ResolvedThinkingConfig | None = None` to `ModelRequestParameters`
- [ ] Add thinking resolution to `Model.prepare_request()` (after output mode, before validation)

### Phase 2: Migrate models (all can be done independently)
For each of the 9 providers:
- [ ] Add `_translate_thinking()` method (pure translator)
- [ ] Move thinking consumption from `prepare_request()` injection → request-time read
- [ ] Remove the `prepare_request()` thinking boilerplate
- [ ] For Groq/Cohere/xAI: remove `prepare_request()` override entirely

### Phase 3: Fix OpenAI inheritance chain
- [ ] Remove thinking injection from `OpenAIChatModel.prepare_request()`
- [ ] Remove `openai_reasoning_effort` stripping from CerebrasModel and OpenRouterModel
- [ ] Each reads `params.resolved_thinking` at request time instead

### Phase 4: Cleanup
- [ ] Remove `anthropic_effort` forwarding from Anthropic's `prepare_request`
- [ ] Update `_build_output_config` to read from `params.resolved_thinking.effort`
- [ ] Rename all `_resolve_thinking_config` → `_translate_thinking`
- [ ] Update `models/AGENTS.md` with new pattern documentation
- [ ] Verify all tests pass

---

## 9. Open Questions

### Should `ModelRequestParameters.resolved_thinking` be public API?

`ModelRequestParameters` is a public dataclass. Adding `resolved_thinking` makes it visible to users who access parameters in `customize_request_parameters()` overrides. This seems fine — it's read-only metadata, analogous to `output_mode`. But if maintainers prefer to keep it internal, it could be prefixed with `_` and excluded from repr.

### Does OpenRouter's profile need `supports_thinking=True`?

OpenRouter's profile is dynamically generated from model names. For unrecognized models, it may default to `supports_thinking=False`, causing the base class to silently drop thinking settings. The proposed solution (OpenRouter overrides to re-resolve without profile) handles this, but alternatively OpenRouter could just set `supports_thinking=True` on all its profiles since it delegates capability detection to the backend.
