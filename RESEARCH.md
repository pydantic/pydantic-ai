# Unified Thinking Settings — Design Plan

> **PR**: #3894 (feat: Add unified thinking settings across all model providers)
> **Issue**: #3756
> **Branch**: `unified-thinking-settings`
> **Date**: 2026-02-14

## Overview

This document captures the research findings from 10 provider-expert agents and the design decisions reached through senior-level debate (Anthropic, Google, OpenAI). The goal: a simple, provider-agnostic thinking settings API for pydantic-ai that covers the 80% use case, while preserving full provider-specific escape hatches for advanced users.

---

## Part 1: Provider-by-Provider Analysis

### Anthropic (Senior)

**Current SDK State:**
- `ThinkingConfigParam` is a union of 3 TypedDicts: `ThinkingConfigEnabledParam` (`type: "enabled"`, `budget_tokens: int`), `ThinkingConfigDisabledParam` (`type: "disabled"`), `ThinkingConfigAdaptiveParam` (`type: "adaptive"`)
- `OutputConfigParam` has `effort: Literal["low", "medium", "high", "max"]` — this affects ALL output tokens (thinking, text, tool calls), not just thinking

**API Trajectory:**
- **Adaptive is the future**: `type: "adaptive"` is recommended for Opus 4.6. `type: "enabled"` with `budget_tokens` is DEPRECATED and will be removed in future model releases
- **Effort replaces budget_tokens**: `output_config.effort` with levels `low | medium | high | max` is the new primary control
- **Interleaved thinking** is automatic with adaptive mode (no beta header needed)
- Supported models: Claude Opus 4.6/4.5/4.1/4, Sonnet 4.5/4, Haiku 4.5, Sonnet 3.7 (deprecated)

**Key Constraints:**
- Thinking + `tool_choice` incompatibility: only `auto` or `none` supported when thinking enabled/adaptive
- Temperature/top_k incompatible with thinking; top_p must be 0.95-1.0
- `budget_tokens` must be >= 1024 and < `max_tokens` (except with interleaved thinking)
- Thinking blocks must be preserved in multi-turn conversations
- `max` effort is Opus 4.6 only; effort only supported on Opus 4.5+
- Cache invalidation when changing thinking parameters

---

### Google / Gemini (Senior)

**Current SDK State:**
- `ThinkingConfig` has 3 params: `include_thoughts: bool`, `thinking_budget: int`, `thinking_level: Literal["minimal", "low", "medium", "high"]`
- Two eras of models with different APIs:

| Era | Models | Control | Can Disable? |
|-----|--------|---------|-------------|
| Gemini 2.5 (retiring March 2026) | 2.5 Pro, 2.5 Flash, 2.5 Flash-Lite | `thinking_budget` (token count) | Flash: yes (budget=0), Pro: no (min 128) |
| Gemini 3 (current) | 3 Pro, 3 Flash | `thinking_level` (enum) | Neither (even `minimal` may still think) |

**API Trajectory:**
- `thinking_budget` accepted on Gemini 3 for backward compat but results in "suboptimal performance" — Google recommends migrating to `thinking_level`
- Cannot use both `thinking_budget` and `thinking_level` in same request
- Thought signatures are REQUIRED for Gemini 3 function calling (pydantic-ai already handles this)
- Design around levels since budgets are being deprecated

**Key Constraints:**
- Pro models CANNOT disable thinking
- `include_thoughts` is independent from thinking being enabled (unique among providers)
- `minimal` on Gemini 3 Flash is NOT truly "off" — may still think for complex tasks
- Gemini 2.5 retirement (March 2026) makes budget-based API less relevant

---

### OpenAI (Senior)

**Current SDK State:**
- `ReasoningEffort`: `Literal["none", "minimal", "low", "medium", "high", "xhigh"]` (6 levels)
- `Reasoning` TypedDict: `effort`, `summary: Literal["auto", "concise", "detailed"]`, `generate_summary` (deprecated)
- Chat Completions: `reasoning_effort` as top-level param; Responses API: nested `reasoning` object

| Model Family | Default Effort | Supports `none`? | Notes |
|-------------|---------------|-------------------|-------|
| o-series (o1, o3, o4-mini) | `medium` | No | Always-on reasoning |
| GPT-5 | `medium` | No | Always-on reasoning |
| GPT-5-pro | `high` (locked) | No | Only supports `high` |
| GPT-5.1, GPT-5.2 | `none` | Yes | Opt-in reasoning |
| GPT-5.1-codex-max+ | `none` | Yes | Adds `xhigh` support |

**API Trajectory:**
- Reasoning shifting to opt-in for newer models (GPT-5.1+ defaults to `none`)
- Expanding effort granularity: started `low/medium/high`, added `none`, `minimal`, `xhigh`
- Responses API as primary (richer reasoning support: summaries, encrypted content)
- Encrypted reasoning for ZDR privacy compliance

**Key Constraints:**
- When reasoning active (effort != `none`): `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `logprobs` NOT supported
- `minimal` and `xhigh` are OpenAI-specific extensions
- Two different API shapes (Chat Completions vs Responses) with different param structures

---

### Bedrock (Router Provider)

**Current State:**
- Already parses `reasoningContent` blocks in responses and sends thinking parts back in multi-turn
- **Zero code** for sending thinking configuration in requests — users must use `bedrock_additional_model_requests_fields` manually
- No first-class thinking field in `InferenceConfiguration` — thinking config goes via freeform `additionalModelRequestFields`

**Router Translation:**
- Forwards provider-native JSON to underlying models: `{"thinking": {"type": "enabled", "budget_tokens": 1024}}` for Claude
- Uses `snake_case` matching Anthropic's API (not Bedrock's usual camelCase)
- Response normalized to Bedrock's `reasoningContent` blocks
- Supports `type: "enabled"` + `budget_tokens`, `type: "adaptive"` + `effort`, `type: "disabled"` for Claude
- DeepSeek R1: always-on, no config needed, no signatures

**Key Constraints:**
- No SDK type safety (freeform dict injection)
- Temperature must be 1 when thinking enabled for Claude
- `additionalModelRequestFields` merge conflicts if user also sets manual fields
- Feature lag behind native Anthropic API (days to weeks)

---

### Cerebras

**Current State:**
- Uses OpenAI-compatible API via `AsyncOpenAI` client — no dedicated Cerebras SDK
- One setting: `cerebras_disable_reasoning: bool` → `extra_body={'disable_reasoning': True/False}`
- Three different toggle mechanisms per model family:

| Model Family | Toggle | Effort Control |
|-------------|--------|---------------|
| GLM (zai-glm-4.6/4.7) | `disable_reasoning: bool` (inverted!) | No |
| GPT-OSS (gpt-oss-120b) | `reasoning_effort: low/medium/high` | Yes |
| Qwen3 (qwen-3-32b) | Always reasons, `reasoning_format: hidden` hides output | No |

**API Trajectory:**
- `reasoning_format` (parsed/text_parsed/raw/hidden/none) added 2025-12-16
- GLM 4.7 added `clear_thinking` for preserved thinking across turns
- Streaming + Reasoning + Constrained Decoding now work together (GA 2026-01-09)

**Key Constraints:**
- Inverted logic (`disable_reasoning=True` = thinking OFF)
- No true disable for Qwen3 (only `hidden` which still bills for reasoning tokens)
- `clear_thinking` has no parallel in other providers
- Profile gaps: `zai_model_profile()` currently returns `None`

---

### Groq

**Current State:**
- `groq_reasoning_format: Literal['hidden', 'raw', 'parsed']` — sole reasoning parameter
- SDK v0.25.0 installed; missing `reasoning_effort` (added v0.28+, latest v0.33.0)
- Two orthogonal axes: format (how to show) and effort (how hard to think)

**API Trajectory (3 parameters now):**
| Parameter | Values | Notes |
|-----------|--------|-------|
| `reasoning_format` | `hidden`, `raw`, `parsed` | Controls output format |
| `reasoning_effort` (newer) | Qwen 3: `none/default`; GPT-OSS: `low/medium/high` | Model-dependent values |
| `include_reasoning` (newer) | `true/false` | Mutually exclusive with `reasoning_format` |

**Key Constraints:**
- `raw` format + tools = 400 error (tools force `parsed`)
- `hidden` doesn't disable thinking (only suppresses output, still bills for reasoning tokens)
- Only Qwen 3's `reasoning_effort='none'` truly disables reasoning
- DeepSeek R1 distills always output `<think>` tags inline
- SDK needs upgrading to access `reasoning_effort`

---

### OpenRouter (Router Provider)

**Current State:**
- Unified `reasoning` object: `{effort, max_tokens, exclude, enabled, summary}`
- `effort` supports 6 levels: `xhigh`, `high`, `medium`, `low`, `minimal`, `none`
- New `summary` field: `Literal["auto", "concise", "detailed"]`
- Current pydantic-ai code only exposes 3 effort levels (behind)

**Router Translation (how OpenRouter maps to providers):**
| Param | Anthropic | OpenAI | Google 3 | Google 2.5 |
|-------|-----------|--------|----------|------------|
| `effort` | → `budget_tokens` via ratio formula | → `reasoning_effort` (1:1) | → `thinkingLevel` (1:1) | → `thinkingBudget` (token count) |
| `max_tokens` | → `thinking.budget_tokens` | N/A | N/A | → `thinkingBudget` |
| `exclude` | Strip from response | Strip from response | Strip from response | Strip from response |

Effort-to-token ratios: `xhigh: ~95%`, `high: ~80%`, `medium: ~50%`, `low: ~20%`, `minimal: ~10%`, `none: disable`

**Key Insight:** OpenRouter has already solved a very similar unified abstraction problem. Our design should leverage their per-provider translation rather than re-implementing it.

---

### Mistral (Magistral)

**Current SDK State (v1.9.11):**
- `prompt_mode: MistralPromptMode | null` — the only reasoning-related parameter
- `MistralPromptMode` enum: single value `"reasoning"`
- Response: `ThinkChunk` with nested `Thinking` objects (`type="text"`)
- No `reasoning_effort`, no `thinking_budget`, no enable/disable toggle

**API Trajectory:**
- Simplest reasoning API of all providers — model just reasons, no knobs
- No signs of `reasoning_effort` or budget controls being added (checked up to SDK v1.12.2, Feb 2026)
- `prompt_mode="reasoning"` only controls the internal system prompt, not thinking behavior

**Key Constraints:**
- Cannot disable thinking on Magistral (always-on)
- Cannot control effort/depth — no API parameter exists
- Non-Magistral models never support thinking
- No temperature or tool_choice incompatibilities documented
- Most similar to DeepSeek R1 in approach (always-on, zero config)

---

### Cohere (Command A Reasoning)

**Current SDK State (v5.20.1):**
- `cohere.Thinking` Pydantic model: `type: Literal["enabled", "disabled"]` (required), `token_budget: int | None` (optional)
- `AsyncV2Client.chat()` accepts `thinking: Optional[cohere.Thinking]` parameter
- pydantic-ai already parses `ThinkingAssistantMessageV2ContentItem` in responses but does NOT pass thinking config in requests

**API Trajectory:**
- Binary on/off toggle + optional token budget — no effort levels
- OpenAI-compatible endpoint supports `reasoning_effort: "none" | "high"` only (maps to disabled/enabled)
- Only one model supports thinking: `command-a-reasoning-08-2025`
- Thinking is enabled by default but CAN be disabled (unlike DeepSeek R1, Magistral)

**Key Constraints:**
- No effort/level control — only binary on/off and token budget
- No temperature, tool_choice, or sampling parameter restrictions (unique among providers)
- Token budget guidance: leave ≥1,000 tokens for final response; max recommended ~31,000
- Profile currently returns `None` — needs `supports_thinking=True`

---

### xAI / Grok

**Current SDK State (xai-sdk, gRPC-based):**
- `reasoning_effort: Literal["low", "high"]` — only for `grok-3-mini`
- Proto definition includes `EFFORT_MEDIUM` but SDK type alias omits it
- Response: `message.reasoning_content` + `message.encrypted_content`
- pydantic-ai already has full response-side handling (`_map_thinking_part()`, `_collect_reasoning_events()`) but passes zero thinking config in requests

**Unique Pattern — Model Variant Selection:**
- Grok 4+ uses separate model names instead of API parameters:
  - `grok-4-fast-reasoning` vs `grok-4-fast-non-reasoning`
  - `grok-4-1-fast-reasoning` vs `grok-4-1-fast-non-reasoning`
- This is model **selection**, not model **configuration** — unique among all providers
- `grok-4` always reasons with no controls
- `grok-3` reasons internally but returns no `reasoning_content`

**API Trajectory:**

| Model | `reasoning_effort` | `reasoning_content` | Can Disable? |
|-------|-------------------|--------------------|----|
| `grok-3-mini` | `"low"` / `"high"` | Yes | No documented mechanism |
| `grok-3` | Error if provided | No (internal only) | No |
| `grok-4` | Error if provided | No | No (always reasons) |
| `grok-4-fast-reasoning` | Error if provided | Yes | No (use non-reasoning variant) |
| `grok-4-fast-non-reasoning` | Error if provided | No | N/A (doesn't reason) |

**Key Constraints:**
- `reasoning_effort` sending to wrong model returns an error (not silent)
- `presence_penalty`, `frequency_penalty`, `stop` unsupported with reasoning models
- Temperature/top_p NOT documented as incompatible (unlike OpenAI/Anthropic)
- `encrypted_content` for ZDR privacy (grok-4 only, provider-specific)
- Profile currently has zero thinking fields

---

## Part 2: Senior Debate & Voting Results

Four contentious design issues were put to democratic vote among the 3 senior agents (Anthropic, Google, OpenAI).

### Issue 1: Field Name for Effort Control

| Agent | Vote | Reasoning |
|-------|------|-----------|
| Anthropic | `thinking_effort` | Describes user intent, not provider implementation; consistent with `thinking` prefix |
| Google | `thinking_effort` | Provider-agnostic; `thinking_level` mirrors Google's term too closely; reads well |
| OpenAI | `thinking_effort` | Scoped to thinking phase; matches OpenAI's `reasoning_effort` naming pattern |

**Result: Unanimous — `thinking_effort`**

### Issue 2: Effort Value Set

| Agent | Vote | Reasoning |
|-------|------|-----------|
| Anthropic | `low \| medium \| high` | Clean cross-provider mapping; `none` belongs on toggle; `max` is provider-specific |
| Google | `low \| medium \| high` | Minimal viable set; mixing "whether to think" with "how hard" violates separation of concerns |
| OpenAI | `low \| medium \| high` | Simplest common denominator; covers 80% use case; extremes via provider-specific |

**Result: Unanimous — `Literal['low', 'medium', 'high']`**

Provider-specific extensions (`minimal`, `xhigh`, `max`, `none`) remain accessible through provider-specific settings.

### Issue 3: Should `include_thinking: bool` Be in Unified API?

| Agent | Vote | Reasoning |
|-------|------|-----------|
| Anthropic | **No** | Not cross-provider enough; niche use case; Anthropic always returns thinking |
| Google | **Yes** | 3+ providers support it; real production pattern; silently ignored where unsupported |
| OpenAI | **No** | Not universal; creates false expectations; no "hide thinking" toggle in OpenAI |

**Result: 2-1 — No.** `include_thinking` stays provider-specific.

### Issue 4: How Should `thinking` Enable/Disable Work?

| Agent | Vote | Reasoning |
|-------|------|-----------|
| Anthropic | `thinking: bool \| None` | Clean semantics; `auto` is provider-specific; effort shouldn't double as toggle |
| Google | `thinking: bool \| None` | Simplest API; `None` already means auto; avoids confusing interaction with effort |
| OpenAI | No toggle (effort only) | One knob avoids confusing `thinking=False, thinking_effort='high'` interaction |

**Result: 2-1 — `thinking: bool | None`**

OpenAI's concern about interaction between `thinking` and `thinking_effort` is valid. Resolution: when `thinking=False`, `thinking_effort` is ignored. When `thinking=True` or `None`, `thinking_effort` controls depth. Document this clearly.

---

## Part 3: Final Unified API Design

### ModelSettings (user-facing)

```python
class ModelSettings(TypedDict, total=False):
    # ... existing fields ...

    # Unified thinking settings
    thinking: bool
    """Enable or disable thinking/reasoning.

    - `True`: Enable thinking. Resolution logic picks the best mode per provider
      (adaptive for Anthropic Opus 4.6, enabled+budget for older Anthropic,
      inherent for OpenAI o-series, etc.)
    - `False`: Disable thinking. Raises UserError if model always thinks
      (o-series, DeepSeek R1, etc.)
    - Omitted/None: Use provider default behavior.

    When `thinking=False`, `thinking_effort` is ignored.
    Provider-specific settings (e.g. `anthropic_thinking`, `openai_reasoning_effort`)
    always take precedence over this unified field.
    """

    thinking_effort: Literal['low', 'medium', 'high']
    """Control the depth of thinking/reasoning.

    - `'low'`: Minimal thinking, faster responses
    - `'medium'`: Balanced thinking depth
    - `'high'`: Deep thinking, most thorough

    Only effective when thinking is enabled (explicitly or by default).
    Provider-specific effort levels (OpenAI's `xhigh`/`minimal`, Anthropic's `max`)
    are available through provider-specific settings.
    """
```

### ModelProfile (capability flags)

```python
@dataclass
class ModelProfile:
    # ... existing fields ...

    supports_thinking: bool = False
    """Whether the model supports thinking/reasoning capabilities."""

    thinking_always_enabled: bool = False
    """Whether thinking cannot be disabled (e.g., o-series, DeepSeek R1).
    If True and user sets thinking=False, raise UserError."""

    supports_thinking_effort: bool = False
    """Whether the model supports thinking effort/level control."""
```

### Provider Mapping Table

| Provider | `thinking=True` | `thinking=False` | `thinking_effort='low'` | `thinking_effort='medium'` | `thinking_effort='high'` |
|----------|----------------|-------------------|------------------------|---------------------------|-------------------------|
| **Anthropic** (Opus 4.6) | `type: "adaptive"` | `type: "disabled"` | `output_config.effort: "low"` | `output_config.effort: "medium"` | `output_config.effort: "high"` |
| **Anthropic** (older) | `type: "enabled", budget_tokens: <default>` | `type: "disabled"` | budget ~4096 | budget ~10000 | budget ~16000 |
| **Google** (Gemini 3) | no-op (default on) | Error on Pro / `thinking_level: "minimal"` on Flash | `thinking_level: "low"` | `thinking_level: "medium"` | `thinking_level: "high"` |
| **Google** (Gemini 2.5) | no-op (default on) | `thinking_budget: 0` (Flash) / Error (Pro) | budget ~2048 | budget ~8192 | budget ~24576 |
| **OpenAI** (o-series) | no-op (always on) | Error (always thinks) | `reasoning_effort: "low"` | `reasoning_effort: "medium"` | `reasoning_effort: "high"` |
| **OpenAI** (GPT-5.1+) | `reasoning_effort: "medium"` | no-op (default off) | `reasoning_effort: "low"` | `reasoning_effort: "medium"` | `reasoning_effort: "high"` |
| **Bedrock** (Claude) | `additionalModelRequestFields.thinking` | `type: "disabled"` | Same as Anthropic via additionalModelRequestFields | Same | Same |
| **Groq** | `reasoning_format: "parsed"` | `reasoning_format: "hidden"` or `reasoning_effort: "none"` | `reasoning_effort: "low"` (GPT-OSS) | `reasoning_effort: "medium"` | `reasoning_effort: "high"` |
| **Cerebras** (GLM) | `disable_reasoning: false` | `disable_reasoning: true` | N/A | N/A | N/A |
| **Cerebras** (GPT-OSS) | default | lowest effort | `reasoning_effort: "low"` | `reasoning_effort: "medium"` | `reasoning_effort: "high"` |
| **OpenRouter** | `reasoning.enabled: true` | `reasoning.effort: "none"` | `reasoning.effort: "low"` | `reasoning.effort: "medium"` | `reasoning.effort: "high"` |
| **Mistral** (Magistral) | no-op (always on) | silent ignore (can't disable) | silent ignore (no effort control) | silent ignore | silent ignore |
| **Cohere** (Command A) | no-op (default on) | `Thinking(type='disabled')` | silent ignore (no effort control) | silent ignore | silent ignore |
| **xAI** (grok-3-mini) | no-op | silent ignore | `reasoning_effort: "low"` | `reasoning_effort: "low"` (downmap) | `reasoning_effort: "high"` |
| **xAI** (grok-4 / reasoning variants) | no-op (always on) | silent ignore (can't disable) | silent ignore | silent ignore | silent ignore |

### Resolution Logic

```python
def resolve_thinking_config(
    settings: ModelSettings,
    profile: ModelProfile,
) -> ResolvedThinkingConfig | None:
    """Single source of truth for thinking configuration resolution.

    Returns None if no thinking settings are specified.
    Raises UserError for invalid combinations (e.g., disable always-on model).
    """
    thinking = settings.get('thinking')
    effort = settings.get('thinking_effort')

    # Nothing set → no unified thinking config
    if thinking is None and effort is None:
        return None

    # Validate model supports thinking
    if (thinking is True or effort is not None) and not profile.supports_thinking:
        raise UserError(f"This model does not support thinking/reasoning")

    # Cannot disable always-on models
    if thinking is False and profile.thinking_always_enabled:
        raise UserError(f"This model's thinking cannot be disabled")

    # Cannot set effort on models without effort support
    if effort is not None and not profile.supports_thinking_effort:
        # Silently ignore? Or warn? TBD — lean toward warning
        pass

    return ResolvedThinkingConfig(
        enabled=thinking,  # True, False, or None
        effort=effort,     # 'low', 'medium', 'high', or None
    )
```

### Precedence Rules

1. **Provider-specific settings always win** over unified settings. If `anthropic_thinking` is set, `thinking` and `thinking_effort` are ignored for Anthropic.
2. **`thinking=False` makes `thinking_effort` irrelevant** — thinking is off, no effort to control.
3. **`thinking=None` (omitted) + `thinking_effort` set** → implicitly enables thinking at that effort level (provider handles the details).
4. **`thinking=True` + `thinking_effort` omitted** → enable thinking at provider default depth.

### Architecture

```
ModelSettings (user input)
    ├── thinking: bool | None
    └── thinking_effort: 'low' | 'medium' | 'high' | None
         │
         ▼
resolve_thinking_config() in thinking.py (centralized validation)
    ├── Checks profile capabilities
    ├── Validates combinations
    └── Returns ResolvedThinkingConfig(enabled, effort)
         │
         ▼
Provider._resolve_thinking_config() (per-provider translation)
    ├── anthropic.py → ThinkingConfigParam + OutputConfigParam
    ├── google.py    → ThinkingConfigDict
    ├── openai.py    → reasoning_effort param
    ├── bedrock.py   → additionalModelRequestFields
    ├── groq.py      → reasoning_format + reasoning_effort
    ├── cerebras.py  → disable_reasoning / reasoning_effort
    ├── openrouter.py → reasoning object
    ├── mistral.py   → no-op (always-on, no config params)
    ├── cohere.py    → cohere.Thinking(type, token_budget)
    └── xai.py       → reasoning_effort (grok-3-mini only)
```

---

## Part 4: Cross-Cutting Concerns

### Router Provider Special Handling

**Bedrock**: Inject thinking config into `additionalModelRequestFields["thinking"]` using Anthropic's native parameter names. Merge strategy: thinking config takes precedence over user-provided `bedrock_additional_model_requests_fields`. Profile needs `supports_thinking`, `thinking_always_enabled`, `supports_adaptive_thinking`.

**OpenRouter**: Map unified settings to `openrouter_reasoning` object. Let OpenRouter handle per-provider translation (they've already solved this). `thinking_effort` → `reasoning.effort`, `thinking=False` → `reasoning.effort: "none"`.

### Profile Detection

Each provider's `*_model_profile()` function needs updating to set:
- `supports_thinking=True` for reasoning-capable models
- `thinking_always_enabled=True` for o-series, GPT-5, DeepSeek R1, Magistral, grok-4/grok-3, reasoning variants
- `supports_thinking_effort=True` for models with effort/level control

### SDK Version Gaps

- **Groq SDK v0.25.0** is missing `reasoning_effort` — needs upgrade to v0.28+ or use `extra_body`
- **Cerebras** has no dedicated SDK — uses OpenAI client with `extra_body`
- **Bedrock** has no SDK type safety for thinking — freeform dict injection

### Temperature/Sampling Conflicts

- **Anthropic**: Thinking incompatible with `temperature` and `top_k`
- **OpenAI**: Reasoning incompatible with `temperature`, `top_p`, penalties, logit_bias
- Resolution logic should warn or auto-adjust when conflicts detected

### What Stays Provider-Specific

| Feature | Provider | Why Not Unified |
|---------|----------|----------------|
| `max` effort | Anthropic | Opus 4.6 only |
| `xhigh`/`minimal`/`none` effort | OpenAI | Provider-specific granularity |
| `include_thoughts` | Google | Not universal (2-1 vote) |
| `reasoning_format` | Groq, Cerebras | Output format, not thinking control |
| `reasoning_summary` | OpenAI, OpenRouter | Provider-specific feature |
| `encrypted_content` | OpenAI | Privacy/compliance feature |
| `clear_thinking` | Cerebras | Multi-turn state, unique to GLM |
| `budget_tokens` (exact) | Anthropic, Google | Precise control for advanced users |
| `token_budget` | Cohere | Only provider with budget-only control (no effort levels) |
| `prompt_mode` | Mistral | Controls system prompt, not thinking behavior |
| `encrypted_content` | xAI | ZDR privacy feature, grok-4 only |
| `reasoning_effort: "low"\|"high"` (2 levels) | xAI | Only grok-3-mini, only 2 levels vs unified 3 |

---

## Part 5: Implementation Checklist

### Core Changes
- [ ] Add `thinking: bool` and `thinking_effort: Literal['low', 'medium', 'high']` to `ModelSettings` in `settings.py`
- [ ] Create/update `thinking.py` with `resolve_thinking_config()` and `ResolvedThinkingConfig`
- [ ] Add `supports_thinking`, `thinking_always_enabled`, `supports_thinking_effort` to `ModelProfile`
- [ ] Export `ThinkingConfig` type from `__init__.py`

### Provider Implementations
- [ ] `anthropic.py`: `_resolve_thinking_config()` — adaptive vs enabled based on profile, effort → `output_config.effort`
- [ ] `google.py`: `_resolve_thinking_config()` — level for Gemini 3, budget for Gemini 2.5
- [ ] `openai.py`: `_resolve_thinking_config()` — effort → `reasoning_effort`
- [ ] `bedrock.py`: `_resolve_thinking_config()` — inject into `additionalModelRequestFields`
- [ ] `groq.py`: `_resolve_thinking_config()` — format + effort mapping
- [ ] `cerebras.py`: `_resolve_thinking_config()` — model-family-specific dispatch
- [ ] `openrouter.py`: `_resolve_thinking_config()` — map to `reasoning` object
- [ ] `mistral.py`: `_resolve_thinking_config()` — no-op (always-on, no config params to send)
- [ ] `cohere.py`: `_resolve_thinking_config()` — map to `cohere.Thinking(type=..., token_budget=...)`
- [ ] `xai.py`: `_resolve_thinking_config()` — map to `reasoning_effort` for grok-3-mini, silent ignore for others

### Profile Updates
- [ ] `profiles/anthropic.py`: Set `supports_thinking`, `supports_thinking_effort` per model
- [ ] `profiles/google.py`: Set `supports_thinking`, `thinking_always_enabled` per model
- [ ] `profiles/openai.py`: Set `supports_thinking`, `thinking_always_enabled`, `supports_thinking_effort`
- [ ] `profiles/groq.py`: Set per-model thinking capabilities
- [ ] `profiles/deepseek.py`: Set `thinking_always_enabled=True` for R1 models
- [ ] Bedrock profiles: Add thinking capability flags
- [ ] Cerebras profiles: Fix `zai_model_profile()` (currently returns `None`)
- [ ] `profiles/mistral.py`: Add `thinking_always_enabled=True` for Magistral models
- [ ] `profiles/cohere.py`: Add `supports_thinking=True` for `command-a-reasoning` model
- [ ] `profiles/grok.py`: Add `supports_thinking`, `thinking_always_enabled` based on model variant (reasoning vs non-reasoning)

### Tests
- [ ] Unit tests for `resolve_thinking_config()` — all valid/invalid combinations
- [ ] Per-provider integration tests — verify correct native params generated
- [ ] Profile detection tests — verify correct flags per model name
- [ ] Precedence tests — provider-specific overrides unified
- [ ] Error case tests — disable always-on, unsupported models, conflicts

### Documentation
- [ ] Update thinking docs with unified API examples
- [ ] Provider-specific docs noting which features are unified vs provider-only
- [ ] Migration guide from provider-specific to unified settings
