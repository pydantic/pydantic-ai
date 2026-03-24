# Plan: Unified Thinking Capability Improvements

> **Branch**: `thinking-cap-improvements` (off `upstream/main`, post-merge of `thinking-cap`)
> **Context**: The unified thinking feature was merged via the capabilities PR. This plan
> addresses review findings from `REVIEW_COMMENTS.md` — correctness bugs, missing
> implementations, effort mapping fixes, and comprehensive tests.

---

## Phase 1: Correctness Fixes (provider translation bugs)

These are behavioral bugs where the unified `thinking` setting produces wrong or no-op
results. Each fix is small and self-contained.

### 1.1 Anthropic: `'medium'` should map to adaptive, not fixed budget

**File:** `pydantic_ai_slim/pydantic_ai/profiles/anthropic.py`

Change `ANTHROPIC_THINKING_BUDGET_MAP` so that for adaptive-capable models:
- `True` → `{'type': 'adaptive'}`
- `'medium'` → `{'type': 'adaptive'}`
- `'low'` → `budget_tokens: 1024` (see 1.2)
- `'high'` → `budget_tokens: 16384` (unchanged)

The translation method needs to check whether the model supports adaptive thinking and
use the appropriate mapping. For budget-only models, keep existing fixed-budget behavior.

### 1.2 Anthropic: `'low'` should use 1024 (documented minimum)

**File:** `pydantic_ai_slim/pydantic_ai/profiles/anthropic.py`

Change `'low'` mapping from `2048` to `1024`. Anthropic's SDK docs state
`budget_tokens >= 1024`. Since `'low'` means "as little thinking as possible,"
the floor is the correct semantic match.

### 1.3 xAI: Invert effort mapping (round down, not up)

**File:** xAI model/helper where thinking translation lives

xAI Grok supports only `'low'` and `'high'`. Current mapping rounds UP for ambiguous
values. Fix to round DOWN (conservative = cheaper, safer):
- `True` → `'low'` (match xAI SDK default)
- `'low'` → `'low'`
- `'medium'` → `'low'` (round down)
- `'high'` → `'high'`

### 1.4 Groq: `thinking=False` should return `'hidden'`

**File:** `pydantic_ai_slim/pydantic_ai/models/groq.py` (lines 245-256)

Groq's SDK accepts `'hidden'`, `'raw'`, `'parsed'` — no true disable exists. But
`'hidden'` is the closest to "disabled" semantics (hides reasoning output). Change:
- `thinking=False` → `'hidden'` (instead of `NOT_GIVEN`)

Add a code comment explaining the SDK limitation. Also add a note in `docs/thinking.md`
in the Groq section.

### 1.5 Cerebras: Handle `thinking=True` (not just `False`)

**File:** Cerebras model/helper (`_cerebras_settings_to_openai_settings()`)

Currently only maps `thinking=False` → `disable_reasoning: True`. Add the reverse:
- `thinking=True` or effort level → `disable_reasoning: False`

---

## Phase 2: Missing Implementations (silent no-ops)

These providers have profile flags set but their model classes never read
`model_request_parameters.thinking`. The unified setting is silently ignored.

### 2.1 Cohere: Implement `_translate_thinking()` or remove flags

**File:** `pydantic_ai_slim/pydantic_ai/models/cohere.py`
**Also:** `pydantic_ai_slim/pydantic_ai/profiles/cohere.py`

Research how Cohere's API controls thinking for Command R+ models. Implement
`_translate_thinking()` that reads `model_request_parameters.thinking` and maps it
to Cohere's API parameter.

**If Cohere has no API-level thinking control**: Remove `supports_thinking=True` and
`thinking_always_enabled=True` from the profile, and add a comment explaining why.
Models that always think with no API knob don't benefit from profile flags.

### 2.2 Mistral: Implement `_translate_thinking()` or remove flags

**File:** `pydantic_ai_slim/pydantic_ai/models/mistral.py`
**Also:** `pydantic_ai_slim/pydantic_ai/profiles/mistral.py`

Same as Cohere — research Magistral's API, implement translation, or remove flags.

### 2.3 Missing profile flags for profile-only providers

**Files:**
- `profiles/meta.py` — Llama-4 thinking models: `supports_thinking=True`, `thinking_tags`
- `profiles/qwen.py` — QwQ models: `supports_thinking=True`, `thinking_always_enabled=True`, `thinking_tags`
- `profiles/moonshotai.py` — `kimi-thinking-*`: `supports_thinking=True`
- `profiles/harmony.py` — `gpt-oss-120b`: `supports_thinking=True`
- `profiles/zai.py` — `zai-glm-4.6`: `supports_thinking=True`

These enable cascading to 15+ hosting providers (Ollama, Fireworks, Together, etc.).

Research each model family to determine correct flag values (especially
`thinking_always_enabled` and whether `thinking_tags` are needed for tag-based parsing).

---

## Phase 3: Safety & Hygiene

### 3.1 Strip `thinking` from `model_settings` after resolution

**File:** `pydantic_ai_slim/pydantic_ai/models/__init__.py` (in `prepare_request()`)

After copying `model_settings['thinking']` to `model_request_parameters.thinking`,
pop it from the dict:
```python
model_settings.pop('thinking', None)
```

Prevents downstream provider code from accidentally reading the raw unresolved value.

### 3.2 Add `thinking_mode` property to `ModelProfile`

**File:** `pydantic_ai_slim/pydantic_ai/profiles/__init__.py`

```python
@property
def thinking_mode(self) -> Literal['unsupported', 'optional', 'required']:
    if self.thinking_always_enabled:
        return 'required'
    if self.supports_thinking:
        return 'optional'
    return 'unsupported'
```

Update `prepare_request()` to use it instead of inline boolean checks.

### 3.3 Rename Google profile field

**File:** `GoogleModelProfile` definition + Google model class

Change `google_supports_thinking_level: bool` to
`google_thinking_api: Literal['budget', 'level'] | None`.

Update all usages in the Google model class and profile factory.

### 3.4 Rename all thinking translation methods to `_translate_thinking()`

**Files:** All provider model classes with thinking translation methods

Rename for consistency and grep-ability:
- Anthropic: `_get_thinking_param()` → `_translate_thinking()`
- OpenAI: `_get_reasoning_effort()` → `_translate_thinking()`
- Groq: `_get_reasoning_format()` → `_translate_thinking()`
- Google: `_get_thinking_config()` → `_translate_thinking()`
- Bedrock: `_get_thinking_fields()` → `_translate_thinking()`

Return types differ per provider — the convention is about naming, not signatures.

---

## Phase 4: Tests

**File:** `tests/test_unified_thinking.py` (new)

Every behavior from Phases 1-3 should have a corresponding test.

### 4.1 Core resolver tests (`TestPrepareRequestThinking`)

Test `Model.prepare_request()` thinking resolution:
- `thinking=True` + `supports_thinking=True` → `params.thinking = True`
- `thinking='high'` + `supports_thinking=True` → `params.thinking = 'high'`
- `thinking='medium'` + `supports_thinking=False` → `params.thinking = None` (silently dropped)
- `thinking=True` + `thinking_always_enabled=True` → `params.thinking = True`
- No `thinking` + `thinking_always_enabled=True` → `params.thinking = True` (auto-enabled)
- No `thinking` + `supports_thinking=True` → `params.thinking = None` (not auto-enabled)

### 4.2 No-mutation regression test

Test that `prepare_request()` does not mutate the original `model_settings` dict.
(This was a real bug on a prior branch.)

### 4.3 Provider-specific settings precedence

For each provider with both unified and provider-specific settings:
- `anthropic_thinking` set + `thinking` set → `anthropic_thinking` wins
- `openai_reasoning_effort` set + `thinking` set → `openai_reasoning_effort` wins
- etc.

### 4.4 Per-provider translation tests

One test class per provider. Each tests the full `ThinkingLevel` → provider parameter
mapping after the Phase 1 fixes:

**Anthropic:**
- `True` → adaptive (adaptive-capable models) / budget (budget-only models)
- `False` → `{'type': 'disabled'}`
- `'low'` → `budget_tokens: 1024`
- `'medium'` → adaptive
- `'high'` → `budget_tokens: 16384`

**OpenAI:**
- `True` → `reasoning_effort='medium'`
- `False` → `reasoning_effort='none'`
- `'low'`/`'medium'`/`'high'` → direct passthrough

**Google (budget models, i.e. Gemini 2.5):**
- `True` → default budget
- `False` → `thinking_budget: 0`
- Effort levels → specific budgets (2048/8192/24576)

**Google (level models, i.e. Gemini 3+):**
- Effort levels → `LOW`/`MEDIUM`/`HIGH` enum values

**Groq:**
- `True`/effort → `reasoning_format='parsed'`
- `False` → `reasoning_format='hidden'`

**xAI:**
- `True` → `'low'`
- `'medium'` → `'low'`
- `'high'` → `'high'`

**Bedrock (Anthropic variant):** Same as Anthropic through Bedrock's layer
**Bedrock (OpenAI variant):** Same as OpenAI through Bedrock's layer

### 4.5 Anthropic conflict test

- `thinking=True` + output type requiring tool → raises `UserError`
- `thinking=False` + output type requiring tool → no error

### 4.6 Settings stripping test

- After `prepare_request()`, `model_settings` should not contain `'thinking'` key

### 4.7 Cross-provider portability

- Same `Thinking(effort='high')` capability produces sensible (non-error, non-no-op)
  results across all providers with `supports_thinking=True`

---

## Phase 5: Documentation

### 5.1 Update `docs/thinking.md`

- Add Groq limitation note (no true disable — `'hidden'` hides output only)
- Verify effort mapping tables match the code after Phase 1 fixes
- Ensure all providers with profile flags are represented

### 5.2 Update effort mapping reference table

After Phase 1 fixes, update the mapping table to reflect:
- Anthropic: `True`/`'medium'` → adaptive, `'low'` → budget 1024, `'high'` → budget 16384
- xAI: `True`/`'medium'` → low, `'high'` → high
- Groq: `False` → hidden

---

## Execution Order

```
Phase 1 (correctness fixes) ─┐
Phase 2 (missing impls)      ├─ can be parallelized
Phase 3 (safety & hygiene)   ─┘
         │
         ▼
Phase 4 (tests) ── validates Phases 1-3
         │
         ▼
Phase 5 (docs) ── reflects final state
```

---

## Out of Scope

- **Thought summaries**: No unified summary field. Provider-specific settings remain.
- **`ResolvedThinkingConfig` dataclass**: Superseded by capabilities architecture.
  Raw `ThinkingLevel` on `ModelRequestParameters` is the right design.
- **OpenAI `'minimal'`/`'xhigh'`**: Valid SDK values, not exposed in unified API.
  Users can reach them via `openai_reasoning_effort` provider-specific setting.
- **Broader Bedrock vendor families**: Existing variants cover current needs.
