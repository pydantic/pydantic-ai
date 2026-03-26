# Plan: Google Structured Output + Tool Combination (Gemini 3)

## Context

**Issues**: [#4801](https://github.com/pydantic/pydantic-ai/issues/4801) (NativeOutput + function tools), [#4788](https://github.com/pydantic/pydantic-ai/issues/4788) (function + builtin tools)
**Linked by**: DouweM (maintainer)

Google Gemini 3 supports [structured outputs with all tools](https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools) including function calling, and [combining builtin tools with function calling](https://ai.google.dev/gemini-api/docs/tool-combination). Currently pydantic-ai blocks both combinations unconditionally.

**Key concept**: `output_tools` (from `ToolOutput`) are function declarations from the API's perspective — `tool_defs = function_tools + output_tools` (line 651 of `models/__init__.py`). They all become `function_declarations` in the Google API. This means the restriction on function_declarations + builtin_tools affects output_tools equally.

**Comparison with other providers**:
- **OpenAI**: Allows NativeOutput + function tools. No restrictions.
- **Anthropic**: Allows NativeOutput + function tools (restricts only when thinking + output tools).
- **Google**: Currently blocks both combinations for all models. Needs version-gating for Gemini 3.

## Implementation

### 1. Profile: add capability flag

**File**: `pydantic_ai_slim/pydantic_ai/profiles/google.py`

Add to `GoogleModelProfile`:
```python
google_supports_function_tools_with_builtin_tools: bool = False
```

Set to `is_3_or_newer` in `google_model_profile()`.

This single flag covers:
- function_tools + builtin_tools (#4788)
- output_tools + builtin_tools (since output_tools ARE function declarations)
- NativeOutput + function_tools (#4801) — see rationale at 2b

We also keep `google_supports_native_output_with_builtin_tools` for Gemini 2 fallback behavior (prompted vs native workaround when the flag above is False). For Gemini 3, the workaround is skipped entirely.

### 2. Model: three changes

**File**: `pydantic_ai_slim/pydantic_ai/models/google.py`

#### 2a. `prepare_request` (line 286-301) — remove workaround for Gemini 3

The existing workaround converts `ToolOutput` to `NativeOutput`/`PromptedOutput` when output_tools + builtin_tools are both present, because older models can't have function_declarations + builtin tools together.

For Gemini 3, this workaround is unnecessary — output_tools (function declarations) can coexist with builtin tools. Skip the workaround when `google_supports_function_tools_with_builtin_tools` is True:

```python
def prepare_request(self, ...):
    google_profile = GoogleModelProfile.from_profile(self.profile)
    if model_request_parameters.builtin_tools and model_request_parameters.output_tools:
        if google_profile.google_supports_function_tools_with_builtin_tools:
            pass  # Gemini 3+: output_tools (function declarations) + builtin tools work fine
        elif model_request_parameters.output_mode == 'auto':
            output_mode = 'native' if google_profile.google_supports_native_output_with_builtin_tools else 'prompted'
            model_request_parameters = replace(model_request_parameters, output_mode=output_mode)
        else:
            output_mode = 'NativeOutput' if google_profile.google_supports_native_output_with_builtin_tools else 'PromptedOutput'
            raise UserError(
                f'This model does not support output tools and built-in tools at the same time. Use `output_type={output_mode}(...)` instead.'
            )
    return super().prepare_request(model_settings, model_request_parameters)
```

#### 2b. `_get_tools` (line 444-446) — function tools + builtin tools

Gate the restriction on the profile flag:

```python
if model_request_parameters.builtin_tools:
    if model_request_parameters.function_tools:
        if not GoogleModelProfile.from_profile(self.profile).google_supports_function_tools_with_builtin_tools:
            raise UserError(
                'This model does not support function tools and built-in tools at the same time.'
            )
```

Error message updated to say "This model" per review feedback.

#### 2c. `_build_content_and_config` (line 537-541) — NativeOutput + function tools

Gate the restriction on the same flag. Rationale: NativeOutput + function_tools is part of the same Gemini 3 'structured output with tools' capability — if the model supports function_declarations + builtin tools, it supports function_declarations + response_schema.

```python
if model_request_parameters.output_mode == 'native':
    if model_request_parameters.function_tools:
        if not GoogleModelProfile.from_profile(self.profile).google_supports_function_tools_with_builtin_tools:
            raise UserError(
                'This model does not support `NativeOutput` and function tools at the same time. Use `output_type=ToolOutput(...)` instead.'
            )
    response_mime_type = 'application/json'
    ...
```

### 3. `_get_tool_config` — no changes needed

When `output_mode='native'`, `allow_text_output=True` (set by base class). `_get_tool_config` returns None (no forced tool calling). The model freely chooses between calling function tools and returning structured text.

### 4. SDK: `include_server_side_tool_invocations` + bump to `>=1.68.0`

The [tool combination docs](https://ai.google.dev/gemini-api/docs/tool-combination) require `include_server_side_tool_invocations=True` when using builtin tools. This flag was added in **google-genai 1.68.0**.

Per DouweM's review: always set this flag when ANY builtin tools are enabled, not just when combined with function tools. This gives us proper `toolCall`/`toolReturn` parts in responses, which we can use to build `BuiltinToolCallPart`/`BuiltinToolReturnPart` instead of reconstructing from `groundingMetadata`.

**Changes**:
- Bump minimum SDK pin from `>=1.56.0` to `>=1.68.0` in `pydantic_ai_slim/pyproject.toml`
- In `_build_content_and_config`: set `config['include_server_side_tool_invocations'] = True` whenever `model_request_parameters.builtin_tools` is non-empty
- Investigate building `BuiltinToolCallPart`/`BuiltinToolReturnPart` from `toolCall`/`toolReturn` response parts, potentially simplifying existing `groundingMetadata` reconstruction logic

**Already handled by pydantic-ai** (no changes needed):
- `thought_signature` on `Part` — round-tripped via `provider_details` (`google.py:984-990`, `google.py:1161-1179`)
- `FunctionCall.id` / `FunctionResponse.id` — already mapped (`google.py:1016`, `google.py:1255-1256`)

**Open question**: How much of the existing `groundingMetadata`-based reconstruction can we remove/simplify? This needs investigation during implementation — we should compare the `toolCall`/`toolReturn` response parts against the current reconstruction logic to see what becomes redundant.

## Edge cases

| Scenario | Gemini 2 | Gemini 3 |
|---|---|---|
| NativeOutput (no tools) | works | works |
| NativeOutput + builtin_tools | error | works (existing) |
| NativeOutput + function_tools | error | **works (new)** |
| NativeOutput + function + builtin | error | **works (new)** |
| function + builtin (no output type) | error | **works (new)** |
| auto + function + builtin + output_type | error | **works (new, auto stays 'tool')** |
| ToolOutput + builtin_tools | error (workaround to native/prompted) | **works (new, no workaround)** |
| ToolOutput + function_tools | works | works (standard tool mode) |

## Tests

### New file: `tests/models/google/test_structured_output.py`

Following `tests/models/anthropic/` pattern with conftest + VCR cassettes.

#### Structure
```
tests/models/google/
  __init__.py
  conftest.py              # google_model factory fixture
  test_structured_output.py
  cassettes/
    test_structured_output/  # VCR cassettes
```

#### conftest.py
- `google_model` factory: creates `GoogleModel(model_name, provider=GoogleProvider(api_key=...))` (mirrors `tests/models/anthropic/conftest.py`)
- Reuse `gemini_api_key` from `tests/conftest.py`, `google_provider` from `tests/models/test_google.py`

#### Test cases

**VCR integration tests** (record against live API):

1. `test_native_output_with_function_tools` - Gemini 3 + NativeOutput(CityLocation) + function tool that returns data -> assert structured output + all_messages snapshot
2. `test_native_output_with_function_tools_stream` - same as above, streaming
3. `test_function_tools_with_builtin_tools` - Gemini 3 + function tool + WebSearchTool -> assert response + messages
4. `test_native_output_with_function_and_builtin_tools` - Gemini 3 + NativeOutput + function tool + WebSearchTool -> full combo
5. `test_native_output_with_builtin_tools` - Gemini 3 + NativeOutput + WebSearchTool (move from test_google.py)
6. `test_tool_output_with_builtin_tools` - Gemini 3 + ToolOutput + WebSearchTool -> works now (no workaround)
7. `test_auto_mode_with_function_and_builtin_tools` - Gemini 3 + output_type=SomeModel + function tool + WebSearchTool -> verify auto stays 'tool' mode

**Error tests** (no VCR needed, error before API call):

8. `test_native_output_with_function_tools_unsupported` - Gemini 2 + NativeOutput + function tool -> UserError
9. `test_function_tools_with_builtin_tools_unsupported` - Gemini 2 + function + builtin -> UserError
10. `test_tool_output_with_builtin_tools_unsupported` - Gemini 2 + ToolOutput + builtin -> UserError (workaround suggests NativeOutput/PromptedOutput)

### Tests to remove from `tests/models/test_google.py`

These tests are superseded by the new file:

| Test in test_google.py | Line | Replaced by |
|---|---|---|
| `test_google_native_output_with_tools` | 2880 | case 8 |
| `test_google_builtin_tools_with_other_tools` | 3279 | cases 9, 10 |
| `test_google_native_output_with_builtin_tools_gemini_3` | 3315 | cases 4, 5, 6 |

Note: `test_google_native_output` (line 2902) and `test_google_native_output_multiple` (line 2955) test NativeOutput WITHOUT tools - could be moved later to keep this PR focused.

## Verification

1. `make format && make lint` - style
2. `make typecheck 2>&1 | tee /tmp/typecheck-output.txt` - types
3. Record VCR cassettes: `source .env && uv run pytest tests/models/google/test_structured_output.py --record-mode=once -x -v`
4. Replay: `uv run pytest tests/models/google/test_structured_output.py -x -v`
5. Verify removed tests don't break: `uv run pytest tests/models/test_google.py -x -v`
6. Full test suite: `uv run pytest tests/ -x --timeout=60`

## Files to modify

- `pydantic_ai_slim/pydantic_ai/profiles/google.py` - add flag
- `pydantic_ai_slim/pydantic_ai/models/google.py` - update `prepare_request`, gate `_get_tools`, gate `_build_content_and_config`, set `include_server_side_tool_invocations`
- `pydantic_ai_slim/pyproject.toml` - bump google-genai minimum to `>=1.68.0`
- `tests/models/google/__init__.py` - new (empty)
- `tests/models/google/conftest.py` - new (fixtures)
- `tests/models/google/test_structured_output.py` - new (tests)
- `tests/models/test_google.py` - remove 3 superseded tests
- `tests/models/cassettes/test_google/test_google_builtin_tools_with_other_tools.yaml` - delete
- `tests/models/cassettes/test_google/test_google_native_output_with_builtin_tools_gemini_3.yaml` - delete
