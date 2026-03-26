# Plan: Google Structured Output + Tool Combination (Gemini 3)

## Context

**Issues**: [#4801](https://github.com/pydantic/pydantic-ai/issues/4801) (NativeOutput + function tools), [#4788](https://github.com/pydantic/pydantic-ai/issues/4788) (function + builtin tools)
**Linked by**: DouweM (maintainer)

Google Gemini 3 supports [structured outputs with all tools](https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools) including function calling, and [combining builtin tools with function calling](https://ai.google.dev/gemini-api/docs/tool-combination). Currently pydantic-ai blocks both combinations unconditionally.

**Three distinct mechanisms in the Google API** (important for understanding the restrictions):

1. **function_tools**: user-defined `function_declarations` the model can call (e.g. `get_weather`)
2. **output_tools** (from `ToolOutput`): also `function_declarations`, but used for structured output extraction — the model "calls" them to return typed data. Defined via `ModelRequestParameters.tool_defs` (`models/__init__.py`), which merges `function_tools + output_tools`. From the API's perspective, output_tools ARE function declarations.
3. **NativeOutput**: uses `response_schema` + `response_mime_type='application/json'` — the model returns structured JSON directly, no function calling involved. Completely separate from function declarations.

The restrictions being lifted are:
- `function_declarations` + `builtin_tools` → affects both function_tools AND output_tools (#4788)
- `response_schema` + `function_declarations` → NativeOutput + function_tools (#4801)

**Comparison with other providers**:
- **OpenAI**: Allows NativeOutput + function tools. No restrictions.
- **Anthropic**: Allows NativeOutput + function tools (restricts only when thinking + output tools).
- **Google**: Currently blocks both combinations for all models. Needs version-gating for Gemini 3.

## Implementation

### 1. Profile: add capability flag

**File**: `pydantic_ai_slim/pydantic_ai/profiles/google.py`

Add to `GoogleModelProfile`:
```python
google_supports_tool_combination: bool = False
```

Set to `is_3_or_newer` in `google_model_profile()`.

Named after [Google's own "tool combination" feature](https://ai.google.dev/gemini-api/docs/tool-combination). This single flag covers all three restriction lifts because they're all part of the same Gemini 3 capability set:
- function_tools + builtin_tools (#4788) — function declarations can coexist with builtin tools
- output_tools + builtin_tools — output_tools ARE function declarations (see Context above)
- NativeOutput + function_tools (#4801) — response_schema can coexist with function declarations

We keep the existing `google_supports_native_output_with_builtin_tools` for Gemini 2 fallback behavior (the `prepare_request` workaround that converts ToolOutput to NativeOutput/PromptedOutput when the new flag is False). For Gemini 3, the workaround is skipped entirely.

### 2. Model: three changes

**File**: `pydantic_ai_slim/pydantic_ai/models/google.py`

#### 2a. `GoogleModel.prepare_request` — skip workaround for Gemini 3

The existing workaround converts `ToolOutput` to `NativeOutput`/`PromptedOutput` when output_tools + builtin_tools are both present, because older models can't have function_declarations + builtin tools together.

For Gemini 3, this workaround is unnecessary — output_tools (function declarations) can coexist with builtin tools. Wrap existing logic in a negated check:

```python
def prepare_request(self, ...):
    google_profile = GoogleModelProfile.from_profile(self.profile)
    if model_request_parameters.builtin_tools and model_request_parameters.output_tools:
        if not google_profile.google_supports_tool_combination:
            if model_request_parameters.output_mode == 'auto':
                output_mode = 'native' if google_profile.google_supports_native_output_with_builtin_tools else 'prompted'
                model_request_parameters = replace(model_request_parameters, output_mode=output_mode)
            else:
                output_mode = 'NativeOutput' if google_profile.google_supports_native_output_with_builtin_tools else 'PromptedOutput'
                raise UserError(
                    f'This model does not support output tools and built-in tools at the same time. Use `output_type={output_mode}(...)` instead.'
                )
    return super().prepare_request(model_settings, model_request_parameters)
```

#### 2b. `GoogleModel._get_tools` — function tools + builtin tools

Gate the restriction on the profile flag:

```python
if model_request_parameters.builtin_tools:
    if model_request_parameters.function_tools:
        if not GoogleModelProfile.from_profile(self.profile).google_supports_tool_combination:
            raise UserError(
                'This model does not support function tools and built-in tools at the same time.'
            )
```

Error message updated to say "This model" per review feedback.

#### 2c. `GoogleModel._build_content_and_config` — NativeOutput + function tools

Gate the restriction on the same flag. Both `response_schema + function_declarations` and `function_declarations + builtin_tools` are part of Gemini 3's tool combination capability:

```python
if model_request_parameters.output_mode == 'native':
    if model_request_parameters.function_tools:
        if not GoogleModelProfile.from_profile(self.profile).google_supports_tool_combination:
            raise UserError(
                'This model does not support `NativeOutput` and function tools at the same time. '
                'Use `output_type=ToolOutput(...)` instead.'
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
- `thought_signature` on `Part` — round-tripped via `provider_details` in `_map_content` and `_map_part_to_content`
- `FunctionCall.id` / `FunctionResponse.id` — already mapped in `_map_content` and `_map_tool_return_part`

**Open question**: How much of the existing `groundingMetadata`-based reconstruction can we remove/simplify? This needs investigation during implementation — we should compare the `toolCall`/`toolReturn` response parts against the current reconstruction logic to see what becomes redundant.

## Edge cases

Error types: **UserError** = pydantic-ai raises before API call; **workaround** = auto mode silently converts output mode

| Scenario | Gemini 2 | Gemini 3 |
|---|---|---|
| NativeOutput (no tools) | works | works |
| NativeOutput + builtin_tools | not blocked by pydantic-ai (untested) | works (existing) |
| NativeOutput + function_tools | UserError in `_build_content_and_config` | **works (new)** |
| NativeOutput + function + builtin | UserError in `_get_tools` (hits function+builtin check first) | **works (new)** |
| function + builtin (no output type) | UserError in `_get_tools` | **works (new)** |
| auto + function + builtin + output_type | UserError in `_get_tools` | **works (new, auto stays 'tool')** |
| ToolOutput + builtin_tools (auto mode) | workaround to native/prompted in `prepare_request` | **works (new, no workaround)** |
| ToolOutput + builtin_tools (explicit) | UserError in `prepare_request` | **works (new, no workaround)** |
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

| Test in test_google.py | Replaced by |
|---|---|
| `test_google_native_output_with_tools` | case 8 |
| `test_google_builtin_tools_with_other_tools` | cases 9, 10 |
| `test_google_native_output_with_builtin_tools_gemini_3` | cases 4, 5, 6 |

Note: `test_google_native_output` and `test_google_native_output_multiple` test NativeOutput WITHOUT tools — could be moved later to keep this PR focused.

## Documentation (deferred per PR flow)

Per our PR flow, docs are deferred until after review confirms logic is correct. These files need updating once logic is finalized:

- `docs/output.md` — remove/update the statement "Gemini cannot use tools at the same time as structured output"
- `docs/builtin-tools.md` — update Google row: "Using built-in tools and function tools (including output tools) at the same time is not supported" is no longer true for Gemini 3+
- `docs/models/google.md` — add section on structured output + tool combination support for Gemini 3

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
