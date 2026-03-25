# Plan: Google Structured Output + Tool Combination (Gemini 3)

## Context

**Issues**: [#4801](https://github.com/pydantic/pydantic-ai/issues/4801) (NativeOutput + function tools), [#4788](https://github.com/pydantic/pydantic-ai/issues/4788) (function + builtin tools)
**Linked by**: DouweM (maintainer)

Google Gemini 3 supports [structured outputs with all tools](https://ai.google.dev/gemini-api/docs/structured-output?example=recipe#structured_outputs_with_tools) including function calling, and [combining builtin tools with function calling](https://ai.google.dev/gemini-api/docs/tool-combination). Currently pydantic-ai blocks both combinations unconditionally.

**Comparison with other providers**:
- **OpenAI**: Allows NativeOutput + function tools. No restrictions.
- **Anthropic**: Allows NativeOutput + function tools (restricts only when thinking + output tools).
- **Google**: Currently blocks both combinations for all models. Needs version-gating for Gemini 3.

## Implementation

### 1. Profile: add two capability flags

**File**: `pydantic_ai_slim/pydantic_ai/profiles/google.py`

Add to `GoogleModelProfile`:
```python
google_supports_native_output_with_function_tools: bool = False
google_supports_function_tools_with_builtin_tools: bool = False
```

In `google_model_profile()`, set both to `is_3_or_newer`.

Two separate flags (vs one combined) follows the existing naming convention (`google_supports_native_output_with_builtin_tools`) and allows independent control.

### 2. Model: gate restrictions on profile flags

**File**: `pydantic_ai_slim/pydantic_ai/models/google.py`

#### 2a. `_get_tools` (line 444-446) - function tools + builtin tools

Current: unconditional `raise UserError(...)` when both present.
Change: read `google_supports_function_tools_with_builtin_tools` from profile; only error if False.

```python
if model_request_parameters.builtin_tools:
    if model_request_parameters.function_tools:
        if not GoogleModelProfile.from_profile(self.profile).google_supports_function_tools_with_builtin_tools:
            raise UserError('Google does not support function tools and built-in tools at the same time.')
```

When the flag is True, the code proceeds to append builtin tool dicts to the same `tools` list that already has function tool dicts. Both types become separate `ToolDict` entries in the list.

#### 2b. `_build_content_and_config` (line 537-541) - NativeOutput + function tools

Current: unconditional `raise UserError(...)` when `output_mode == 'native'` and function_tools present.
Change: read `google_supports_native_output_with_function_tools` from profile; only error if False.

```python
if model_request_parameters.output_mode == 'native':
    if model_request_parameters.function_tools:
        if not GoogleModelProfile.from_profile(self.profile).google_supports_native_output_with_function_tools:
            raise UserError(
                'Google does not support `NativeOutput` and function tools at the same time. Use `output_type=ToolOutput(...)` instead.'
            )
    response_mime_type = 'application/json'
    ...
```

When allowed, `response_mime_type` + `response_json_schema` are set alongside the function tool declarations. The model can call tools when needed and must return structured JSON conforming to the schema for its final text response.

#### 2c. No changes to `prepare_request`

The existing `prepare_request` (line 286-301) handles `output_tools + builtin_tools` using `google_supports_native_output_with_builtin_tools`. This already correctly resolves auto mode to 'native' for Gemini 3 when builtin_tools + output_tools are present.

For auto mode + function_tools + builtin_tools on Gemini 3:
1. `prepare_request`: output_tools + builtin_tools → auto converts to 'native' (existing logic)
2. Base class: clears output_tools, populates output_object
3. `_get_tools`: function_tools + builtin_tools → now allowed (fix 2a)
4. `_build_content_and_config`: native + function_tools → now allowed (fix 2b)

Auto mode WITHOUT builtin_tools resolves to 'tool' (via `default_structured_output_mode='tool'`), so output_tools handle structured output. No change needed.

### 3. `_get_tool_config` - no changes needed

When `output_mode='native'`, `allow_text_output=True` (set by base class). `_get_tool_config` returns None (no forced tool calling). The model freely chooses between calling function tools and returning structured text. Correct behavior per Google docs.

## Edge cases

| Scenario | Gemini 2 | Gemini 3 |
|---|---|---|
| NativeOutput (no tools) | works | works |
| NativeOutput + builtin_tools | error | works (existing) |
| NativeOutput + function_tools | error | **works (new)** |
| NativeOutput + function + builtin | error | **works (new)** |
| function + builtin (no output type) | error | **works (new)** |
| auto + function + builtin + output_type | error | **works (new, auto->native)** |
| ToolOutput + builtin_tools | error | error (correct, use NativeOutput) |
| ToolOutput + function_tools | works | works (standard tool mode) |

### 4. SDK investigation: `include_server_side_tool_invocations`

The [tool combination docs](https://ai.google.dev/gemini-api/docs/tool-combination) say `include_server_side_tool_invocations=True` is required when combining function tools + builtin tools.

**SDK support**:
- `include_server_side_tool_invocations` was added to `GenerateContentConfig` in **google-genai 1.68.0** ([release](https://github.com/googleapis/python-genai/releases/tag/v1.68.0))
- Current minimum pin: `>=1.56.0` (in `pydantic_ai_slim/pyproject.toml`)
- Current installed: 1.56.0

**Already handled by pydantic-ai** (no changes needed):
- `thought_signature` on `Part` — exists in SDK 1.56.0, pydantic-ai already round-trips it via `provider_details` (`google.py:984-990`, `google.py:1161-1179`)
- `FunctionCall.id` / `FunctionResponse.id` — exists in SDK, already mapped (`google.py:1016`, `google.py:1255-1256`)

**Approach**:
1. For **#4801** (NativeOutput + function tools): no SDK changes needed. Just lift the restriction. This doesn't involve builtin tools, so `include_server_side_tool_invocations` is irrelevant.
2. For **#4788** (function + builtin tools): test against the live API first WITHOUT setting `include_server_side_tool_invocations`:
   - Existing builtin tool tests (Google Search, Code Execution) already work without this flag
   - If the API rejects the combination without the flag, add it conditionally:
     ```python
     # In _build_content_and_config, when both tool types present:
     if has_function_tools and has_builtin_tools:
         config['include_server_side_tool_invocations'] = True
     ```
     Since `GenerateContentConfigDict` is a TypedDict (dict at runtime), this key is silently passed through. On SDK >=1.68.0 it's recognized; on older versions the SDK ignores unknown dict keys when constructing the protobuf.
   - If that also fails, we may need to bump the minimum SDK version to `>=1.68.0` for the tool combination feature — but this would be a last resort since it forces upgrades on all users.

**No minimum version bump** unless API testing proves it necessary. Per AGENTS.md: "Verify provider limitations through testing before implementing workarounds."

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

1. `test_native_output_with_function_tools` - Gemini 3 + NativeOutput(CityLocation) + function tool that returns data → assert structured output + all_messages snapshot
2. `test_native_output_with_function_tools_stream` - same as above, streaming
3. `test_function_tools_with_builtin_tools` - Gemini 3 + function tool + WebSearchTool → assert response + messages
4. `test_native_output_with_function_and_builtin_tools` - Gemini 3 + NativeOutput + function tool + WebSearchTool → full combo
5. `test_native_output_with_builtin_tools` - Gemini 3 + NativeOutput + WebSearchTool (existing behavior, move from test_google.py)
6. `test_auto_mode_with_function_and_builtin_tools` - Gemini 3 + output_type=SomeModel + function tool + WebSearchTool → verify auto resolves to native

**Error tests** (no VCR needed, error before API call):

7. `test_native_output_with_function_tools_unsupported` - Gemini 2 + NativeOutput + function tool → UserError
8. `test_function_tools_with_builtin_tools_unsupported` - Gemini 2 + function + builtin → UserError
9. `test_tool_output_with_builtin_tools_error` - Gemini 3 + ToolOutput + builtin → UserError (correct - use NativeOutput)

### Tests to remove from `tests/models/test_google.py`

These tests are superseded by the new file:

| Test in test_google.py | Line | Replaced by |
|---|---|---|
| `test_google_native_output_with_tools` | 2880 | case 7 |
| `test_google_builtin_tools_with_other_tools` | 3279 | cases 8, 9 |
| `test_google_native_output_with_builtin_tools_gemini_3` | 3315 | cases 4, 5, 6 |

Note: `test_google_native_output` (line 2902) and `test_google_native_output_multiple` (line 2955) test NativeOutput WITHOUT tools - they could be moved too but are not replaced by our new tests. We can defer moving them to keep this PR focused.

## Verification

1. `make format && make lint` - style
2. `make typecheck 2>&1 | tee /tmp/typecheck-output.txt` - types
3. Record VCR cassettes: `source .env && uv run pytest tests/models/google/test_structured_output.py --record-mode=once -x -v`
4. Replay: `uv run pytest tests/models/google/test_structured_output.py -x -v`
5. Verify removed tests don't break: `uv run pytest tests/models/test_google.py -x -v`
6. Full test suite: `uv run pytest tests/ -x --timeout=60`

## Files to modify

- `pydantic_ai_slim/pydantic_ai/profiles/google.py` - add 2 flags
- `pydantic_ai_slim/pydantic_ai/models/google.py` - gate 2 restrictions
- `tests/models/google/__init__.py` - new (empty)
- `tests/models/google/conftest.py` - new (fixtures)
- `tests/models/google/test_structured_output.py` - new (tests)
- `tests/models/test_google.py` - remove 3 superseded tests
- `tests/models/cassettes/test_google/test_google_builtin_tools_with_other_tools.yaml` - delete (superseded cassette)
- `tests/models/cassettes/test_google/test_google_native_output_with_builtin_tools_gemini_3.yaml` - delete (superseded cassette)
- Note: `test_google_native_output_with_tools` has no cassette (errors before API call)
