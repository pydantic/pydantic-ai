# Handoff: Rewrite Private-API Tests to Use Public APIs

> **Status**: Blocking — must complete before PR is shippable
> **Branch**: `skill-support-v2`
> **Context**: Code review found 63 `# pyright: ignore[reportPrivateUsage]` suppressions across test files. Per `tests/AGENTS.md` rule:177, tests must go through public APIs, not private methods.

---

## Current State

- **Pyright**: 0 errors across entire repo
- **Tests**: 657 passed, 0 failed
- **But**: 63 tests call private methods directly (`_get_tools`, `_process_response`, `_map_messages`, `_build_shell_tool_param`, etc.) and suppress the type errors with `# pyright: ignore[reportPrivateUsage]`

## What Needs to Happen

Replace direct private-method calls with tests that go through `Agent.run()` (or `Agent.run_stream()`) using either:
- **VCR cassettes** (preferred for adapter integration tests) — record with `doppler run -- uv run pytest ... --vcr-record=new_episodes`
- **TestModel** (for unit-level behavior tests that don't need real API responses)

## Files to Rewrite

### `tests/models/test_anthropic.py` (14 private-method calls)

| Line(s) | Private Method | What It Tests | Rewrite Strategy |
|---------|---------------|---------------|-----------------|
| 117 | `_map_usage` | Usage mapping | TestModel + assert `result.usage()` |
| 5467, 5499 | `_add_builtin_tools` | Builtin tool assembly | VCR cassette — verify tools in request via mock_client kwargs |
| 8734 | `_map_message` | Shell history round-trip | VCR multi-turn cassette with `message_history` |
| 9084, 9142 | `_get_tools` | Native tool definition emission, text_editor max_characters | VCR — inspect request tools param |
| 9201 | `_map_message` | Shell tool return round-trip | VCR multi-turn |
| 9260 | `_process_response` | Server tool use block mapping | VCR — assert `result.all_messages()` |
| 9283 | `_map_message` | Uploaded file target handling | VCR multi-turn with UploadedFile |
| 9368 | `_process_streamed_response` | Streaming text editor result | VCR stream cassette |
| 9609-9610 | `_get_tools` | Fallback warning for unsupported native tools | TestModel with custom profile (no VCR needed — just verify warning) |

### `tests/models/test_openai.py` (36 private-method calls)

| Line(s) | Private Method | What It Tests | Rewrite Strategy |
|---------|---------------|---------------|-----------------|
| 97 | `_resolve_openai_image_generation_size` | Image size resolution | Pure function — can stay as unit test (make function non-private or accept the suppress) |
| 1094, 1121 | `_map_messages` | Message mapping basics | VCR multi-turn |
| 3691, 3710 | `_completions_create`, `_responses_create` | API call construction | VCR — verify request structure |
| 3808, 3821 | `_map_user_prompt` | CachePoint handling | VCR with cache points |
| 4315 | `_map_messages` | Message ordering | VCR multi-turn |
| 5160-5161 | `_get_tools` | Native tool fallback warning | TestModel with custom profile |
| 5176 | `_map_user_prompt` | Uploaded file container target | VCR with UploadedFile |
| 5211, 5259 | `_map_messages` | Shell/apply-patch round-trip | VCR multi-turn |
| 5281, 5302 | `_native_tool_names` + `_process_response` | Local shell response processing | VCR — use ShellToolset with Agent |
| 5320, 5340, 5363 | `_native_tool_names` + `_process_response` + `_map_messages` | Apply-patch response + round-trip | VCR multi-turn with ApplyPatchToolset |
| 5389 | `_map_messages` | Apply-patch round-trip | VCR multi-turn |
| 5421-5473 | `_build_shell_tool_param` | Shell tool param construction | VCR — verify shell param in outgoing request |
| 5565 | `_native_tool_names` | Streaming apply-patch | VCR stream cassette |
| 5616 | `_map_user_prompt` | Uploaded file for shell container | VCR with UploadedFile + ShellTool |
| 5649 | `_map_messages` | Mixed shell + apply_patch round-trip | VCR multi-turn |
| 5684-5726 | `_native_tool_names` + `_process_response` | Combined shell + apply_patch processing | VCR |
| 5765-5767 | `_native_tool_names` + `_map_messages` | Apply-patch round-trip with operations | VCR multi-turn |
| 5796 | `_build_shell_tool_param` | Container with uploaded files | VCR |
| 5856 | `_process_response` | Hosted shell + apply_patch parts | VCR |
| 5904 | `_get_tools` | Shell + apply_patch tool emission | VCR — inspect tools in request |
| 5920 | `_map_shell_tool_call` | Shell call mapping | VCR — assert all_messages() |
| 5955 | `_map_shell_tool_call_output` | Shell output mapping | VCR — assert all_messages() |
| 5983 | `_native_tool_names` | Streaming apply_patch + shell | VCR stream |
| 6058, 6123 | `_process_streamed_response` | Streaming shell/apply_patch | VCR stream cassettes |

### `tests/test_shell_toolset.py` (6 private-method calls)

| Line(s) | Private Method | What It Tests | Rewrite Strategy |
|---------|---------------|---------------|-----------------|
| 648, 661, 670 | `_map_apply_patch_operation` | OpenAI operation type mapping | Move to test_openai.py, test via VCR cassette |
| 677, 686, 693 | `_build_apply_patch_call_operation` | Round-trip operation building | Move to test_openai.py, test via VCR multi-turn |

## Recording VCR Cassettes

```bash
# Anthropic cassettes
doppler run -- uv run pytest tests/models/test_anthropic.py::<test_name> --vcr-record=new_episodes

# OpenAI cassettes
doppler run -- uv run pytest tests/models/test_openai.py::<test_name> --vcr-record=new_episodes
```

## Principles (from tests/AGENTS.md)

1. **Test through public APIs** — `Agent.run()` / `Agent.run_stream()`, not `model._process_response()`
2. **Use VCR cassettes** — real HTTP recordings, not mocks
3. **Assert final output AND snapshot `result.all_messages()`** — validate complete execution trace
4. **Use `snapshot()`** for complex structured outputs with `IsStr()`, `IsDatetime()` matchers for variable fields
5. **Maintain 1:1 test file correspondence** — keep provider tests in `test_{provider}.py`

## Grouping Strategy

Many of the private-method tests can be consolidated into fewer, richer VCR tests:

1. **Anthropic shell round-trip** (1 VCR cassette): Covers `_get_tools` native emission, `_process_response` tool call mapping, `_map_message` round-trip — all via a single `Agent.run()` with ShellToolset + multi-turn `message_history`
2. **Anthropic text editor round-trip** (1 VCR cassette): Same pattern for TextEditorToolset
3. **Anthropic fallback warning** (TestModel): No cassette needed — just custom profile
4. **OpenAI local shell round-trip** (1 VCR cassette): Covers tool emission, response processing, round-trip
5. **OpenAI apply_patch round-trip** (1 VCR cassette): Same pattern
6. **OpenAI hosted shell** (1 VCR cassette): ShellTool builtin with container
7. **OpenAI fallback warning** (TestModel): Custom profile, no cassette

This reduces ~56 private-method calls to ~7 well-structured integration tests + 2 TestModel tests.

## Non-Goals

- Rewriting pre-existing private-method tests that aren't related to this branch's changes (e.g., `_map_usage` at line 117, `_resolve_openai_image_generation_size` at line 97)
- These can be addressed in a separate cleanup effort
