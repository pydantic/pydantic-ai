# fix(vercel_ai): emit `tool-input-(available|error)` on `FunctionToolCallEvent`

> Temporary design note attached to the PR for review discussion. Will be
> removed once reviewers have weighed in.

## Context

In the Vercel AI adapter, `ToolInputAvailableChunk` was emitted from
`handle_tool_call_end`, which fires on `PartEndEvent` for a `ToolCallPart` —
i.e. as soon as the model finishes streaming the call, *before* validation.
That has two consequences:

1. For deferred / client-side tools, where the chunk effectively acts as a
   "go execute this" signal, the frontend could be handed args the server
   would have rejected.
2. There was no way to surface a validation *failure* as a distinct chunk —
   the input chunk fired regardless.

This change moves the emission to `handle_function_tool_call`, which fires
post-validation in `_agent_graph.py`, and tightens a few related loose ends.

## What shipped

### Vercel AI adapter — `pydantic_ai/ui/vercel_ai/_event_stream.py`

- Replace `handle_tool_call_end` emission with `handle_function_tool_call`.
- On `args_valid=True`: emit `ToolInputAvailableChunk`. Use
  `event.validated_args` for `input` when populated, falling back to
  `part.args_as_dict()` (so the frontend sees the *coerced* value).
- On `args_valid=False`: emit `ToolInputErrorChunk` with `errorText` from
  `validation_error.tool_retry.model_response()`.
- Skip emission entirely when `args_valid is None` *and* no
  `validation_error` — covers resume of non-`ToolApproved` deferred results
  (`ToolDenied`, `ModelRetry`, direct return) and the output-tool
  end-strategy-skipped path. Re-announcing "input available" for those would
  be misleading. The `args_valid=False, validation_error=None` case
  (`UnexpectedModelBehavior` raised during validation) is also skipped — the
  outer error path closes the pending tool call with `tool-output-error`.

### Event surface — `pydantic_ai/messages.py`

- Add `validated_args: dict[str, Any] | None` and
  `validation_error: ToolRetryError | None` to `FunctionToolCallEvent`,
  mirroring `ValidatedToolCall`.
- `validation_error` is `compare=False`: the same information is carried by
  the following `FunctionToolResultEvent.RetryPromptPart`, and including it
  in equality breaks snapshot tooling (since `ToolRetryError` doesn't roundtrip
  cleanly through `copy.deepcopy` + repr).
- `validated_args` carries the documented "mild lie" — it's typed as a dict
  but holds a model instance for bare-`BaseModel` output tools, matching
  `ValidatedToolCall.validated_args` semantics.

### Agent graph — `pydantic_ai/_agent_graph.py`

- Propagate `validated_args` / `validation_error` through every
  `FunctionToolCallEvent` emission site that has a `ValidatedToolCall` in
  scope, including widening `_emit_skipped_output_tool` to accept and forward
  them.
- **Fix a pre-existing gap**: the output-tool happy path (validation passes
  *and* execution succeeds) emitted no `FunctionToolCallEvent`. The two
  failure branches emitted; the success branch did not. Add the success-path
  emission so consumers (UI adapters, tracing) finally see successful output
  tool calls. *(Open question for reviewers: whether the original asymmetry
  between function tools and output tools was intentional design or an
  oversight — see "Open questions" below.)*

### UI base — `pydantic_ai/ui/_event_stream.py`

- In the `AgentRunResultEvent` handler that synthesizes a
  `FunctionToolResultEvent` for the dangling output-tool call, drop the
  redundant `_turn_to('request')` call (the new output-tool
  `FunctionToolCallEvent` already does this transition upstream) and pop the
  output tool from `_pending_tool_calls` (the synthesized result event is
  dispatched directly into `handle_function_tool_result` and bypasses the
  main loop's pop, which would otherwise leave a stale entry that the
  on-error close path could double-handle).

### Misc — `pydantic_ai/exceptions.py`

- Add `__eq__` / `__hash__` to `ToolRetryError` based on `tool_retry`
  equality, so events containing it survive `copy.deepcopy` (required by
  inline-snapshot tooling and by general structural-equality consumers).

## Tests

- `test_event_stream_tool_input_error_with_provider_metadata` — pins that
  `FunctionToolCallEvent(args_valid=False, validation_error=…)` produces
  `tool-input-error` with the provider metadata propagated and `errorText`
  from `tool_retry.model_response()`.
- `test_event_stream_tool_call_part_end_does_not_emit_input_available` —
  pins the negative direction: `PartEnd` of a `ToolCallPart` *without* a
  following `FunctionToolCallEvent` produces neither `tool-input-available`
  nor `tool-input-error`.
- `test_tool_retry_error_equality_and_hash` — covers the new
  `ToolRetryError.__eq__` / `__hash__` (same / different / non-`ToolRetryError`
  operands).
- Existing `test_event_stream_tool_call_end_with_provider_metadata_v5/v6`
  updated to drive `FunctionToolCallEvent` explicitly so they exercise the
  new emission point.
- Existing snapshots updated where the change altered observable behavior:
  - `test_run_stream_response_error` and `test_run_stream_output_tool_error`
    now assert `tool-input-error` instead of `tool-input-available`.
  - `test_tool_output_denied_chunk_emission` shows the new suppression
    behavior on resume of a denied deferred tool, and the wired-through
    coerced `validated_args` on resume of an approved deferred tool.
  - `test_run_stream_with_explicit_deferred_tool_results` shows the
    coerced-args behavior on resume of an approved deferred tool.
  - `test_run_stream_output_tool` (test_ui.py),
    `test_complex_agent_run*` (test_temporal.py / test_dbos.py), and the
    bedrock / google streaming tests now show the new
    `FunctionToolCallEvent` emission for the output-tool happy path.

## Open questions for reviewers

- **Was the original asymmetry intentional?** Pre-PR, function tools always
  emitted `FunctionToolCallEvent` + `FunctionToolResultEvent`, but the
  output-tool happy path emitted neither — instead, `_event_stream.py`'s
  `AgentRunResultEvent` handler synthesized a `FunctionToolResultEvent` from
  the trailing `FinalResultEvent`. The two failure branches inside
  `process_tool_calls` (validation failed, `ToolRetryError` on execute) *did*
  emit the call event, suggesting the success-path skip might have been
  deliberate (since output-tool success doesn't round-trip back to the
  model). This PR aligns the success path with the failure paths so all
  tool calls surface uniformly to consumers; if the asymmetry was load-bearing,
  this part should be reverted.

- **Should the Vercel adapter still emit `tool-input-available` for output
  tools if the agent-graph emission is reverted?** Pre-PR, output tools got
  the chunk via `handle_tool_call_end` on `PartEnd` (pre-validation). If we
  reverted the agent-graph change, the new `handle_function_tool_call` path
  wouldn't fire for output tools and the chunk would disappear unless we
  also restored a `handle_tool_call_end` override in the Vercel adapter.

## Verification

```
uv run pytest
# all targeted suites pass

uv run pyright \
  pydantic_ai_slim/pydantic_ai/messages.py \
  pydantic_ai_slim/pydantic_ai/exceptions.py \
  pydantic_ai_slim/pydantic_ai/_agent_graph.py \
  pydantic_ai_slim/pydantic_ai/ui/_event_stream.py \
  pydantic_ai_slim/pydantic_ai/ui/vercel_ai/_event_stream.py
# 0 errors, 0 warnings, 0 information
```
