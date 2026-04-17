# PR #4859 Review Handoff

This document is a prompt/context package for a new coding agent to continue work on PR `pydantic/pydantic-ai#4859` from the local branch `output-hooks` at commit `f424b92060cf16c2cec1e22a5624cbbbdff593c0`.

## Goal

Address the remaining review concerns for output hooks, with a bias toward making the behavior reliably match the intended contract rather than just weakening the docs.

The maintainer preference is:

- prefer making hooks work reliably over documenting that they do not

## What was already reviewed

I reviewed:

- the current PR diff on this branch against `origin/main`
- the latest PR review round and prior comments for context
- the repo instructions in `AGENTS.md`, `agent_docs/index.md`, `pydantic_ai_slim/pydantic_ai/AGENTS.md`, and `tests/AGENTS.md`
- the auto-review workflow prompt in `.github/workflows/bots.yml`

I also ran:

- `uv run pytest -q tests/test_output_hooks.py`
- `uv run pytest -q tests/test_output_hooks.py -k streaming`

Both passed on this branch at review time.

## Main findings

### 1. Streaming docs/docstrings overstate output-hook behavior

Current docs/docstrings claim output hooks fire on every partial chunk during streaming.

Relevant locations:

- `docs/hooks.md`
- `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py`

The problem is specifically with execute hooks, not validate hooks.

What the code currently does:

- `AgentStream.stream_output()` calls `validate_response_output(..., allow_partial=True)` for partial streamed responses in `pydantic_ai_slim/pydantic_ai/result.py:66-93`.
- For text output, `validate_response_output()` calls `run_output_with_hooks(... allow_partial=True, wrap_validation_errors=False)` in `pydantic_ai_slim/pydantic_ai/result.py:221-243`.
- In `run_output_validate_hooks()`, validation errors during partial validation are re-raised immediately instead of going through error recovery in `pydantic_ai_slim/pydantic_ai/_output.py:143-186`.
- `stream_output()` then swallows `ValidationError` and `ModelRetry` in `pydantic_ai_slim/pydantic_ai/result.py:80-83`.
- For output tools, `validate_response_output()` calls `handle_output_tool_call(... allow_partial=True, wrap_validation_errors=False)` in `pydantic_ai_slim/pydantic_ai/result.py:210-212`, which also requires partial validation to succeed before execute hooks can run.

Consequence:

- output validation hooks do run during partial validation attempts
- output execute hooks only run when a partial value successfully validates
- therefore the current wording "fires on every partial chunk" is too strong for execute hooks

### 2. The new tests do not yet match the repo's preferred trace-level style

Relevant guidance from `tests/AGENTS.md`:

- keep tests aligned with source modules where possible
- in agent/model/stream tests, assert final output and snapshot `result.all_messages()`

Current state of `tests/test_output_hooks.py`:

- the new file is feature-oriented rather than module-oriented
- most tests assert outputs/hook logs only
- there is only one ad hoc `all_messages()` inspection, not proper snapshots

This matters because this PR changes retry prompt generation and output-tool message flow. Output correctness alone does not validate those traces.

## Recommended direction

The maintainer explicitly prefers making hooks work reliably over documenting limitations.

So the preferred order is:

1. See whether execute hooks can be made to run more reliably for partial streamed output.
2. If full "every partial chunk" semantics are still not realistically achievable without inventing poor heuristics, then tighten the docs to describe the actual contract precisely.
3. In either case, upgrade tests to snapshot message history for the important integration cases.

## Suggested implementation tasks

### Task A: Investigate whether streamed partial execute hooks can be improved

Relevant files:

- `pydantic_ai_slim/pydantic_ai/result.py`
- `pydantic_ai_slim/pydantic_ai/_output.py`
- `pydantic_ai_slim/pydantic_ai/_tool_manager.py`
- `tests/test_streaming.py`
- `tests/test_output_hooks.py`

Key question:

- Can partial structured/tool output produce a stable enough intermediate validated value on each streamed chunk so execute hooks can actually fire "reliably" during streaming?

Useful existing behavior reference:

- `tests/test_streaming.py:299-318`
- `tests/test_streaming.py:3301-3319`

Those tests show the library already expects partial structured validation to emit progressive values when Pydantic partial parsing can recover them. That suggests a more accurate contract might be based on "each partial value that validates successfully" unless you can extend recovery beyond that.

Things to inspect closely:

- `run_output_with_hooks()` in `pydantic_ai_slim/pydantic_ai/_output.py:229-271`
- `run_output_validate_hooks()` in `pydantic_ai_slim/pydantic_ai/_output.py:143-186`
- `run_output_execute_hooks()` in `pydantic_ai_slim/pydantic_ai/_output.py:189-226`
- `validate_output_tool_call()` in `pydantic_ai_slim/pydantic_ai/_tool_manager.py:448-545`
- `execute_output_tool_call()` in `pydantic_ai_slim/pydantic_ai/_tool_manager.py:546-597`

Potential directions:

- preserve current validation-first design but make partial recovery produce more intermediate validated values
- if not possible, define the contract explicitly around successful partial validation rather than raw chunks

### Task B: Fix docs/docstrings to match the final implemented behavior

Relevant locations:

- `docs/hooks.md`
- `pydantic_ai_slim/pydantic_ai/capabilities/abstract.py`

If execute hooks still cannot be guaranteed on every raw partial chunk, update wording accordingly.

A precise version would be something like:

- validation hooks fire for partial validation attempts and the final result
- execute hooks fire for each partial value that validates successfully, and the final result

Do not leave the current broader claim in place if the runtime still does not support it.

### Task C: Upgrade tests to repo standard

Relevant locations:

- `tests/test_output_hooks.py`
- possibly move or split tests if you want to align better with `tests/AGENTS.md`

Minimum recommended test improvements:

- add `all_messages()` snapshots for representative text-output retry flows
- add `all_messages()` snapshots for representative output-tool flows
- add stronger streaming tests that assert both:
  - actual partial hook call sequences
  - streamed message history / final message history

A particularly useful new streaming test would:

- use `agent.run_stream(...)`
- iterate `stream.stream_output(debounce_by=None)`
- record validate-hook and execute-hook invocations with `ctx.partial_output`
- assert exactly which partial validated values reached execute hooks
- snapshot `stream.all_messages()`

## Specific current comments that were still relevant

These were the only comments I considered materially still open after re-checking current code:

1. Overstated streaming claims for output hooks
2. Missing `all_messages()` snapshots / weaker-than-repo-standard test style

Other recent bot comments mostly looked addressed already, or were non-blocking cleanup/preferences.

## Auto-review standards context

The review workflow in `.github/workflows/bots.yml` prioritizes:

- public API
- concepts and behavior
- documentation
- tests
- code style

It also explicitly says not to repeat old comments unless the issue still persists.

## Commands that were useful

Run these from the repo root:

```bash
uv run pytest -q tests/test_output_hooks.py
uv run pytest -q tests/test_output_hooks.py -k streaming
uv run pytest -q tests/test_streaming.py -k structured_response_iter
```

Potentially also:

```bash
uv run pytest -q tests/test_capabilities.py -k stream
```

## Important environment note

The dedicated `apply_patch` tool was unusable in this environment because sandboxed `bwrap` namespace creation failed. Regular escalated shell commands worked.

If the next agent needs to edit files, it may need to use an escalated shell write path instead of `apply_patch` unless the environment changes.
