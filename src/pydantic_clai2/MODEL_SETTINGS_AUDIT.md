# Model settings parity audit

Reference: the local Code Puppy checkout at `a862bf47`.
This is a partial implementation, not a claim of full runtime parity.

## Source comparison

- [Menu and navigation](https://github.com/mpfaffenberger/code_puppy/blob/a862bf47/code_puppy/command_line/model_settings_menu.py):
  `build_models_menu`, `run_model_settings_flow`, `run_settings_flow`.
- [Fields, defaults, and bounds](https://github.com/mpfaffenberger/code_puppy/blob/a862bf47/code_puppy/command_line/model_settings_defs.py).
- [Model support filtering](https://github.com/mpfaffenberger/code_puppy/blob/a862bf47/code_puppy/config.py):
  `model_supports_setting`.
- [Request translation](https://github.com/mpfaffenberger/code_puppy/blob/a862bf47/code_puppy/model_factory.py):
  `make_model_settings`.
- [Retry policy](https://github.com/mpfaffenberger/code_puppy/blob/a862bf47/code_puppy/agents/retry_profiles.py).
- [Streaming recovery](https://github.com/mpfaffenberger/code_puppy/blob/a862bf47/code_puppy/agents/_runtime.py):
  `streaming_retry`, `should_retry_streaming`.

## Implemented here

| Behavior | Result |
| --- | --- |
| Command without arguments | Searchable model picker, not a direct editor of the globally active model |
| Configure another model | No model switch; returning to the picker preserves the last edited selection |
| Empty list | Disabled instruction inside the menu |
| Preview | Model settings summary and field descriptions/current/default/choices |
| Settings list | Human-readable names, lowercase `r` reset, Esc back, no search consuming reset keys |
| Fixed choices | Choice picker without the generic "type a value" and "keep current" rows |
| Repeated edits | Immediate persistence; selection retained; Esc in an editor discards its unsubmitted input |
| Relevant settings | Remove broad generic fields; hide sampling controls for reasoning GPT models; preserve saved fields for cleanup |
| GPT-6/GPT-5.6 defaults | Apply the requested seven defaults before saved overrides, independent of provider prefix; reset restores inheritance |
| Custom params | Existing add/edit/rename/delete flow retained; custom body values applied last |
| GLM controls | Native thinking type and clear-thinking body fields from GLM-4.5; reasoning effort from GLM-5.2 |
| Claude controls | Classic interleaved thinking, Fable 5.1 thinking display, and preserved-thinking prefix mismatch behavior |
| Shared-store compatibility | Ignore unknown saved keys, preserve them on edit/reset, keep new edits strict; invalid known values return to the prompt |

The shared `FieldMenu` still owns editing. `/set` retains searchable keys and
its custom-value choice row. The store holds only explicit model overrides.

## Remaining parity work

The full-runtime scope requested by the user is **not finished**:

- Code Puppy's main/sub-agent retries: global fallback, per-model overrides,
  exception classification, jittered backoff, progress reset, total-attempt cap,
  partial-output handling, logging, and sub-agent inheritance.
- GLM proxy-specific translation to `chat_template_kwargs`; the current controls
  emit GLM's native body shape. Gemini's separate thinking-enabled/level controls
  and Claude's separate clear-thinking control are still missing.
- Code Puppy's numeric ranges, rounding/step hints, initial numeric input, and
  display formatting. CLAI still uses its existing validated Pydantic bounds.
- Catalog output-token defaults (catalog value, then 15% of context clamped to
  2048-65536). CLAI still leaves the output cap unset without an override.
- Complete provider/alias capability metadata. CLAI currently uses core profiles
  for native providers and conservative fallback rows for other providers, not
  Code Puppy's `supported_settings` catalog. Persisted unsupported rows remain
  visible for reset and can block validation until removed.
- Protocol-specific equivalents for all seven requested GPT defaults on every
  provider. The defaults are resolved for every matching model identity, but
  an OpenAI-specific key does not make a non-OpenAI API support that feature.
- Custom parameter parsing differs intentionally today: CLAI also accepts JSON
  arrays/objects/null; Code Puppy parses scalar booleans/numbers/text.

## Core recovery prerequisite

Repository ownership rules prohibit implementing a second agent loop in CLAI.
Code Puppy's recovery retries a whole run from completed-step checkpoints.
A simple retry around core's `wrap_model_request` is not equivalent.

A local probe using the installed core, a `FunctionModel` that yields `partial`
and then raises `ConnectionError`, and `Hooks.on.model_request` attempting to
catch and retry that error produced:

```text
simulated mid-stream disconnect: requests=1, errors seen by wrap_model_request=0
```

Core's streamed `ModelRequestNode` passes the stream to its consumer and cancels
the wrapper task if consumption fails. The wrapper does not receive that failure
as a retryable request error. The source is the installed
`pydantic_ai/_agent_graph.py`, `ModelRequestNode.stream`, normal-stream cleanup.
Wrapping the entire run instead would need checkpoint recovery and careful tool
side-effect accounting, not just a sleep and another call to `Agent.run`.

Proposed core work: expose a supported recovery boundary for interrupted streams
that resumes from the last completed step, accounts for partial usage, and avoids
repeating completed tool execution. Define how emitted text is marked/replaced,
how cancellation bypasses recovery, and which hook observes recovery attempts.
Test failures before the first chunk, after text, after tool arguments, and after
a completed tool step. Verify cancellation, usage limits, and nested runs.

Then harness can own the reusable policy: per-role attempts (Code Puppy defaults
5 main / 9 sub-agent), three named strategies, equal jitter with a 1-30 second
bound, progress-aware budget reset and a 200-attempt backstop. CLAI should only
persist validated policy, bind the capability, and render recovery events.
Retry policy must not be sent in provider request settings. Do not expose the
four retry rows until both main and delegated runs consume their values.

## Verification

- The full CLAI suite passed: 980 tests, one skipped. The final provider-fallback
  regression suite also passed (62 tests).
- Repository Ruff check/format and strict Pyright on modified source/tests passed.
- Combined full-suite and final regression coverage reached 100% of CLAI
  statements and branches using its own coverage configuration.
- Fresh tmux sessions running `uv run clai2` with isolated settings verified the
  model picker, GPT defaults, choice edits and reset, editing a different Claude
  model without changing the active GPT model, numeric input, return navigation,
  and `/set` cancellation. No live model calls were made.
