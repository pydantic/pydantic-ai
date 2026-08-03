### [18] NEEDS-DECISION — Realtime tool contexts retain the opening history snapshot

Confirmed state: with a new session whose connection records a user turn and then emits a tool call, the tool's `RunContext.messages` is still `[]`; with seeded history it contains only that seed. `_open_realtime_session` creates one list before constructing `RealtimeSession`, while the session copies the seed into `_seeded` and records live messages in its private `_history`. `all_messages()` also returns a copy, so the agent cannot provide a live list from the assigned function.

The narrow fix belongs in prohibited `realtime/_session.py`: immediately before each `ToolManager.handle_call` in `_execute_tool`, synchronize the manager context's existing message list in place from `self.all_messages()`. This matches `_agent_graph`, which builds each request/tool context from the current graph history. Recommendation: make that change and add a tool-call regression test asserting the context contains the finalized user turn and calling assistant response. No code or test changed for this finding.

### [21] NEEDS-DECISION — Realtime skips model-request authorization hooks

Inventory: realtime resolves `for_agent`/`for_run`, contributed instructions/toolsets/native tools and wrapper toolsets, runs `prepare_tools`, and runs the complete tool-validation (`before`/`wrap`/`after`/`on_error`) and tool-execution (`before`/`wrap`/`after`/`on_error`) hook stacks plus `handle_deferred_tool_calls`. It does not run run/node/event-stream hooks, any model-request hook (`before_model_request`, `wrap_model_request`, `after_model_request`, `on_model_request_error`), output validation/processing hooks, or `prepare_output_tools`; capability model settings are explicitly ignored, dynamic native-tool functions are not resolved, and deferred loading is disabled.

Concrete bypass: a capability can remove a privileged function tool from `ModelRequestContext.model_request_parameters` in `before_model_request` during normal runs, but realtime connects using definitions produced before any such context or hook exists, so that tool remains advertised and executable. Applying `before_model_request` once would also expose message/settings rewriting with unclear one-shot semantics; rejecting every capability that overrides model-request hooks is safer but backward-incompatible. Recommendation: define a realtime-specific connect/authorization hook, or until then fail clearly for capabilities with unsupported model-request hooks. No code or test changed pending that product decision.

### [2] CONFIRMED — Optionality leaked between unsupported native tools sharing an ID

Concrete input: `ModelRequestParameters(native_tools=[WebSearchTool(optional=True), WebSearchTool()])` with no supported native types or fallback returned without error because `optional_ids == {'web_search'}` exempted both instances. Conflicting duplicates within one capability layer are rejected earlier, but identical IDs are otherwise reachable: cross-layer overrides are intentionally last-wins, and realtime calls `resolve_native_tool_swap` before the classic model path's ID deduplication.

Changed the missing-fallback check to evaluate `optional` per unsupported instance rather than per `unique_id`. Regression test: `test_native_tool_swap_does_not_apply_optional_by_unique_id`.

### [24] CONFIRMED — A WAV header can declare more PCM frames than its body contains

Concrete input: a valid mono PCM16 WAV declaring one frame, truncated immediately after its header. `wave.readframes(1)` returns `b''` without raising, so the previous code accepted and returned empty PCM.

Changed `seed_pcm_audio` to require the returned byte count to equal declared frames × channels × sample width, using the existing invalid-WAV error path on mismatch. Regression test: `test_seed_pcm_audio_rejects_truncated_wav`.

Verification: both regression tests passed with `--inline-snapshot=disable`; Ruff check/format and Pyright passed for all touched implementation and test files.
