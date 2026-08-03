### [25] CONFIRMED — Boolean `anyOf` members crashed schema conversion

`GoogleOpenAPISchemaTransformer` preserves `False` in `{'anyOf': [False, {'type': 'string'}]}`. The subsequent recursive call passed that boolean to `_drop_unsupported_keywords`, producing `AttributeError: 'bool' object has no attribute 'items'`. Boolean members are now reduced according to JSON Schema semantics: `False` members are omitted and `True` members become the unconstrained schema `{}`. Regression: `test_schema_drops_false_any_of_member`.

### [14] CONFIRMED — Provider-specific disabled VAD created an unusable session

`AutomaticVAD(disabled=True)` serialized as `automaticActivityDetection.disabled=true`. The SDK requires explicit activity markers in that mode, but `GoogleRealtimeConnection.send()` exposes neither those markers nor equivalent manual turn controls. `_realtime_input_config` now rejects this setting with `UserError`, matching `turn_detection=False`. Regression: `test_google_vad_disabled_is_rejected`.

### [19] CONFIRMED — Receive-side `OSError` bypassed transport handling

A session whose `receive()` raised `ConnectionResetError('connection reset')` propagated the raw exception because `__aiter__` caught only `ConnectionClosed` and `APIError`, despite `transport_errors` including `OSError`. The receive loop now catches `self.transport_errors`, so the existing reconnect/fatal-event policy applies. Regression: `test_iter_ends_on_oserror`.

### [23] CONFIRMED — Cancelled tool calls remained retained

After mapping calls `c1`, `c2`, and `active`, a cancellation for `c1` and `c2` emitted `ToolCallCancelled` but retained all three entries in `_tool_calls`. Cancellation handling now removes the cancelled IDs while preserving unrelated active calls. Regression: `test_map_tool_call_cancellation`.

Verification: all four targeted regressions pass; Ruff check and format pass; `google.py` passes Pyright; `git diff --check` passes. Pyright on the touched test module still reports its pre-existing partially-unknown `gateway_provider` import at `tests/realtime/test_google.py:78`.
