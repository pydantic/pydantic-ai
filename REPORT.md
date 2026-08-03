### [26] CONFIRMED — validation exceptions could leave realtime dispatch waiting forever

`ToolManager.handle_call()` notified `on_validate(False)` only for `UnexpectedModelBehavior`. A `RuntimeError` from `validate_tool_call()` propagated before the callback, while realtime dispatch was blocked on `validation_done.wait()`. Changed the validation guard to catch `BaseException`, notify failure, and re-raise, including cancellation. Regression test: `test_validation_hook_exception_reports_failed_validation`.

### [27] CONFIRMED — lost-state reconnects left obsolete tool tasks live

With a tool blocked after validation and `SessionReconnectEvent(state_restored=False)`, `_handle_reconnected()` finalized the old response but left its task in `_pending_tool_calls`; releasing it would send the obsolete call ID on the fresh connection. The reconnect path now removes and cancels each pending task and records an interrupted return, matching provider-driven tool cancellation. Regression test: `test_reconnect_cancels_obsolete_tool_call`.

### [32] CONFIRMED — one finalized item cleared the active state of overlapping user items

After partial transcripts opened item IDs `u1` and `u2`, finalizing `u1` set `_user_turn_active=False` while `u2` remained in `_active_users_by_id`. `_finalize_user()` now derives the flag from remaining identified, id-less, or not-yet-identified active turns. Regression test: `test_finalizing_overlapping_user_item_keeps_turn_active`.

### [30] CONFIRMED — reverse completion produced reverse tool-return history

For a response containing calls `A, B`, inserting `B` first and then `A` produced `response, result_B, result_A`: the old loop always skipped every existing return. Insertion now compares each return's call position in the response, matching the normal graph's index-sorted `output_parts`. Regression test: `test_parallel_tool_returns_are_inserted_in_call_order`.

### [6] CONFIRMED — concurrent image and text sends could disagree with wire order

Concurrent sends are supported and serialized by `_send_lock`, but an image blocked inside `connection.send()` was recorded only after the await; a concurrent text send recorded immediately, producing wire order `image, text` and history order `text, image`. Image history is now reserved before sending and rolled back on failure, matching text handling. Regression test: `test_concurrent_image_and_text_history_matches_wire_order`.

### [4] CONFIRMED — realtime incorrectly gated tool definitions with serialized parameters

`InstrumentationSettings.include_model_request_parameters` documents that only `model_request_parameters` is optional and that `gen_ai.tool.definitions` is always emitted; the classic instrumentation path builds tool definitions outside that gate. Realtime now does the same while continuing to gate the serialized parameter/settings blobs. Regression test: `test_request_config_respects_include_model_request_parameters`.
