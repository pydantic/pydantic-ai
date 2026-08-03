### [1] CONFIRMED — reconnect failed on media-bearing tool returns

`ToolReturnPart(content=['done', BinaryContent(...image/png...)])` passed its generated image `user_content` into seeding with image support disabled, raising `UserError`. `_without_media` now replaces media-bearing tool content with its text-only model response. Covered by `test_replay_items_strips_media_and_keeps_tagged_text`.

### [3] CONFIRMED — reconnect dropped tagged user text

`UserPromptPart(content=[TextContent('earlier question')])` produced no replay item because only plain `str` values survived. `_without_media` now retains `TextContent`. Covered by `test_replay_items_strips_media_and_keeps_tagged_text`.

### [31] CONFIRMED — reconnect failed on response file parts

`ModelResponse(parts=[FilePart(...)])` reached `_seed_response_items` and raised `UserError`. `_without_media` now drops response `FilePart`s. Covered by `test_replay_items_strips_media_and_keeps_tagged_text`.

### [12] CONFIRMED — malformed completion finalized a turn

`{'type': 'response.done'}` mapped to `ResponseCompleteEvent(provider_details={'status': None})`. It now maps to a recoverable `SessionErrorEvent` and does not finalize the turn. Covered by `test_map_response_done_without_response_object` and `test_response_done_without_response_object_is_recoverable`.

### [20] CONFIRMED — malformed completion left response state wedged

With `_response_active=True` and `_pending_response=True`, a `response.done` without `response` returned before clearing state, so the pending request was never sent. The malformed terminal frame now clears tracked response state and releases the deferred `response.create`, while surfacing the recoverable error from [12]. Covered by `test_response_done_without_response_object_is_recoverable`.

### [10] CONFIRMED — invalid base64 audio decoded silently

Audio delta `"!!!!"` decoded to `b''` under permissive base64 decoding. Audio deltas now use `base64.b64decode(..., validate=True)`, routing the decoding error through the existing recoverable-frame path. Covered by `test_connection_iter_recovers_from_malformed_frame`.

### [11] CONFIRMED — reconnect reused stale authentication headers

An async API-key provider returning `sk-initial` then `sk-refreshed` was resolved only once, so both handshakes used `sk-initial`. Authentication and trace-context injection now run inside each `dial()`. Covered by `test_reconnect_refreshes_async_api_key`.

### [13] CONFIRMED — positive fractional transcription duration was discarded

`seconds=0.5` rounded to zero and returned no usage. Positive durations now have a minimum reported integer value of one second. Covered by `test_map_transcription_usage`.

### [28] CONFIRMED — infinite transcription duration escaped recoverable parsing

`usage={'type': 'duration', 'seconds': Infinity}` passed shape validation, then `round(inf)` raised `OverflowError` outside the recoverable `ValueError` path. Duration validation now rejects non-finite numbers with `ValueError`. Covered by `test_connection_iter_recovers_from_malformed_frame`.

Scoped verification: `177 passed`; Ruff passed for all touched code/test files; Pyright reported zero errors for all touched code/test files.
