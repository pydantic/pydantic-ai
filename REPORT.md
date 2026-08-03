### [8]+[29] CONFIRMED — xAI model names could inject or truncate query parameters

`model='voice&conversation_id=stolen#fragment'` produced an unescaped WebSocket URL, so the model value ended at `voice` and supplied a second query parameter. `connect()` now applies `quote(..., safe='')`, matching the resume ID handling. Regression test: `test_connect_url_encodes_model_name`.

### [15] CONFIRMED — Entra-only Azure clients exposed the SDK API-key sentinel

With the installed OpenAI SDK, `azure_ad_token` and `azure_ad_token_provider` clients expose the truthy string sentinel `API_KEY_SENTINEL` as `client.api_key`, so realtime would send `<missing API key>`. `AzureProvider` now converts that sentinel to `None`, making `api_key` raise the existing clear `UserError`. The callable API-key subcase is not currently vulnerable: the SDK stores it as `''`, which the existing `or None` already rejects. Regression test: `test_azure_provider_realtime_rejects_entra_auth` covers all three SDK states.

### [16] CONFIRMED — Azure realtime model arguments are deployment names

Both examples implied that `gpt-realtime` must be the model identifier. The guide now states that Azure resolves a user-chosen deployment name and uses `my-realtime-deployment` in both construction forms. Tests: `test_docs_examples[docs/realtime/azure.md:19]` and `test_docs_examples[docs/realtime/azure.md:38]`.

### [17] CONFIRMED — the overview interrupted ordinary user turns

`RealtimeSession.interrupt()` is not a no-op when no response is playing: it always sends `CancelResponse`. The event-loop example now flushes playback and calls `interrupt(played_ms=...)` only when buffered output existed, consistent with the detailed barge-in section. Test: `test_docs_examples[docs/realtime/index.md:105]`.

### [9] CONFIRMED — `PlaybackBuffer` counted carry bytes it could not evict

With `max_bytes=6`, adding `abcdef`, filling two bytes, then adding `ghij` retained eight bytes (`cdefghij`); an oversized single chunk also exceeded the bound. `add()` now evicts the oldest carry bytes first and retains only the newest bounded suffix of an oversized chunk. Regression tests: `test_playback_buffer_evicts_carry_before_adding_audio` and `test_playback_buffer_truncates_oversized_chunk`.

### [5] CONFIRMED — camera defaults could terminate the inline script

With `VOICE='</script><script>alert(1)</script>'`, the raw JSON contained a literal closing script tag. The embedded JSON now escapes `<` and `>` as JSON Unicode escapes. Regression test: `test_camera_defaults_are_safe_to_embed_in_script`.

### [7] CONFIRMED — matching attacker-controlled Origin and Host values passed

`Origin: http://attacker.example:8000` with `Host: attacker.example:8000` passed the old equality check. The check now additionally requires the parsed hostname to be `localhost`, `127.0.0.1`, or `::1`. Regression test: `test_camera_websocket_origin_requires_loopback_host`.

### [22] NEEDS-DECISION — the sandbox blocks top navigation but not iframe self-navigation

The iframe already uses an empty `sandbox`, so scripts, popups, and top-level navigation are blocked; an injected link can still navigate the iframe itself. No sandbox or `allow` flag disables self-navigation, and CSP `default-src 'none'` does not govern navigations. Options are to disable all pointer interaction (small but changes diagram UX and does not robustly cover keyboard activation) or sanitize navigation-capable markup/attributes (robust but materially expands this example). Recommendation: leave this PR unchanged and decide whether generated drawings must be interactive before choosing sanitization versus a non-interactive overlay. No test added because code was unchanged.
