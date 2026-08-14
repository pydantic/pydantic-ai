# pydantic_ai_slim/pydantic_ai/realtime/ Guidelines

Start with the architecture walkthrough in [docs/api/realtime.md](../../../docs/api/realtime.md)
and the user-facing guides under [docs/realtime/](../../../docs/realtime/): a
`RealtimeModel.connect()` opens a provider-specific `RealtimeConnection` (the *codec*, vocabulary in
`codec.py`), and `RealtimeSession` (`_session.py`) wraps it — translating codec events into the
shared message/part vocabulary from `pydantic_ai.messages`, building ordinary `ModelMessage`
history, and running the tool loop. The provider-agnostic layout mirrors the request-response side:
`model.py` holds the `RealtimeModel` ABC and `infer_realtime_model`, `settings.py` the settings
vocabulary, `profiles.py` the `RealtimeModelProfile` type (per-provider tables live in
`pydantic_ai/profiles/{google,openai,grok}.py` as `*_realtime_model_profile` helpers), and
`codec.py` the connection layer; concrete providers live in `openai.py` / `azure.py` / `google.py` /
`xai.py`. The [models/ guidelines](../models/AGENTS.md) apply in spirit throughout.

## Policy lives in the shared core, never in the session

- The session executes tools through the same `ToolManager` core as the agent graph
  (`validate_tool_call` / `execute_tool_call` / `build_tool_return_part`), so hooks, retries, and
  usage behave identically by construction. Any *policy* the graph applies before dispatch
  (declarative tool kinds: `'unapproved'` → `ApprovalRequired`, `'external'` → `CallDeferred`) is
  enforced inside `ToolManager.handle_call` — do not reimplement or filter above it. A reimplemented
  policy layer that drifts is a security bug, not a style problem: the realtime approval-bypass was
  exactly this.
- When behavior differs from a standard run (no graph nodes, no `before_model_request`, deferred
  results resolved inline), the difference must be deliberate, documented in
  [docs/realtime/](../../../docs/realtime/), and covered by a parity test in `tests/realtime/`.

## Provider adapters

- Parse provider frames through typed SDK models, never ad-hoc `dict` access. OpenAI-protocol event
  types come from the OpenAI SDK; Gemini's from `google-genai`.
- Azure and xAI reuse the OpenAI codec (`_openai_protocol.py`). Fix protocol bugs there so all
  three benefit; put genuinely provider-specific divergence in the provider's own module.
- Capability differences go through `RealtimeModelProfile` flags (resolved defaults → provider →
  user `profile=`), never `isinstance`/provider-name checks at use sites. A change that branches on
  a profile flag needs a test pinning each side of the flag.
- One event means one thing on every provider. If providers disagree about what a frame implies
  (speech start, turn end), normalize in the codec — the session must not branch per provider.
  `RealtimeTurnCompleteEvent` is synthesized by the session, never read off the wire.
- Wrap connect/handshake failures in `ModelAPIError`/`ModelHTTPError`; recoverable in-session
  provider errors become `RealtimeSessionErrorEvent` and leave the session usable.

## History fidelity

Session history must always be a valid input for `Agent.run(message_history=...)`: settle
everything on close/reconnect-loss (partial replies recorded as interrupted responses, running
tools given cancelled returns) so history never ends on a dangling `ToolCallPart`.

## Testing

- Realtime WebSocket cassettes live in `tests/realtime/cassettes/` (raw frames, secrets scrubbed,
  audio truncated); record with `uv run --env-file .env pytest --record-mode=rewrite <test>`. The
  cross-provider matrix test is the parity net — extend it when adding provider behavior.
- Docs examples run against the scripted `MockRealtimeConnection` in `tests/test_examples.py`; an
  agent defining a `check_availability` tool triggers the scripted conversation reserved for the
  quickstart, so don't use that tool name elsewhere.
