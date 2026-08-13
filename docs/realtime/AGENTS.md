# docs/realtime/ Guidelines

All of [docs/AGENTS.md](../AGENTS.md) applies. Realtime-specific rules, distilled from maintainer
review; the source-tree counterpart is
[pydantic_ai/realtime/AGENTS.md](../../pydantic_ai_slim/pydantic_ai/realtime/AGENTS.md).

## Page charters

Each page owns its concept; a rule is stated once on its owning page and linked from everywhere
else: `overview.md` (front door, provider matrix, limitations table — every limitation row links a
tracking issue), `audio.md` (media I/O: audio, images, transcripts), `events.md` (event vocabulary
and its overlap with standard run events; the turn-boundary rule lives here), `turns.md`,
`tools.md` (tools only), `capabilities.md` (per-hook support story), `history.md`, `deployment.md`
(frontend transports), `lifecycle.md` (connection lifecycle only), `observability.md`,
`troubleshooting.md` (the symptom-first index — per-page "Edge cases" must not duplicate it), and
the four provider pages (canonical for installs, model names, settings, quirks).

## House rules for these pages

- Cross-link relentlessly to the non-realtime docs (tools, message history, capabilities, model
  settings, profiles, multimodal input, Logfire, deferred tools); where behavior matches a standard
  run, say so and link instead of re-explaining. The word "gateway" links
  [gateway.md](../gateway.md) on first use per page. The embeddings docs are the register/pattern
  reference.
- Examples use the string model form (`agent.realtime('openai:gpt-realtime')`,
  `'gateway/openai:gpt-realtime'`); import a model class only when demonstrating model-level
  configuration. Tools are `async def`. Install blocks use the `pip/uv-add` macro with the
  per-provider extras (`openai-realtime`, `google-realtime`, `xai-realtime`). Complete examples get
  the standard runnable banner; rely on `async with` exit to close the session rather than an
  explicit `close()` unless the example is about `close()`.
- Docs examples execute in `tests/test_examples.py` against a scripted connection: the default
  script speaks one assistant turn ('Hello from the realtime assistant.'); an agent defining a
  `check_availability` tool triggers the quickstart's scripted reservation conversation — never use
  that tool name elsewhere.
- Browser WebRTC ships in Pydantic AI: `docs/realtime/deployment.md#browser-webrtc-server-sideband` is the canonical
  owner of the topology, and don't present third-party media platforms as the WebRTC story. Azure
  Voice Live is still an in-flight PR (#6642): mention it as coming to Pydantic AI where relevant.
