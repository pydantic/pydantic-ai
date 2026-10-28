<!-- braindump: rules extracted from PR review patterns -->

# pydantic_ai_slim/pydantic_ai/models/ Guidelines

Rendering history into a request must be deterministic and must keep every message in its trust channel: see [Message History & Prompt Caching](../../../agent_docs/message-history-and-caching.md), "Canonicalize on send" and "Trust tiers map to channels".

## API Design

<!-- rule:912 -->
- Silently ignore unsupported generic tuning settings (`temperature`, sampling params, penalties, …) at runtime and document them in docstrings — a model that simply no-ops an unsupported knob keeps client code portable across models; failing noisily would break that portability (provider-namespaced settings like `google_*`/`openai_*` are governed by a separate rule below, not this one). Declare every generic setting a model class ignores in its `unsupported_settings` (all but `request_timeout`, `thinking`, `cache` and `tool_choice`, which are resolved outside the adapter), never with an ad hoc warning: `Model.prepare_request` warns with `UnsupportedSettingWarning` for the ones that change what a request guarantees (`timeout`, `max_retries`, `stop_sequences`, `parallel_tool_calls=False`, `extra_headers`, `extra_body`), and stays silent for the tuning knobs
<!-- rule:81 -->
- Apply identical response processing to both `request()` and `request_stream()` — if `request()` calls `_process_response()`, `request_stream()` must apply it to each chunk — Ensures streaming and non-streaming code paths support the same message types (`ToolCallPart`, `NativeToolCallPart`, `TextPart`, etc.) with consistent behavior, preventing bugs where features work in one mode but fail in the other
<!-- rule:598 -->
- Expose provider-specific data via `ModelResponse.provider_details` or `TextPart.provider_details` — prevents API bloat and maintains consistent provider integration patterns — Keeps the core response interface clean while allowing providers to expose logprobs, safety filters, content filtering, and usage metrics without breaking consistency across integrations
<!-- rule:26 -->
- Don't add preemptive client-side guards that reject provider-namespaced settings (`google_*`, `openai_*`, …) based on assumed capability limits; forward the setting the user opted into and let the provider API surface the actual incompatibility — the API is the authority on what it currently supports, so a client-side guard degrades functionality on outdated assumptions
<!-- rule:478 -->
- Token counting must mirror actual request parameters (`tools`, `system_prompt`, configs) and use identical message formatting — Ensures token count estimates match actual API usage, preventing billing surprises and quota errors
- Per-request injections or mutations of request content (message blocks, tool defs, instructions, cache breakpoints) must land on a set of positions chosen by message identity, not by history length (e.g. every user message). Never anchor to a position *defined as* the tail — "the last message", or any length-based index — because that position moves every turn, which shifts the cacheable prefix, so the provider silently re-processes the tail instead of reading from cache, a cost/latency regression that surfaces no error. Covering the last message is correct when it falls out of an identity rule that covers the others too; what is forbidden is making the tail itself the anchor. Stability is necessary but not sufficient: pinning to the first user message alone is stable and still wrong when the wire needs the injection later in the conversation, which is how `container_upload` blocks stopped reaching a fresh container (https://github.com/pydantic/pydantic-ai/issues/7775). Cover every position the API acts on *and will accept the injection at*, then check each one is identity-anchored. Those are different sets: Anthropic acts on a `container_upload` in a user message holding only `tool_result` blocks, and rejects the request for it, so that position is excluded on purpose

## Error Handling

<!-- rule:562 -->
- Raise explicit errors for unsupported model features (e.g. function tools, JSON/native output modes) that can't be formed for a given model — never silently skip or degrade — makes capability limits discoverable at runtime; unsupported settings are governed by the settings rules above, and unrepresentable content/message-part types by the rule below
<!-- rule:65 -->
- Use exhaustive pattern matching for message part/content types in model adapters; raise explicit errors for unsupported types instead of filtering or assertions — Prevents silent data loss during message mapping and provides clear feedback when model APIs don't support certain content types (e.g., `FileContent`), making integration failures debuggable rather than mysterious
<!-- rule:433 -->
- Return `ModelResponse` with empty `parts=[]` but populated metadata (`finish_reason`, `timestamp`, `provider_response_id`) for recoverable API failures (content filters, empty content) — enables graceful degradation instead of cascading errors — Allows the system to handle provider-level failures gracefully by preserving response metadata for observability while signaling no usable content, preventing unnecessary exception propagation in model adapters

## Type System

<!-- rule:73 -->
- Use typed settings classes (e.g., `OpenAISettings`, `AnthropicSettings`) with provider-prefixed fields instead of `extra_body` or dict literals — Enables type checking and autocomplete for provider-specific config, preventing runtime errors from typos or invalid values
<!-- rule:972 -->
- Define Pydantic models to validate API responses — avoids `.get()` fragility and catches schema changes early — Prevents runtime errors from missing/malformed fields and provides type safety when parsing external API data

## General

<!-- rule:9 -->
- Place provider-specific code in the provider's own module (`models/{provider}.py`, or the `models/{provider}/` package), not shared modules — add functions consistently across all providers even if some are simple — Maintains clear architectural boundaries and prevents shared compatibility layers from accumulating provider-specific logic that becomes hard to maintain
- Turn a provider module into a `models/{provider}/` package once its module-level helpers form concerns that a reader of the `Model` class can skip. `models/anthropic/` is the reference layout:
  - `__init__.py` defines the `Model` subclass and exposes the public API. Import each public API name a submodule defines (`AnthropicStreamedResponse`, `AnthropicStaleThinkingBlockWarning`) into `__init__.py` and list it in `__all__`, along with every name the module exported before it became a package (the `*_content_ta` type adapters): the API docs render an imported name only when `__all__` lists it. Never list any other submodule helper there.
  - Each other concern gets its own `_*.py` submodule. A function in a `_*.py` submodule has no leading underscore: the module name already marks it private.
  - The package's own `AGENTS.md` holds the rules for that provider alone. List it in the root `AGENTS.md`.

A model should read reveal modes through `self.tool_deferral_mode` and
`self.tool_addition_mode`, never directly from the corresponding profile keys. An adapter that
implements a reveal renderer must declare its supported values in
`supported_tool_deferral_modes` and `supported_tool_addition_modes`; the inherited empty sets are
the safe default for adapters with no renderer.

An adapter that honors `CompactionPart`s on the wire declares
`compaction_requires_encrypted_content` and `compaction_retains_standing_prompt`, sets
`_honors_compaction_boundaries = True`, and calls `self._trim_before_compaction()` from its own message-prep
step — never `_trim_messages_before_compaction` directly, and never restating what those declarations imply.
Each of the two `compaction_*` declarations states only what the API does
(does it need the encrypted blob to honor an item; does the item keep serving the window's system
items); turning that into trim behavior belongs to the one helper. The two are independent — today's
two adapters happen to answer both the same way, so don't infer one from the other when adding a
third. The flag and the call go together: the flag is what lets tool dispatch judge a compaction boundary with
the same `accepts_stateless_artifacts` the trim used. Without it, tool dispatch falls back to matching the
response's provider name, which can disagree with what the request kept. The declarations belong on the
adapter, not the profile: eight providers route a profile of their own through `OpenAIResponsesModel`, so a profile key would be absent exactly where the wire format is
most certain. Where in a request build the trim belongs stays adapter mechanics (OpenAI Responses
resolves server-side state from the *untrimmed* history, so it keeps a separate trimmed view).

An adapter whose message mapping refuses a kind of file (image, document, audio, video) declares it
in `_unsupported_file_inputs`, and one that sends only some document media types declares those in
`_supported_document_media_types`. `Model.profile` narrows the file-input flags to them after every
other layer, which is what lets `Model.prepare_messages` replace a tool's file the adapter would refuse
with a note instead of failing the run. A refusal that depends on the model rather than the API belongs
in the profile instead (`supports_video_input` in `google_model_profile`).

### Third-party model fallback

Custom `Model` subclasses that continue reading `tool_defs` degrade gracefully: all tools are fully
declared and availability deltas fall back to a system-text announcement. Deferral is not withheld
on that model. Reading `declared_tool_defs` and `visibility_of()` opts the adapter into withholding.

<!-- /braindump -->
- When a model forwards a generic `ModelSettings` field, add it to that field's `Supported by:` list in `pydantic_ai/settings.py` and leave it out of the class's `unsupported_settings`, and give a new `Model` class a case in `tests/models/test_model_settings_support.py` — that test probes each class's outgoing request and fails when a list, the declaration and the wire disagree.
- Narrow a capability by client class, never by the client's `base_url`. Where the gateway serves a model it must behave exactly as the provider's canonical API does, and the Pydantic AI Gateway — like an ordinary corporate proxy — reaches that API through the provider's normal SDK client carrying a proxy base URL. A host test therefore splits those callers off from the transport they actually reach and silently degrades them, which is why no capability here is decided that way. Genuinely separate transports (`AsyncAnthropicBedrock`, `AsyncAnthropicVertex`, `AsyncAnthropicFoundry`, …) are distinct client classes and earn their own gates. Gate on the transport (the client class, or the client's own mode such as `genai.Client.vertexai`) only in the model's `_instance_profile_defaults()` or `profile`: turn the result into a profile flag there. Where the capability is used, read only a profile flag or the adapter's own declaration. A branch that sends one feature in each transport's own wire form is mapping, not a gate. Probe the gateway leg (`Model('<id>', provider='gateway')`) rather than reasoning about it; a model the gateway genuinely does not serve belongs in `UNSUPPORTED_GATEWAY_MODEL_NAMES`, not in a carve-out that leaves the id advertised and degraded.
- The one place a base URL is read as identity is `_canonical_system.py::stateless_artifact_system`, for history recorded before responses carried `canonical_system`: a `google` or `google-cloud` response whose `provider_url` is the *other* first-party Google transport's host gets that transport's canonical system, so its thought signatures aren't replayed where they're rejected. It only tells the two Google transports apart, it reads only their first-party hosts, and any other URL, a gateway's or a proxy's included, keeps the name-based behavior. Don't add another: new history carries `canonical_system`.

## Adapter structure

Each rule's precedent, and the rules for tool-call IDs, replay data, thinking, empty content, tool-return media and cache markers, are in [Provider Adapters](../../../agent_docs/provider-adapters.md). Read it before you add a part type, a native tool kind or a capability gate to an adapter.

- Map each part type in its own method, called from one exhaustive dispatch per message kind. An arm of one statement stays inline. A new part type adds one arm, never a branch nested in the mapping loop.
- Keep request-wide mapping state in a private dataclass that holds data only, unless a subclass overrides its methods.
- Map each response wire item in one function, called from both `_process_response` and the stream.
- Lower a part type most APIs can't render in `Model.prepare_messages`. An adapter that doesn't render it raises `_unprepared_part_error(part)` in one arm.
- Map native tool parts through one table per adapter, from `kind` to wire name.

## `TestModel`

Changes to the tool-call arguments and text `TestModel` generates are test scaffolding, not application behavior, so the "be scoped to the problem you are solving" and "leave behavior unchanged" requirements in the root `AGENTS.md` don't apply to them. A PR may change these values beyond the schema it was opened for, and may fold in fixes for other schemas `TestModel` crashes on or generates invalid data for, each fix with a test that fails on `main`. Arguments that are valid for their schema on `main` must stay valid, and a run must not newly crash: either is a regression. A changed value in a run that succeeds on `main` is a compatibility impact under [agent_docs/api-design.md](../../../agent_docs/api-design.md): add the `compatibility impact` label and a `[!WARNING]` section whose migration is updating snapshots, or pinning values with `custom_output_text`, `custom_output_args` or `FunctionModel`. Every other requirement still applies, and the exemption does not extend to `TestModel`'s parameters, its tool-call ids, how it estimates usage (usage that moves because the arguments or text changed is covered), its tool-calling flow (which tools it calls, on which request, and how it answers retry prompts), or `FunctionModel`.
