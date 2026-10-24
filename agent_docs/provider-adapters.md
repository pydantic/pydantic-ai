# Provider Adapters

> Rules for the code in `models/` that maps Pydantic AI messages to a provider's wire format and back

**When to check**: Whenever you write or review a model adapter's request mapping (`_map_messages`, `_map_user_prompt`, a tool-return mapper), its response or stream parsing (`_process_response`, `_get_event_iterator`), a capability check in an adapter, or a profile flag an adapter reads

Every cross-provider feature reaches every adapter. When an adapter absorbs one as another branch inside its mapper, the mapper nests deeper with each feature, and each adapter solves the same concern its own way. These rules give each concern one shape, so a new feature adds one flat arm per adapter.

Apply them to the code you add. Aligning existing code that diverges is its own PR, never part of an unrelated change.

- A structural alignment, such as moving a mapper to the dispatch shape or onto a shared cache helper, keeps request bodies byte-identical. Prove it by diffing the request bodies captured on `main` and on the branch. Cassette replay doesn't compare bodies (`tests/conftest.py::RequestCapture`), so a green replay proves nothing.
- A behavioral alignment, such as dropping empty thinking or raising on an unknown native tool kind, changes what the adapter sends. Treat it as a fix under the root `AGENTS.md` scope rules, and call out each change it makes.

Paths are relative to `pydantic_ai_slim/pydantic_ai/`.

## Rules

### One method per part type, one dispatch per message kind

- Map each part type in its own method. Call those methods from one exhaustive dispatch per message kind: an `isinstance` chain or a `match` over the part union, ending in `assert_never`. Precedents: `models/anthropic.py::AnthropicModel._map_model_response` and `models/openai.py::OpenAIChatModel._MapModelResponseContext.map_assistant_message`. — A new part type then adds one flat arm and one method, not a branch nested inside the loop.
- A per-part method with one call site is an arm of the dispatch. Make it a method on the model class, never a `def` nested in the mapper. An arm of one statement stays inline in the dispatch. — A nested function counts toward the mapper's complexity, so it removes nothing.
- Keep the request-wide state those methods share in a private dataclass that holds data only. Build it once before the loop: the facts each part is mapped against (resolved flags, tool definitions by name) and the wire lists being built. The per-part methods live on the model class. Precedent: `models/anthropic.py::_MapMessagesContext`. — `_MapModelResponseContext` carries methods because `OpenRouterModel` and `SnowflakeModel` subclass it to override them; a context that nothing subclasses holds data only.
- Map each wire item of a response in one function, and call it from both `_process_response` and the stream. Precedent: `models/anthropic.py::_map_server_tool_use_block` and its sibling `_map_*_block` functions. — `request()` and `request_stream()` must build the same parts, and two copies of a mapping drift.
- In a stream, map each event type in its own sync method that returns an `Iterable` of events, and forward it from `_get_event_iterator` in one loop. Precedent: `models/openai.py::OpenAIStreamedResponse._map_part_delta`. — An async generator can't `yield from`, so every parts-manager call made directly in `_get_event_iterator` needs its own re-yield loop; a sync method can `yield from` each one.

### Lower a part type most APIs can't render, before the adapter

- When a new part type has no counterpart on most provider APIs, translate it in `Model.prepare_messages` into parts every adapter already maps. Key the pass on a profile flag or an adapter property. Precedents: `models/__init__.py::_convert_speech_parts` and `_project_instruction_state`. — One pass replaces a branch in every adapter.
- An adapter that renders the part natively turns the flag or property on and maps the part in its own method, as `Model._renders_retry_feedback` lets `DecisionModel` keep `RetryFeedbackPart`. Every other adapter keeps one arm for it: add the type to `models/__init__.py::_UnpreparedPart` and raise `_unprepared_part_error(part)`. — `Model.request` and `Model.count_tokens` don't run `prepare_messages`, so a direct caller can still hand the part to an adapter, and the shared error names the missing step.
- Render a lowered part type in its pass only, never again in an adapter or another module. Precedent: `models/__init__.py::_wrap_non_leading_system_prompts` renders the `<system>` fallback. — Two renderings of one part drift apart, and the request bytes drift with them.

### Gate a capability on a profile flag

- Where a capability is used, read only a profile flag or the adapter's own declaration, such as `_unsupported_file_inputs`. Never check the transport, the model name, the provider name or the `base_url` there. The transport is the client class, or the client's own mode, such as `genai.Client.vertexai` behind `GoogleModel._is_google_cloud`. — A use-site check keeps the fact out of the profile, so a user's `profile=` can't override it and every other use site needs its own copy.
- A gate decides whether to send a feature. A branch that sends the same feature in each transport's own wire form is mapping, not a gate, and may branch on the transport: Vertex takes the service tier as headers, the Gemini API as a request field.
- Set each flag where its fact lives:
  - a model-family fact in the family's `*_model_profile()` in `profiles/`;
  - an API fact in the provider's `Provider.model_profile()`;
  - a fact that depends on the SDK client class in the model's `_instance_profile_defaults()`, which the user's `profile=` still overrides;
  - a limit the client class can't get past, such as a native tool it can't send, in the model's `profile` property, which applies after the user's `profile=`.

  Precedents: `models/anthropic.py::AnthropicModel._instance_profile_defaults` and `AnthropicModel.profile`. — `Provider.model_profile()` is a static method that sees only the model name, so it can't hold a client-class fact. Those two model methods are the only places that see both.
- Express a difference between model families as a `Literal`-valued flag, never as a model-name prefix check in the adapter. Precedent: `bedrock_thinking_variant` in `providers/bedrock.py::BedrockModelProfile`.

### Tool-call IDs

- Store the provider's call ID in `tool_call_id` as the provider sent it. When the provider sends none, mint one with `_utils.generate_tool_call_id()`. Never encode another value into `tool_call_id`. — Other adapters replay the ID as-is, and [`pydantic_ai/AGENTS.md`](../pydantic_ai_slim/pydantic_ai/AGENTS.md) says to "Store provider-specific metadata in structured `provider_details` or `provider_metadata` fields, not in `id`, `content`, or `args`".
- Fit IDs to an API's format only by setting the profile's `tool_call_id_length`. `Model.prepare_messages` then rewrites the outgoing copy of a call and its result together, with a hash of the original ID (`models/__init__.py::_fit_tool_call_ids`). — A remapped ID must be a pure function of the original, or the cached prefix moves (see "Canonicalize on send" in [Message History & Prompt Caching](message-history-and-caching.md)).
- Recognize a minted ID by `_utils.TOOL_CALL_ID_PREFIX`, never by a string literal.

### Replay data

- Replay a part's provider data only when the part's `provider_name` is this adapter's `self.system`. When an adapter accepts more than one name, define that set once and read it at every replay site. Precedent: `models/google.py::GoogleModel._matching_provider_names`. — Each copy of an alias set drifts on its own.
- Put an opaque payload the API checks on replay (a signature, an encrypted blob) in the part's `signature` field when the part has one. Put any other replay data in `provider_details`.
- Read each replay field back through one function that validates it before use, with a Pydantic model or an explicit type check. Never read it through scattered, unvalidated `.get()` calls. Precedents: `models/_reasoning_details.py::into_reasoning_detail` and `models/anthropic.py::_get_anthropic_code_execution_tool_name`. — History comes back from storage and from UI clients, so one function has to catch a malformed value, by raising or by falling back to a safe default, not each place that reads it.
- Mark a variant of a part in `provider_details`, never with a sentinel `id`, for the same `pydantic_ai/AGENTS.md` rule.

### Thinking round-trips

- Send a `ThinkingPart` on a signed or encrypted reasoning channel only when this adapter owns it and it carries the signature or encrypted content the API checks.
- Render every `ThinkingPart` the adapter doesn't send on its native channel with `_thinking_part.py::render_replayed_thinking`, and place it where its second return value says. Never join `thinking_tags` by hand. Precedent: `models/anthropic.py::AnthropicModel._map_thinking_part`. — The helper owns the tags and `mimics_assistant_message_formatting`; a hand-written join ignores that flag.
- Send nothing for a `ThinkingPart` that has neither content nor replay data.

### Native tool parts

- Map native tool parts through one module-level table per adapter, from the native tool's `kind` to its wire name, and derive the reverse table from it. Precedent: `models/google.py::_TOOL_TYPE_TO_NATIVE_TOOL_NAME` and `_NATIVE_TOOL_NAME_TO_TOOL_TYPE`. — A branch per kind that only swaps a name adds a level of nesting for a lookup.
- Branch on the kind only where the wire shape differs.
- On replay, skip native tool parts from another provider. Raise `UnexpectedModelBehavior` for this adapter's own part when its kind is missing from the table. Precedent: `models/google.py::_native_tool_call_part_dict`. — An own part with an unknown kind is history this code didn't write, and a guessed wire shape sends a malformed request.
- Pair native calls with their results in one pass over the built wire messages, after mapping. Precedent: `models/anthropic.py::_drop_unpaired_native_tool_calls`. — Pairing decided inside each arm repeats the decision per kind.

### Empty content

- Filter empty content while building the wire message, never by removing the part from history. — An empty part can still carry an `id` or `provider_details` the API needs back.
- Omit a wire message left with nothing to send. Never send it empty, and never fill it with placeholder text. Precedent: `models/openai.py::OpenAIChatModel._MapModelResponseContext._into_message_param`.

### Media in tool returns

- Send a tool return's files inside the tool result where the API accepts them there. Send the rest in a following user message.
- When the API takes no media in a tool result, build the result with `BaseToolReturnPart.model_response_str_and_user_content()` and send the files it returns in the next user message. Precedent: `models/openai.py::OpenAIChatModel._map_user_message`. — The helper writes the `See file` references and the provenance tags that tell the model the files came from the tool.
- When the API accepts some media in a tool result, decide per file in the adapter's tool-return method, from one profile flag that lists the accepted MIME types. Precedent: `models/google.py::GoogleModel._map_tool_return` with `google_supported_mime_types_in_tool_returns`.
- Never send `model_response_str()` alone for a tool return that carries files. — It leaves the files out, so the model never sees them.
- A file kind the model can't read on any channel is not the tool-return mapper's case: declare it in the adapter's `_unsupported_file_inputs` (see `models/AGENTS.md`), and `Model.prepare_messages` replaces the file with a note.

### Prompt-cache markers

- Enforce the API's breakpoint limit with `models/_prompt_cache.py::excess_cache_points`. The adapter passes a wire accessor that recognizes a breakpoint, and strips the excess in its own format. Precedent: `models/anthropic.py::AnthropicModel._limit_cache_points`. — Breakpoint arithmetic written per adapter has already produced several copies of the same rule.
- Put a new cache algorithm that two adapters need in `models/_prompt_cache.py`, with a public name and adapter-supplied wire accessors. Never copy it into a second adapter.
- An adapter without prompt caching drops `CachePoint`s. The `CachePoint` docstring promises that models without caching filter them out.
- Anchor every breakpoint by message identity: see the injection rule in `models/AGENTS.md` and [Message History & Prompt Caching](message-history-and-caching.md).
