# Input and History

Read this file when the user wants multimodal or structured input, message history, `run_id` / `conversation_id` correlation, or context trimming.

## Send Images, Audio, Video, or Documents to the Model

Pass multimodal content as a list mixing text with `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`, or `BinaryContent`.

```python
from pydantic_ai import Agent, ImageUrl

agent = Agent(model='openai:gpt-5.2', name='multimodal_agent')
result = agent.run_sync(
    [
        'What company is this logo from?',
        ImageUrl(url='https://example.com/logo.png'),
    ]
)
print(result.output)
```

Use `BinaryContent(...)` when the asset is already in memory instead of at a URL.

Not every model supports every input type. Keep provider expectations in mind when the user chooses a specific model.

For Gemini models that support agentic video processing, set `vendor_metadata={'media_processing': 'AGENTIC'}` on a `VideoUrl`, video `BinaryContent`, or video `UploadedFile`. See Google's [agentic video understanding guide](https://ai.google.dev/gemini-api/docs/video-understanding#agentic-video-understanding).

On follow-up requests, Pydantic AI omits Gemini media-processing response parts while keeping the ordinary final response and thought signature. When follow-up history still includes the original video, it is sent and processed again, which can incur additional video-processing tokens.

## Send Structured Data in the Prompt

Wrap a structured value (UI state, a record to judge) in `StructuredContent` instead of serializing it yourself. It stays structured in message history, and the model is sent text rendered once when the item is created: compact JSON by default, wrapped in `tag` when given, with `<`, `>` and `&` escaped.

```python
from pydantic_ai import Agent, StructuredContent

agent = Agent(model='openai:gpt-5.2')
result = agent.run_sync(
    [StructuredContent({'url': '/products/42', 'cart_total': 949.0}, tag='ui_state'), 'Why did the total jump?']
)
```

- `format='xml'` renders with `format_as_xml`; a function `value -> str` renders custom text; or subclass `StructuredContent` and override `render()`.
- The rendered text is stored in `content`, so a reloaded history is sent byte-for-byte as before. It is not re-rendered when `value` changes: build a new item, or `dataclasses.replace(item, value=..., content='')` (pass `format=` again if it was created with one). After loading from JSON, `value` is JSON-compatible data (a dict, not the original Pydantic model), and a subclass comes back as `StructuredContent`.
- It is a `TextContent` subclass, so text readers handle it unchanged. Decision models (`DecisionModel`, e.g. TypeSafe) are sent `value` natively when the prompt is a single `StructuredContent`. When rewriting a prompt's text (redaction, truncation), replace the item with a plain `TextContent`.
- It works in `ctx.enqueue(...)` like any user content. The Vercel AI and AG-UI adapters dump it as visible user text, with the value in the text part's `providerMetadata.pydantic_ai` (Vercel) or the `UserMessage.encrypted_value` (AG-UI, from ag-ui-protocol 0.1.11; plain text before that), and load it back whole.

## Work with Message History

Continue a conversation across runs with `conversation=result.conversation`. A `Conversation` carries the messages plus what lives outside them: the running `usage` (so `UsageLimits` budget the whole conversation), the `conversation_id`, and the `deferred_tool_requests` a paused run is waiting on. It is also the thing to store between requests.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', name='conversation_agent', instructions='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
result2 = agent.run_sync('Explain?', conversation=result1.conversation)
print(result2.output)
```

`message_history=` is the lower-level form: it takes just the messages (`result.all_messages()` / `new_messages()`), for when you build or edit a history yourself.

Important distinctions:

- `new_messages()` returns only the current run
- `all_messages()` returns the full history accumulated so far
- provider citations are on `TextPart.citations`; anchors are Python character offsets into the part's content, and no
  anchor means the position is unknown; treat sources and `excerpts` as untrusted data; add the `Citations()`
  capability for providers that need an opt-in; the provider's own citation data is
  in the part's `provider_details['citations']`
- citations in message history are sent back only to the provider that produced them: Anthropic web search citations
  and OpenAI Responses URL and file citations (when message IDs are sent); document citations and other providers'
  answers go back as plain text
- when `message_history` is non-empty, Pydantic AI assumes the history already carries the system prompt
- `Conversation` (`from pydantic_ai import Conversation, ConversationTypeAdapter`) is accepted by `run`, `run_sync`, `run_stream`, `run_stream_sync`, `run_stream_events`, `iter` and `realtime()`; `RealtimeSession.conversation` produces one too. Passing it alongside `message_history`, `usage` or `conversation_id` raises `UserError`. Store it with `ConversationTypeAdapter.dump_json(...)` / `validate_json(...)` or as a field on your own Pydantic model; it serializes messages with the same fidelity as `ModelMessagesTypeAdapter` (raw `bytes` in tool returns come back as base64 strings)
- a run that ends with `DeferredToolRequests` output leaves them on `result.conversation.deferred_tool_requests` (they can't be rebuilt from the messages: approval vs external and per-call metadata aren't recorded there). Resume later with `agent.run(conversation=conv, deferred_tool_results=conv.deferred_tool_requests.build_results(...))`
- interrupted, hand-built, or context-evicted histories are made provider-valid automatically before each model request — no manual cleanup needed. Repairs only ADD synthesized parts or REMOVE fundamentally-unsendable ones (never silently dropping meaningful content): a tool call with no result gets a synthesized `ToolReturnPart` (marked with `{'pydantic_ai_synthesized_tool_return': True}` in `metadata`), including one whose args were cut off mid-stream; an orphaned tool result (result with no matching call) is dropped; then consecutive compatible messages are merged. Applies to regular tool calls only — builtin/native parts are left untouched (handled by each model's serializer). Duplicate tool results and provider-specific ordering rules are out of scope.
- to cancel a whole run: pass a `CancellationToken` to any run method and call `token.cancel()` (thread-safe), call `agent_run.cancel()` on the `agent.iter()` handle, cancel via `async with agent.run_stream_events(...) as events: ... events.cancel()`, or call `ctx.cancel()` from a tool, `event_stream_handler`, or capability hook. Inside the `agent.iter()` block this surfaces as `CancelledError`; once the context exits it raises `RunCancelled`. `RunCancelled.all_messages()` returns a complete snapshot of the history (completed tool results included) and can be passed as `message_history` to a new run to resume — dangling calls are repaired per the previous bullet. Cancellation is terminal: capability hooks may clean up but cannot recover the run to success. External `asyncio.Task.cancel()` keeps raising `CancelledError` (never translated; wins if both race); catch it and call `RunCancelled.from_cancellation(exc)` to access the attached run state. `StreamedRunResult.cancel()` is different: it only stops the current model response, the run continues.
- to bound a whole run's wall-clock time: pass `timeout=` (seconds) to any run method (`run`, `run_sync`, `run_stream`, `run_stream_sync`, `run_stream_events`, `iter`), or to a UI adapter's `run_stream`/`dispatch_request` (which report expiry to the client like any first-party cancellation and pass the `RunTimedOut` to `on_cancel`) or `run_stream_native` (which raises it). Expiry cancels the run like `cancel()` and raises `RunTimedOut` (a subclass of both `RunCancelled` and `TimeoutError`) carrying the same resumable `all_messages()`. `RunContext.deadline` (UTC `datetime` or `None`) and `ctx.remaining_time()` (seconds or `None`) expose it to tools and hooks. A sub-agent run from a tool inherits the deadline; its own `timeout=` can only shorten it, and a sub-agent's own shorter timeout fails just the delegate tool. A `FallbackModel` never falls back on expiry. Durability capabilities keep the original deadline across replay/recovery (Temporal: `workflow.now()` + workflow timer); deprecated `TemporalAgent`/`DBOSAgent`/`PrefectAgent` reject `timeout=`. Prefer this over wrapping the run in `asyncio.timeout()`, which surfaces as an external `CancelledError`/`TimeoutError`.

## Correlate Runs with `run_id` and `conversation_id`

Each message carries two identifiers:

- `run_id` — unique per `Agent.run` call (including deferred-tool resume). Surfaces on `RunContext.run_id`, `AgentRunResult.run_id`, message stamps, and OTel `gen_ai.agent.call.id`.
- `conversation_id` — shared across turns that pass the same `message_history`. Surfaces on `AgentRunResult.conversation_id`, message stamps, and OTel `gen_ai.conversation.id`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent(TestModel())

result1 = agent.run_sync('Tell me a joke.', run_id='run-from-api-42')
result2 = agent.run_sync('Explain?', message_history=result1.all_messages())

assert result1.run_id == 'run-from-api-42'
assert result1.run_id != result2.run_id  # never inherited from history
assert result1.conversation_id == result2.conversation_id  # inherited from history
```

Rules of thumb:

- Pass `run_id=` when your app mints an id before the run starts (e.g. one created, stored, or handed out to a client first) and you want `ctx.run_id` / stamps / OTel to match it.
- Do **not** pass `run_id=''`, or reuse a `run_id` that already appears on `message_history` — both raise `UserError` because they break `new_messages()` boundary detection. Correlate pause/resume and multi-turn work with `conversation_id` instead. When retrying a failed run with the same `run_id`, rebuild `message_history` without the failed attempt's messages.
- Pass `conversation_id='new'` to fork a thread off existing history; `'new'` is **not** a sentinel for `run_id`.
- UI adapters (`VercelAIAdapter`, `AGUIAdapter`) put a `message_history` you pass in front of the client's messages by default (`manage_message_history='client'`), and warn when a client message's UI message id is already in it (only histories that went through an adapter carry those ids, so a hand-built one duplicates silently). `useChat` and `@ag-ui/client` send the whole conversation every request, so when you persist `result.all_messages()` server-side (e.g. in `on_complete`) and pass it back, also pass `manage_message_history='server'` (adapter field, `from_request`, `dispatch_request`): only the client's new turn (its messages after the last assistant response) is used, a new turn starting with a stored user message id (regenerate/edit) cuts the stored history before it (a regenerate of an id the stored history lacks is rejected as a protocol error event), and client tool results only count for calls still unanswered at the end of the stored history.
- UI adapters auto-wire protocol thread/chat ids into `conversation_id`. Protocol run ids (e.g. AG-UI `runId`) are **not** mapped into agent `run_id` — pass `run_id=` on the adapter/`Agent.run` if you need them aligned. A `UIEventStream` used standalone (no adapter, e.g. encoding events out of a durable execution workflow or a queue) has no protocol ids to wire: `AGUIEventStream(thread_id=..., run_id=...)` takes them directly, and defaults each to a fresh UUID that matches nothing agent-side, so pass the run's own `conversation_id`/`run_id` to keep them aligned.
- AG-UI live failed tool outcomes round-trip through a namespaced payload on `ReasoningEncryptedValueEvent.encrypted_value` with `ag-ui-protocol >= 0.1.11`. Earlier event streams have no outcome carrier, so reloading them reconstructs the tool result as successful.

## Manage Context Size

Use `capabilities=[ProcessHistory(...)]` to trim or rewrite message history before each model request. `ProcessHistory` is a thin wrapper around the `before_model_request` lifecycle hook — for richer control (access to `RunContext`/`ModelRequestContext`, ability to short-circuit the model call), hook the event directly via `capabilities=[Hooks(before_model_request=fn)]`.

In `before_model_request` and `wrap_model_request`, use the two message views deliberately:

- Mutating or assigning `request_context.messages` changes only the current model request.
- `ctx.messages[:] = rewritten` changes persistent history and later requests, but not the current model request.
- Use both assignments when both effects are intended.
- Deprecated: in `before_model_request`, `append`/`extend`/`+=` on `request_context.messages` also appends to `ctx.messages` and warns. Assign a new list instead, e.g. `request_context.messages = [*request_context.messages, m]` plus `ctx.messages.append(m)`.

The separation is only at the outer collection. `request_context.messages` is an independent shallow list, but retained messages and nested parts may be the same objects as those in `ctx.messages`. Apart from the deprecated appends above, changing the outer list is isolated; mutating a contained message or part in place is not and can affect persistent history. For request-only changes below the message level, use `dataclasses.replace` to construct new messages and parts down to the level being changed.

`ProcessHistory` and compaction intentionally update both contexts. A processor transforms the current request view and makes its complete result persistent, so capability order still matters around processors.

```python
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.capabilities import ProcessHistory


async def keep_recent(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-10:] if len(messages) > 10 else messages


agent = Agent('openai:gpt-5.2', name='trimmed_history_agent', capabilities=[ProcessHistory(keep_recent)])
```

Good uses:

- trimming long conversations
- removing PII before provider calls
- summarizing old messages: put the summary in a `ModelRequest(parts=[CompactionPart(content=summary)])` that replaces the messages it summarizes, not in a `SystemPromptPart`. Every provider model receives it as framed user-voice text that says a separate model wrote it, so a model-written summary stays out of the system prompt.
- applying app-specific history policies

To decide *when* to trim or summarize, a context-aware processor can check `ctx.context_window_used` — the fraction of the model's context window occupied as of the last response. Treat `None` as unknown and leave history unchanged; it means there is no response yet or the window or usage is unknown. The window size itself is `model.context_window`, read from the profile's `context_window` (filled automatically from genai-prices data, or set explicitly via `profile={'context_window': 128_000}` for custom/local models); a `FallbackModel` reports the smallest window among its candidates.

## Inject Messages Mid-Run

Use `RunContext.enqueue(...)` (from a tool or capability hook), `AgentRun.enqueue(...)` (from external code driving `agent.iter()`), or `RealtimeSession.enqueue(...)` (from external code driving a realtime session) to add content to the conversation while a run is in progress — e.g. a tool adding follow-up context, or an external event "steering" the agent.

`enqueue` is variadic; each positional arg is one item: a piece of `UserContent` (a `str` or multi-modal content like an `ImageUrl`), a `ModelRequestPart` (e.g. a `SystemPromptPart`), or a complete `ModelRequest`/`ModelResponse`. Adjacent user content is gathered into one `UserPromptPart`. Pass an existing list by spreading it (`enqueue(*items)`). All three entry points return an `enqueue_id` (`str`) for non-empty calls, or `None` for empty calls. Standard-run and realtime event streams yield an `EnqueuedMessagesEvent` (with that `enqueue_id` and the delivered messages) once those messages enter history, so a client can observe when its steering message took effect. Parts enqueued with `'asap'` during a tool call go into the request carrying that call's `ToolReturnPart`, right after it, and the event's `parts` holds them as they sit in history (its `messages` still holds them, as a request of their own). The Vercel AI and AG-UI adapters stream them to the frontend as history it keeps and sends back without rendering (a `data-enqueued-messages` data part, a `pydantic_ai_enqueued_messages` activity), and `load_messages` restores them in place when the frontend sends them back. `@ag-ui/client` leaves activity messages out of its requests, so on AG-UI keep the history on the server or re-add the activities in a client middleware (see [Records carried as activity messages](https://pydantic.dev/docs/ai/integrations/ui/ag-ui/#activity-records)). Under the default `manage_system_prompt='server'`, enqueued `SystemPromptPart`s are not sent. Realtime sessions accept text and `SystemPromptPart`s only. System parts enqueued alone are delivered like `session.send(SystemPromptPart(...))`, on the provider's operator channel, and recorded as `SystemPromptPart`s; next to text they're rendered as `<system>…</system>` in the user turn, recorded as one `UserPromptPart`. A system part marks provenance, not silence: the model still gets a turn on it (use `session.send(text, respond=False)` for context that should not prompt a turn).

An enqueued `SystemPromptPart` is a mid-conversation instruction: it's sent at its position in the history rather than hoisted into the provider's top-level system prompt, so it doesn't invalidate a cached prefix ahead of it. This does not enable caching by itself; configure prompt caching (the unified `cache` model setting — see [Configure Prompt Caching Across Providers](./CAPABILITIES-AND-HOOKS.md#configure-prompt-caching-across-providers) — or a provider-specific setting) or include a `CachePoint`. On models that honor `CachePoint`, one at the end of an `enqueue(...)` batch covers every preceding item in that batch, including a `SystemPromptPart`; one with more content after it caches up to where you put it and leaves the instruction outside, since the instruction is sent after the content it accompanies. Where the provider's API accepts a system message inline it's sent as one, with real operator authority; elsewhere it's rendered as `<system>`-tagged user content at that position, which a model treats as a strong preference rather than a system-level rule. On those models the standing prompt also gets one fixed sentence saying what the tag means, and `<system>` tags in user prompt and tool return text are escaped as `&lt;system>`, so that text can't open or close the tag. Support varies by model *and* transport, and Pydantic AI picks the rendering automatically — don't gate your own code on a model list.

For an instruction that's only true for the model request it's sent with (the budget left right now, a nudge against instruction fade), use `SystemPromptPart(content, scope='request')`. It stays in history, but the model sees it only in the model request it's part of: the next request, even the next step of the same run, no longer shows it. That fits a value that changes every request, sent with every request. Fixed text belongs in instructions, and a note the model must still act on after the next request ("the user can't see `run_tests` output, summarize its failures in your final answer") keeps the default `scope='conversation'` or goes in the tool's return value, since a second tool call would otherwise hide it before the final answer. On Anthropic models that take mid-conversation system messages it's sent as a `clear_at: 'next_user_message'` system message, every copy kept on the wire, which keeps the prompt cache and preserved thinking valid; elsewhere it's sent with its own request only. Automatic cache breakpoints land before it, so don't add a `CachePoint` ahead of it yourself. A capability that adds one every request calls `ctx.enqueue(SystemPromptPart(..., scope='request'))` in `before_model_request`, which puts it in the request being prepared and in history. Don't add reminders to `request_context.messages` alone, as per-request copies that never enter history: deleting one on the next request is a history edit that, on models with preserved thinking, invalidates every earlier thinking block.

To find a system prompt your code added again later (count reminders, replace an earlier note, tell your records from other prompts), set `SystemPromptPart(..., metadata={'origin': 'my_app.note'})` and match on that, never on the visible text. The metadata is never sent to the model, and it stays with the part through `all_messages()`, `ModelMessagesTypeAdapter` and continued conversations. `ModelRequest.metadata` doesn't: when consecutive requests are merged, the merged request keeps only Pydantic AI's own entries. The UI adapters drop it, like other application-only fields.

Only enqueue a `SystemPromptPart` for an instruction you authored. A system prompt carries operator authority, so building one out of tool output, a retrieved document, or a webhook payload hands that content the same authority and makes a prompt injection buried in it load-bearing. Late-arriving results are the case to watch, since they're a common reason to reach for `enqueue`: a background job whose tool returned `'started'` long before the work finished. Enqueue those as user content, and if the result should also change how the agent behaves, write that instruction yourself and enqueue the payload separately.

Never mutate messages already in the history in place (e.g. `ctx.messages[0].parts[0].content = '...'`, or `append`/item assignment on an existing `parts` list) — enqueue new content, or rewrite history via `ProcessHistory` by building new message objects, e.g. with `dataclasses.replace` passing a new `parts` list (replacing a message in the history and reassigning its `parts` list are both safe). In-place mutation is unsupported: instrumentation caches each message's serialized form per run, so later request spans record stale `gen_ai.input.messages` (a `MessageHistoryMutatedWarning` is emitted at run end when detected).

```python
from pydantic_ai import Agent, RunContext

agent = Agent('anthropic:claude-opus-4-7', name='alerting_agent')


@agent.tool
def trigger_alert(ctx: RunContext) -> str:
    ctx.enqueue('Alert: production is degraded, prioritize triage.')
    return 'alert raised'
```

A `priority` controls delivery:

- `'asap'` (default): delivered at the earliest opportunity — added to the next model request, or, if the agent would otherwise terminate, used to redirect the run into one more request. This is "steering" an in-flight agent.
- `'when_idle'`: delivered only when the agent would otherwise terminate, after any `'asap'` messages — a follow-up task that shouldn't interrupt in-flight work.

Both priorities drain however you drive the run — `agent.run()`, explicit `AgentRun.next()`, and a bare `async for node in agent_run:` loop all deliver enqueued messages. See [message history docs](https://pydantic.dev/docs/ai/core-concepts/message-history/#injecting-messages-mid-run) for details.
