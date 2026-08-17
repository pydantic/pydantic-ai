# Input and History

Read this file when the user wants multimodal input, message history, `run_id` / `conversation_id` correlation, or context trimming.

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

## Work with Message History

Use `message_history=` to continue a conversation across runs.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', name='conversation_agent', instructions='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.output)
```

Important distinctions:

- `new_messages()` returns only the current run
- `all_messages()` returns the full history accumulated so far
- when `message_history` is non-empty, Pydantic AI assumes the history already carries the system prompt
- interrupted, hand-built, or context-evicted histories are made provider-valid automatically before each model request — no manual cleanup needed. Repairs only ADD synthesized parts or REMOVE fundamentally-unsendable ones (never silently dropping meaningful content): a tool call with no result gets a synthesized `ToolReturnPart` (marked with `{'pydantic_ai_synthesized_tool_return': True}` in `metadata`), including one whose args were cut off mid-stream; an orphaned tool result (result with no matching call) is dropped; then consecutive compatible messages are merged. Applies to regular tool calls only — builtin/native parts are left untouched (handled by each model's serializer). Duplicate tool results and provider-specific ordering rules are out of scope.
- to cancel a whole run: pass a `CancellationToken` to any run method and call `token.cancel()` (thread-safe), call `agent_run.cancel()` on the `agent.iter()` handle, cancel via `async with agent.run_stream_events(...) as events: ... events.cancel()`, or call `ctx.cancel()` from a tool, `event_stream_handler`, or capability hook. Inside the `agent.iter()` block this surfaces as `CancelledError`; once the context exits it raises `RunCancelled`. `RunCancelled.all_messages()` returns a complete snapshot of the history (completed tool results included) and can be passed as `message_history` to a new run to resume — dangling calls are repaired per the previous bullet. Cancellation is terminal: capability hooks may clean up but cannot recover the run to success. External `asyncio.Task.cancel()` keeps raising `CancelledError` (never translated; wins if both race); catch it and call `RunCancelled.from_cancellation(exc)` to access the attached run state. `StreamedRunResult.cancel()` is different: it only stops the current model response, the run continues.

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
- UI adapters auto-wire protocol thread/chat ids into `conversation_id`. Protocol run ids (e.g. AG-UI `runId`) are **not** mapped into agent `run_id` — pass `run_id=` on the adapter/`Agent.run` if you need them aligned. A `UIEventStream` used standalone (no adapter, e.g. encoding events out of a durable execution workflow or a queue) has no protocol ids to wire: `AGUIEventStream(thread_id=..., run_id=...)` takes them directly, and defaults each to a fresh UUID that matches nothing agent-side, so pass the run's own `conversation_id`/`run_id` to keep them aligned.
- AG-UI live failed tool outcomes round-trip through a namespaced payload on `ReasoningEncryptedValueEvent.encrypted_value` with `ag-ui-protocol >= 0.1.13`. Earlier event streams have no outcome carrier, so reloading them reconstructs the tool result as successful.

## Manage Context Size

Use `capabilities=[ProcessHistory(...)]` to trim or rewrite message history before each model request. `ProcessHistory` is a thin wrapper around the `before_model_request` lifecycle hook — for richer control (access to `RunContext`/`ModelRequestContext`, ability to short-circuit the model call), hook the event directly via `capabilities=[Hooks(before_model_request=fn)]`.

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
- summarizing old messages
- applying app-specific history policies

## Inject Messages Mid-Run

Use `RunContext.enqueue(...)` (from a tool or capability hook) or `AgentRun.enqueue(...)` (from external code driving `agent.iter()`) to add content to the conversation while a run is in progress — e.g. a tool adding follow-up context, or an external event "steering" the agent.

`enqueue` is variadic; each positional arg is one item: a piece of `UserContent` (a `str` or multi-modal content like an `ImageUrl`), a `ModelRequestPart` (e.g. a `SystemPromptPart`), or a complete `ModelRequest`/`ModelResponse`. Adjacent user content is gathered into one `UserPromptPart`. Pass an existing list by spreading it (`enqueue(*items)`). Both `enqueue` methods return an `enqueue_id` (`str`) for non-empty calls, or `None` for empty calls. The event stream yields an `EnqueuedMessagesEvent` (with that `enqueue_id` and the delivered messages) once those messages enter run history, so a client can observe when its steering message took effect.

An enqueued `SystemPromptPart` is a mid-conversation instruction: it's sent at its position in the history rather than hoisted into the provider's top-level system prompt, so it doesn't invalidate a cached prefix ahead of it. This does not enable caching by itself; configure the model's prompt caching or include a `CachePoint`. On models that honor `CachePoint`, one at the end of an `enqueue(...)` batch covers every preceding item in that batch, including a `SystemPromptPart`; one with more content after it caches up to where you put it and leaves the instruction outside, since the instruction is sent after the content it accompanies. Where the provider's API accepts a system message inline it's sent as one, with real operator authority; elsewhere it's rendered as `<system>`-tagged user content at that position, which a model treats as a strong preference rather than a system-level rule. Support varies by model *and* transport, and Pydantic AI picks the rendering automatically — don't gate your own code on a model list.

Only enqueue a `SystemPromptPart` for an instruction you authored. A system prompt carries operator authority, so building one out of tool output, a retrieved document, or a webhook payload hands that content the same authority and makes a prompt injection buried in it load-bearing. Late-arriving results are the case to watch, since they're a common reason to reach for `enqueue`: a background job whose tool returned `'started'` long before the work finished. Enqueue those as user content, and if the result should also change how the agent behaves, write that instruction yourself and enqueue the payload separately.

Never mutate messages already in the history in place (e.g. `ctx.messages[0].parts[0].content = '...'`, or `append`/item assignment on an existing `parts` list) — enqueue new content, or rewrite history via `ProcessHistory` by building new message objects, e.g. with `dataclasses.replace` passing a new `parts` list (replacing a message in the history and reassigning its `parts` list are both safe). In-place mutation is unsupported: instrumentation caches each message's serialized form per run, so later request spans record stale `gen_ai.input.messages` (a `MessageHistoryMutatedWarning` is emitted at run end when detected).

```python
from pydantic_ai import Agent, RunContext

agent = Agent('anthropic:claude-opus-4-7', name='alerting_agent')


@agent.tool
def trigger_alert(ctx: RunContext[None]) -> str:
    ctx.enqueue('Alert: production is degraded, prioritize triage.')
    return 'alert raised'
```

A `priority` controls delivery:

- `'asap'` (default): delivered at the earliest opportunity — added to the next model request, or, if the agent would otherwise terminate, used to redirect the run into one more request. This is "steering" an in-flight agent.
- `'when_idle'`: delivered only when the agent would otherwise terminate, after any `'asap'` messages — a follow-up task that shouldn't interrupt in-flight work.

Both priorities drain however you drive the run — `agent.run()`, explicit `AgentRun.next()`, and a bare `async for node in agent_run:` loop all deliver enqueued messages. See [message history docs](https://ai.pydantic.dev/message-history/#injecting-messages-mid-run) for details.
