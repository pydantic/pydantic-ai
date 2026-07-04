# Input and History

Read this file when the user wants multimodal input, message history, or context trimming.

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

```python
from pydantic_ai import Agent, RunContext

agent = Agent('anthropic:claude-opus-4-7', name='alerting_agent')


@agent.tool
def trigger_alert(ctx: RunContext[None]) -> str:
    enqueue_id = ctx.enqueue('Alert: production is degraded, prioritize triage.')
    return f'alert raised: {enqueue_id}'
```

A `priority` controls delivery:

- `'asap'` (default): delivered at the earliest opportunity — added to the next model request, or, if the agent would otherwise terminate, used to redirect the run into one more request. This is "steering" an in-flight agent.
- `'when_idle'`: delivered only when the agent would otherwise terminate, after any `'asap'` messages — a follow-up task that shouldn't interrupt in-flight work.

`'when_idle'` redirects need `agent.run()` or explicit `AgentRun.next()` driving; they aren't drained inside a bare `async for node in agent_run:` loop. See [message history docs](https://ai.pydantic.dev/message-history/#injecting-messages-mid-run) for details.
