# Messages and chat history

Pydantic AI provides access to messages exchanged during an agent run. These messages can be used both to continue a coherent conversation, and to understand how an agent performed.

### Accessing Messages from Results

After running an agent, you can access the messages exchanged during that run from the `result` object.

Both [`RunResult`][pydantic_ai.agent.AgentRunResult]
(returned by [`Agent.run`][pydantic_ai.agent.AbstractAgent.run], [`Agent.run_sync`][pydantic_ai.agent.AbstractAgent.run_sync])
and [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] (returned by [`Agent.run_stream`][pydantic_ai.agent.AbstractAgent.run_stream]) have the following methods:

- [`all_messages()`][pydantic_ai.agent.AgentRunResult.all_messages]: returns all messages, including messages from prior runs. There's also a variant that returns JSON bytes, [`all_messages_json()`][pydantic_ai.agent.AgentRunResult.all_messages_json].
- [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages]: returns only the messages from the current run. There's also a variant that returns JSON bytes, [`new_messages_json()`][pydantic_ai.agent.AgentRunResult.new_messages_json].

!!! info "StreamedRunResult and complete messages"
    On [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult], the messages returned from these methods will only include the final result message once the stream has finished.

    E.g. you've awaited one of the following coroutines:

    * [`StreamedRunResult.stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output]
    * [`StreamedRunResult.stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text]
    * [`StreamedRunResult.stream_response()`][pydantic_ai.result.StreamedRunResult.stream_response]
    * [`StreamedRunResult.get_output()`][pydantic_ai.result.StreamedRunResult.get_output]

    **Note:** The final result message will NOT be added to result messages if you use [`.stream_text(delta=True)`][pydantic_ai.result.StreamedRunResult.stream_text] since in this case the result content is never built as one string.

Example of accessing methods on a [`RunResult`][pydantic_ai.agent.AgentRunResult] :

```python {title="run_result_messages.py" hl_lines="10"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

result = agent.run_sync('Tell me a joke.')
print(result.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

# all messages from the run
print(result.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            )
        ],
        timestamp=datetime.datetime(...),
        instructions='Be a helpful assistant.',
        run_id='...',
        conversation_id='...',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=RequestUsage(input_tokens=55, output_tokens=12),
        model_name='gpt-5.2',
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
    ),
]
"""
```

_(This example is complete, it can be run "as is")_

Example of accessing methods on a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] :

```python {title="streamed_run_result_messages.py" hl_lines="9 40"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')


async def main():
    async with agent.run_stream('Tell me a joke.') as result:
        # incomplete messages before the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    )
                ],
                timestamp=datetime.datetime(...),
                instructions='Be a helpful assistant.',
                run_id='...',
                conversation_id='...',
            )
        ]
        """

        async for text in result.stream_text():
            print(text)
            #> Did you hear
            #> Did you hear about the toothpaste
            #> Did you hear about the toothpaste scandal? They called
            #> Did you hear about the toothpaste scandal? They called it Colgate.

        # complete messages once the stream finishes
        print(result.all_messages())
        """
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Tell me a joke.',
                        timestamp=datetime.datetime(...),
                    )
                ],
                timestamp=datetime.datetime(...),
                instructions='Be a helpful assistant.',
                run_id='...',
                conversation_id='...',
            ),
            ModelResponse(
                parts=[
                    TextPart(
                        content='Did you hear about the toothpaste scandal? They called it Colgate.'
                    )
                ],
                usage=RequestUsage(input_tokens=50, output_tokens=12),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
                conversation_id='...',
            ),
        ]
        """
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

### Using Messages as Input for Further Agent Runs

The primary use of message histories in Pydantic AI is to maintain context across multiple agent runs.

To use existing messages in a run, pass them to the `message_history` parameter of
[`Agent.run`][pydantic_ai.agent.AbstractAgent.run], [`Agent.run_sync`][pydantic_ai.agent.AbstractAgent.run_sync] or
[`Agent.run_stream`][pydantic_ai.agent.AbstractAgent.run_stream].

If `message_history` is set and not empty, a new system prompt is not generated — we assume the existing message history includes a system prompt. If your history comes from a source that doesn't round-trip system prompts (a UI frontend, a database that didn't persist them, a compaction pipeline), add the [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability so the agent's configured `system_prompt` is reinjected at the head of the first request when it's missing.

Mid-conversation `SystemPromptPart`s (those in any `ModelRequest` after the first) are sent inline at their original position by providers whose API accepts system messages at arbitrary positions. For providers whose API doesn't, they're instead rendered as `<system>`-tagged `UserPromptPart`s at the same position, preserving the prefix cache and positional intent. Leading `SystemPromptPart`s always hoist to the provider's top-level system parameter.

```python {title="Reusing messages in a conversation" hl_lines="9 13"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync('Explain?', message_history=result1.new_messages())
print(result2.output)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            )
        ],
        timestamp=datetime.datetime(...),
        instructions='Be a helpful assistant.',
        run_id='...',
        conversation_id='...',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=RequestUsage(input_tokens=55, output_tokens=12),
        model_name='gpt-5.2',
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ],
        timestamp=datetime.datetime(...),
        instructions='Be a helpful assistant.',
        run_id='...',
        conversation_id='...',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.'
            )
        ],
        usage=RequestUsage(input_tokens=56, output_tokens=26),
        model_name='gpt-5.2',
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
    ),
]
"""
```

_(This example is complete, it can be run "as is")_

### Making histories provider-valid

Model providers reject a request whose message history has broken tool-call/tool-result pairing — a tool call with no result, or a result with no call. A run that is cancelled or crashes partway through can leave the history in exactly this state, and so can a hand-built, truncated, or context-evicted history. You don't need to clean these up yourself: before each model request, Pydantic AI repairs the history it was given so the provider accepts it.

The guiding rule is to massage the history into a shape the provider accepts without ever discarding something you meant to send. Repairs only **add** synthesized parts or **remove** parts that are fundamentally unsendable (no provider could accept them); nothing meaningful is silently dropped. Concretely, before each request Pydantic AI:

- **Adds** a synthesized [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] for a tool call that has no result, telling the model the call was interrupted before a result was produced. It has [`outcome='interrupted'`][pydantic_ai.messages.BaseToolReturnPart.outcome] — a neutral outcome that (unlike `'failed'`) is not surfaced as a provider error — and carries `{'pydantic_ai_synthesized_tool_return': True}` in its [`metadata`][pydantic_ai.messages.BaseToolReturnPart.metadata] so your code can tell it apart from real tool results. This also covers a call whose arguments were cut off mid-stream: the call is kept as-is and closed out the same way.
- **Removes** an orphaned tool result — a [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] or [`RetryPromptPart`][pydantic_ai.messages.RetryPromptPart] whose tool call is absent from the history (including a result placed before its call). If this empties an interior [`ModelRequest`][pydantic_ai.messages.ModelRequest] the request is removed; if it empties the last message, an empty request is kept so the history still ends on a `ModelRequest`.

After the invalid parts are handled, consecutive compatible messages are **merged** into one (two adjacent [`ModelRequest`][pydantic_ai.messages.ModelRequest]s become a single turn, with tool results ordered ahead of user parts). This changes message boundaries but preserves all content, so processed history you inspect afterwards may have fewer messages than you passed in.

The repair is deterministic and idempotent: repairing the same history always produces the same output, running a repaired history through another run leaves it untouched, and synthesized parts contain no wall-clock data, so reuse doesn't invalidate provider prompt caches.

Tool calls that can still receive a real result are left alone: when the history ends on a `ModelResponse` with tool calls, running without a new `user_prompt` executes them, and [deferred tool calls](deferred-tools.md) are matched to their `deferred_tool_results` — including when a 'complete' `ModelRequest` with the already-executed results follows the response. Repair of that live frontier only happens when the interruption is evident: a final response with [`state='interrupted'`][pydantic_ai.messages.ModelResponse.state] or a trailing request with [`state='interrupted'`][pydantic_ai.messages.ModelRequest.state] (e.g. from a [cancelled stream](output.md#cancelling-streams) or a crash during tool execution) whose tool calls will never be executed.

This pipeline handles regular, locally-executed tool calls only. Builtin (server-side) tool parts — produced and resulted by the provider inline — are left untouched and repaired by each model's own serializer instead. Some other provider-invalid shapes are also out of scope and may be rejected: duplicate tool results for one call, and provider-specific ordering rules beyond call/result pairing.

### Correlating runs with `conversation_id`

Each `ModelRequest` and `ModelResponse` carries two identifiers:

- [`run_id`][pydantic_ai.messages.ModelRequest.run_id] — unique per agent run; emitted on the OpenTelemetry agent run span as `gen_ai.agent.call.id`.
- [`conversation_id`][pydantic_ai.messages.ModelRequest.conversation_id] — shared across all runs that build on the same `message_history`; emitted as `gen_ai.conversation.id`.

A fresh `conversation_id` is generated on the first run, stamped onto every message produced by that run, and inherited by subsequent runs that pass the messages back via `message_history`. This means you can correlate traces from a multi-turn conversation in [Logfire](logfire.md) (or any OpenTelemetry backend) without tracking anything yourself — as long as the message history round-trips, the conversation ID does too.

```python {title="conversation_id is shared across runs in the same conversation"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')

result1 = agent.run_sync('Tell me a joke.')
result2 = agent.run_sync('Explain?', message_history=result1.all_messages())

assert result1.conversation_id == result2.conversation_id
```

To override or fork:

- Pass `conversation_id='<your-id>'` to use an ID from your own application (e.g. a chat thread ID stored in your database).
- Pass `conversation_id='new'` to start a fresh conversation that ignores any `conversation_id` already on `message_history` — useful for branching off an existing thread without making the caller generate an ID.

```python {title="forking a conversation"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')

result1 = agent.run_sync('Tell me a joke.')
forked = agent.run_sync(
    'Tell me a different joke.',
    message_history=result1.all_messages(),
    conversation_id='new',
)

assert forked.conversation_id != result1.conversation_id
```

The [UI adapters](ui/overview.md) auto-populate `conversation_id` from the protocol's own thread/chat ID, so frontends using these protocols get correlation for free.

## Storing and loading messages (to JSON)

While maintaining conversation state in memory is enough for many applications, often times you may want to store the messages history of an agent run on disk or in a database. This might be for evals, for sharing data between Python and JavaScript/TypeScript, or any number of other use cases.

The intended way to do this is using a `TypeAdapter`.

We export [`ModelMessagesTypeAdapter`][pydantic_ai.messages.ModelMessagesTypeAdapter] that can be used for this, or you can create your own.

Here's an example showing how:

```python {title="serialize messages to json"}
from pydantic_core import to_jsonable_python

from pydantic_ai import (
    Agent,
    ModelMessagesTypeAdapter,  # (1)!
)

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
history_step_1 = result1.all_messages()
as_python_objects = to_jsonable_python(history_step_1)  # (2)!
same_history_as_step_1 = ModelMessagesTypeAdapter.validate_python(as_python_objects)

result2 = agent.run_sync(  # (3)!
    'Tell me a different joke.', message_history=same_history_as_step_1
)
```

1. Alternatively, you can create a `TypeAdapter` from scratch:
   ```python {lint="skip" format="skip"}
   from pydantic import TypeAdapter
   from pydantic_ai import ModelMessage
   ModelMessagesTypeAdapter = TypeAdapter(list[ModelMessage])
   ```
2. Alternatively you can serialize to/from JSON directly:
   ```python {test="skip" lint="skip" format="skip"}
   from pydantic_core import to_json
   ...
   as_json_objects = to_json(history_step_1)
   same_history_as_step_1 = ModelMessagesTypeAdapter.validate_json(as_json_objects)
   ```
3. You can now continue the conversation with history `same_history_as_step_1` despite creating a new agent run.

_(This example is complete, it can be run "as is")_

### Loading untrusted history

The `message_history` parameter is trusted server-side state. If you load history that came from a browser request or another untrusted boundary, sanitize it before passing it to the agent.

[`sanitize_messages`][pydantic_ai.messages.sanitize_messages] applies the same default message sanitization used by the [UI adapters](ui/overview.md): it strips client-supplied system prompts, drops non-HTTP file URL schemes, resets non-allowlisted [`FileUrl.force_download`][pydantic_ai.messages.FileUrl.force_download] values to `False`, drops uploaded file references, and removes unresolved tool calls at the end of the history.

```python {title="sanitize untrusted message history" test="skip" lint="skip"}
from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.messages import sanitize_messages

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

# `request_json` is the body submitted by an untrusted client.
loaded_history = ModelMessagesTypeAdapter.validate_python(request_json['message_history'])
message_history = sanitize_messages(loaded_history)

result = agent.run_sync('Tell me a different joke.', message_history=message_history)
```

Each sanitization can be turned off individually when the corresponding parts were created by trusted server-side code: pass `strip_system_prompts=False`, add schemes to `allowed_file_url_schemes`, add values to `allowed_file_url_force_download`, or set `allow_uploaded_files=True`. See [file URL input security](input.md#user-side-download-vs-direct-file-url) for the file input trust model.

## Other ways of using messages

Since messages are defined by simple dataclasses, you can manually create and manipulate, e.g. for testing.

The message format is independent of the model used, so you can use messages in different agents, or the same agent with different models.

In the example below, we reuse the message from the first agent run, which uses the `openai:gpt-5.2` model, in a second agent run using the `google:gemini-3-pro-preview` model.

```python {title="Reusing messages with a different model" hl_lines="17"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', instructions='Be a helpful assistant.')

result1 = agent.run_sync('Tell me a joke.')
print(result1.output)
#> Did you hear about the toothpaste scandal? They called it Colgate.

result2 = agent.run_sync(
    'Explain?',
    model='google:gemini-3-pro-preview',
    message_history=result1.new_messages(),
)
print(result2.output)
#> This is an excellent joke invented by Samuel Colvin, it needs no explanation.

print(result2.all_messages())
"""
[
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Tell me a joke.',
                timestamp=datetime.datetime(...),
            )
        ],
        timestamp=datetime.datetime(...),
        instructions='Be a helpful assistant.',
        run_id='...',
        conversation_id='...',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='Did you hear about the toothpaste scandal? They called it Colgate.'
            )
        ],
        usage=RequestUsage(input_tokens=55, output_tokens=12),
        model_name='gpt-5.2',
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
    ),
    ModelRequest(
        parts=[
            UserPromptPart(
                content='Explain?',
                timestamp=datetime.datetime(...),
            )
        ],
        timestamp=datetime.datetime(...),
        instructions='Be a helpful assistant.',
        run_id='...',
        conversation_id='...',
    ),
    ModelResponse(
        parts=[
            TextPart(
                content='This is an excellent joke invented by Samuel Colvin, it needs no explanation.'
            )
        ],
        usage=RequestUsage(input_tokens=56, output_tokens=26),
        model_name='gemini-3-pro-preview',
        timestamp=datetime.datetime(...),
        run_id='...',
        conversation_id='...',
    ),
]
"""
```

_(This example is complete, it can be run "as is")_

## Sharing messages between agents

The same `message_history` parameter also works when the next run uses a
different [`Agent`][pydantic_ai.Agent]. This is useful for
[programmatic agent hand-off](multi-agent-applications.md#programmatic-agent-hand-off),
where your application runs one agent, then gives another agent the conversation
so far as context.

```python {title="sharing_messages_between_agents.py" hl_lines="19"}
from pydantic_ai import Agent

biography_agent = Agent(
    'openai:gpt-5.2',
    instructions='Answer biographical questions concisely.',
)

science_agent = Agent(
    'anthropic:claude-sonnet-4-6',
    instructions='Answer science questions for a general audience.',
)

biography_result = biography_agent.run_sync('Who was Albert Einstein?')
print(biography_result.output)
#> Albert Einstein was a German-born theoretical physicist.

science_result = science_agent.run_sync(
    'What was his most famous equation?',
    message_history=biography_result.new_messages(),
)
print(science_result.output)
#> Albert Einstein's most famous equation is (E = mc^2).
```

_(This example is complete, it can be run "as is")_

!!! tip "Handing off from a realtime session"
    A [realtime speech-to-speech session](realtime/index.md) accumulates the same message history, so you
    can pass [`session.all_messages()`][pydantic_ai.realtime.RealtimeSession.all_messages] straight
    into `agent.run(message_history=...)` to summarize or extract structured data from a voice
    conversation. See [Realtime message history](realtime/index.md#message-history).

!!! note "Instructions, system prompts, and tools"
    When you pass `message_history` to another agent, previous
    [`ModelRequest`][pydantic_ai.messages.ModelRequest] messages still contain
    the instructions used by the originating agent, but those instructions are
    not sent to the model again. The receiving agent uses its own
    `instructions`; see [Instructions](agent.md#instructions) for how this
    differs from [system prompts](agent.md#system-prompts) when
    `message_history` is provided.

    `system_prompt` is different: system prompt parts are part of the message
    history. If the receiving agent has its own `system_prompt` and you need to
    ensure it is present when reusing history, see
    [`ReinjectSystemPrompt`](capabilities/reinject-system-prompt.md). Use
    `replace_existing=True` when a system prompt from another agent should not
    remain authoritative.

    Tool call and tool return parts also remain in the history. Prefer sharing
    history between agents that can understand the same tool context, or pass
    only the messages that make sense for the receiving agent.

For more complex multi-agent patterns, see the [multi-agent applications](multi-agent-applications.md) documentation.

## Injecting messages mid-run

Tools, capability hooks, and external code driving an agent run can inject extra content
into the conversation mid-run with [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue]
(when a `RunContext` is in scope, e.g. inside a tool or capability hook) or
[`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue] (from external code driving
[`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter]). Use this when something happens during a
run that the agent should know about — a tool wants to add follow-up context, an external event
needs to *steer* the agent's plan, or background work needs to reach the agent when it completes.

A `priority` controls when the enqueued content is delivered:

- `'asap'` (default): delivered at the earliest opportunity — added to the next [`ModelRequest`][pydantic_ai.messages.ModelRequest], or, if the agent would otherwise terminate before another request, used to redirect the run into one more request. Use when the new context should reach the model as soon as possible; this is what other frameworks often call **steering** an in-flight agent.
- `'when_idle'`: delivered only when the agent would otherwise terminate, after any `'asap'` messages. Use when the agent shouldn't be interrupted but should pick up the new work — a follow-up task — once it's done with what it's doing.

`enqueue` is variadic — each positional argument is one item, and can be:

- a piece of [`UserContent`][pydantic_ai.messages.UserContent] — a `str` or multi-modal content like an [`ImageUrl`][pydantic_ai.messages.ImageUrl]. Adjacent user content is gathered into a single [`UserPromptPart`][pydantic_ai.messages.UserPromptPart], so `enqueue('caption', image)` forms one user turn. To pass an existing list, spread it: `enqueue(*items)`;
- a [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart], such as a [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart];
- a complete [`ModelRequest`][pydantic_ai.messages.ModelRequest] or [`ModelResponse`][pydantic_ai.messages.ModelResponse], to control request-level fields like `instructions`/`metadata` or to inject a synthetic prior turn.

Adjacent part-style items (user content and [`ModelRequestPart`][pydantic_ai.messages.ModelRequestPart]s) are coalesced into one [`ModelRequest`][pydantic_ai.messages.ModelRequest]; complete messages stay separate. This lets a single call inject an interleaved exchange — for example a synthetic tool call (a [`ModelResponse`][pydantic_ai.messages.ModelResponse]) followed by its result (a [`ModelRequest`][pydantic_ai.messages.ModelRequest]). The content must end in a request, so the agent has something to respond to.

Both `enqueue` methods return an `enqueue_id` (`str`) for a non-empty call, or `None` when called with no content. When the queued content is actually delivered into run history, the [event stream](agent.md#streaming-all-events) yields an [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] carrying that `enqueue_id` and the delivered messages (exactly as they landed in history), so a client can observe when its steering message took effect. The event carries the delivered message objects themselves — the same objects held in the run's message history. A history processor that replaces history with new message objects does not affect the event, but in-place mutation of a delivered message will be visible through it.

### From inside a tool or hook

Use [`RunContext.enqueue`][pydantic_ai.tools.RunContext.enqueue] when you have a
`RunContext` in scope:

```python {title="enqueue_from_tool.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart

agent = Agent('anthropic:claude-opus-4-7')


@agent.tool
def trigger_alert(ctx: RunContext[None]) -> str:
    ctx.enqueue('Alert: production is degraded, prioritize triage.')
    return 'alert raised'


@agent.tool
def enter_incident_mode(ctx: RunContext[None]) -> str:
    # Enqueue a `SystemPromptPart` to adjust the agent's standing instructions mid-run.
    ctx.enqueue(SystemPromptPart(content='You are now in incident mode: be terse and action-oriented.'))
    return 'incident mode enabled'
```

The `'asap'` message is appended to the agent's message history and is visible to the
model on the next request, alongside any tool returns from the same step. A
[`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart] is delivered the same way; on
providers that hoist system prompts (e.g. Anthropic, Google) a non-leading one is sent as a
`<system>`-tagged user-role message, so it keeps its mid-conversation position rather than being
lifted to the top.

### From external code driving `agent.iter()`

Use [`AgentRun.enqueue`][pydantic_ai.run.AgentRun.enqueue] when you're driving a run
from outside (e.g. forwarding events from a webhook, chat platform, or job queue):

```python {title="enqueue_from_agent_run.py"}
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('anthropic:claude-opus-4-7')


async def main():
    async with agent.iter('Summarize the latest deploy report') as agent_run:
        # An external system pushes a follow-up while the agent is working.
        # When the agent would otherwise finish, the message redirects it
        # into a fresh model request so it can incorporate the new context.
        agent_run.enqueue(
            'A new error was just reported — include it in the summary.',
            priority='when_idle',
        )
        node = agent_run.next_node
        while not isinstance(node, End):
            node = await agent_run.next(node)
```

The example drives the run with [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] +
[`AgentRun.next()`][pydantic_ai.run.AgentRun.next] because `'when_idle'` messages are only
drained when the agent would otherwise reach an `End` — that drain happens in `after_node_run`,
which doesn't fire inside a bare `async for node in agent_run:` loop. `'asap'` messages are
drained in `before_model_request` (which fires either way) and also at the same end-of-run point
if anything arrived during the final step. Reaching the end of a bare `async for` loop with
undrained pending messages raises [`UndrainedPendingMessagesError`][pydantic_ai.exceptions.UndrainedPendingMessagesError],
since those messages would otherwise be silently lost.

!!! info "Limitations"
    - End-of-run redirects need [`Agent.run`][pydantic_ai.agent.AbstractAgent.run] or
      explicit [`AgentRun.next()`][pydantic_ai.run.AgentRun.next] driving — they
      aren't drained inside a bare `async for node in agent_run:` loop (which raises
      [`UndrainedPendingMessagesError`][pydantic_ai.exceptions.UndrainedPendingMessagesError]
      if it ends with undrained messages). Messages delivered into a
      `before_model_request` work in either case.
    - Inside a [Temporal](durable_execution/temporal.md) workflow, tools run in
      activities and don't share state with the workflow, so `ctx.enqueue` from a
      tool doesn't currently propagate back to the run. Enqueue from the workflow
      context (e.g. via `AgentRun.enqueue`) instead.
    - Each end-of-run redirect opens a new model request. If something keeps
      enqueueing on every step (e.g. a tool that always enqueues, or a
      system-prompt callback that re-enqueues on each reinjection), the run will
      loop indefinitely. Set [`UsageLimits`][pydantic_ai.usage.UsageLimits] on the
      run as a safety net.
    - `enqueue` is designed to be called from the same event loop that drives the
      agent run. Inside the run that's automatic: async tools, sync tools (which
      Pydantic AI auto-wraps in a thread executor), and capability hooks all
      enqueue safely because the drain only iterates between graph nodes, never
      concurrently with a tool body. If you're forwarding events from a *different*
      thread or loop (e.g. a webhook handler), marshal the call onto the agent's
      loop first — e.g. `loop.call_soon_threadsafe(agent_run.enqueue, msg)`. The
      drain isn't atomic against concurrent cross-thread appends.

## Processing Message History

Sometimes you may want to modify the message history before it's sent to the model. This could be for privacy
reasons (filtering out sensitive information), to save costs on tokens, to give less context to the LLM, or
custom processing logic.

Pydantic AI provides the [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] capability that allows
you to intercept and modify the message history before each model request.

!!! note "`ProcessHistory` is a thin wrapper over `before_model_request`"
    [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] is a migration-friendly wrapper
    around the [`before_model_request`](hooks.md) lifecycle hook. If you want richer control
    over the message history — access to the full [`RunContext`][pydantic_ai.tools.RunContext]
    and [`ModelRequestContext`][pydantic_ai.models.ModelRequestContext], the ability to short-circuit
    the model call, etc. — hook the event directly via
    `capabilities=[Hooks(before_model_request=fn)]`.

!!! warning "History processors replace the message history"
    History processors replace the message history in the state with the processed messages, including the new user prompt part.
    This means that if you want to keep the original message history, you need to make a copy of it.

!!! warning "History processors can affect `new_messages()` results"
    [`new_messages()`][pydantic_ai.agent.AgentRunResult.new_messages] returns the messages
    produced during the current run. Messages provided via `message_history` are excluded —
    including the trailing `ModelRequest` when resuming without a user prompt, even though
    the framework may stamp it with the current run's `run_id` for observability.

    To keep this working when your processor mutates or adds messages:

    - If you rebuild the trailing `ModelRequest`, preserve its `parts`, `timestamp`,
      `instructions`, and `metadata` so it can still be identified as prior context.
    - If you insert a new message that should appear in `new_messages()`, use a
      [context-aware processor](#runcontext-parameter) and set `run_id=ctx.run_id` on it.

### Usage

Each [`ProcessHistory`][pydantic_ai.capabilities.ProcessHistory] wraps a callable that takes a list of
[`ModelMessage`][pydantic_ai.messages.ModelMessage] and returns a modified list of the same type.

Each processor is applied in sequence, and processors can be either synchronous or asynchronous.

```python {title="simple_history_processor.py"}
from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import ProcessHistory


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Remove all ModelResponse messages, keeping only ModelRequest messages."""
    return [msg for msg in messages if isinstance(msg, ModelRequest)]

# Create agent with history processor
agent = Agent('openai:gpt-5.2', capabilities=[ProcessHistory(filter_responses)])

# Example: Create some conversation history
message_history = [
    ModelRequest(parts=[UserPromptPart(content='What is 2+2?')]),
    ModelResponse(parts=[TextPart(content='2+2 equals 4')]),  # This will be filtered out
]

# When you run the agent, the history processor will filter out ModelResponse messages
# result = agent.run_sync('What about 3+3?', message_history=message_history)
```

#### Keep Only Recent Messages

You can use the `history_processor` to only keep the recent messages:

```python {title="keep_recent_messages.py"}
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.capabilities import ProcessHistory


async def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep only the last 5 messages to manage token usage."""
    return messages[-5:] if len(messages) > 5 else messages

agent = Agent('openai:gpt-5.2', capabilities=[ProcessHistory(keep_recent_messages)])

# Example: Even with a long conversation history, only the last 5 messages are sent to the model
long_conversation_history: list[ModelMessage] = []  # Your long conversation history here
# result = agent.run_sync('What did we discuss?', message_history=long_conversation_history)
```

!!! warning "Be careful when slicing the message history"
    When slicing the message history, you need to make sure that tool calls and returns are paired, otherwise the LLM may return an error. For more details, refer to [this GitHub issue](https://github.com/pydantic/pydantic-ai/issues/2050#issuecomment-3019976269).

#### `RunContext` parameter

History processors can optionally accept a [`RunContext`][pydantic_ai.tools.RunContext] parameter to access
additional information about the current run, such as dependencies, model information, and usage statistics:

```python {title="context_aware_processor.py"}
from pydantic_ai import Agent, ModelMessage, RunContext
from pydantic_ai.capabilities import ProcessHistory


def context_aware_processor(
    ctx: RunContext,
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    # Access current usage
    current_tokens = ctx.usage.total_tokens

    # Filter messages based on context
    if current_tokens > 1000:
        return messages[-3:]  # Keep only recent messages when token usage is high
    return messages

agent = Agent('openai:gpt-5.2', capabilities=[ProcessHistory(context_aware_processor)])
```

This allows for more sophisticated message processing based on the current state of the agent run.

#### Summarize Old Messages

Use an LLM to summarize older messages to preserve context while reducing tokens.

```python {title="summarize_old_messages.py"}
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.capabilities import ProcessHistory

# Use a cheaper model to summarize old messages.
summarize_agent = Agent(
    'openai:gpt-5-mini',
    instructions="""
Summarize this conversation, omitting small talk and unrelated topics.
Focus on the technical discussion and next steps.
""",
)


async def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    # Summarize the oldest 10 messages
    if len(messages) > 10:
        oldest_messages = messages[:10]
        summary = await summarize_agent.run(message_history=oldest_messages)
        # Return the last message and the summary
        return summary.new_messages() + messages[-1:]

    return messages


agent = Agent('openai:gpt-5.2', capabilities=[ProcessHistory(summarize_old_messages)])
```

!!! warning "Be careful when summarizing the message history"
    When summarizing the message history, you need to make sure that tool calls and returns are paired, otherwise the LLM may return an error. For more details, refer to [this GitHub issue](https://github.com/pydantic/pydantic-ai/issues/2050#issuecomment-3019976269), where you can find examples of summarizing the message history.

### Testing History Processors

You can test what messages are actually sent to the model provider using
[`FunctionModel`][pydantic_ai.models.function.FunctionModel]:

```python {title="test_history_processor.py"}
import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.fixture
def received_messages() -> list[ModelMessage]:
    return []


@pytest.fixture
def function_model(received_messages: list[ModelMessage]) -> FunctionModel:
    def capture_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Capture the messages that the provider actually receives
        received_messages.clear()
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Provider response')])

    return FunctionModel(capture_model_function)


def test_history_processor(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, capabilities=[ProcessHistory(filter_responses)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
    ]

    agent.run_sync('Question 2', message_history=message_history)
    assert received_messages == [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelRequest(parts=[UserPromptPart(content='Question 2')]),
    ]
```

### Multiple Processors

You can also use multiple processors:

```python {title="multiple_history_processors.py"}
from pydantic_ai import Agent, ModelMessage, ModelRequest
from pydantic_ai.capabilities import ProcessHistory


def filter_responses(messages: list[ModelMessage]) -> list[ModelMessage]:
    return [msg for msg in messages if isinstance(msg, ModelRequest)]


def summarize_old_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    return messages[-5:]


agent = Agent(
    'openai:gpt-5.2',
    capabilities=[ProcessHistory(filter_responses), ProcessHistory(summarize_old_messages)],
)
```

In this case, the `filter_responses` processor will be applied first, and the
`summarize_old_messages` processor will be applied second.

## Examples

For a more complete example of using messages in conversations, see the [chat app](examples/chat-app.md) example.
