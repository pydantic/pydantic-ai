# Streaming

An agent run has two things worth showing before it finishes: the **output** the model is producing, and the **events** that happen along the way, like the model's thinking, the tool calls it makes, and the results those tools return. Pydantic AI streams both, and most applications want some of each: text appearing token by token in a chat window, above a list of the tools that ran to produce it.

## Choosing how to stream {#choosing}

Four ways to drive a run give you a stream. They differ in what they hand you and in whether the agent graph always runs to completion:

| | Streams output | Streams events | Runs the graph to completion |
|---|---|---|---|
| [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] | yes, validated and accumulated | with `event_stream_handler=` | no, it stops at the first final output |
| [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] | as raw events you reassemble | yes | yes |
| [`agent.run(event_stream_handler=...)`][pydantic_ai.agent.AbstractAgent.run] | no, you get the result at the end | yes | yes |
| [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] | yes, per graph node | yes, per graph node | yes |

Reach for `run_stream()` when you want the final output as it is produced and nothing more. Reach for `run_stream_events()` when you want everything that happens and are willing to piece the output back together yourself. Reach for `iter()` when you want both at full fidelity and can accept the extra complexity of [driving the graph](agent.md#iterating-over-an-agents-graph) yourself.

!!! note "`run_stream()` stops at the first final output"
    The `run_stream()` and `run_stream_sync()` methods will consider the first output that matches the [output type](output.md#structured-output) (which could be text, an [output tool](output.md#tool-output) call, or a [deferred](deferred-tools.md) tool call) to be the final output of the agent run, even when the model generates (additional) tool calls after this "final" output.

    These "dangling" tool calls will not be executed unless the agent's [`end_strategy`][pydantic_ai.agent.Agent.end_strategy] is set to `'graceful'` or `'exhaustive'`, and even then their results will not be sent back to the model as the agent run will already be considered completed. In short, if the model returns both tool calls and text, and the agent's output type is `str`, **the tool calls will not run** in streaming mode with the default setting.

    If you want to always keep running the agent when it performs tool calls, use [`run_stream_events()`](#streaming-all-events) or [`iter()`](#streaming-all-events-and-output) instead.

## Streaming output {#streaming-output}

There are two challenges with streamed output:

1. Validating structured responses before they're complete, which is achieved by "partial validation" in Pydantic ([pydantic/pydantic#10748](https://github.com/pydantic/pydantic/pull/10748)).
2. When receiving a response, we don't know if it's the final response without starting to stream it and peeking at the content. Pydantic AI streams just enough of the response to sniff out if it's a tool call or an output, then streams the whole thing and calls tools, or returns the stream as a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult].

### Streaming text {#streaming-text}

Example of streamed text output:

```python {title="streamed_hello_world.py" line_length="120"}
from pydantic_ai import Agent

agent = Agent('google:gemini-3-flash-preview')  # (1)!


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:  # (2)!
        async for message in result.stream_text():  # (3)!
            print(message)
            #> The first known
            #> The first known use of "hello,
            #> The first known use of "hello, world" was in
            #> The first known use of "hello, world" was in a 1974 textbook
            #> The first known use of "hello, world" was in a 1974 textbook about the C
            #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.
```

1. Streaming works with the standard [`Agent`][pydantic_ai.Agent] class, and doesn't require any special setup, just a model that supports streaming (currently all models support streaming).
2. The [`Agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] method is used to start a streamed run, this method returns a context manager so the connection can be closed when the stream completes.
3. Each item yield by [`StreamedRunResult.stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text] is the complete text response, extended as new data is received.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

The optional `debounce_by` argument of [`stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text] controls how long Pydantic AI groups incoming chunks before yielding. The default `0.1` groups chunks for up to 0.1 seconds; pass `None` to yield as soon as each chunk arrives. Debouncing is especially helpful for long structured responses, where it reduces the overhead of validating each chunk as it arrives.

We can also stream text as deltas rather than the entire text in each item:

```python {title="streamed_delta_hello_world.py"}
from pydantic_ai import Agent

agent = Agent('google:gemini-3-flash-preview')


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text(delta=True):  # (1)!
            print(message)
            #> The first known
            #> use of "hello,
            #> world" was in
            #> a 1974 textbook
            #> about the C
            #> programming language.
```

1. [`stream_text`][pydantic_ai.result.StreamedRunResult.stream_text] will error if the response is not text.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

!!! warning "Output message not included in `messages`"
    The final output message will **NOT** be added to result messages if you use `.stream_text(delta=True)`,
    see [Messages and chat history](message-history.md) for more information.

!!! note "`stream_text()` skips `TextOutput` functions"
    [`stream_text()`][pydantic_ai.result.StreamedRunResult.stream_text] does **not** apply [`TextOutput`](output.md#text-output) functions. With `delta=False` it applies [output validators](output.md#output-validator-functions) to each accumulated text snapshot, so a validator can transform what's yielded; with `delta=True` it yields the raw text deltas and skips validators. To stream the value produced by your `TextOutput` function, use [`stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output] instead.

### Streaming structured output {#streaming-structured-output}

Here's an example of streaming a user profile as it's built:

```python {title="streamed_user_profile.py" line_length="120"}
from datetime import date

from typing_extensions import NotRequired, TypedDict

from pydantic_ai import Agent


class UserProfile(TypedDict):
    name: str
    dob: NotRequired[date]
    bio: NotRequired[str]


agent = Agent(
    'openai:gpt-5.2',
    output_type=UserProfile,
    instructions='Extract a user profile from the input',
)


async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for profile in result.stream_output():
            print(profile)
            #> {'name': 'Ben'}
            #> {'name': 'Ben'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

#### Making structured responses appear faster {#making-structured-responses-appear-faster}

If a structured response takes a long time to appear in your application, make sure you stream validated partial output rather than waiting for the full run to finish. [`stream_output()`][pydantic_ai.result.StreamedRunResult.stream_output] yields the accumulated output as the model produces it, with partial validation applied to each snapshot and full validation applied to the final output.

When you also need events from intermediate model requests and tool calls, use [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter]. Iterate over each `AgentStream` until the model starts producing the final result, then switch to `stream_output()` for validated partial output:

```python {title="stream_structured_output_and_events.py"}
from pydantic import BaseModel

from pydantic_ai import Agent, AgentStreamEvent, FinalResultEvent


class Client(BaseModel):
    id: int
    name: str


agent = Agent(
    'openai:gpt-5.2',
    output_type=list[str | Client],
    instructions='Find the requested clients and explain each match.',
)


def record_event(event: AgentStreamEvent) -> None:
    ...


def render_output(output: list[str | Client]) -> None:
    ...


async def main():
    async with agent.iter('Find clients named Jane') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    final_result_started = False
                    async for event in stream:
                        record_event(event)
                        if isinstance(event, FinalResultEvent):
                            final_result_started = True
                            break

                    if final_result_started:
                        async for output in stream.stream_output():
                            render_output(output)
            elif Agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        record_event(event)
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

Each value from `stream_output()` is an accumulated snapshot, not a delta. An incomplete field or list item may be absent until enough data has arrived for it to pass partial validation, so update the rendered value from each snapshot rather than appending every yield.

`AgentStream` is a single iterator. Once you switch to `stream_output()`, it consumes the remaining final-output events while validating them, so those raw events are not also yielded to the preceding loop. If you need to retain every raw event, use [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] and reconstruct and validate the output yourself.

As setting an `output_type` uses the [Tool Output](output.md#tool-output) mode by default, this will only work if the model supports streaming tool arguments. For models that don't, try [Native Output](output.md#native-output) or [Prompted Output](output.md#prompted-output) instead. With Gemini 3, use Native Output; with earlier Gemini models that also use function tools, use Prompted Output.

### Streaming model responses

If you want fine-grained control of validation, you can use the following pattern to get the entire partial [`ModelResponse`][pydantic_ai.messages.ModelResponse]:

```python {title="streamed_user_profile.py" line_length="120"}
from datetime import date

from pydantic import ValidationError
from typing_extensions import TypedDict

from pydantic_ai import Agent


class UserProfile(TypedDict, total=False):
    name: str
    dob: date
    bio: str


agent = Agent('openai:gpt-5.2', output_type=UserProfile)


async def main():
    user_input = 'My name is Ben, I was born on January 28th 1990, I like the chain the dog and the pyramid.'
    async with agent.run_stream(user_input) as result:
        async for message in result.stream_response(debounce_by=0.01):  # (1)!
            try:
                profile = await result.validate_response_output(  # (2)!
                    message,
                    allow_partial=message.state == 'incomplete',
                )
            except ValidationError:
                continue
            print(profile)
            #> {'name': 'Ben'}
            #> {'name': 'Ben'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the '}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyr'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
            #> {'name': 'Ben', 'dob': date(1990, 1, 28), 'bio': 'Likes the chain the dog and the pyramid'}
```

1. [`stream_response`][pydantic_ai.result.StreamedRunResult.stream_response] streams the data as [`ModelResponse`][pydantic_ai.messages.ModelResponse] objects, thus iteration can't fail with a `ValidationError`.
2. [`validate_response_output`][pydantic_ai.result.StreamedRunResult.validate_response_output] validates the data, `allow_partial=True` enables pydantic's [`experimental_allow_partial` flag on `TypeAdapter`][pydantic.type_adapter.TypeAdapter.validate_json].

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

## Streaming events

Whatever streams the output, the same run also produces a stream of [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] describing what is happening: parts of the model's response arriving, tool calls being made, tool results coming back, and events your own code or a capability emitted. Consumers receive them through an `event_stream_handler`, through [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events], by streaming a node during [`iter()`][pydantic_ai.agent.AbstractAgent.iter], or through an [event hook](hooks.md#event-stream-hooks).

### The event stream {#event-reference}

Match events with `isinstance` against the classes below. Where an event reaches a frontend, the [AG-UI](ui/ag-ui.md) and [Vercel AI](ui/vercel-ai.md) columns name what the adapter emits for it; a dash means the adapter does not forward that event.

Events describing the model's response:

| Event | Fires when | AG-UI | Vercel AI |
|---|---|---|---|
| [`PartStartEvent`][pydantic_ai.messages.PartStartEvent] | A [response part](message-history.md) starts, or replaces the part at its index | depends on the part | depends on the part |
| [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] | A part receives an incremental update | depends on the delta | depends on the delta |
| [`PartEndEvent`][pydantic_ai.messages.PartEndEvent] | A part is complete | depends on the part | depends on the part |
| [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] | The response is recognized as producing the run's output | - | - |

What a part event becomes on the wire depends on the part it carries: text parts become the protocol's text events, [thinking](capabilities/thinking.md) parts its reasoning events, tool call parts its tool-input events, and [native tool](native-tools.md) returns its tool-output events. A [`PartDeltaEvent`][pydantic_ai.messages.PartDeltaEvent] carries a [`TextPartDelta`][pydantic_ai.messages.TextPartDelta], [`ThinkingPartDelta`][pydantic_ai.messages.ThinkingPartDelta], [`ToolCallPartDelta`][pydantic_ai.messages.ToolCallPartDelta], or, during a [realtime session](realtime/audio.md), a [`SpeechPartDelta`][pydantic_ai.messages.SpeechPartDelta], saying what changed.

Events describing tool execution:

| Event | Fires when | AG-UI | Vercel AI |
|---|---|---|---|
| [`FunctionToolCallEvent`][pydantic_ai.messages.FunctionToolCallEvent] | A [function tool](tools.md) is about to be called, including with invalid arguments | - | `tool-input-available` |
| [`FunctionToolResultEvent`][pydantic_ai.messages.FunctionToolResultEvent] | A function tool returned or asked the model to retry | `TOOL_CALL_RESULT` | `tool-output-*` |
| [`OutputToolCallEvent`][pydantic_ai.messages.OutputToolCallEvent] | The [output tool](output.md#tool-output) is called | - | `tool-input-available` |
| [`OutputToolResultEvent`][pydantic_ai.messages.OutputToolResultEvent] | The output tool produced its result | `TOOL_CALL_RESULT` | `tool-output-*` |
| [`ToolAvailabilityDeltaEvent`][pydantic_ai.messages.ToolAvailabilityDeltaEvent] | Tools were [revealed](tools-advanced.md) partway through the run | `ACTIVITY_SNAPSHOT`, on `ag-ui-protocol >= 0.1.19` | data chunk |
| [`DeferredToolRequestsEvent`][pydantic_ai.messages.DeferredToolRequestsEvent] | A batch of calls needs [approval or external execution](deferred-tools.md) | at run end | at run end |
| [`DeferredToolResultsEvent`][pydantic_ai.messages.DeferredToolResultsEvent] | A handler resolved deferred calls within the run | - | - |

To handle a function tool and the output tool the same way, match the [`ToolCallEvent`][pydantic_ai.messages.ToolCallEvent] and [`ToolResultEvent`][pydantic_ai.messages.ToolResultEvent] base classes instead of the four subclasses.

Events describing the run itself, and events your own code emits:

| Event | Fires when | AG-UI | Vercel AI |
|---|---|---|---|
| [`EnqueuedMessagesEvent`][pydantic_ai.messages.EnqueuedMessagesEvent] | [Enqueued messages](message-history.md) enter the run's history | - | - |
| [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] | The run finished; carries the result. Only `run_stream_events()` yields it | `RUN_FINISHED` | `message-metadata` |
| [`CustomEvent`][pydantic_ai.messages.CustomEvent] subclass | Your application [emits one](#custom-events) | `CUSTOM` | `data-{name}` chunk |
| [`CapabilityEvent`][pydantic_ai.messages.CapabilityEvent] subclass | A [capability](capabilities/overview.md#capability-events) emits one | - | - |

`AgentRunResultEvent` is not an `AgentStreamEvent`: it is appended by `run_stream_events()` so that one loop can see both the events and the result, which is why handlers and node streams never receive it.

To annotate something narrower than the full union, the members are also grouped: [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent] covers the four model-response events, [`HandleResponseEvent`][pydantic_ai.messages.HandleResponseEvent] the seven tool events, and [`RealtimeSessionEvent`][pydantic_ai.messages.RealtimeSessionEvent] the realtime ones.

During a [realtime session](realtime/overview.md), the same stream additionally carries the realtime-only [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members that report speech boundaries, interruptions, reconnects, and session errors. See [Realtime events](realtime/events.md) for that vocabulary.

### Streaming events with the final output {#streaming-events-and-final-output}

As shown above, [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] makes it easy to stream the agent's final output as it comes in.
It also takes an optional `event_stream_handler` argument that you can use to gain insight into what is happening during the run before the final output is produced.
During a realtime session, the same handler stream can also contain realtime-only [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members.

The example below shows how to stream events and text output. You can also [stream structured output](#streaming-structured-output).

Remember that `run_stream()` [stops at the first final output](#choosing), so tool calls the model makes alongside that output do not run. Use [`run_stream_events()`](#streaming-all-events) or [`iter()`](#streaming-all-events-and-output) when the whole graph must run.

```python {title="run_stream_event_stream_handler.py"}
import asyncio
from collections.abc import AsyncIterable
from datetime import date

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)

weather_agent = Agent(
    'openai:gpt-5.2',
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext,
    location: str,
    forecast_date: date,
) -> str:
    return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'


output_messages: list[str] = []

async def handle_event(event: AgentStreamEvent):
    if isinstance(event, PartStartEvent):
        output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
    elif isinstance(event, PartDeltaEvent):
        if isinstance(event.delta, TextPartDelta):
            output_messages.append(f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ThinkingPartDelta):
            output_messages.append(f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}')
        elif isinstance(event.delta, ToolCallPartDelta):
            output_messages.append(f'[Request] Part {event.index} args delta: {event.delta.args_delta}')
    elif isinstance(event, FunctionToolCallEvent):
        output_messages.append(
            f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
        )
    elif isinstance(event, FunctionToolResultEvent):
        output_messages.append(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.part.content}')
    elif isinstance(event, FinalResultEvent):
        output_messages.append(f'[Result] The model starting producing a final result (tool_name={event.tool_name})')


async def event_stream_handler(
    ctx: RunContext,
    event_stream: AsyncIterable[AgentStreamEvent],
):
    async for event in event_stream:
        await handle_event(event)

async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async with weather_agent.run_stream(user_prompt, event_stream_handler=event_stream_handler) as run:
        async for output in run.stream_text():
            output_messages.append(f'[Output] {output}')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """
```

_(This example is complete, it can be run "as is")_

### Streaming all events {#streaming-all-events}

Like `agent.run_stream()`, [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] takes an optional `event_stream_handler`
argument that lets you stream all events from the model's streaming response and the agent's execution of tools.
Unlike `run_stream()`, it always runs the agent graph to completion even if text was received ahead of tool calls that looked like it could've been the final result.
During a realtime session, an event stream handler can also receive realtime-only [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members.

For convenience, a [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] method is also available as a wrapper around `run(event_stream_handler=...)`. It is an async context manager that yields an async iterator over [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] ending with an [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] carrying the final run result.

!!! note
    As they return raw events as they come in, the `run_stream_events()` and `run(event_stream_handler=...)` methods require you to piece together the streamed text and structured output yourself from the `PartStartEvent` and subsequent `PartDeltaEvent`s.

    To get the best of both worlds, at the expense of some additional complexity, you can use [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] as described in [Streaming events and output with `iter`](#streaming-all-events-and-output), which lets you [iterate over the agent graph](agent.md#iterating-over-an-agents-graph) and [stream both events and output](#streaming-all-events-and-output) at every step. See [Making structured responses appear faster](#making-structured-responses-appear-faster) for a focused example using validated structured output.

```python {title="run_events.py" requires="run_stream_event_stream_handler.py"}
import asyncio

from pydantic_ai import AgentRunResultEvent

from run_stream_event_stream_handler import handle_event, output_messages, weather_agent


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async with weather_agent.run_stream_events(user_prompt) as events:
        async for event in events:
            if isinstance(event, AgentRunResultEvent):
                output_messages.append(f'[Final Output] {event.result.output}')
            else:
                await handle_event(event)

if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model starting producing a final result (tool_name=None)',
        "[Request] Part 0 text delta: 'warm and sunny '",
        "[Request] Part 0 text delta: 'in Paris on '",
        "[Request] Part 0 text delta: 'Tuesday.'",
        '[Final Output] It will be warm and sunny in Paris on Tuesday.',
    ]
    """
```

_(This example is complete, it can be run "as is")_

### Streaming events and output with `iter` {#streaming-all-events-and-output}

[`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] hands you the agent's graph one node at a time, as described in [Iterating over an agent's graph](agent.md#iterating-over-an-agents-graph). Each node can be streamed individually with `node.stream(run.ctx)`, which is what lets you take raw events from some nodes and validated output from others in the same run, at the cost of driving the loop yourself.

Here is an example of streaming an agent run in combination with `async for` iteration:

```python {title="streaming_iter.py"}
import asyncio
from dataclasses import dataclass
from datetime import date

from pydantic_ai import (
    Agent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
)


@dataclass
class WeatherService:
    async def get_forecast(self, location: str, forecast_date: date) -> str:
        # In real code: call weather API, DB queries, etc.
        return f'The forecast in {location} on {forecast_date} is 24°C and sunny.'

    async def get_historic_weather(self, location: str, forecast_date: date) -> str:
        # In real code: call a historical weather API or DB
        return f'The weather in {location} on {forecast_date} was 18°C and partly cloudy.'


weather_agent = Agent[WeatherService, str](
    'openai:gpt-5.2',
    deps_type=WeatherService,
    output_type=str,  # We'll produce a final answer as plain text
    system_prompt='Providing a weather forecast at the locations the user provides.',
)


@weather_agent.tool
async def weather_forecast(
    ctx: RunContext[WeatherService],
    location: str,
    forecast_date: date,
) -> str:
    if forecast_date >= date.today():
        return await ctx.deps.get_forecast(location, forecast_date)
    else:
        return await ctx.deps.get_historic_weather(location, forecast_date)


output_messages: list[str] = []


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    # Begin a node-by-node, streaming iteration
    async with weather_agent.iter(user_prompt, deps=WeatherService()) as run:
        async for node in run:
            if Agent.is_user_prompt_node(node):
                # A user prompt node => The user has provided input
                output_messages.append(f'=== UserPromptNode: {node.user_prompt} ===')
            elif Agent.is_model_request_node(node):
                # A model request node => We can stream tokens from the model's request
                output_messages.append('=== ModelRequestNode: streaming partial request tokens ===')
                async with node.stream(run.ctx) as request_stream:
                    final_result_found = False
                    async for event in request_stream:
                        if isinstance(event, PartStartEvent):
                            output_messages.append(f'[Request] Starting part {event.index}: {event.part!r}')
                        elif isinstance(event, PartDeltaEvent):
                            if isinstance(event.delta, TextPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} text delta: {event.delta.content_delta!r}'
                                )
                            elif isinstance(event.delta, ThinkingPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}'
                                )
                            elif isinstance(event.delta, ToolCallPartDelta):
                                output_messages.append(
                                    f'[Request] Part {event.index} args delta: {event.delta.args_delta}'
                                )
                        elif isinstance(event, FinalResultEvent):
                            output_messages.append(
                                f'[Result] The model started producing a final result (tool_name={event.tool_name})'
                            )
                            final_result_found = True
                            break

                    if final_result_found:
                        # Once the final result is found, we can call `AgentStream.stream_text()` to stream the text.
                        # A similar `AgentStream.stream_output()` method is available to stream structured output.
                        async for output in request_stream.stream_text():
                            output_messages.append(f'[Output] {output}')
            elif Agent.is_call_tools_node(node):
                # A handle-response node => The model returned some data, potentially calls a tool
                output_messages.append('=== CallToolsNode: streaming partial response & tool usage ===')
                async with node.stream(run.ctx) as handle_stream:
                    async for event in handle_stream:
                        if isinstance(event, FunctionToolCallEvent):
                            output_messages.append(
                                f'[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})'
                            )
                        elif isinstance(event, FunctionToolResultEvent):
                            output_messages.append(
                                f'[Tools] Tool call {event.tool_call_id!r} returned => {event.part.content}'
                            )
            elif Agent.is_end_node(node):
                # Once an End node is reached, the agent run is complete
                assert run.result is not None
                assert run.result.output == node.data.output
                output_messages.append(f'=== Final Agent Output: {run.result.output} ===')


if __name__ == '__main__':
    asyncio.run(main())

    print(output_messages)
    """
    [
        '=== UserPromptNode: What will the weather be like in Paris on Tuesday? ===',
        '=== ModelRequestNode: streaming partial request tokens ===',
        "[Request] Starting part 0: ToolCallPart(tool_name='weather_forecast', tool_call_id='0001')",
        '[Request] Part 0 args delta: {"location":"Pa',
        '[Request] Part 0 args delta: ris","forecast_',
        '[Request] Part 0 args delta: date":"2030-01-',
        '[Request] Part 0 args delta: 01"}',
        '=== CallToolsNode: streaming partial response & tool usage ===',
        '[Tools] The LLM calls tool=\'weather_forecast\' with args={"location":"Paris","forecast_date":"2030-01-01"} (tool_call_id=\'0001\')',
        "[Tools] Tool call '0001' returned => The forecast in Paris on 2030-01-01 is 24°C and sunny.",
        '=== ModelRequestNode: streaming partial request tokens ===',
        "[Request] Starting part 0: TextPart(content='It will be ')",
        '[Result] The model started producing a final result (tool_name=None)',
        '[Output] It will be ',
        '[Output] It will be warm and sunny ',
        '[Output] It will be warm and sunny in Paris on ',
        '[Output] It will be warm and sunny in Paris on Tuesday.',
        '=== CallToolsNode: streaming partial response & tool usage ===',
        '=== Final Agent Output: It will be warm and sunny in Paris on Tuesday. ===',
    ]
    """
```

_(This example is complete, it can be run "as is")_

### Which event type do I use? {#which-event-type}

Pydantic AI has two families of user-defined events. They ride the same stream and are defined the same way, but which one you define is decided by **who owns the code doing the emitting**, and that split is enforced at runtime: emitting the wrong family raises a [`UserError`][pydantic_ai.exceptions.UserError].

| | [`CustomEvent`][pydantic_ai.messages.CustomEvent] | [`CapabilityEvent`][pydantic_ai.messages.CapabilityEvent] |
|---|---|---|
| **Use it when** | your application wants to tell its own stream consumer or frontend something | your [capability](capabilities/overview.md) wants to tell other capabilities and the host application something |
| **Emit from** | an application tool, an [output validator](output.md#output-validator-functions), a [hook](hooks.md), an `event_stream_handler`, or [`AgentRun.emit()`][pydantic_ai.run.AgentRun.emit] | a [capability](capabilities/custom.md) hook or a tool the capability contributes |
| **Naming** | flat and process-wide, like `progress` | namespaced, like `file_system.file_read` |
| **Reaches the frontend** | yes, via the [AG-UI](ui/ag-ui.md) and [Vercel AI](ui/vercel-ai.md) adapters | no, it is an internal signal; re-publish it as a `CustomEvent` if the frontend needs it |
| **Can carry a decision** | no | yes, with `dispatch='inline'` |

If you are writing a **capability**, define [`CapabilityEvent`][pydantic_ai.messages.CapabilityEvent]s, as described in [Capability events](capabilities/overview.md#capability-events): its events are part of its contract with the rest of the run, and the namespace is what keeps two capabilities from colliding on a name. If you are writing an **application**, define `CustomEvent`s. To surface a capability's event to your frontend, listen for it with an [event hook](hooks.md#event-stream-hooks) and emit your own `CustomEvent` carrying the public payload.

### Custom events {#custom-events}

Alongside the framework's own events, a tool or code driving [`agent.iter()`](agent.md#iterating-over-an-agents-graph) can emit its own [`CustomEvent`][pydantic_ai.messages.CustomEvent]s into the same stream. This is useful for surfacing progress updates, intermediate results, or status information from long-running work to whoever is consuming the stream, without adding anything to the model's context.

#### Defining and emitting an event

Define an event as a dataclass subclass of `CustomEvent` — its fields are the payload, and consumers can use an `isinstance` check against the class. Await [`ctx.emit()`][pydantic_ai.tools.RunContext.emit] with an event instance from any of your application's async code that receives a [`RunContext`][pydantic_ai.tools.RunContext]; code driving `agent.iter()` uses [`AgentRun.emit()`][pydantic_ai.run.AgentRun.emit] instead. Sync tools cannot emit events; write async tools when they need to emit events. When emitted from within a tool call, the event's [`tool_call_id`][pydantic_ai.messages.CustomEvent.tool_call_id] and [`tool_name`][pydantic_ai.messages.CustomEvent.tool_name] are stamped automatically so consumers can attribute it to the originating call. The event reaches the `event_stream_handler`, `run_stream_events()`, `agent.iter()` streaming, and the [AG-UI](ui/ag-ui.md) and [Vercel AI](ui/vercel-ai.md) UI adapters.

```python {title="custom_events.py"}
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass

from pydantic_ai import Agent, AgentStreamEvent, CustomEvent, RunContext
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)


@dataclass(kw_only=True)
class SyncProgressEvent(CustomEvent):
    done: int
    total: int


async def model_function(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[DeltaToolCalls | str]:
    if any(
        isinstance(part, ToolReturnPart)
        for message in messages
        for part in message.parts
    ):
        yield 'All 3 files synchronized.'
    else:
        yield {
            0: DeltaToolCall(
                name='sync_files', json_args='{"count": 3}', tool_call_id='sync'
            )
        }


agent = Agent(FunctionModel(stream_function=model_function))


@agent.tool
async def sync_files(ctx: RunContext, count: int) -> str:
    for i in range(1, count + 1):
        # Do some long-running work, emitting a progress event after each step.
        await ctx.emit(SyncProgressEvent(done=i, total=count))
    return f'Synchronized {count} files.'


progress: list[str] = []


async def handle_events(
    ctx: RunContext, events: AsyncIterable[AgentStreamEvent]
) -> None:
    async for event in events:
        if isinstance(event, SyncProgressEvent):
            progress.append(
                f'{event.done}/{event.total} from {event.tool_name} ({event.tool_call_id})'
            )


async def main():
    await agent.run('Synchronize my files', event_stream_handler=handle_events)
    print(progress)
    """
    [
        '1/3 from sync_files (sync)',
        '2/3 from sync_files (sync)',
        '3/3 from sync_files (sync)',
    ]
    """
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

An event is delivered to stream consumers as soon as it is emitted, so a progress event surfaces while the emitting tool is still running. Payload fields can hold any object, but to flow through [durable execution](durable_execution/overview.md) and the UI adapters they need to be serializable by pydantic. Events emitted from tools running concurrently interleave in emission order (best-effort ordering).

The payload cannot use the field names the envelope needs for itself: `data`, `tool_call_id`, `tool_name`, and `event_kind` are rejected when the class is defined, so pick another name (`payload`, `call_id`) for a field that would collide.

Emitting only works while the run is in progress and only from the family the emitting code owns, so each of these raises a [`UserError`][pydantic_ai.exceptions.UserError]: emitting a `CustomEvent` from a capability, emitting a [`CapabilityEvent`][pydantic_ai.messages.CapabilityEvent] from application code, and calling [`AgentRun.emit()`][pydantic_ai.run.AgentRun.emit] after the run has finished. Events emitted with `AgentRun.emit()` reach consumers that stream the run's nodes with `node.stream(run.ctx)`, as shown in [Streaming All Events and Output](#streaming-all-events-and-output); a bare `async for node in run` does not consume any event stream, so nothing surfaces.

Custom event names are derived from the class name by removing `Event` and converting the rest to snake case, so `SyncProgressEvent` uses `sync_progress`. Override the name with a class argument, for example `class SyncProgressEvent(CustomEvent, name='sync_status')`. Names are registered when the class is defined and must be unique within the process; re-executing the same class definition (as when re-running a notebook cell) replaces the registration.

The name is the event's wire identifier, not just a label: it's what a serialized event carries, so renaming the class renames the tag along with it. A rename is a compatibility break wherever events outlive the process that emitted them — [durable execution](durable_execution/overview.md) histories and caches, persisted event logs, a frontend matching on the name. Pass an explicit `name=` to pin the tag when you want the class free to be renamed.

Events round-trip through [`AgentStreamEvent`][pydantic_ai.messages.AgentStreamEvent] serialization as their original class. If an event is deserialized before its class is registered, it becomes an [`UnknownCustomEvent`][pydantic_ai.messages.UnknownCustomEvent], with its payload preserved in `data`, and a `UserWarning` is emitted. Import the module that defines your event before creating the adapter that deserializes it; each pydantic `TypeAdapter` captures the event classes registered when it is created. A registered event's payload schema follows the same compatibility expectations as [message](message-history.md) types: a payload that no longer validates against the local class fails loudly rather than degrading, so keep the serializing and deserializing sides on compatible versions of the module that defines the event.

Event names share one application-wide registry, and defining a second class with an already-registered name raises immediately. Custom events belong to the application, so a library that emits events into agent runs should define [capability events](capabilities/overview.md#capability-events) on a capability, which are namespaced. Only a library that reaches the run outside a capability -- a bare tool it hands the user to register -- can emit application-level events at all, and it should then register them under a dotted prefix (`name='mylib.progress'`) so they can't collide with the application's own event names.

UI adapters get the frontend payload by calling [`CustomEvent.to_payload()`][pydantic_ai.messages.CustomEvent.to_payload], which defaults to the event's own fields; override it when the UI should receive a different payload.

Custom events are forwarded to the frontend by default, as the application that emits them is also the one serving that frontend. An event that exists only for server-side consumers — metrics, an audit log, an `event_stream_handler` of your own — opts out with `ui=False`, and then reaches every in-process consumer while the [AG-UI](ui/ag-ui.md) and [Vercel AI](ui/vercel-ai.md) adapters skip it:

```python {title="internal_custom_event.py" noqa="F841"}
from dataclasses import dataclass

from pydantic_ai import CustomEvent


@dataclass(kw_only=True)
class IndexProgressEvent(CustomEvent, ui=False):
    done: int
    total: int
```

Subclasses inherit the setting, and because the check happens before the protocol-specific handler, adapters for other protocols honor it too. To send a *different* payload rather than nothing, override `to_payload()` instead — returning `None` from it sends an event with a null payload, which is how you send a name-only signal.

The flag lives on the class rather than on the wire, so an event deserialized where its defining module hasn't been imported arrives as an [`UnknownCustomEvent`][pydantic_ai.messages.UnknownCustomEvent] whose `ui` says nothing about what the application declared. Those aren't forwarded either, so an event crossing a process boundary can't leak a payload its class had opted out of. If events reach your frontend from another process — a [durable execution](durable_execution/overview.md) workflow, a queue, a websocket fan-out, as in [encoding events without a request](ui/overview.md#encoding-events-without-a-request) — import the modules that define them there, or none of your custom events will reach the frontend.

## Cancelling streams

Sometimes you need to stop a streaming response before it completes: a user clicks "stop generating" in a chat UI, you've received enough data to make a decision, or you want to avoid receiving more tokens. [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] and [`iter()`][pydantic_ai.agent.Agent.iter] support explicit cancellation by closing the underlying model stream. [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] is an async context manager, so cleanup runs deterministically when you stop consuming events — leaving the `async with` block cancels the background run task. To stop a non-streaming run, see [Cancelling a Run](agent.md#cancelling-a-run); to bound how long a single step may take, see [Timeouts](timeouts.md).

!!! note "Model support"
    The Google, xAI, and Hugging Face SDKs expose streaming only as async iterators, without documented per-stream transport handles. Pydantic AI safely interrupts its active iterator pulls, but the SDKs do not guarantee that closing the local iterator immediately stops remote generation or billing. See the [Google](models/google.md#streaming-cancellation), [xAI](models/xai.md#streaming-cancellation), and [Hugging Face](models/huggingface.md#streaming-cancellation) provider notes.

### Cleaning up `run_stream_events`

[`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] is an async context manager that yields an async iterator over events:

```python {title="stream_cancel_stream_events.py"}
from pydantic_ai import Agent, FinalResultEvent, PartStartEvent

agent = Agent('openai:gpt-5.2')


async def main():
    async with agent.run_stream_events('Write a long essay about Python') as events:
        async for event in events:
            if isinstance(event, PartStartEvent):
                print(f'Started: {event.part!r}')
                #> Started: TextPart(content='Python is a ')
            elif isinstance(event, FinalResultEvent):
                break  # (1)!
```

1. Breaking out of the loop leaves the `async with` block, which cancels the background run task and closes the HTTP connection.

_(This example is complete, it can be run "as is" -- you'll need to add `asyncio.run(main())` to run `main`)_

The yielded [`AgentRunEvents`][pydantic_ai.agent.AgentRunEvents] handle exposes `cancel()` to cancel the whole run (see [Cancelling a Run](agent.md#cancelling-a-run)); continued iteration then raises [`RunCancelled`][pydantic_ai.exceptions.RunCancelled]. It also provides `all_messages()`, `new_messages()`, `usage`, and the completed `result`. From inside a tool or `event_stream_handler`, use [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel] instead. As a response-level alternative, [`StreamedRunResult.cancel()`][pydantic_ai.result.StreamedRunResult.cancel] from `run_stream()` stops only the current model response.

### Cancelling `run_stream`

Call `cancel()` on the [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult] to cancel the stream:

```python {title="stream_cancel_stream.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')


async def main():
    async with agent.run_stream('Write a long essay about Python') as result:
        text = ''
        async for chunk in result.stream_text(delta=True):
            text += chunk
            if len(text) > 100:  # (1)!
                await result.cancel()  # (2)!
                break
        print(result.cancelled)  # (3)!
        #> True
        print(result.response.state == 'interrupted')  # (4)!
        #> True
```

1. Check a condition during streaming, for example whether enough text has been received.
2. `cancel()` tells the model provider to stop generating tokens and closes the HTTP connection when the model integration supports it.
3. The `cancelled` property reflects the cancellation state.
4. The final [`ModelResponse`][pydantic_ai.messages.ModelResponse] is marked with `state='interrupted'` so that downstream code can identify incomplete responses.

_(This example is complete, it can be run "as is" -- you'll need to add `asyncio.run(main())` to run `main`)_

If you `break` out of `stream_text()` and then leave the surrounding `async with` block, the stream is cleaned up as the context exits. Use `cancel()` when you want to stop generation immediately instead of only stopping local consumption.

!!! warning "Interrupted tool calls"
    Cancelling or breaking out of a model response stream can leave the final [`ModelResponse`][pydantic_ai.messages.ModelResponse] with incomplete tool-call arguments. Pydantic AI records the response with `state='interrupted'`, and when the history is reused in another run the partial tool calls are [repaired automatically](message-history.md#making-histories-provider-valid). If you are controlling the graph with [`agent.iter()`][pydantic_ai.agent.Agent.iter], call [`agent_run.cancel()`][pydantic_ai.run.AgentRun.cancel] to stop the whole run as well, or check `response.state == 'interrupted'` before allowing the run to continue into tool execution.

### Cancelling with `iter`

When using [`agent.iter()`][pydantic_ai.agent.Agent.iter] for fine-grained control over the agent graph, you can cancel the [`AgentStream`][pydantic_ai.result.AgentStream] inside a `ModelRequestNode.stream()` context:

```python {title="stream_cancel_iter.py"}
from pydantic_ai import Agent, FinalResultEvent

agent = Agent('openai:gpt-5.2')


async def main():
    async with agent.iter('Write a long essay about Python') as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        if isinstance(event, FinalResultEvent):
                            await stream.cancel()  # (1)!
                            break
```

1. `AgentStream.cancel()` cancels the stream at the model request level.

_(This example is complete, it can be run "as is" -- you'll need to add `asyncio.run(main())` to run `main`)_

To abort the run itself rather than just the current response -- and for how cancellation is recorded in message history -- see [Cancelling a Run](agent.md#cancelling-a-run).

## Streaming elsewhere

The event stream is also what several other features are built on:

- [UI event streams](ui/overview.md) turn it into a protocol a frontend understands, over [AG-UI](ui/ag-ui.md) or [Vercel AI](ui/vercel-ai.md).
- [Realtime sessions](realtime/events.md) carry the same events plus speech and session-lifecycle events.
- [Durable execution](durable_execution/overview.md) streams events across replays, with per-backend restrictions on where you can emit.
- The [`ProcessEventStream`](capabilities/process-event-stream.md) capability attaches a handler to an agent instead of passing one per run, and [event hooks](hooks.md#event-stream-hooks) observe or replace individual events.
- [Capability events](capabilities/overview.md#capability-events) are how a reusable capability reports what it did.
- [Cancelling a run](agent.md#cancelling-a-run) covers aborting a whole run, streamed or not, and [Timeouts](timeouts.md) bounds how long a single step may take.
- [`pydantic_ai.messages`](api/messages.md) is the API reference for every event class.

## Examples

The following examples demonstrate how to use streamed responses in Pydantic AI:

- [Stream markdown](examples/stream-markdown.md)
- [Stream Whales](examples/stream-whales.md)
