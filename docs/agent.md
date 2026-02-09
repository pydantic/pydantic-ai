## Introduction

Agents are Pydantic AI's primary interface for interacting with LLMs.

In some use cases a single Agent will control an entire application or component,
but multiple agents can also interact to embody more complex workflows.

The [`Agent`][pydantic_ai.Agent] class has full API documentation, but conceptually you can think of an agent as a container for:

| **Component**                                             | **Description**                                                                                           |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| [Instructions](#instructions)                             | A set of instructions for the LLM written by the developer.                                               |
| [Function tool(s)](tools.md) and [toolsets](toolsets.md)  | Functions that the LLM may call to get information while generating a response.                           |
| [Structured output type](output.md)                       | The structured datatype the LLM must return at the end of a run, if specified.                            |
| [Dependency type constraint](dependencies.md)             | Dynamic instructions functions, tools, and output functions may all use dependencies when they're run.          |
| [LLM model](api/models/base.md)                           | Optional default LLM model associated with the agent. Can also be specified when running the agent.       |
| [Model Settings](#additional-configuration)               | Optional default model settings to help fine tune requests. Can also be specified when running the agent. |

In typing terms, agents are generic in their dependency and output types, e.g., an agent which required dependencies of type `#!python Foobar` and produced outputs of type `#!python list[str]` would have type `Agent[Foobar, list[str]]`. In practice, you shouldn't need to care about this, it should just mean your IDE can tell you when you have the right type, and if you choose to use [static type checking](#static-type-checking) it should work well with Pydantic AI.

Here's a toy example of an agent that simulates a roulette wheel:

```python {title="roulette_wheel.py"}
from pydantic_ai import Agent, RunContext

roulette_agent = Agent(  # (1)!
    'openai:gpt-5.2',
    deps_type=int,
    output_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
)


@roulette_agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  # (2)!
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18  # (3)!
result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.output)  # (4)!
#> True

result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
print(result.output)
#> False
```

1. Create an agent, which expects an integer dependency and produces a boolean output. This agent will have type `#!python Agent[int, bool]`.
2. Define a tool that checks if the square is a winner. Here [`RunContext`][pydantic_ai.tools.RunContext] is parameterized with the dependency type `int`; if you got the dependency type wrong you'd get a typing error.
3. In reality, you might want to use a random number here e.g. `random.randint(0, 36)`.
4. `result.output` will be a boolean indicating if the square is a winner. Pydantic performs the output validation, and it'll be typed as a `bool` since its type is derived from the `output_type` generic parameter of the agent.

!!! tip "Agents are designed for reuse, like FastAPI Apps"
    You can instantiate one agent and use it globally throughout your application, as you would a small [FastAPI][fastapi.FastAPI] app or an [APIRouter][fastapi.APIRouter], or dynamically create as many agents as you want. Both are valid and supported ways to use agents.

## Running Agents

There are five ways to run an agent:

1. [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] — an async function which returns a [`RunResult`][pydantic_ai.agent.AgentRunResult] containing a completed response.
2. [`agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] — a plain, synchronous function which returns a [`RunResult`][pydantic_ai.agent.AgentRunResult] containing a completed response (internally, this just calls `loop.run_until_complete(self.run())`).
3. [`agent.run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] — an async context manager which returns a [`StreamedRunResult`][pydantic_ai.result.StreamedRunResult], which contains methods to stream text and structured output as an async iterable. [`agent.run_stream_sync()`][pydantic_ai.agent.AbstractAgent.run_stream_sync] is a synchronous variation that returns a [`StreamedRunResultSync`][pydantic_ai.result.StreamedRunResultSync] with synchronous versions of the same methods.
4. [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] — a function which returns an async iterable of [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] and a [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] containing the final run result.
5. [`agent.iter()`][pydantic_ai.agent.Agent.iter] — a context manager which returns an [`AgentRun`][pydantic_ai.agent.AgentRun], an async iterable over the nodes of the agent's underlying [`Graph`][pydantic_graph.graph.Graph].

Here's a simple example demonstrating the first four:

```python {title="run_agent.py"}
from pydantic_ai import Agent, AgentRunResultEvent, AgentStreamEvent

agent = Agent('openai:gpt-5.2')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> The capital of Italy is Rome.


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> The capital of France is Paris.

    async with agent.run_stream('What is the capital of the UK?') as response:
        async for text in response.stream_text():
            print(text)
            #> The capital of
            #> The capital of the UK is
            #> The capital of the UK is London.

    events: list[AgentStreamEvent | AgentRunResultEvent] = []
    async for event in agent.run_stream_events('What is the capital of Mexico?'):
        events.append(event)
    print(events)
    """
    [
        PartStartEvent(index=0, part=TextPart(content='The capital of ')),
        FinalResultEvent(tool_name=None, tool_call_id=None),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Mexico is Mexico ')),
        PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='City.')),
        PartEndEvent(
            index=0, part=TextPart(content='The capital of Mexico is Mexico City.')
        ),
        AgentRunResultEvent(
            result=AgentRunResult(output='The capital of Mexico is Mexico City.')
        ),
    ]
    """
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

You can also pass messages from previous runs to continue a conversation or provide context, as described in [Messages and Chat History](message-history.md).

### Streaming Events and Final Output

As shown in the example above, [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] makes it easy to stream the agent's final output as it comes in.
It also takes an optional `event_stream_handler` argument that you can use to gain insight into what is happening during the run before the final output is produced.

The example below shows how to stream events and text output. You can also [stream structured output](output.md#streaming-structured-output).

!!! note
    The `run_stream()` and `run_stream_sync()` methods will consider the first output that matches the [output type](output.md#structured-output) (which could be text, an [output tool](output.md#tool-output) call, or a [deferred](deferred-tools.md) tool call) to be the final output of the agent run, even when the model generates (additional) tool calls after this "final" output.

	These "dangling" tool calls will not be executed unless the agent's [`end_strategy`][pydantic_ai.agent.Agent.end_strategy] is set to `'exhaustive'`, and even then their results will not be sent back to the model as the agent run will already be considered completed. In short, if the model returns both tool calls and text, and the agent's output type is `str`, **the tool calls will not run** in streaming mode with the default setting.

    If you want to always keep running the agent when it performs tool calls, and stream all events from the model's streaming response and the agent's execution of tools,
    use [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] or [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] instead, as described in the following sections.

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
        output_messages.append(f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}')
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

### Streaming All Events

Like `agent.run_stream()`, [`agent.run()`][pydantic_ai.agent.AbstractAgent.run_stream] takes an optional `event_stream_handler`
argument that lets you stream all events from the model's streaming response and the agent's execution of tools.
Unlike `run_stream()`, it always runs the agent graph to completion even if text was received ahead of tool calls that looked like it could've been the final result.

For convenience, a [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] method is also available as a wrapper around `run(event_stream_handler=...)`, which returns an async iterable of [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] and a [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] containing the final run result.

!!! note
    As they return raw events as they come in, the `run_stream_events()` and `run(event_stream_handler=...)` methods require you to piece together the streamed text and structured output yourself from the `PartStartEvent` and subsequent `PartDeltaEvent`s.

    To get the best of both worlds, at the expense of some additional complexity, you can use [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] as described in the next section, which lets you [iterate over the agent graph](#iterating-over-an-agents-graph) and [stream both events and output](#streaming-all-events-and-output) at every step.

```python {title="run_events.py" requires="run_stream_event_stream_handler.py"}
import asyncio

from pydantic_ai import AgentRunResultEvent

from run_stream_event_stream_handler import handle_event, output_messages, weather_agent


async def main():
    user_prompt = 'What will the weather be like in Paris on Tuesday?'

    async for event in weather_agent.run_stream_events(user_prompt):
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

### Iterating Over an Agent's Graph

Under the hood, each `Agent` in Pydantic AI uses **pydantic-graph** to manage its execution flow. **pydantic-graph** is a generic, type-centric library for building and running finite state machines in Python. It doesn't actually depend on Pydantic AI — you can use it standalone for workflows that have nothing to do with GenAI — but Pydantic AI makes use of it to orchestrate the handling of model requests and model responses in an agent's run.

In many scenarios, you don't need to worry about pydantic-graph at all; calling `agent.run(...)` simply traverses the underlying graph from start to finish. However, if you need deeper insight or control — for example to inject your own logic at specific stages — Pydantic AI exposes the lower-level iteration process via [`Agent.iter`][pydantic_ai.agent.Agent.iter]. This method returns an [`AgentRun`][pydantic_ai.agent.AgentRun], which you can async-iterate over, or manually drive node-by-node via the [`next`][pydantic_ai.agent.AgentRun.next] method. Once the agent's graph returns an [`End`][pydantic_graph.nodes.End], you have the final result along with a detailed history of all steps.

#### `async for` iteration

Here's an example of using `async for` with `iter` to record each node the agent executes:

```python {title="agent_iter_async_for.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')


async def main():
    nodes = []
    # Begin an AgentRun, which is an async-iterable over the nodes of the agent's graph
    async with agent.iter('What is the capital of France?') as agent_run:
        async for node in agent_run:
            # Each node represents a step in the agent's execution
            nodes.append(node)
    print(nodes)
    """
    [
        UserPromptNode(
            user_prompt='What is the capital of France?',
            instructions_functions=[],
            system_prompts=(),
            system_prompt_functions=[],
            system_prompt_dynamic_functions={},
        ),
        ModelRequestNode(
            request=ModelRequest(
                parts=[
                    UserPromptPart(
                        content='What is the capital of France?',
                        timestamp=datetime.datetime(...),
                    )
                ],
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        CallToolsNode(
            model_response=ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(input_tokens=56, output_tokens=7),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    """
    print(agent_run.result.output)
    #> The capital of France is Paris.
```

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

- The `AgentRun` is an async iterator that yields each node (`BaseNode` or `End`) in the flow.
- The run ends when an `End` node is returned.

#### Using `.next(...)` manually

You can also drive the iteration manually by passing the node you want to run next to the `AgentRun.next(...)` method. This allows you to inspect or modify the node before it executes or skip nodes based on your own logic, and to catch errors in `next()` more easily:

```python {title="agent_iter_next.py"}
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-5.2')


async def main():
    async with agent.iter('What is the capital of France?') as agent_run:
        node = agent_run.next_node  # (1)!

        all_nodes = [node]

        # Drive the iteration manually:
        while not isinstance(node, End):  # (2)!
            node = await agent_run.next(node)  # (3)!
            all_nodes.append(node)  # (4)!

        print(all_nodes)
        """
        [
            UserPromptNode(
                user_prompt='What is the capital of France?',
                instructions_functions=[],
                system_prompts=(),
                system_prompt_functions=[],
                system_prompt_dynamic_functions={},
            ),
            ModelRequestNode(
                request=ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='What is the capital of France?',
                            timestamp=datetime.datetime(...),
                        )
                    ],
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(input_tokens=56, output_tokens=7),
                    model_name='gpt-5.2',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                )
            ),
            End(data=FinalResult(output='The capital of France is Paris.')),
        ]
        """
```

1. We start by grabbing the first node that will be run in the agent's graph.
2. The agent run is finished once an `End` node has been produced; instances of `End` cannot be passed to `next`.
3. When you call `await agent_run.next(node)`, it executes that node in the agent's graph, updates the run's history, and returns the _next_ node to run.
4. You could also inspect or mutate the new `node` here as needed.

_(This example is complete, it can be run "as is" — you'll need to add `asyncio.run(main())` to run `main`)_

#### Accessing usage and final output

You can retrieve usage statistics (tokens, requests, etc.) at any time from the [`AgentRun`][pydantic_ai.agent.AgentRun] object via `agent_run.usage()`. This method returns a [`RunUsage`][pydantic_ai.usage.RunUsage] object containing the usage data.

Once the run finishes, `agent_run.result` becomes an [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] object containing the final output (and related metadata).

#### Streaming All Events and Output

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
                                f'[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}'
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

### Additional Configuration

#### Usage Limits

Pydantic AI offers a [`UsageLimits`][pydantic_ai.usage.UsageLimits] structure to help you limit your
usage (tokens, requests, and tool calls) on model runs.

You can apply these settings by passing the `usage_limits` argument to the `run{_sync,_stream}` functions.

Consider the following example, where we limit the number of response tokens:

```py
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-5')

result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with just the city.',
    usage_limits=UsageLimits(response_tokens_limit=10),
)
print(result_sync.output)
#> Rome
print(result_sync.usage())
#> RunUsage(input_tokens=62, output_tokens=1, requests=1)

try:
    result_sync = agent.run_sync(
        'What is the capital of Italy? Answer with a paragraph.',
        usage_limits=UsageLimits(response_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    #> Exceeded the output_tokens_limit of 10 (output_tokens=32)
```

Restricting the number of requests can be useful in preventing infinite loops or excessive tool calling:

```py
from typing_extensions import TypedDict

from pydantic_ai import Agent, ModelRetry, UsageLimitExceeded, UsageLimits


class NeverOutputType(TypedDict):
    """
    Never ever coerce data to this type.
    """

    never_use_this: str


agent = Agent(
    'anthropic:claude-sonnet-4-5',
    retries=3,
    output_type=NeverOutputType,
    system_prompt='Any time you get a response, call the `infinite_retry_tool` to produce another response.',
)


@agent.tool_plain(retries=5)  # (1)!
def infinite_retry_tool() -> int:
    raise ModelRetry('Please try again.')


try:
    result_sync = agent.run_sync(
        'Begin infinite retry loop!', usage_limits=UsageLimits(request_limit=3)  # (2)!
    )
except UsageLimitExceeded as e:
    print(e)
    #> The next request would exceed the request_limit of 3
```

1. This tool has the ability to retry 5 times before erroring, simulating a tool that might get stuck in a loop.
2. This run will error after 3 requests, preventing the infinite tool calling.

##### Capping tool calls

If you need a limit on the number of successful tool invocations within a single run, use `tool_calls_limit`:

```py
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

agent = Agent('anthropic:claude-sonnet-4-5')

@agent.tool_plain
def do_work() -> str:
    return 'ok'

try:
    # Allow at most one executed tool call in this run
    agent.run_sync('Please call the tool twice', usage_limits=UsageLimits(tool_calls_limit=1))
except UsageLimitExceeded as e:
    print(e)
    #> The next tool call(s) would exceed the tool_calls_limit of 1 (tool_calls=2).
```

!!! note
    - Usage limits are especially relevant if you've registered many tools. Use `request_limit` to bound the number of model turns, and `tool_calls_limit` to cap the number of successful tool executions within a run.
    - The `tool_calls_limit` is checked before executing tool calls. If the model returns parallel tool calls that would exceed the limit, no tools will be executed.

#### Model (Run) Settings

Pydantic AI offers a [`settings.ModelSettings`][pydantic_ai.settings.ModelSettings] structure to help you fine tune your requests.
This structure allows you to configure common parameters that influence the model's behavior, such as `temperature`, `max_tokens`,
`timeout`, and more.

There are three ways to apply these settings, with a clear precedence order:

1. **Model-level defaults** - Set when creating a model instance via the `settings` parameter. These serve as the base defaults for that model.
2. **Agent-level defaults** - Set during [`Agent`][pydantic_ai.agent.Agent] initialization via the `model_settings` argument. These are merged with model defaults, with agent settings taking precedence.
3. **Run-time overrides** - Passed to `run{_sync,_stream}` functions via the `model_settings` argument. These have the highest priority and are merged with the combined agent and model defaults.

For example, if you'd like to set the `temperature` setting to `0.0` to ensure less random behavior,
you can do the following:

```py
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel

# 1. Model-level defaults
model = OpenAIChatModel(
    'gpt-5.2',
    settings=ModelSettings(temperature=0.8, max_tokens=500)  # Base defaults
)

# 2. Agent-level defaults (overrides model defaults by merging)
agent = Agent(model, model_settings=ModelSettings(temperature=0.5))

# 3. Run-time overrides (highest priority)
result_sync = agent.run_sync(
    'What is the capital of Italy?',
    model_settings=ModelSettings(temperature=0.0)  # Final temperature: 0.0
)
print(result_sync.output)
#> The capital of Italy is Rome.
```

The final request uses `temperature=0.0` (run-time), `max_tokens=500` (from model), demonstrating how settings merge with run-time taking precedence.

!!! note "Model Settings Support"
    Model-level settings are supported by all concrete model implementations (OpenAI, Anthropic, Google, etc.). Wrapper models like [`FallbackModel`](models/overview.md#fallback-model), [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel], and [`InstrumentedModel`][pydantic_ai.models.instrumented.InstrumentedModel] don't have their own settings - they use the settings of their underlying models.

#### Run metadata

Run metadata lets you tag each agent execution with contextual details (for example, a tenant ID to filter traces and logs)
and read it after completion via [`AgentRun.metadata`][pydantic_ai.agent.AgentRun],
[`AgentRunResult.metadata`][pydantic_ai.agent.AgentRunResult], or
[`StreamedRunResult.metadata`][pydantic_ai.result.StreamedRunResult].
The resolved metadata is attached to the [`RunContext`][pydantic_ai.tools.RunContext] during the run and,
when instrumentation is enabled, added to the run span attributes for observability tools.

Configure metadata on an [`Agent`][pydantic_ai.agent.Agent] or pass it to a run.
Both accept either a static dictionary or a callable that receives the [`RunContext`][pydantic_ai.tools.RunContext].
Metadata is computed (if a callable) and applied when the run starts, then recomputed after a run ends successfully,
so it can include end-of-run values.
Agent-level metadata and per-run metadata are merged, with per-run values overriding agent-level ones.

```python {title="run_metadata.py"}
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class Deps:
    tenant: str


agent = Agent[Deps](
    'openai:gpt-5.2',
    deps_type=Deps,
    metadata=lambda ctx: {'tenant': ctx.deps.tenant},  # agent-level metadata
)

result = agent.run_sync(
    'What is the capital of France?',
    deps=Deps(tenant='tenant-123'),
    metadata=lambda ctx: {'num_requests': ctx.usage.requests},  # per-run metadata
)
print(result.output)
#> The capital of France is Paris.
print(result.metadata)
#> {'tenant': 'tenant-123', 'num_requests': 1}
```

#### Concurrency Limiting

You can limit the number of concurrent agent runs using the `max_concurrency` parameter.
This is useful when you want to prevent overwhelming external resources or enforce rate limits when running many agent instances in parallel.

```python {title="agent_concurrency.py"}
import asyncio

from pydantic_ai import Agent, ConcurrencyLimit

# Simple limit: allow up to 10 concurrent runs
agent = Agent('openai:gpt-5', max_concurrency=10)


# With backpressure: limit concurrent runs and queue depth
agent_with_backpressure = Agent(
    'openai:gpt-5',
    max_concurrency=ConcurrencyLimit(max_running=10, max_queued=100),
)


async def main():
    # These will be rate-limited to 10 concurrent runs
    results = await asyncio.gather(
        *[agent.run(f'Question {i}') for i in range(20)]
    )
    print(len(results))
    #> 20
```

When the concurrency limit is reached, additional calls to [`agent.run()`][pydantic_ai.agent.AbstractAgent.run] or [`agent.iter()`][pydantic_ai.agent.Agent.iter]
will wait until a slot becomes available. If you configure `max_queued` and the queue fills up,
a [`ConcurrencyLimitExceeded`][pydantic_ai.exceptions.ConcurrencyLimitExceeded] exception is raised.

When instrumentation is enabled, waiting operations appear as "waiting for concurrency" spans
with attributes showing queue depth and limits.

### Model specific settings

If you wish to further customize model behavior, you can use a subclass of [`ModelSettings`][pydantic_ai.settings.ModelSettings], like
[`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings], associated with your model of choice.

For example:

```py
from pydantic_ai import Agent, UnexpectedModelBehavior
from pydantic_ai.models.google import GoogleModelSettings

agent = Agent('google-gla:gemini-3-flash-preview')

try:
    result = agent.run_sync(
        'Write a list of 5 very rude things that I might say to the universe after stubbing my toe in the dark:',
        model_settings=GoogleModelSettings(
            temperature=0.0,  # general model settings can also be specified
            gemini_safety_settings=[
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_LOW_AND_ABOVE',
                },
            ],
        ),
    )
except UnexpectedModelBehavior as e:
    print(e)  # (1)!
    """
    Content filter 'SAFETY' triggered, body:
    <safety settings details>
    """
```

1. This error is raised because the safety thresholds were exceeded.

## Runs vs. Conversations

An agent **run** might represent an entire conversation — there's no limit to how many messages can be exchanged in a single run. However, a **conversation** might also be composed of multiple runs, especially if you need to maintain state between separate interactions or API calls.

Here's an example of a conversation comprised of multiple runs:

```python {title="conversation_example.py" hl_lines="13"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')
print(result1.output)
#> Albert Einstein was a German-born theoretical physicist.

# Second run, passing previous messages
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),  # (1)!
)
print(result2.output)
#> Albert Einstein's most famous equation is (E = mc^2).
```

1. Continue the conversation; without `message_history` the model would not know who "his" was referring to.

_(This example is complete, it can be run "as is")_

## Type safe by design {#static-type-checking}

Pydantic AI is designed to work well with static type checkers, like mypy and pyright.

!!! tip "Typing is (somewhat) optional"
    Pydantic AI is designed to make type checking as useful as possible for you if you choose to use it, but you don't have to use types everywhere all the time.

    That said, because Pydantic AI uses Pydantic, and Pydantic uses type hints as the definition for schema and validation, some types (specifically type hints on parameters to tools, and the `output_type` arguments to [`Agent`][pydantic_ai.Agent]) are used at runtime.

    We (the library developers) have messed up if type hints are confusing you more than helping you, if you find this, please create an [issue](https://github.com/pydantic/pydantic-ai/issues) explaining what's annoying you!

In particular, agents are generic in both the type of their dependencies and the type of the outputs they return, so you can use the type hints to ensure you're using the right types.

Consider the following script with type mistakes:

```python {title="type_mistakes.py" hl_lines="18 28"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class User:
    name: str


agent = Agent(
    'test',
    deps_type=User,  # (1)!
    output_type=bool,
)


@agent.system_prompt
def add_user_name(ctx: RunContext[str]) -> str:  # (2)!
    return f"The user's name is {ctx.deps}."


def foobar(x: bytes) -> None:
    pass


result = agent.run_sync('Does their name start with "A"?', deps=User('Anne'))
foobar(result.output)  # (3)!
```

1. The agent is defined as expecting an instance of `User` as `deps`.
2. But here `add_user_name` is defined as taking a `str` as the dependency, not a `User`.
3. Since the agent is defined as returning a `bool`, this will raise a type error since `foobar` expects `bytes`.

Running `mypy` on this will give the following output:

```bash
➤ uv run mypy type_mistakes.py
type_mistakes.py:18: error: Argument 1 to "system_prompt" of "Agent" has incompatible type "Callable[[RunContext[str]], str]"; expected "Callable[[RunContext[User]], str]"  [arg-type]
type_mistakes.py:28: error: Argument 1 to "foobar" has incompatible type "bool"; expected "bytes"  [arg-type]
Found 2 errors in 1 file (checked 1 source file)
```

Running `pyright` would identify the same issues.

## System Prompts

System prompts might seem simple at first glance since they're just strings (or sequences of strings that are concatenated), but crafting the right system prompt is key to getting the model to behave as you want.

!!! tip
    For most use cases, you should use `instructions` instead of "system prompts".

    If you know what you are doing though and want to preserve system prompt messages in the message history sent to the
    LLM in subsequent completions requests, you can achieve this using the `system_prompt` argument/decorator.

    See the section below on [Instructions](#instructions) for more information.

Generally, system prompts fall into two categories:

1. **Static system prompts**: These are known when writing the code and can be defined via the `system_prompt` parameter of the [`Agent` constructor][pydantic_ai.agent.Agent.__init__].
2. **Dynamic system prompts**: These depend in some way on context that isn't known until runtime, and should be defined via functions decorated with [`@agent.system_prompt`][pydantic_ai.agent.Agent.system_prompt].

You can add both to a single agent; they're appended in the order they're defined at runtime.

Here's an example using both types of system prompts:

```python {title="system_prompts.py"}
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5.2',
    deps_type=str,  # (1)!
    system_prompt="Use the customer's name while replying to them.",  # (2)!
)


@agent.system_prompt  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.system_prompt
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

1. The agent expects a string dependency.
2. Static system prompt defined at agent creation time.
3. Dynamic system prompt defined via a decorator with [`RunContext`][pydantic_ai.tools.RunContext], this is called just after `run_sync`, not when the agent is created, so can benefit from runtime information like the dependencies used on that run.
4. Another dynamic system prompt, system prompts don't have to have the `RunContext` parameter.

_(This example is complete, it can be run "as is")_

## Instructions

Instructions are similar to system prompts. The main difference is that when an explicit `message_history` is provided
in a call to `Agent.run` and similar methods, _instructions_ from any existing messages in the history are not included
in the request to the model — only the instructions of the _current_ agent are included.

You should use:

- `instructions` when you want your request to the model to only include system prompts for the _current_ agent
- `system_prompt` when you want your request to the model to _retain_ the system prompts used in previous requests (possibly made using other agents)

In general, we recommend using `instructions` instead of `system_prompt` unless you have a specific reason to use `system_prompt`.

Instructions, like system prompts, can be specified at different times:

1. **Static instructions**: These are known when writing the code and can be defined via the `instructions` parameter of the [`Agent` constructor][pydantic_ai.agent.Agent.__init__].
2. **Dynamic instructions**: These rely on context that is only available at runtime and should be defined using functions decorated with [`@agent.instructions`][pydantic_ai.agent.Agent.instructions]. Unlike dynamic system prompts, which may be reused when `message_history` is present, dynamic instructions are always reevaluated.
3. **Runtime instructions*: These are additional instructions for a specific run that can be passed to one of the [run methods](#running-agents) using the `instructions` argument.

All three types of instructions can be added to a single agent, and they are appended in the order they are defined at runtime.

Here's an example using a static instruction as well as dynamic instructions:

```python {title="instructions.py"}
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5.2',
    deps_type=str,  # (1)!
    instructions="Use the customer's name while replying to them.",  # (2)!
)


@agent.instructions  # (3)!
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:  # (4)!
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

1. The agent expects a string dependency.
2. Static instructions defined at agent creation time.
3. Dynamic instructions defined via a decorator with [`RunContext`][pydantic_ai.tools.RunContext],
   this is called just after `run_sync`, not when the agent is created, so can benefit from runtime
   information like the dependencies used on that run.
4. Another dynamic instruction, instructions don't have to have the `RunContext` parameter.

_(This example is complete, it can be run "as is")_

Note that returning an empty string will result in no instruction message added.

## Reflection and self-correction

Validation errors from both function tool parameter validation and [structured output validation](output.md#structured-output) can be passed back to the model with a request to retry.

You can also raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] from within a [tool](tools.md) or [output function](output.md#output-functions) to tell the model it should retry generating a response.

- The default retry count is **1** but can be altered for the [entire agent][pydantic_ai.agent.Agent.__init__], a [specific tool][pydantic_ai.agent.Agent.tool], or [outputs][pydantic_ai.agent.Agent.__init__].
- You can access the current retry count from within a tool, output validator, or output function via [`ctx.retry`][pydantic_ai.tools.RunContext.retry].

Here's an example:

```python {title="tool_retry.py"}
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, ModelRetry

from fake_database import DatabaseConn


class ChatResult(BaseModel):
    user_id: int
    message: str


agent = Agent(
    'openai:gpt-5.2',
    deps_type=DatabaseConn,
    output_type=ChatResult,
)


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[DatabaseConn], name: str) -> int:
    """Get a user's ID from their full name."""
    print(name)
    #> John
    #> John Doe
    user_id = ctx.deps.users.get(name=name)
    if user_id is None:
        raise ModelRetry(
            f'No user found with name {name!r}, remember to provide their full name'
        )
    return user_id


result = agent.run_sync(
    'Send a message to John Doe asking for coffee next week', deps=DatabaseConn()
)
print(result.output)
"""
user_id=123 message='Hello John, would you be free for coffee sometime next week? Let me know what works for you!'
"""
```

## Debugging and Monitoring

Agents require a different approach to observability than traditional software. With traditional web endpoints or data pipelines, you can largely predict behavior by reading the code. With agents, this is much harder. The model's decisions are stochastic, and that stochasticity compounds through the agentic loop as the agent reasons, calls tools, observes results, and reasons again. You need to actually see what happened.

This means setting up your application to record what's happening in a way you can review afterward, both during development (to understand and iterate) and in production (to debug issues and monitor behavior). The ergonomics matter too: a plaintext dump of everything that happened isn't a practical way to review agent behavior, even during development. You want tooling that lets you step through each decision and tool call interactively.

We recommend [Pydantic Logfire](https://logfire.pydantic.dev/docs/), which has been designed with Pydantic AI workflows in mind.

### Tracing with Logfire

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
```

With Logfire instrumentation enabled, every agent run creates a detailed trace showing:

- **Messages exchanged** with the model (system, user, assistant)
- **Tool calls** including arguments and return values
- **Token usage** per request and cumulative
- **Latency** for each operation
- **Errors** with full context

This visibility is invaluable for:

- Understanding why an agent made a specific decision
- Debugging unexpected behavior
- Optimizing performance and costs
- Monitoring production deployments

### Systematic Testing with Evals

For systematic evaluation of agent behavior beyond runtime debugging, [Pydantic Evals](evals.md) provides a code-first framework for testing AI systems:

```python {test="skip" lint="skip" format="skip"}
from pydantic_evals import Case, Dataset

dataset = Dataset(
    cases=[
        Case(name='capital_question', inputs='What is the capital of France?', expected_output='Paris'),
    ]
)
report = dataset.evaluate_sync(my_agent_function)
```

Evals let you define test cases, run them against your agent, and score the results. When combined with Logfire, evaluation results appear in the web UI for visualization and comparison across runs. See the [Logfire integration guide](evals/how-to/logfire-integration.md) for setup.

### Using Other Backends

Pydantic AI's instrumentation is built on [OpenTelemetry](https://opentelemetry.io/), so you can send traces to any compatible backend. Even if you use the Logfire SDK for its convenience, you can configure it to send data to other backends. See [alternative backends](logfire.md#using-opentelemetry) for setup instructions.

[Full Logfire integration guide →](logfire.md)

## Model errors

If models behave unexpectedly (e.g., the retry limit is exceeded, or their API returns `503`), agent runs will raise [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior].

In these cases, [`capture_run_messages`][pydantic_ai.capture_run_messages] can be used to access the messages exchanged during the run to help diagnose the issue.

```python {title="agent_model_errors.py"}
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

agent = Agent('openai:gpt-5.2')


@agent.tool_plain
def calc_volume(size: int) -> int:  # (1)!
    if size == 42:
        return size**3
    else:
        raise ModelRetry('Please try again.')


with capture_run_messages() as messages:  # (2)!
    try:
        result = agent.run_sync('Please get me the volume of a box with size 6.')
    except UnexpectedModelBehavior as e:
        print('An error occurred:', e)
        #> An error occurred: Tool 'calc_volume' exceeded max retries count of 1
        print('cause:', repr(e.__cause__))
        #> cause: ModelRetry('Please try again.')
        print('messages:', messages)
        """
        messages:
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Please get me the volume of a box with size 6.',
                        timestamp=datetime.datetime(...),
                    )
                ],
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(input_tokens=62, output_tokens=4),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
            ModelRequest(
                parts=[
                    RetryPromptPart(
                        content='Please try again.',
                        tool_name='calc_volume',
                        tool_call_id='pyd_ai_tool_call_id',
                        timestamp=datetime.datetime(...),
                    )
                ],
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(input_tokens=72, output_tokens=8),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
            ),
        ]
        """
    else:
        print(result.output)
```

1. Define a tool that will raise `ModelRetry` repeatedly in this case.
2. [`capture_run_messages`][pydantic_ai.capture_run_messages] is used to capture the messages exchanged during the run.

_(This example is complete, it can be run "as is")_

!!! note
    If you call [`run`][pydantic_ai.agent.AbstractAgent.run], [`run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], or [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] more than once within a single `capture_run_messages` context, `messages` will represent the messages exchanged during the first call only.
