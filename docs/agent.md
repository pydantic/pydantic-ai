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
| [Dependency type constraint](dependencies.md)             | Dynamic instruction functions, tools, and output functions may all use dependencies when they're run.          |
| [LLM model](api/models/base.md)                           | Optional default LLM model associated with the agent. Can also be specified when running the agent.       |
| [Model Settings](#additional-configuration)               | Optional default model settings to help fine tune requests. Can also be specified when running the agent. |
| [Capabilities](capabilities/overview.md)                           | Reusable bundles of tools, hooks, instructions, and model settings that extend agent behavior.            |

While each of these can be configured individually, [capabilities](capabilities/overview.md) let you bundle related behavior into reusable units that are easier to compose, share, and [load from configuration files](agent-spec.md).

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
4. [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] — an async context manager which yields an async iterator over [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] ending with an [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] containing the final run result.
5. [`agent.iter()`][pydantic_ai.agent.Agent.iter] — a context manager which returns an [`AgentRun`][pydantic_ai.agent.AgentRun], an async iterable over the nodes of the agent's underlying [`Graph`][pydantic_graph.graph_builder.Graph].

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

    collected: list[AgentStreamEvent | AgentRunResultEvent] = []
    async with agent.run_stream_events('What is the capital of Mexico?') as events:
        async for event in events:
            collected.append(event)
    print(collected)
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

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

You can also pass messages from previous runs to continue a conversation or provide context, as described in [Messages and Chat History](message-history.md).

### Streaming Events and Final Output

As shown in the example above, [`run_stream()`][pydantic_ai.agent.AbstractAgent.run_stream] makes it easy to stream the agent's final output as it comes in.
It also takes an optional `event_stream_handler` argument that you can use to gain insight into what is happening during the run before the final output is produced.
During a realtime session, the same handler stream can also contain realtime-only [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members.

The example below shows how to stream events and text output. You can also [stream structured output](output.md#streaming-structured-output).

!!! note
    The `run_stream()` and `run_stream_sync()` methods will consider the first output that matches the [output type](output.md#structured-output) (which could be text, an [output tool](output.md#tool-output) call, or a [deferred](deferred-tools.md) tool call) to be the final output of the agent run, even when the model generates (additional) tool calls after this "final" output.

	These "dangling" tool calls will not be executed unless the agent's [`end_strategy`][pydantic_ai.agent.Agent.end_strategy] is set to `'graceful'` or `'exhaustive'`, and even then their results will not be sent back to the model as the agent run will already be considered completed. In short, if the model returns both tool calls and text, and the agent's output type is `str`, **the tool calls will not run** in streaming mode with the default setting.

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

### Streaming All Events

Like `agent.run_stream()`, [`agent.run()`][pydantic_ai.agent.AbstractAgent.run_stream] takes an optional `event_stream_handler`
argument that lets you stream all events from the model's streaming response and the agent's execution of tools.
Unlike `run_stream()`, it always runs the agent graph to completion even if text was received ahead of tool calls that looked like it could've been the final result.
During a realtime session, an event stream handler can also receive realtime-only [`RealtimeEvent`][pydantic_ai.realtime.RealtimeEvent] members.

For convenience, a [`agent.run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] method is also available as a wrapper around `run(event_stream_handler=...)`. It is an async context manager that yields an async iterator over [`AgentStreamEvent`s][pydantic_ai.messages.AgentStreamEvent] ending with an [`AgentRunResultEvent`][pydantic_ai.run.AgentRunResultEvent] carrying the final run result.

!!! note
    As they return raw events as they come in, the `run_stream_events()` and `run(event_stream_handler=...)` methods require you to piece together the streamed text and structured output yourself from the `PartStartEvent` and subsequent `PartDeltaEvent`s.

    To get the best of both worlds, at the expense of some additional complexity, you can use [`agent.iter()`][pydantic_ai.agent.AbstractAgent.iter] as described in the next section, which lets you [iterate over the agent graph](#iterating-over-an-agents-graph) and [stream both events and output](#streaming-all-events-and-output) at every step. See [Making structured responses appear faster](output.md#making-structured-responses-appear-faster) for a focused example using validated structured output.

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

### Iterating Over an Agent's Graph

Under the hood, each `Agent` in Pydantic AI uses **pydantic-graph** to manage its execution flow. **pydantic-graph** is a generic, type-centric library for building and running finite state machines in Python. It doesn't actually depend on Pydantic AI — you can use it standalone for workflows that have nothing to do with GenAI — but Pydantic AI makes use of it to orchestrate the handling of model requests and model responses in an agent's run.

In many scenarios, you don't need to worry about pydantic-graph at all; calling `agent.run(...)` simply traverses the underlying graph from start to finish. However, if you need deeper insight or control — for example to inject your own logic at specific stages — Pydantic AI exposes the lower-level iteration process via [`Agent.iter`][pydantic_ai.agent.Agent.iter]. This method returns an [`AgentRun`][pydantic_ai.agent.AgentRun], which you can async-iterate over, or manually drive node-by-node via the [`next`][pydantic_ai.agent.AgentRun.next] method. Once the agent's graph returns an [`End`][pydantic_graph.basenode.End], you have the final result along with a detailed history of all steps.

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
                conversation_id='...',
            )
        ),
        CallToolsNode(
            model_response=ModelResponse(
                parts=[TextPart(content='The capital of France is Paris.')],
                usage=RequestUsage(
                    cost=Decimal('0.000196'), input_tokens=56, output_tokens=7
                ),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
                conversation_id='...',
            )
        ),
        End(data=FinalResult(output='The capital of France is Paris.')),
    ]
    """
    print(agent_run.result.output)
    #> The capital of France is Paris.
```

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

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
                    conversation_id='...',
                )
            ),
            CallToolsNode(
                model_response=ModelResponse(
                    parts=[TextPart(content='The capital of France is Paris.')],
                    usage=RequestUsage(
                        cost=Decimal('0.000196'), input_tokens=56, output_tokens=7
                    ),
                    model_name='gpt-5.2',
                    timestamp=datetime.datetime(...),
                    run_id='...',
                    conversation_id='...',
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

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

#### Accessing usage and final output

You can retrieve usage statistics (tokens, requests, etc.) at any time from the [`AgentRun`][pydantic_ai.agent.AgentRun] object via `agent_run.usage`. This property returns a [`RunUsage`][pydantic_ai.usage.RunUsage] object containing the usage data.

[`RunUsage.cost`][pydantic_ai.usage.RunUsage.cost] additionally holds a best-effort estimate of the run's total cost in USD, calculated from each request's usage with [genai-prices](https://github.com/pydantic/genai-prices). Requests to models or providers that genai-prices doesn't have pricing data for don't contribute to the total.

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

### Cancelling a Run

A run in flight can be cancelled entirely -- e.g. when a user hits a "stop" button. Create a [`CancellationToken`][pydantic_ai.CancellationToken], pass it to the run, and call `cancel()` from the stop handler. Cancellation raises [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] with the completed message history and usage so you can persist and resume the conversation:

```python {title="run_cancel.py"}
import asyncio

from pydantic_ai import Agent, CancellationToken, RunCancelled

agent = Agent('test')
tool_started = asyncio.Event()


@agent.tool_plain
async def slow_lookup() -> str:
    tool_started.set()
    await asyncio.sleep(10)
    return 'result'


async def main():
    token = CancellationToken()
    run = asyncio.create_task(
        agent.run('Look something up', cancellation_token=token)
    )
    await tool_started.wait()
    token.cancel()  # (1)!

    try:
        await run
    except RunCancelled as exc:
        messages = exc.all_messages()
        print(f'Cancelled after {len(messages)} messages')
        #> Cancelled after 3 messages
        await agent.run(message_history=messages)  # (2)!
```

1. `cancel()` is idempotent and thread-safe. One token may govern multiple concurrent runs, cancelling all of them. A token is single-use: once cancelled it stays cancelled, and passing an already-cancelled token to a run prevents that run from starting (which also closes the "cancel raced ahead of the run" gap). So mint a fresh token per run or per stop gesture -- reusing one token across a session would cancel every run after the first before it starts.
2. [`RunCancelled.all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages] contains everything completed before cancellation, including completed tool results. Any dangling tool call is [repaired automatically](message-history.md#making-histories-provider-valid) when the history is resumed.

[UI adapter](ui/overview.md) users can persist this resumable history with the `on_cancel` callback.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

[`agent.run_sync()`][pydantic_ai.agent.AbstractAgent.run_sync] accepts the same token. Calling `token.cancel()` from another thread is the only way to interrupt a synchronous run while it is blocked.

!!! note "Which mechanism, and which exception"
    A [`CancellationToken`][pydantic_ai.CancellationToken] is the one to reach for by default -- it's the only surface that works from outside the run, from another thread, and against `run_sync()`, and one token can govern several runs at once. The others exist for where a token can't reach:

    | Where you are when you cancel | Use | Run ends with |
    | --- | --- | --- |
    | Outside the run (a "stop" button, another thread) | [`CancellationToken`][pydantic_ai.CancellationToken] | [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] |
    | Inside a tool, `event_stream_handler`, or capability hook | [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel] | [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] |
    | Consuming [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events] | [`AgentRunEvents.cancel()`][pydantic_ai.agent.AgentRunEvents.cancel] on the yielded handle | [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] |
    | Driving the graph yourself via [`agent.iter()`][pydantic_ai.agent.Agent.iter] | [`AgentRun.cancel()`][pydantic_ai.run.AgentRun.cancel] | [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] |
    | The environment cancelled you (`asyncio.timeout()`, a [`TaskGroup`][asyncio.TaskGroup], shutdown) | *(you don't call anything)* | [`CancelledError`][asyncio.CancelledError] |

    The first four are **first-party**: Pydantic AI stops the run itself and raises `RunCancelled`, an ordinary catchable exception carrying the resumable history. The last is **external**: the `CancelledError` keeps propagating unchanged -- so `asyncio.timeout()` still raises `TimeoutError`, a `TaskGroup` still tears down, and Temporal still ends the workflow *Cancelled* -- with the same history *attached* for [`RunCancelled.from_cancellation()`][pydantic_ai.exceptions.RunCancelled.from_cancellation]. Pydantic AI can't turn an external `CancelledError` into `RunCancelled` without breaking those semantics; that's why cancellation has two kinds, covered next.

When the surrounding environment cancels the run -- for example through `asyncio.timeout()`, a [`TaskGroup`][asyncio.TaskGroup], or application shutdown -- the [`CancelledError`][asyncio.CancelledError] remains unchanged. [`RunCancelled.from_cancellation()`][pydantic_ai.exceptions.RunCancelled.from_cancellation] provides the attached run state:

```python {title="run_external_cancel.py"}
import asyncio

from pydantic_ai import Agent, RunCancelled

agent = Agent('test')
tool_started = asyncio.Event()


@agent.tool_plain
async def slow_lookup() -> str:
    tool_started.set()
    await asyncio.sleep(10)
    return 'result'


async def main():
    task = asyncio.create_task(agent.run('Look something up'))
    await tool_started.wait()
    task.cancel()  # (1)!

    try:
        await task
    except asyncio.CancelledError as exc:
        cancelled = RunCancelled.from_cancellation(exc)  # (2)!
        assert cancelled is not None
        messages = cancelled.all_messages()
        print(f'Cancelled after {len(messages)} messages')
        #> Cancelled after 3 messages
        await agent.run(message_history=messages)  # (3)!
```

1. This demonstrates cancellation imposed by the surrounding asyncio environment. For application stop gestures, prefer a `CancellationToken`.
2. External cancellation is never converted: `asyncio.timeout()`, [`TaskGroup`][asyncio.TaskGroup], and [Temporal](durable_execution/temporal.md) cancellation semantics are preserved. The run state rides along on the original `CancelledError`.
3. [`RunCancelled.all_messages()`][pydantic_ai.exceptions.RunCancelled.all_messages] contains everything completed before cancellation, including completed tool results. Any dangling tool call is [repaired automatically](message-history.md#making-histories-provider-valid) when the history is resumed.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

On Python 3.10, asyncio recreates `CancelledError` across an `await task` boundary, but chains the original exception -- carrying the attached run state -- via `__context__`, which `from_cancellation()` traverses. The chain is attached only to the first `await` of the cancelled task, so later awaits of the same task see an unchained exception; [`capture_run_messages()`][pydantic_ai.agent.capture_run_messages] is the fallback when only history is needed.

When consuming [`run_stream_events()`][pydantic_ai.agent.AbstractAgent.run_stream_events], the yielded [`AgentRunEvents`][pydantic_ai.agent.AgentRunEvents] handle offers a first-party alternative that needs no task juggling: [`AgentRunEvents.cancel()`][pydantic_ai.agent.AgentRunEvents.cancel] is safe to call from another task (e.g. a UI's "stop" handler) and surfaces as `RunCancelled` on continued iteration:

```python {title="run_cancel_stream_events.py"}
from pydantic_ai import Agent, RunCancelled

agent = Agent('test')


async def main():
    async with agent.run_stream_events('Write a long essay about Python') as events:
        try:
            async for _event in events:
                events.cancel()  # (1)!
        except RunCancelled as exc:
            print(f'Cancelled after {len(exc.all_messages())} messages')
            #> Cancelled after 2 messages
```

1. Idempotent, a no-op once the run has finished, and callable before the first iteration to prevent the run from starting at all.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

Externally cancelling the consuming task works here too: the background run tears down, the propagating `CancelledError` carries the run state for `from_cancellation()`, and the handle's `all_messages()` and `usage` remain accessible afterwards.

To request cancellation from a tool, an `event_stream_handler`, or a capability hook, call [`RunContext.cancel()`][pydantic_ai.tools.RunContext.cancel]. This requests first-party cancellation, so the run ends with [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] rather than an external `CancelledError`. `cancel()` itself returns normally — the cancellation is delivered at the calling code's next `await`, and the tool's return value is discarded — so a tool can still run cleanup after requesting it:

```python {title="run_cancel_from_tool.py"}
from pydantic_ai import Agent, RunCancelled, RunContext

agent = Agent('test')


@agent.tool
async def stop(ctx: RunContext) -> str:
    ctx.cancel()
    return 'discarded'  # cancel() returned; this value is never sent to the model


async def main():
    try:
        await agent.run('Stop now')
    except RunCancelled as exc:
        print(f'Cancelled after {len(exc.all_messages())} messages')
        #> Cancelled after 2 messages
```

!!! note "Cancellation is cooperative"
    Pydantic AI requests cancellation of in-flight work, discards results that arrive after cancellation, and closes the resources it owns. Async tools receive `CancelledError` at a suspension point. A synchronous (`def`) tool runs in a worker thread, which Python cannot safely terminate; depending on the run mode, cancellation may wait for the worker or let it finish in the background. Either way, its result is discarded, but any side effects are not rolled back. Cancelling provider-side model generation is best-effort and depends on the provider.

You may not control which way cancellation will arrive: a caller wraps `agent.run()` in a task for a stop gesture, while a tool -- perhaps from another library -- calls `ctx.cancel()` internally. Handle each on its own terms -- consume the first-party `RunCancelled`, but let an external `CancelledError` keep propagating so timeouts and task groups still tear down correctly, capturing its state first if you need it:

```python {title="run_cancel_either_way.py"}
import asyncio

from pydantic_ai import Agent, RunCancelled, RunContext

agent = Agent('test')


@agent.tool
async def imported_tool(ctx: RunContext) -> str:
    ctx.cancel()  # (1)!
    return 'discarded'


async def main():
    task = asyncio.create_task(agent.run('Go'))
    try:
        await task
    except RunCancelled as exc:  # (2)!
        print(f'Cancelled after {len(exc.all_messages())} messages')
        #> Cancelled after 2 messages
    except asyncio.CancelledError as exc:  # (3)!
        cancelled = RunCancelled.from_cancellation(exc)
        if cancelled is not None:
            ...  # persist cancelled.all_messages() before re-raising
        raise
```

1. Here the tool cancels first-party, so `await task` raises `RunCancelled`. Had a stop button called `task.cancel()` instead, `await task` would raise `CancelledError` and the second handler would run.
2. First-party cancellation is a `RunCancelled` you can consume: the run stopped because your own code asked it to, so returning normally is fine.
3. External cancellation stays `CancelledError`, and a stop button's `task.cancel()` is indistinguishable from a timeout or a [`TaskGroup`][asyncio.TaskGroup] tearing down -- so re-raise it (swallowing it would break those teardowns), reaching for [`from_cancellation()`][pydantic_ai.exceptions.RunCancelled.from_cancellation] only to capture the partial state first. It returns `None` when nothing is attached, e.g. an application shutdown unrelated to this run.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

!!! note "Why two exception types?"
    Cancellation can originate from two different places, and only one of them is Pydantic AI's to name:

    - **Your application** decides to stop the run, through one of the dedicated cancellation methods. Pydantic AI issued that cancellation itself, so it can consume it before asyncio interprets it and raise `RunCancelled` instead: the run ends with an ordinary, catchable application error.
    - **The asyncio environment** cancels the task the run happens to be on: `asyncio.Task.cancel()`, `asyncio.timeout()` expiring, a [`TaskGroup`][asyncio.TaskGroup] tearing down after a sibling failed, a server shutting down, workflow cancellation under [durable execution](durable_execution/overview.md). All of these deliver the very same `CancelledError` signal, so Pydantic AI cannot tell a stop button from a timeout -- and the exception's type is load-bearing for everything built on it: `asyncio.timeout()` only produces `TimeoutError`, a `TaskGroup` only treats the task as cleanly cancelled, and Temporal only ends the workflow as *Cancelled* if `CancelledError` itself keeps propagating. Raising `RunCancelled` in its place would silently break each of those. So the run state is *attached to* the propagating `CancelledError` for [`from_cancellation()`][pydantic_ai.exceptions.RunCancelled.from_cancellation], rather than replacing it.

Cancellation is terminal: capability hooks may observe it and clean up, but cannot recover the run to success — on Python 3.11+ this holds even if user code absorbs the delivered cancellation; on Python 3.10 it is best-effort. When first-party and external cancellation race, external cancellation wins. On Python 3.10, that race cannot be distinguished, so first-party cancellation wins instead.

For fine-grained control over the agent graph, call [`AgentRun.cancel()`][pydantic_ai.run.AgentRun.cancel] on the handle returned by [`agent.iter()`][pydantic_ai.agent.Agent.iter]:

```python {title="run_cancel_iter.py"}
from pydantic_ai import Agent, RunCancelled

agent = Agent('test')


async def main():
    try:
        async with agent.iter('Write a long essay about Python') as agent_run:
            async for node in agent_run:
                if Agent.is_call_tools_node(node):
                    agent_run.cancel()  # (1)!
    except RunCancelled as exc:
        print(f'Cancelled after {len(exc.all_messages())} messages')  # (2)!
        #> Cancelled after 2 messages
```

1. `AgentRun.cancel()` is safe to call from another task and is a no-op once the run has finished.
2. Inside the `agent.iter()` block, cancellation surfaces as `asyncio.CancelledError`; after the context exits, first-party cancellation raises `RunCancelled` with a detached state snapshot.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

#### Message History After Cancellation

When a stream is cancelled mid-generation, the response is recorded with `state='interrupted'` in the message history. The history includes any partial content that was received before cancellation:

```python {title="stream_cancel_history.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2')


async def main():
    async with agent.run_stream('Tell me about Python') as result:
        async for text in result.stream_text(delta=True):
            break
        await result.cancel()

    messages = result.all_messages()  # (1)!
    print(messages[-1].state)  # (2)!
    #> interrupted
```

1. The message history includes the interrupted response with any partial content that was received before cancellation.
2. The interrupted response state lets your application decide whether to keep, inspect, or discard the partial response before reusing the history.

_(To run this example, ensure `asyncio` is imported and add `asyncio.run(main())`; no other changes are needed.)_

!!! note "Reusing interrupted history"
    Interrupted history can be passed directly into another run. Before the next model request, Pydantic AI [repairs the transcript](message-history.md#making-histories-provider-valid): any tool call that never received a result — including one whose arguments were cut off mid-stream — is answered with a synthesized [`ToolReturnPart`][pydantic_ai.messages.ToolReturnPart] telling the model it was interrupted.

!!! info "Usage tracking for cancelled streams"
    Token usage reported by `usage` after cancellation is partial and provider-dependent. Pydantic AI stops pulling from the stream immediately, so final usage events may never arrive; some provider SDKs may also continue generation server-side after the local stream is closed. Do not rely on cancelled-stream usage for cost-critical accounting.
    For OpenAI chat completions, [`openai_continuous_usage_stats`][pydantic_ai.models.openai.OpenAIChatModelSettings] can improve in-stream usage reporting by requesting cumulative usage data with each chunk, but cancelled-stream usage is still best-effort.

#### Cancellation and sub-agents

Cancellation is **run-scoped**: `cancel()` cancels the run its `RunContext` belongs to, and a `CancellationToken` cancels the runs it's attached to. This matters when you use [agent delegation](multi-agent-applications.md#agent-delegation) — a tool that runs another agent with `await sub_agent.run(...)`:

- **A sub-agent cancelling itself does not cancel the parent** — when it's `await`ed inside a tool body. If the sub-agent (or one of its tools) calls `ctx.cancel()`, that cancels the *sub-agent's* run. The delegate tool sees a [`RunCancelled`][pydantic_ai.exceptions.RunCancelled], which — if it isn't caught — surfaces to the parent as a *failed tool return* the parent's model can react to, not as a cancellation of the parent run. This isolation is specific to tool bodies: a sub-agent `await`ed from an `event_stream_handler`, an [output validator](output.md#output-validator-functions), or a [capability](capabilities/overview.md) hook runs directly on the parent's task, so its `cancel()` *does* surface as the parent's own `RunCancelled`.
- **To cancel the parent too, opt in from the delegate tool** by catching `RunCancelled` and calling `ctx.cancel()` on the parent's context (or re-raising a different error).
- **To cancel a whole tree of runs at once, share one `CancellationToken`** across the parent and its sub-agents — cancelling it stops all of them. A parent cancelled this way (or by an external `asyncio.CancelledError`) also tears down any sub-agent run it is `await`ing inline, since they run on the same task.

### Additional Configuration

#### Usage Limits

Pydantic AI offers a [`UsageLimits`][pydantic_ai.usage.UsageLimits] structure to help you limit your
usage (tokens, requests, tool calls, and cost) on model runs.

You can apply these settings by passing the `usage_limits` argument to the `run{_sync,_stream}` functions.

Consider the following example, where we limit the number of output tokens:

```py
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-6')

result_sync = agent.run_sync(
    'What is the capital of Italy? Answer with just the city.',
    usage_limits=UsageLimits(output_tokens_limit=10),
)
print(result_sync.output)
#> Rome
print(result_sync.usage)
#> RunUsage(cost=Decimal('0.000201'), input_tokens=62, output_tokens=1, requests=1)

try:
    result_sync = agent.run_sync(
        'What is the capital of Italy? Answer with a paragraph.',
        usage_limits=UsageLimits(output_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    """
    Exceeded the output_tokens_limit of 10 (output_tokens=32). Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
    """
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
    'anthropic:claude-sonnet-4-6',
    retries={'tools': 3},
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
    """
    The next request would exceed the request_limit of 3. Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
    """
```

1. This tool has the ability to retry 5 times before erroring, simulating a tool that might get stuck in a loop.
2. This run will error after 3 requests, preventing the infinite tool calling.

##### Capping tool calls

If you need a limit on the number of successful tool invocations within a single run, use `tool_calls_limit`:

```py
from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

agent = Agent('anthropic:claude-sonnet-4-6')

@agent.tool_plain
def do_work() -> str:
    return 'ok'

try:
    # Allow at most one executed tool call in this run
    agent.run_sync('Please call the tool twice', usage_limits=UsageLimits(tool_calls_limit=1))
except UsageLimitExceeded as e:
    print(e)
    """
    The next tool call(s) would exceed the tool_calls_limit of 1 (tool_calls=2). Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
    """
```

!!! note
    - Usage limits are especially relevant if you've registered many tools. Use `request_limit` to bound the number of model turns, and `tool_calls_limit` to cap the number of successful tool executions within a run.
    - The `tool_calls_limit` is checked before executing tool calls. If the model returns parallel tool calls that would exceed the limit, no tools will be executed.

Tools and [capabilities](capabilities/overview.md) can read the run's limits from [`ctx.usage_limits`][pydantic_ai.tools.RunContext.usage_limits] (alongside [`ctx.usage`][pydantic_ai.tools.RunContext.usage] for usage so far), so a budget-aware tool or capability can disclose or adapt to the remaining budget without being configured with a duplicate copy of the limits. It reflects what the run is already enforcing and is read-only by convention.

##### Limiting per-request input size

The token limits above are cumulative across the whole run. To instead cap the size of any single request's input (the context window actually sent to the model), use `per_request_input_tokens_limit`. This is useful when prompt caching makes cumulative input a poor proxy for cost: re-sent cached prefixes are cheap, while a single oversized context is what degrades model performance and drives cache-miss cost.

```py
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-6')

try:
    agent.run_sync(
        'What is the capital of Italy? Answer with just the city.',
        usage_limits=UsageLimits(per_request_input_tokens_limit=10),
    )
except UsageLimitExceeded as e:
    print(e)
    """
    Exceeded the per_request_input_tokens_limit of 10 (request_input_tokens=62). Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
    """
```

By default the limit is checked against the provider-reported input tokens after the response, so the oversized request is still sent and billed (matching `input_tokens_limit`). Set `count_tokens_before_request=True` to run a token-counting pass and enforce the limit before the request is sent.

##### Capping run cost

Token limits are a proxy for spend: the same token count costs wildly different amounts on different models, so a limit tuned for one model is wrong for the next. To bound the actual dollars a run can spend, use [`cost_limit`][pydantic_ai.usage.UsageLimits.cost_limit], which caps [`RunUsage.cost`][pydantic_ai.usage.RunUsage.cost] in USD:

```py
from decimal import Decimal

from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits

agent = Agent('anthropic:claude-sonnet-4-6')

try:
    agent.run_sync(
        'What is the capital of Italy? Answer with just the city.',
        usage_limits=UsageLimits(cost_limit=Decimal('0.0001')),
    )
except UsageLimitExceeded as e:
    print(e)
    """
    Exceeded the `cost_limit` of 0.0001 (`usage.cost`=Decimal('0.000201')). Consider raising the limit, or see the docs on usage limits for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits
    """
```

Like `output_tokens_limit`, this is checked after each response, since a response's output cost isn't known until it arrives. Setting `count_tokens_before_request=True` additionally prices the counted input tokens and rejects the request up front when that lower bound alone exceeds the limit.

!!! note
    Cost is best-effort: it's `None` for models and providers [genai-prices](https://github.com/pydantic/genai-prices) has no pricing data for. With a [`cost_limit`][pydantic_ai.usage.UsageLimits.cost_limit], a run that could not be priced at all emits [`CostNotFoundWarning`][pydantic_ai.exceptions.CostNotFoundWarning] rather than being silently unconstrained; an unexpected pricing failure emits [`CostCalculationFailedWarning`][pydantic_ai.exceptions.CostCalculationFailedWarning]. Don't rely on `cost_limit` as a hard billing guarantee — pair it with [`request_limit`][pydantic_ai.usage.UsageLimits.request_limit] or your provider's own spend controls.

#### Model (Run) Settings

Pydantic AI offers a [`settings.ModelSettings`][pydantic_ai.settings.ModelSettings] structure to help you fine tune your requests.
This structure allows you to configure common parameters that influence the model's behavior, such as `temperature`, `max_tokens`, `top_k`,
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

##### Dynamic model settings

Both agent-level and run-level `model_settings` accept a callable that receives a
[`RunContext`][pydantic_ai.tools.RunContext] and returns [`ModelSettings`][pydantic_ai.settings.ModelSettings].
The callable is invoked before each model request, so settings can vary per step.
The current resolved settings so far are available via `ctx.model_settings` inside the callable.

Settings are resolved in layers, each merged on top of the previous:

1. **Model defaults** (`model.settings`)
2. **Agent-level** (`Agent(model_settings=...)`)
3. **Capability-level** (e.g. from [`Thinking()`][pydantic_ai.capabilities.Thinking] — see [Capabilities](capabilities/custom.md#providing-model-settings))
4. **Run-level** (`agent.run(model_settings=...)`)

Inside a callable, `ctx.model_settings` contains the merged result of all *previous* layers (position-dependent). For example, an agent-level callable sees only model defaults, while a run-level callable sees model defaults + agent-level + capability-level settings. To reset a field set by a previous layer, set it explicitly (e.g. `{'temperature': None}`).

```python
from pydantic_ai import Agent, ModelSettings

agent = Agent(
    'test',
    model_settings=lambda ctx: ModelSettings(
        temperature=0.0 if ctx.run_step <= 1 else 0.7,
    ),
)
```

!!! note "Model Settings Support"
    Model-level settings are supported by all concrete model implementations (OpenAI, Anthropic, Google, etc.). Wrapper models like [`FallbackModel`](models/overview.md#fallback-model) and [`WrapperModel`][pydantic_ai.models.wrapper.WrapperModel] don't have their own settings - they use the settings of their underlying models.

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

agent = Agent('google:gemini-3-flash-preview')

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
3. **Runtime instructions**: These are additional instructions for a specific run that can be passed to one of the [run methods](#running-agents) using the `instructions` argument.

All three types of instructions can be added to a single agent, and they are appended in the order they are defined at runtime. Each instruction is internally classified as either **static** (literal strings from the `instructions` parameter) or **dynamic** (from `@agent.instructions` functions, runtime instructions, or [toolset](toolsets.md) instructions). Static instructions are always sorted before dynamic ones. This ordering enables providers that support prompt caching (like [Anthropic](models/anthropic.md#smart-instruction-caching) and [Bedrock](models/bedrock.md#prompt-caching)) to cache the stable static prefix while leaving dynamic instructions outside the cache boundary.

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

Instructions can also come from [capabilities](capabilities/overview.md) via [`get_instructions()`][pydantic_ai.capabilities.AbstractCapability.get_instructions], [toolsets](toolsets.md#building-a-custom-toolset) via [`get_instructions()`][pydantic_ai.toolsets.AbstractToolset.get_instructions], or from [template strings](agent-spec.md#template-strings) rendered against the agent's dependencies.

### Instruction parts {#instruction-parts}

Each source contributes its own instruction part. Parts are sent to the model as one string, separated by a blank line, and are also available individually as [`InstructionPart`][pydantic_ai.messages.InstructionPart]s on [`ModelRequestParameters.instruction_parts`][pydantic_ai.models.ModelRequestParameters.instruction_parts].

You declare a [`name`][pydantic_ai.messages.InstructionPart.name]; the framework issues an [`id`][pydantic_ai.messages.InstructionPart.id]. Name a part relative to what you own — `'limits'`, not `'toolset:weather:limits'` — and the source contributing it supplies the rest, so you never repeat your own identity and can never claim another source's key. The [`InstructionId`][pydantic_ai.messages.InstructionId] you get back pairs the [`source`][pydantic_ai.messages.InstructionId.source] that contributed the part with the [`name`][pydantic_ai.messages.InstructionId.name], if there is one, and renders as its segments joined by `:`:

| `str(part.id)` | Addresses |
|---|---|
| `'agent'` | The agent's own `instructions` |
| `'toolset:<toolset id>'` | Everything a [toolset](toolsets.md) with an [`id`][pydantic_ai.toolsets.AbstractToolset.id] contributes |
| `'capability:<capability id>'` | Everything a [capability](capabilities/overview.md) with an [`id`][pydantic_ai.capabilities.AbstractCapability.id] contributes |
| `'agent:<name>'` | One part the agent named |
| `'toolset:<toolset id>:<name>'` | One part that toolset named |
| `'capability:<capability id>:<name>'` | One part that capability named |

There are two ways to declare a name, depending on what the part is:

- **A function** is named where it is registered: [`@agent.instructions(name=...)`][pydantic_ai.agent.Agent.instructions], [`@capability.instructions(name=...)`][pydantic_ai.capabilities.Capability.instructions].
- **Literal text** carries its name on the text itself, by passing an [`InstructionPart`][pydantic_ai.messages.InstructionPart] anywhere instructions are accepted — `Agent(instructions=...)`, `Capability(instructions=...)`, `FunctionToolset(instructions=...)`, or a `get_instructions()` implementation on a [capability](capabilities/custom.md) or [toolset](toolsets.md#building-a-custom-toolset). The part also decides whether it counts as [`dynamic`][pydantic_ai.messages.InstructionPart.dynamic], which is what keeps it outside the cacheable prefix (see [prompt caching](models/anthropic.md#prompt-caching)), and a part is always kept whole rather than merged with its neighbours.

Because a part's id is stable across runs, an application that stores instruction configuration elsewhere (say, a UI where a user edits the instructions an MCP server contributes) can key that configuration on the id instead of on the part's position or wording, both of which change as the agent evolves.

A source key is what everything else is built from, so it keeps its meaning permanently: giving more sources ids later can only add keys, never change what an existing key addresses.

Capability ids, toolset ids, and instruction names cannot contain `:`, because the character is reserved as the delimiter between these segments. The name `'agent'` is also reserved, because on its own it is the key of the agent's own instructions.

```python {title="instruction_parts.py"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.models.test import TestModel

model = TestModel()
agent = Agent(
    model,
    instructions='Be concise.',
    capabilities=[Capability(instructions='Cite your sources.', id='research')],
)


@agent.instructions(name='local_time')
def local_time() -> str:
    return 'The time is 10:00.'


@agent.instructions
def user_name(ctx: RunContext[None]) -> str:
    return 'The user is Frank.'


agent.run_sync('What is the capital of Italy?')

parts = model.last_model_request_parameters.instruction_parts or []
print([(part.name, str(part.id) if part.id is not None else None, part.content) for part in parts])
"""
[
    (None, 'agent', 'Be concise.'),
    (None, 'capability:research', 'Cite your sources.'),
    ('local_time', 'agent:local_time', 'The time is 10:00.'),
    (None, None, 'The user is Frank.'),
]
"""
```

_(This example is complete, it can be run "as is")_

Two consequences worth knowing before you key configuration on an id:

- **A key covers everything under it.** Where a source contributes several parts and none of them are named, they all carry the source key, so replacing that key's text replaces all of them — computed parts included. That is the honest meaning of "I control what this capability tells the model", but it means parts a source didn't name can't be addressed one by one. `'agent'` is the deliberate exception: it covers only the literal instructions the agent was built with, so taking over the base prompt doesn't silently swallow an `@agent.instructions` function that injects the date or the user's name.
- **Some parts can't be addressed at all.** They take part in the prompt like any other, but nothing keys them: an instructions function with no `name` (a function's own name isn't unique, and a lambda or [template string](agent-spec.md#template-strings) has none), a callable passed to `Agent(instructions=...)`, anything passed as runtime instructions to a specific run, and anything from a toolset or capability without an `id`. If you need one of your own callables to be overridable, register it with `@agent.instructions(name=...)` or [`@capability.instructions(name=...)`][pydantic_ai.capabilities.Capability.instructions] instead of passing it to the constructor.

    Naming a part whose source has no `id` leaves its `id` as `None`, because there is no source key to qualify the name against. The name still travels with the part, so you can see what its author called it, but nothing addresses it.

## Reflection and self-correction

Validation errors from both function tool parameter validation and [structured output validation](output.md#structured-output) can be passed back to the model with a request to retry.

You can also raise [`ModelRetry`][pydantic_ai.exceptions.ModelRetry] from within a [tool](tools.md) or [output function](output.md#output-functions) to tell the model it should retry generating a response.

This is one of [several layers that can retry](retries.md) during a run, each with its own budget.

- The default retry count is **1** but can be altered for the [entire agent][pydantic_ai.agent.Agent.__init__] with `retries` or [`AgentRetries`][pydantic_ai.agent.AgentRetries], a [specific tool][pydantic_ai.agent.Agent.tool], or [outputs][pydantic_ai.agent.Agent.__init__]. Both the tool and output sides of the agent retry budget can also be overridden per run via `agent.run(retries={'tools': ..., 'output': ...})` and friends (or for a block of runs via [`agent.override()`][pydantic_ai.agent.Agent.override]). At these call sites a bare `int` overrides both budgets, just like at construction — pass a dict such as `retries={'tools': ...}` to override just one. The tool-retry default and its per-run override apply to function tools, output tools, and MCP tools.
- You can access the current retry count from within a tool, output validator, or output function via [`ctx.retry`][pydantic_ai.tools.RunContext.retry].

### How output retries are enforced

Pydantic AI enforces the output retry budget differently depending on how the model returns its final output:

- **Text output path** (`output_type=str`, text-only outputs, empty or unusable model responses): a single global budget is shared across the whole run. Each invalid response consumes one unit of the budget; when it's exhausted, the run raises [`UnexpectedModelBehavior`][pydantic_ai.exceptions.UnexpectedModelBehavior] with message `'Exceeded maximum output retries (N)'`.
- **Tool output path** ([`output_type=ToolOutput(...)`](output.md#tool-output), structured outputs): the output retry budget is the *default per-tool limit*. See [Tool Output](output.md#tool-output) for per-tool overrides via [`ToolOutput(max_retries=N)`][pydantic_ai.output.ToolOutput.max_retries].

For how the budget appears inside [output validators](output.md#output-validator-functions) — including what `ctx.max_retries` and `ctx.retry` reflect on each path — see the [Output validators](output.md#output-validator-functions) section.

Tool retries are tracked per tool — see [Tool Execution, Retries, and Failures](tools-advanced.md#tool-retries) for the per-tool counter model and the three configuration levels.

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
    name='agent_eval',
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

For a run that was cancelled rather than failed, [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] and [`RunCancelled.from_cancellation()`][pydantic_ai.exceptions.RunCancelled.from_cancellation] carry the run's history directly -- see [Cancelling a Run](#cancelling-a-run).

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
        """
        An error occurred:
        Tool 'calc_volume' exceeded max retries count of 1. Consider raising the retry limit, or see the docs on tool retries: https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-retries
        """
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
                conversation_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(
                    cost=Decimal('0.0001645'), input_tokens=62, output_tokens=4
                ),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
                conversation_id='...',
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
                conversation_id='...',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='calc_volume',
                        args={'size': 6},
                        tool_call_id='pyd_ai_tool_call_id',
                    )
                ],
                usage=RequestUsage(
                    cost=Decimal('0.000238'), input_tokens=72, output_tokens=8
                ),
                model_name='gpt-5.2',
                timestamp=datetime.datetime(...),
                run_id='...',
                conversation_id='...',
            ),
            ModelRequest(
                parts=[],
                timestamp=datetime.datetime(...),
                run_id='...',
                conversation_id='...',
                state='interrupted',
            ),
        ]
        """
    else:
        print(result.output)
```

1. Define a tool that will raise `ModelRetry` repeatedly in this case.
2. [`capture_run_messages`][pydantic_ai.capture_run_messages] is used to capture the messages exchanged during the run.

_(This example is complete, it can be run "as is")_

When a run is cut short by an exception while streaming, an exception inside a tool, or external cancellation, Pydantic AI still captures partial state where it can. Partial [`ModelResponse`][pydantic_ai.messages.ModelResponse] and [`ModelRequest`][pydantic_ai.messages.ModelRequest] messages have `state='interrupted'` so persistence layers and UIs can distinguish them from complete messages.

For model responses, interrupted messages contain the response parts streamed before the interruption. For model requests, interrupted messages contain the tool results that completed before tool execution stopped — if none did, the request is still recorded, with no parts, marking the point where the response's tool calls were abandoned. The captured messages reflect exactly what happened — half-finished tool call parts are not turned into synthetic tool results at capture time. When an interrupted history is passed back into a run, it is [repaired automatically](message-history.md#making-histories-provider-valid) before the next model request.

In this example, `get_volume` completes before `get_mass` raises, so the interrupted request contains the completed `get_volume` return:

```python {title="capture_interrupted_run.py"}
from pydantic_ai import Agent, ModelRequest, capture_run_messages
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel


def call_tools(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(
        parts=[
            ToolCallPart(tool_name='get_volume', args={'size': 6}, tool_call_id='volume_call'),
            ToolCallPart(tool_name='get_mass', args={'size': 6}, tool_call_id='mass_call'),
        ]
    )


agent = Agent(FunctionModel(function=call_tools))


@agent.tool_plain(sequential=True)
def get_volume(size: int) -> int:
    return size**3


@agent.tool_plain(sequential=True)
def get_mass(size: int) -> int:
    raise RuntimeError('missing density')


with capture_run_messages() as messages:
    try:
        agent.run_sync('Calculate volume and mass.')
    except RuntimeError as exc:
        print(f'Run failed: {exc}')
        #> Run failed: missing density

interrupted_request = next(
    message for message in messages if isinstance(message, ModelRequest) and message.state == 'interrupted'
)
assert any(
    isinstance(part, ToolReturnPart) and part.tool_name == 'get_volume' and part.content == 216
    for part in interrupted_request.parts
)
```

!!! note
    If you call [`run`][pydantic_ai.agent.AbstractAgent.run], [`run_sync`][pydantic_ai.agent.AbstractAgent.run_sync], or [`run_stream`][pydantic_ai.agent.AbstractAgent.run_stream] more than once within a single `capture_run_messages` context, `messages` will represent the messages exchanged during the first call only.

    `capture_run_messages` contexts can be nested: each context captures the runs for which it is the innermost active context. A run started inside a nested context is captured by that nested context, not by any enclosing one. This means you can wrap a nested agent run (for example inside a tool that calls another agent) in its own `capture_run_messages` to inspect that inner run's messages independently.

## Agent Specs

Agents can also be defined declaratively in YAML or JSON using [agent specs](agent-spec.md). This separates agent configuration from application code:

```yaml {test="skip"}
model: anthropic:claude-opus-4-6
instructions: You are a helpful assistant.
capabilities:
  - WebSearch
  - Thinking:
      effort: high
```

```python {test="skip" lint="skip"}
from pydantic_ai import Agent

agent = Agent.from_file('agent.yaml')
```

See [Agent Specs](agent-spec.md) for the full spec format, template strings, and custom capability registration.
