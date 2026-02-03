# Streaming Reference

Source: `pydantic_ai_slim/pydantic_ai/agent/abstract.py`, `pydantic_ai_slim/pydantic_ai/result.py`

## run_stream() — Streaming Text

```python {title="streamed_hello_world.py" line_length="120"}
from pydantic_ai import Agent

agent = Agent('google-gla:gemini-2.5-flash')


async def main():
    async with agent.run_stream('Where does "hello world" come from?') as result:
        async for message in result.stream_text():
            print(message)
            #> The first known
            #> The first known use of "hello,
            #> The first known use of "hello, world" was in
            #> The first known use of "hello, world" was in a 1974 textbook
            #> The first known use of "hello, world" was in a 1974 textbook about the C
            #> The first known use of "hello, world" was in a 1974 textbook about the C programming language.
```

### StreamedRunResult Methods

- `stream_text(debounce_by=None)` — yields text chunks as they arrive
- `stream_output(debounce_by=None)` — yields validated structured output as it streams
- `get_output()` — await the final validated output

### Streaming Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Story(BaseModel):
    title: str
    content: str

agent = Agent('openai:gpt-4o', output_type=Story)

async with agent.run_stream('Write a story') as result:
    async for partial in result.stream_output():
        # partial is a partially populated Story
        print(partial)
```

## run_stream_events() — Event Stream

Lower-level API that yields typed events during streaming:

```python
async with agent.run_stream_events('prompt') as events:
    async for event in events:
        # event is one of:
        # - PartStartEvent (new part starting)
        # - PartDeltaEvent (incremental update)
        # - PartEndEvent (part complete)
        # - FunctionToolCallEvent (tool being called)
        # - FunctionToolResultEvent (tool result)
        pass
```

### Event Types

| Event | Description |
|-------|-------------|
| `PartStartEvent` | A new response part has started |
| `PartDeltaEvent` | Incremental update to a part (text chunk, tool args) |
| `PartEndEvent` | A response part is complete |
| `FunctionToolCallEvent` | A function tool is about to be called |
| `FunctionToolResultEvent` | A function tool has returned a result |
| `FinalResultEvent` | The final validated result is available |
| `HandleResponseEvent` | The model response is being handled |

### Delta Types

- `TextPartDelta` — text chunk delta
- `ThinkingPartDelta` — thinking/reasoning delta
- `ToolCallPartDelta` — tool call argument delta

## iter() — Step-by-Step Control

For maximum control, iterate over graph nodes:

```python
async with agent.iter('prompt', deps=my_deps) as agent_run:
    async for node in agent_run:
        if isinstance(node, pydantic_ai.CallToolsNode):
            # Inspect or modify tool calls
            pass
    result = agent_run.result
```

### AgentRun Properties

- `agent_run.result` — `AgentRunResult` (available after iteration completes)
- `agent_run.next_node` — the next node to be processed
- `agent_run.ctx` — the `GraphRunContext`

## Debounce

Control how often streaming yields by setting `debounce_by` (seconds):

```python
async with agent.run_stream('prompt') as result:
    async for text in result.stream_text(debounce_by=0.1):
        # Yields at most every 100ms
        print(text)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `AgentRun` | `pydantic_ai.AgentRun` | Stateful run from `iter()` |
| `AgentRunResult` | `pydantic_ai.AgentRunResult` | Final result |
| `AgentRunResultEvent` | `pydantic_ai.AgentRunResultEvent` | Result event |
| `PartStartEvent` | `pydantic_ai.PartStartEvent` | Stream event |
| `PartDeltaEvent` | `pydantic_ai.PartDeltaEvent` | Stream event |
| `PartEndEvent` | `pydantic_ai.PartEndEvent` | Stream event |
| `FinalResultEvent` | `pydantic_ai.FinalResultEvent` | Stream event |
| `TextPartDelta` | `pydantic_ai.TextPartDelta` | Text delta |
