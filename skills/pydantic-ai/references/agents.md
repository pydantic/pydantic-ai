# Agents Reference

Source: `pydantic_ai_slim/pydantic_ai/agent/__init__.py`, `pydantic_ai_slim/pydantic_ai/agent/_agent.py`

## Agent Constructor

```python
Agent(
    model='openai:gpt-4o',          # Model string or Model instance
    *,
    output_type=str,                 # Pydantic model, dataclass, or type
    instructions=None,               # Static str, callable, or list of either
    system_prompt=(),                # Legacy: str or Sequence[str]
    deps_type=NoneType,              # Type for dependency injection
    name=None,                       # Agent name for logging/tracing
    model_settings=None,             # ModelSettings dict
    retries=1,                       # Default max retries for tools
    output_retries=None,             # Max retries for output validation
    tools=(),                        # Sequence of Tool or tool functions
    builtin_tools=(),                # Provider builtin tools
    toolsets=None,                   # Sequence of AbstractToolset
    prepare_tools=None,              # ToolsPrepareFunc for all tools
    prepare_output_tools=None,       # ToolsPrepareFunc for output tools
    end_strategy='early',            # 'early' | 'exhaustive'
    instrument=None,                 # InstrumentationSettings | bool
    defer_model_check=False,         # Skip model validation at init
    metadata=None,                   # AgentMetadata dict
    history_processors=None,         # Message history processors
    event_stream_handler=None,       # EventStreamHandler
    tool_timeout=None,               # Default timeout for tool execution
    validation_context=None,         # Pydantic validation context
)
```

## Run Methods

### `agent.run()` — Async

```python
result = await agent.run(
    'user prompt',
    *,
    output_type=None,                # Override output type for this run
    message_history=None,            # Continue from previous messages
    model=None,                      # Override model for this run
    instructions=None,               # Override instructions
    deps=None,                       # Dependencies for this run
    model_settings=None,             # Override model settings
    usage_limits=None,               # UsageLimits for this run
    usage=None,                      # Pre-existing RunUsage to accumulate
    toolsets=None,                   # Additional toolsets for this run
    builtin_tools=None,              # Override builtin tools
)
# result: AgentRunResult[OutputDataT]
# result.output — the validated output
# result.all_messages() — full message history
# result.new_messages() — messages from this run only
# result.usage() — RunUsage with token counts
```

### `agent.run_sync()` — Synchronous

Same parameters as `run()`. Runs the async method in a new event loop.

### `agent.run_stream()` — Streaming

```python
async with agent.run_stream('user prompt', deps=my_deps) as result:
    async for text in result.stream_text():
        print(text)
    # or: async for chunk in result.stream_output():
```

Returns `StreamedRunResult`. See [streaming.md](streaming.md).

### `agent.iter()` — Step-by-Step Iteration

```python
async with agent.iter('user prompt', deps=my_deps) as agent_run:
    async for node in agent_run:
        # node is UserPromptNode, ModelRequestNode, CallToolsNode, or End
        print(type(node).__name__)
    print(agent_run.result.output)
```

Returns `AgentRun`. Gives fine-grained control over each step.

## Instructions (System Prompts)

### Static instructions

```python {title="static_instructions.py"}
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-5',
    instructions='You are a helpful assistant that speaks like a pirate.',
)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

### Dynamic instructions via decorator

```python {title="dynamic_instructions.py"}
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5', deps_type=str)


@agent.instructions
def custom_instructions(ctx: RunContext[str]) -> str:
    return f'The user prefers responses in {ctx.deps} language.'


result = agent.run_sync('What is the capital of France?', deps='English')
print(result.output)
#> The capital of France is Paris.
```

## Agent.override()

Override agent configuration for testing or runtime changes. Returns a context manager.

```python {title="agent_override.py"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5')

# Override model for testing
with agent.override(model=TestModel()):
    result = agent.run_sync('test prompt')
    print(result.output)
    #> success (no tool calls)
```

`override()` accepts: `model=`, `deps=`, `model_settings=`, `instrument=`.

## WrapperAgent

Wraps an existing agent to modify behavior without subclassing.

```python
from pydantic_ai.agent import WrapperAgent

class MyWrapper(WrapperAgent[MyDeps, str]):
    @property
    def wrapped_agent(self) -> Agent[MyDeps, str]:
        return self._inner_agent
```

Source: `pydantic_ai_slim/pydantic_ai/agent/wrapper.py`

## EndStrategy

- `'early'` (default): Stop after first output from the model.
- `'exhaustive'`: Process all tool calls before stopping, even after output is received.

## Tool Registration on Agent

```python
@agent.tool
async def my_tool(ctx: RunContext[MyDeps], arg: str) -> str:
    """Tool description."""
    return f'Result: {arg}'

@agent.tool_plain
def simple_tool(arg: str) -> str:
    """No RunContext needed."""
    return f'Result: {arg}'
```

See [tools.md](tools.md) for complete tool documentation.

## Debugging Agent Runs

When an agent behaves unexpectedly, enable instrumentation to trace the full execution:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
# Now all agent runs are traced with full message history, tool calls, and model responses
```

This captures every model request and response, making it easy to understand what the model "saw" and why it made certain decisions. See [observability.md](observability.md) for details.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Agent` | `pydantic_ai.Agent` | Main agent class |
| `AgentRunResult` | `pydantic_ai.AgentRunResult` | Result of `run()`/`run_sync()` |
| `AgentRun` | `pydantic_ai.AgentRun` | Stateful run from `iter()` |
| `EndStrategy` | `pydantic_ai.EndStrategy` | `'early'` or `'exhaustive'` |
| `RunContext` | `pydantic_ai.RunContext` | Dependency injection context |
| `InstrumentationSettings` | `pydantic_ai.InstrumentationSettings` | Tracing config |
