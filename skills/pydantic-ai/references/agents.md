# Agents Reference

Source: `pydantic_ai_slim/pydantic_ai/agent/__init__.py`, `pydantic_ai_slim/pydantic_ai/agent/_agent.py`

## Agent Execution Flow

```
                    ┌─────────────────┐
                    │  agent.run()    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ UserPromptNode  │  Build initial request
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              ▼              │
              │     ┌─────────────────┐     │
              │     │ModelRequestNode │     │  Send to LLM
              │     └────────┬────────┘     │
              │              │              │
              │              ▼              │
              │      Has tool calls?        │
              │         │       │           │
              │        Yes      No          │
              │         │       │           │
              │         ▼       │           │
              │  ┌─────────────┐│           │
              │  │CallToolsNode││           │  Execute tools
              │  └──────┬──────┘│           │
              │         │       │           │
              │         ▼       │           │
              │    More calls?  │           │
              │      │    │     │           │
              │     Yes   No    │           │
              │      │    │     │           │
              └──────┘    └─────┼───────────┘
                                │
                                ▼
                         ┌─────────────┐
                         │     End     │  Return result
                         └─────────────┘
```

The agent loop continues until:
- The model returns output without tool calls (`end_strategy='early'`)
- All tool calls are processed and model returns output (`end_strategy='exhaustive'`)
- Usage limits are exceeded
- Maximum retries are exhausted

## Agent Constructor

```python
Agent(
    model='openai:gpt-5',          # Model string or Model instance
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

### `agent.run_stream_sync()` — Synchronous Streaming

Synchronous version of `run_stream()`:

```python
with agent.run_stream_sync('user prompt', deps=my_deps) as result:
    for text in result.stream_text():
        print(text)
```

### Event Stream Handler

Observe events during streaming (tool calls, deltas, etc.):

```python
from collections.abc import AsyncIterable

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
)

agent = Agent('openai:gpt-5')


async def event_stream_handler(
    ctx: RunContext,
    event_stream: AsyncIterable[AgentStreamEvent],
):
    async for event in event_stream:
        if isinstance(event, PartStartEvent):
            print(f'Starting part {event.index}: {event.part!r}')
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                print(f'Text delta: {event.delta.content_delta!r}')
        elif isinstance(event, FunctionToolCallEvent):
            print(f'Tool call: {event.part.tool_name}({event.part.args})')
        elif isinstance(event, FunctionToolResultEvent):
            print(f'Tool result: {event.result.content}')
        elif isinstance(event, FinalResultEvent):
            print(f'Final result starting (tool_name={event.tool_name})')


async def main():
    async with agent.run_stream('prompt', event_stream_handler=event_stream_handler) as run:
        async for output in run.stream_text():
            print(output)
```

Also works with `agent.run()` for non-streaming runs.

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

### Advanced Instruction Patterns

**Combining static and dynamic instructions:**

```python {title="instructions.py"}
from datetime import date

from pydantic_ai import Agent, RunContext

agent = Agent(
    'openai:gpt-5',
    deps_type=str,
    instructions="Use the customer's name while replying to them.",
)


@agent.instructions
def add_the_users_name(ctx: RunContext[str]) -> str:
    return f"The user's name is {ctx.deps}."


@agent.instructions
def add_the_date() -> str:
    return f'The date is {date.today()}.'


result = agent.run_sync('What is the date?', deps='Frank')
print(result.output)
#> Hello Frank, the date today is 2032-01-02.
```

**When to use instructions vs system_prompt:**

| Feature | `instructions` | `system_prompt` |
|---------|---------------|-----------------|
| In multi-agent runs | Only current agent's instructions used | Retained from previous messages |
| Re-evaluated | Always (each run) | Only if dynamic |
| Recommended for | Most use cases | Preserving context across agents |

**Returning empty string:** If an instruction function returns `""`, no instruction is added for that function.

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

## Run Metadata

Tag agent runs with contextual data for monitoring and filtering:

```python {title="agent_metadata.py"}
from dataclasses import dataclass

from pydantic_ai import Agent


@dataclass
class Deps:
    tenant: str


agent = Agent[Deps](
    'openai:gpt-5',
    deps_type=Deps,
    metadata=lambda ctx: {'tenant': ctx.deps.tenant},  # Agent-level metadata
)

result = agent.run_sync(
    'What is the capital of France?',
    deps=Deps(tenant='tenant-123'),
    metadata=lambda ctx: {'num_requests': ctx.usage.requests},  # Per-run metadata
)
print(result.metadata)
#> {'tenant': 'tenant-123', 'num_requests': 1}
```

Metadata is:
- Computed when run starts and again after successful completion
- Merged (per-run overrides agent-level)
- Attached to `RunContext` during the run
- Added to span attributes when instrumentation is enabled

Access via `result.metadata`, `agent_run.metadata`, or `streamed_result.metadata`.

## capture_run_messages() — Debug Failed Runs

Capture messages from a failed run for debugging:

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior, capture_run_messages

agent = Agent('openai:gpt-5')


@agent.tool_plain
def flaky_tool(size: int) -> int:
    if size != 42:
        raise ModelRetry('Please try again.')
    return size**3


with capture_run_messages() as messages:
    try:
        result = agent.run_sync('Get volume of box with size 6.')
    except UnexpectedModelBehavior as e:
        print('Error:', e)
        print('Messages exchanged:')
        for msg in messages:
            print(f'  {type(msg).__name__}: {msg}')
```

The `messages` list contains all `ModelRequest` and `ModelResponse` objects exchanged during the run, useful for understanding why retries failed.

Note: If multiple runs occur in one context, only the first run's messages are captured.

## Runs vs. Conversations

A **run** can be a complete conversation or just one turn. A **conversation** can span multiple runs using `message_history`:

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')

# First run
result1 = agent.run_sync('Who was Albert Einstein?')

# Second run, continuing the conversation
result2 = agent.run_sync(
    'What was his most famous equation?',
    message_history=result1.new_messages(),  # Continue conversation
)
```

Without `message_history`, the model wouldn't know who "his" refers to.

## Model Settings Precedence

Settings merge with later values overriding earlier:

```
Model settings < Agent settings < Run settings
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

# Model-level settings
model = OpenAIChatModel('gpt-5', settings={'max_tokens': 500})

# Agent-level settings (override model)
agent = Agent(model, model_settings={'temperature': 0.7})

# Run-level settings (override agent and model)
result = agent.run_sync('prompt', model_settings={'temperature': 0.0})
# Final: max_tokens=500 (model), temperature=0.0 (run)
```

## Debugging Agent Runs

When an agent behaves unexpectedly, enable instrumentation to trace the full execution:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
# Now all agent runs are traced with full message history, tool calls, and model responses
```

This captures every model request and response, making it easy to understand what the model "saw" and why it made certain decisions. See [observability.md](observability.md) for details.

## Performance and Cost Optimization

### Reduce Token Usage

**Use `instructions` instead of repeating context:**

```python
# Inefficient - context in every prompt
result = agent.run_sync('User: John. Task: What is 2+2?')
result = agent.run_sync('User: John. Task: What is 3+3?')

# Efficient - context in instructions
agent = Agent('openai:gpt-5', deps_type=str)

@agent.instructions
def user_context(ctx: RunContext[str]) -> str:
    return f'User: {ctx.deps}'

result = agent.run_sync('What is 2+2?', deps='John')
```

**Summarize long message histories:**

```python
from pydantic_ai.history_processors import SummarizingHistoryProcessor

agent = Agent(
    'openai:gpt-5',
    history_processors=[
        SummarizingHistoryProcessor(
            max_tokens=4000,  # Summarize when history exceeds this
            summary_model='openai:gpt-5-mini',  # Cheap model for summarization
        ),
    ],
)
```

**Set max_tokens to limit response length:**

```python
agent = Agent(
    'openai:gpt-5',
    model_settings={'max_tokens': 500},  # Prevent runaway responses
)
```

### Prevent Runaway Costs

**Always use `UsageLimits` in production:**

```python
from pydantic_ai import UsageLimits

result = await agent.run(
    user_prompt,
    usage_limits=UsageLimits(
        request_limit=10,           # Max model requests
        total_tokens_limit=10000,   # Max total tokens
    ),
)
```

**Track usage across delegated agents:**

```python
from pydantic_ai import RunUsage, UsageLimits

usage = RunUsage()
limits = UsageLimits(total_tokens_limit=50000)

# All agents share the same usage tracking
result1 = await agent1.run(prompt, usage=usage, usage_limits=limits)
result2 = await agent2.run(prompt, usage=usage, usage_limits=limits)

print(f'Total tokens: {usage.total_tokens}')
```

### Choose the Right Model

**Use cheaper models for simple tasks:**

```python
# Routing/classification - use fast, cheap model
router = Agent('openai:gpt-5-mini', output_type=RouteDecision)

# Complex reasoning - use capable model
reasoner = Agent('anthropic:claude-sonnet-4-5', output_type=DetailedAnalysis)
```

**Use FallbackModel for cost optimization:**

```python
from pydantic_ai.models.fallback import FallbackModel

# Try cheap model first, fall back to expensive on failure
agent = Agent(
    FallbackModel('openai:gpt-5-mini', 'openai:gpt-5'),
)
```

### Optimize Tool Schemas

**Keep tool parameters simple:**

```python
# Generates large schema - avoid
class ComplexInput(BaseModel):
    nested: dict[str, list[dict[str, Any]]]

# Simple schema - preferred
@agent.tool_plain
def process_item(item_id: str, action: str) -> str:
    """Process an item. action must be 'approve' or 'reject'."""
    return f'Processed {item_id}: {action}'
```

**Filter tools to reduce schema size:**

```python
from pydantic_ai.toolsets import FilteredToolset

# Only expose relevant tools based on context
filtered = FilteredToolset(
    full_toolset,
    filter_func=lambda ctx, td: td.name in ['tool1', 'tool2'],
)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Agent` | `pydantic_ai.Agent` | Main agent class |
| `AgentRunResult` | `pydantic_ai.AgentRunResult` | Result of `run()`/`run_sync()` |
| `AgentRun` | `pydantic_ai.AgentRun` | Stateful run from `iter()` |
| `EndStrategy` | `pydantic_ai.EndStrategy` | `'early'` or `'exhaustive'` |
| `RunContext` | `pydantic_ai.RunContext` | Dependency injection context |
| `InstrumentationSettings` | `pydantic_ai.InstrumentationSettings` | Tracing config |
| `capture_run_messages` | `pydantic_ai.capture_run_messages` | Debug context manager |
| `AgentStreamEvent` | `pydantic_ai.AgentStreamEvent` | Base event type for streaming |

## See Also

- [streaming.md](streaming.md) — Streaming responses and events
- [output.md](output.md) — Structured output configuration
- [dependencies.md](dependencies.md) — Dependency injection
- [tools.md](tools.md) — Tool registration
- [multi-agent.md](multi-agent.md) — Multi-agent patterns
