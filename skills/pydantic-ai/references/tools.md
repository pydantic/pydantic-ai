# Tools Reference

Source: `pydantic_ai_slim/pydantic_ai/tools.py`, `pydantic_ai_slim/pydantic_ai/toolsets/`

## Tool Registration

### `@agent.tool` — With RunContext

The first parameter must be `RunContext[DepsT]`. Provides access to dependencies, usage, messages, and retry info.

```python {title="tool_with_ctx.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    api_key: str


agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.tool
async def fetch_data(ctx: RunContext[Deps], query: str) -> str:
    """Fetch data from the API.

    Args:
        ctx: The run context with dependencies.
        query: The search query.
    """
    return f'Results for {query} using key {ctx.deps.api_key}'


result = agent.run_sync(
    'What is the capital of France?', deps=Deps(api_key='test-key')
)
print(result.output)
#> The capital of France is Paris.
```

### `@agent.tool_plain` — Without RunContext

No `RunContext` parameter. Use when the tool does not need dependencies.

```python {title="tool_plain_example.py"}
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')


@agent.tool_plain
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307


result = agent.run_sync('What is 7 plus 5?')
print(result.output)
#> The answer is 12.
```

### `tools=` Constructor Kwarg

Pass tools at agent creation instead of using decorators:

```python
from pydantic_ai import Agent, Tool

def my_func(x: int) -> str:
    """Double a number."""
    return str(x * 2)

agent = Agent('openai:gpt-4o', tools=[Tool(my_func)])
```

## Tool Class Constructor

```python
Tool(
    function,                            # The tool function
    *,
    takes_ctx=None,                      # Auto-detected if None
    max_retries=None,                    # Override agent default
    name=None,                           # Custom tool name
    description=None,                    # Override docstring description
    prepare=None,                        # ToolPrepareFunc
    docstring_format='auto',             # 'google' | 'numpy' | 'sphinx' | 'auto'
    require_parameter_descriptions=False,
    strict=None,                         # Strict JSON schema mode
    sequential=False,                    # Run sequentially (not in parallel)
    requires_approval=False,             # Requires human approval
    metadata=None,                       # Arbitrary metadata dict
    timeout=None,                        # Timeout in seconds
)
```

## RunContext Fields

```python
ctx.deps              # The dependencies object (AgentDepsT)
ctx.model             # The Model instance being used
ctx.usage             # RunUsage — token counts so far
ctx.prompt            # Original user prompt
ctx.messages          # Messages exchanged so far
ctx.retry             # Current retry count for this tool
ctx.max_retries       # Maximum retries configured
ctx.last_attempt      # bool — True if this is the final attempt
ctx.tool_name         # Name of the current tool
ctx.tool_call_id      # ID of the current tool call
ctx.run_step          # Current step in the agent run
ctx.run_id            # Unique run identifier
```

## ModelRetry — Tool Retry

Raise `ModelRetry` inside a tool to send an error message back to the model and retry the tool call.

```python {title="tool_retry_example.py"}
from pydantic_ai import Agent, ModelRetry

agent = Agent('openai:gpt-5')


@agent.tool_plain(retries=3)
def validate_input(value: str) -> str:
    """Validate and process input."""
    if not value.strip():
        raise ModelRetry('Input cannot be empty, please provide a valid value.')
    return f'Processed: {value}'


result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

## ToolPrepareFunc — Dynamic Tool Configuration

A `prepare` function can modify or hide a tool at runtime based on context.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.tools import ToolDefinition


async def prepare_tool(
    ctx: RunContext[str], tool_def: ToolDefinition
) -> ToolDefinition | None:
    # Return None to hide the tool, or modify tool_def
    if ctx.deps == 'admin':
        return tool_def
    return None  # Hide tool for non-admin users
```

## FunctionToolset

Register multiple tools from a class or collection:

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

toolset = FunctionToolset[None]()

@toolset.tool_plain
def greet(name: str) -> str:
    """Greet someone."""
    return f'Hello, {name}!'

agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

## Docstring Formats

Tool descriptions and parameter descriptions are extracted from docstrings.
Supported formats: `'google'`, `'numpy'`, `'sphinx'`, `'auto'` (default).

**Google format** (recommended):

```python
def my_tool(city: str, date: str) -> str:
    """Get weather forecast.

    Args:
        city: The city name.
        date: The date in YYYY-MM-DD format.
    """
```

## Deferred Tools and Approval

Tools can be deferred for human-in-the-loop approval:

```python
from pydantic_ai.exceptions import ApprovalRequired

@agent.tool_plain(requires_approval=True)
def dangerous_action(target: str) -> str:
    """Perform an action that needs approval."""
    return f'Action performed on {target}'
```

When `requires_approval=True`, the tool raises `ApprovalRequired`. Use `DeferredToolResults` to resume.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Tool` | `pydantic_ai.Tool` | Tool wrapper class |
| `ToolDefinition` | `pydantic_ai.ToolDefinition` | Schema + metadata for a tool |
| `RunContext` | `pydantic_ai.RunContext` | Context passed to tools |
| `ModelRetry` | `pydantic_ai.ModelRetry` | Exception for tool retry |
| `FunctionToolset` | `pydantic_ai.FunctionToolset` | Toolset from decorated functions |
| `DeferredToolRequests` | `pydantic_ai.DeferredToolRequests` | Deferred tool call data |
| `DeferredToolResults` | `pydantic_ai.DeferredToolResults` | Results for deferred tools |
| `ToolApproved` | `pydantic_ai.ToolApproved` | Approval response |
| `ToolDenied` | `pydantic_ai.ToolDenied` | Denial response |
