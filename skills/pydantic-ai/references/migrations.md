# Migrations Reference

Upgrade patterns from deprecated APIs to current best practices.

## system_prompt to instructions

The `system_prompt` parameter still works but `instructions` is recommended for new code.

```python
# Old (still supported)
agent = Agent(
    'openai:gpt-5',
    system_prompt='Be helpful and concise.',
)

# New (recommended)
agent = Agent(
    'openai:gpt-5',
    instructions='Be helpful and concise.',
)
```

**Key difference:** In multi-agent delegation, `instructions` are always re-evaluated for the current agent, while `system_prompt` may persist from previous messages.

## Individual Tools to Toolsets

For better organization and reusability, group related tools into toolsets.

```python
# Old pattern - tools defined directly on agent
agent = Agent('openai:gpt-5')

@agent.tool_plain
def search_web(query: str) -> str:
    """Search the web."""
    return f'Results for {query}'

@agent.tool_plain
def fetch_page(url: str) -> str:
    """Fetch a web page."""
    return f'Content from {url}'
```

```python
# New pattern - toolset is reusable across agents
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset

web_toolset = FunctionToolset[None]()

@web_toolset.tool_plain
def search_web(query: str) -> str:
    """Search the web."""
    return f'Results for {query}'

@web_toolset.tool_plain
def fetch_page(url: str) -> str:
    """Fetch a web page."""
    return f'Content from {url}'

# Reuse across multiple agents
agent1 = Agent('openai:gpt-5', toolsets=[web_toolset])
agent2 = Agent('anthropic:claude-sonnet-4-5', toolsets=[web_toolset])
```

## result.data to result.output

The `result.data` attribute was renamed to `result.output` for clarity.

```python
# Old
result = agent.run_sync('prompt')
print(result.data)  # Deprecated

# New
result = agent.run_sync('prompt')
print(result.output)  # Current
```

## output_type List Syntax

For union output types, explicit `ToolOutput` wrappers with custom names are now preferred.

```python
# Old - simple list (still works but less control)
agent = Agent('openai:gpt-5', output_type=[Fruit, Vehicle])

# New - explicit ToolOutput with descriptive names
from pydantic_ai import ToolOutput

agent = Agent(
    'openai:gpt-5',
    output_type=[
        ToolOutput(Fruit, name='return_fruit', description='Return fruit info'),
        ToolOutput(Vehicle, name='return_vehicle', description='Return vehicle info'),
    ],
)
```

## result_tool_name to ToolOutput

The `result_tool_name` parameter was replaced by the more flexible `ToolOutput` wrapper.

```python
# Old
agent = Agent(
    'openai:gpt-5',
    output_type=MyModel,
    result_tool_name='extract_data',  # Deprecated
)

# New
from pydantic_ai import ToolOutput

agent = Agent(
    'openai:gpt-5',
    output_type=ToolOutput(MyModel, name='extract_data'),
)
```

## Plain Functions to Tool Class

When you need more control over tool behavior, migrate from decorated functions to explicit `Tool` instances.

```python
# Old - limited configuration
@agent.tool_plain
def my_tool(arg: str) -> str:
    """My tool description."""
    return arg

# New - full control
from pydantic_ai import Agent, Tool

def my_tool(arg: str) -> str:
    """My tool description."""
    return arg

agent = Agent(
    'openai:gpt-5',
    tools=[
        Tool(
            my_tool,
            name='custom_name',
            description='Custom description',
            retries=3,
            strict=True,
            requires_approval=True,
        ),
    ],
)
```

## Direct Model Classes to Model Strings

Model strings are now the preferred way to specify models.

```python
# Old - explicit model class import
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

agent = Agent(OpenAIChatModel('gpt-5'))

# New - model string (preferred)
from pydantic_ai import Agent

agent = Agent('openai:gpt-5')
```

Model strings provide:
- Automatic provider detection
- Built-in profile selection
- Cleaner code

Use explicit model classes only when you need custom configuration:

```python
from pydantic_ai.models.openai import OpenAIChatModel

# Custom client, profile, or provider
model = OpenAIChatModel(
    'gpt-5',
    openai_client=custom_client,
    provider='azure',
)
agent = Agent(model)
```

## @agent.system_prompt to @agent.instructions

The `@agent.system_prompt` decorator was renamed to `@agent.instructions`.

```python
# Old
@agent.system_prompt
def add_context(ctx: RunContext[str]) -> str:
    return f'User: {ctx.deps}'

# New
@agent.instructions
def add_context(ctx: RunContext[str]) -> str:
    return f'User: {ctx.deps}'
```

## retries Parameter Location

The `retries` parameter can now be set at multiple levels with clear precedence.

```python
# Old - only on agent
agent = Agent('openai:gpt-5', retries=3)

# New - at tool level for fine-grained control
@agent.tool_plain(retries=5)  # This tool gets 5 retries
def critical_tool(arg: str) -> str:
    return arg

@agent.tool_plain(retries=1)  # This tool gets 1 retry
def simple_tool(arg: str) -> str:
    return arg
```

Precedence: Tool-level > Agent-level > Default (1)

## Mutable deps to Dataclass deps

Using mutable dictionaries for dependencies is discouraged.

```python
# Old - mutable dict (avoid)
agent = Agent('openai:gpt-5', deps_type=dict)

result = agent.run_sync('prompt', deps={'api_key': 'secret'})

# New - immutable dataclass (preferred)
from dataclasses import dataclass

@dataclass
class Deps:
    api_key: str
    client: httpx.AsyncClient

agent = Agent('openai:gpt-5', deps_type=Deps)

result = agent.run_sync('prompt', deps=Deps(api_key='secret', client=client))
```

Benefits:
- Type safety in tools via `RunContext[Deps]`
- IDE autocompletion
- Clear contract for required dependencies

## History Processing

Message history manipulation moved from manual code to processors.

```python
# Old - manual history manipulation
messages = result.all_messages()
# Manual trimming, filtering, etc.
trimmed = messages[-10:]  # Keep last 10
result2 = agent.run_sync('next', message_history=trimmed)

# New - declarative history processors
from pydantic_ai.history_processors import SummarizingHistoryProcessor

agent = Agent(
    'openai:gpt-5',
    history_processors=[
        SummarizingHistoryProcessor(max_tokens=4000),
    ],
)

# History is automatically processed
result2 = agent.run_sync('next', message_history=result.all_messages())
```

## Deprecation Timeline

| API | Status | Removal Target |
|-----|--------|----------------|
| `system_prompt` kwarg | Deprecated, use `instructions` | v1.0 |
| `result.data` | Deprecated, use `result.output` | v1.0 |
| `result_tool_name` | Deprecated, use `ToolOutput` | v1.0 |
| `@agent.system_prompt` | Deprecated, use `@agent.instructions` | v1.0 |

## Checking for Deprecation Warnings

Run your tests with deprecation warnings visible:

```bash
python -W default::DeprecationWarning -m pytest tests/
```

Or in your pytest configuration:

```ini
# pyproject.toml
[tool.pytest.ini_options]
filterwarnings = ["default::DeprecationWarning"]
```

## See Also

- [agents.md](agents.md) — Current agent configuration
- [tools.md](tools.md) — Current tool patterns
- [output.md](output.md) — Current output configuration
- [toolsets.md](toolsets.md) — Toolset patterns
