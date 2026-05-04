# Code Mode

Replace individual tool calls with a single sandboxed Python execution environment.

## The problem

Standard tool calling requires one model round-trip per tool call. An agent that needs to fetch 10 items and process each one makes 11+ model calls -- slow, expensive, and context-heavy.

## The solution

`CodeMode` wraps your tools into a single `run_code` tool. The model writes Python code that calls multiple tools with loops, conditionals, variables, and `asyncio.gather` -- all inside a sandboxed [Monty](https://github.com/pydantic/monty) runtime.

| Standard tool calling | Code mode |
|---|---|
| 1 model call per tool | 1 model call for N tools |
| Sequential by default | Parallel via `asyncio.gather` |
| No local computation | Filter, transform, aggregate in code |
| Large conversation history | Compact -- fewer messages |

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[CodeMode()])

@agent.tool_plain
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}

@agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)

result = agent.run_sync("What's the weather in Paris and Tokyo, in Celsius?")
print(result.output)
```

The model writes code like:

```python
paris, tokyo = await asyncio.gather(
    get_weather(city='Paris'),
    get_weather(city='Tokyo'),
)
paris_c = await convert_temp(fahrenheit=paris['temp_f'])
tokyo_c = await convert_temp(fahrenheit=tokyo['temp_f'])
{'paris': paris_c, 'tokyo': tokyo_c}
```

## Installation

Code mode requires the Monty sandbox:

```bash
uv add "pydantic-ai-harness[codemode]"
```

The `code-mode` extra is also supported as an alias.

## Selective tool sandboxing

By default, `CodeMode(tools='all')` sandboxes every tool. You can control which tools go through the sandbox:

```python
# By name -- only these tools are available inside run_code
CodeMode(tools=['search', 'fetch'])

# By predicate
CodeMode(tools=lambda ctx, td: td.name != 'dangerous_tool')

# By metadata -- combine with SetToolMetadata or .with_metadata()
CodeMode(tools={'code_mode': True})
```

Tools that match the selector are wrapped inside `run_code`. Non-matching tools remain available as regular tool calls.

### Metadata-based selection

```python
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import CodeMode

search_tools = FunctionToolset(tools=[search, fetch]).with_metadata(code_mode=True)

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    toolsets=[search_tools],
    capabilities=[CodeMode(tools={'code_mode': True})],
)
```

## Return values

The last expression in the code snippet is automatically captured as the return value -- the model does not need to `print()`.

| Scenario | Return |
|---|---|
| No print output | Last expression value |
| With print output | `{"output": "<printed text>", "result": <last expression>}` |
| Multimodal content (e.g. images) | Returned natively for model processing |

## REPL state

State persists between `run_code` calls within the same agent run -- variables, imports, and function definitions carry over. Pass `restart: true` in the tool call to reset state.

## Observability

Nested tool calls inside `run_code` produce their own spans when instrumented with [Logfire](https://pydantic.dev/logfire) or any OpenTelemetry backend. The `run_code` tool return includes metadata with all nested calls:

```python
for msg in result.all_messages():
    for part in msg.parts:
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
            tool_calls = part.metadata['tool_calls']    # dict[str, ToolCallPart]
            tool_returns = part.metadata['tool_returns'] # dict[str, ToolReturnPart]
```

## Sandbox restrictions

Code runs inside [Monty](https://github.com/pydantic/monty), a sandboxed Python subset. Key restrictions:

- No class definitions
- No third-party imports (allowed stdlib: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`)
- No `import *`
- Tools requiring approval or with deferred execution are excluded from the sandbox

## API

```python
CodeMode(
    tools: ToolSelector = 'all',   # 'all', list[str], callable, or dict
    max_retries: int = 3,          # retries on sandbox execution errors
)
```

## Agent spec (YAML/JSON)

CodeMode works with Pydantic AI's [agent spec](https://ai.pydantic.dev/agent-spec/) feature for defining agents in YAML:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - CodeMode: {}
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import CodeMode

agent = Agent.from_file('agent.yaml', custom_capability_types=[CodeMode])
result = agent.run_sync('...')
print(result.output)
```

Pass `custom_capability_types` so the spec loader knows how to instantiate `CodeMode`. You can also pass arguments in the YAML:

```yaml
capabilities:
  - CodeMode:
      tools: ['search', 'fetch']
      max_retries: 5
```

## Further reading

- [Tool use via code](https://www.anthropic.com/engineering/code-execution-with-mcp) (Anthropic)
- [Code mode in production](https://blog.cloudflare.com/code-mode/) (Cloudflare)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
