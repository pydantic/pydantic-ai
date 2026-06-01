# Code Mode

Standard tool calling takes one model round-trip per tool call. `CodeMode` wraps eligible tools into a
single `run_code` tool so the model can write Python that loops, branches, aggregates, and parallelizes tool work inside a sandbox.

Use it when the agent needs to call several tools, transform intermediate results, run concurrent
tool work with `asyncio.gather`, or anywhere a simple Python script is more reliable than the model alone, such as for mathematics.

## Install

Code Mode needs the Monty sandbox, pulled in by the `codemode` extra:

```bash
uv add "pydantic-ai-harness[codemode]"   # `code-mode` is also accepted as an alias
```

## Basic Pattern

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import CodeMode

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[CodeMode()])


@agent.tool_plain
def get_weather(city: str) -> dict:
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}


@agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    return round((fahrenheit - 32) * 5 / 9, 1)
```

The model could generate code like:

```python {test="skip" lint="skip"}
paris, tokyo = await asyncio.gather(
    get_weather(city='Paris'),
    get_weather(city='Tokyo'),
)
paris_c = await convert_temp(fahrenheit=paris['temp_f'])
tokyo_c = await convert_temp(fahrenheit=tokyo['temp_f'])
{'paris': paris_c, 'tokyo': tokyo_c}
```

This reduces four model round-trips to one.

## Choose Which Tools Are Sandboxed

The `tools` parameter controls which tools move behind `run_code`.

```python
from pydantic_ai_harness import CodeMode

CodeMode(tools='all')
CodeMode(tools=['search', 'fetch'])
CodeMode(tools=lambda ctx, td: td.name != 'dangerous_tool')
CodeMode(tools={'code_mode': True})
```

Metadata-based selection is useful when the project already groups tools into toolsets:

```python {test="skip" lint="skip"}
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

Non-matching tools remain regular tool calls.

## Return Values

`run_code` captures the last expression automatically.

| Scenario | Return |
| --- | --- |
| No print output | Last expression value |
| With print output | `{"output": "<printed text>", "result": <last expression>}` |
| Multimodal content | Returned natively for model processing |

If the user expects a raw dict or list back, avoid unnecessary `print()` statements.

## Retries

```python
from pydantic_ai_harness import CodeMode

CodeMode(
    tools='all',
    max_retries=3,
)
```

Use `max_retries` when sandbox execution errors are expected to be recoverable.
If the generated code fails during sandbox execution, the error message is sent back to the model, and it is asked to redraft the code in light of that failure. This process is repeated up to `max_retries` times, or until the code executes successfully.

## REPL State

State persists between `run_code` calls during the same agent run. Imports, variables, and helper
functions carry over until the run ends. Use `restart: true` in the tool call to reset state when the
conversation has drifted or stale state is causing wrong behavior.

## Sandbox Restrictions

Code runs inside Monty, an implementation of Python which intentionally supports only a subset of features.

Key restrictions:

- No class definitions
- No third-party imports
- No `import *`
- Only a small stdlib subset is allowed: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `datetime`, `os`, `pathlib`
- No wall-clock or timing primitives: `asyncio.sleep`, `datetime.datetime.now()`, `datetime.date.today()`, and the `time` module are unavailable
- Tools that need approval or deferred execution are excluded from the sandbox

When a generated example keeps failing, check these restrictions before changing the rest of the agent.

## API

```python {test="skip" lint="skip"}
CodeMode(
    tools: ToolSelector = 'all',   # 'all', list[str], callable, or dict
    max_retries: int = 3,          # retries on sandbox execution errors
)
```

## Agent Specs

When the user defines the agent in YAML or JSON, the loader needs to know how to build `CodeMode`:

```yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - CodeMode: {}
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import CodeMode

agent = Agent.from_file('agent.yaml', custom_capability_types=[CodeMode])
```

The same pattern applies when passing arguments:

```yaml
capabilities:
  - CodeMode:
      tools: ['search', 'fetch']
      max_retries: 5
```

## Observability

With Logfire or another OpenTelemetry backend, nested tool calls inside `run_code` produce child spans.
That makes CodeMode much easier to debug than a plain blob of generated code.

```python {test="skip" lint="skip"}
for msg in result.all_messages():
    for part in msg.parts:
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
            tool_calls = part.metadata['tool_calls']    # dict[str, ToolCallPart]
            tool_returns = part.metadata['tool_returns'] # dict[str, ToolReturnPart]
```
