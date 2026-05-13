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

## In practice

The [harness Quick start](../../README.md#quick-start) wires `CodeMode` up against an MCP server and a web search and asks it to find the most-discussed Hacker News story across three feeds, pull the comment thread and the submitter's profile, and search the web for follow-up coverage. CodeMode collapses that into two `run_code` calls: the first fetches all three feeds in parallel via `asyncio.gather`, dedupes by id, filters by score, and ranks by comment count -- in plain Python; the second batches the three follow-up calls (`hn_get_thread`, `hn_get_user`, `duckduckgo_search`) together.

[![CodeMode's first run_code: parallel asyncio.gather over three HN feeds, then a dedupe and a score filter](../../docs/images/code-mode-trace.png)](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946)

**[See the full Logfire trace →](https://logfire-us.pydantic.dev/public-trace/84bcf123-2106-49da-9f6f-5c26395339bb?spanId=7650806a0785b946)** Each `run_code` span fans out into the tool calls the model issued from inside the sandbox -- the easiest way to understand what code mode actually did. See the [Pydantic AI Logfire docs](https://ai.pydantic.dev/logfire/) for setup details.

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

### Tool Search

When you mark tools or whole toolsets `defer_loading=True` ([Tool Search](https://ai.pydantic.dev/tools-advanced/#tool-search)), `CodeMode` keeps them out of `run_code` while they're undiscovered — they pass straight through, so Tool Search drives them as usual (sent on the wire with `defer_loading` on providers with native tool search; otherwise dropped until discovered, with a `search_tools` tool alongside `run_code`). Once the model discovers a tool it comes back with `defer_loading=False`, and from then on `CodeMode` folds it into `run_code` like any other tool, so it's callable from generated code.

That fold-in grows `run_code`'s description, which invalidates the prompt-cache prefix once at the moment of discovery (turns with no discovery stay cache-warm). To instead keep a Tool Search corpus fully native — never folded into `run_code`, fully cache-stable, but not callable from inside it — exclude it with a `tools` selector; corpus members carry `with_native` set to the managing native tool:

```python
CodeMode(tools=lambda ctx, td: td.with_native is None)
```

A future Pydantic AI change will let `run_code`'s description stay static — newly discovered tools announced separately — so the fold-in costs nothing; until then, the selector above is the escape hatch.

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
- No wall-clock or timing primitives: `asyncio.sleep`, `datetime.datetime.now()`/`datetime.date.today()`, and the `time` module are unavailable
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
