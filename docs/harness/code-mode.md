# Code Mode

Code mode is one of the capabilities in [**Pydantic AI Harness**](overview.md), the official capability library for Pydantic AI. The full docs live in the [harness repo](https://github.com/pydantic/pydantic-ai-harness) -- this page is a short intro.

[`CodeMode`](https://github.com/pydantic/pydantic-ai-harness/blob/main/pydantic_ai_harness/code_mode/README.md) wraps your tools into a single `run_code` tool powered by our [Monty](https://github.com/pydantic/monty) sandbox. The model writes Python that calls multiple tools with loops, conditionals, variables, and `asyncio.gather` -- all inside one tool call.

Standard tool calling requires one model round-trip per tool call. An agent that needs to fetch 10 items and process each one makes 11+ model calls -- slow, expensive, and context-heavy. Code mode collapses that into one.

## Usage

```bash
uv add "pydantic-ai-harness[code-mode]"
```

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

## Full documentation

See the [Code Mode README](https://github.com/pydantic/pydantic-ai-harness/blob/main/pydantic_ai_harness/code_mode/README.md) in the harness repo for selective tool sandboxing, metadata-based selection, return value handling, REPL state, observability, sandbox restrictions, the full API, and agent spec usage.
