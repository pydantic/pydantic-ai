# Code Mode

!!! warning "Experimental"
    Code mode is an experimental feature under active development. The API may change in future releases.

Instead of calling [tools](tools.md) one at a time via the standard tool-calling protocol, code mode lets the model **write and execute Python code** that orchestrates multiple tool calls in a single step — with loops, conditionals, variables, and parallel execution.

This approach was pioneered by Anthropic's research on [tool use via code](https://www.anthropic.com/engineering/code-execution-with-mcp) and is now available in Pydantic AI as a first-class feature, with [Cloudflare's production deployment](https://blog.cloudflare.com/code-mode/) as a notable early adopter.

## Why Code Mode?

With standard tool calling, each tool call is a separate round-trip to the model. If an agent needs to fetch 10 items and then look up details for each one, that's 11 model calls — slow and expensive.

With code mode, the model writes a script that does it all at once:

```python {test="skip" lint="skip"}
# The model writes this code, which runs in a sandbox
items = await get_items(category="electronics")

# Fire all detail lookups at once (each returns a Future immediately)
futures = [get_details(id=item["id"]) for item in items]
# Await results — all calls are already in flight
details = [await f for f in futures]

# Process locally — no model calls needed
total = sum(d["price"] for d in details if d["in_stock"])
{"total": total, "count": len(details)}
```

One model call instead of eleven. The model thinks in code, which it's already good at, and the runtime handles executing it safely.

**Key benefits:**

- **Fewer model round-trips** — complex multi-step workflows run in a single `run_code` call
- **Natural parallelism** — fire multiple tool calls concurrently with fire-then-await
- **Richer logic** — loops, conditionals, and data transformations happen in code, not in the model's "head"
- **Type safety** — with Monty, generated code is type-checked before execution

## Install

Code mode requires a runtime to execute the generated code. For the recommended [Monty](#monty) runtime:

```bash
pip/uv-add "pydantic-ai-slim[monty]"
```

## Quick Start

```python {title="code_mode_example.py" test="skip"}
from pydantic_ai import Agent
from pydantic_ai.runtime import MontyRuntime
from pydantic_ai.toolsets import CodeModeToolset, FunctionToolset


def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    # In production, this would call a real API
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}


def convert_temp(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return round((fahrenheit - 32) * 5 / 9, 1)


tools = FunctionToolset(tools=[get_weather, convert_temp])

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    toolsets=[CodeModeToolset(tools)],
)

result = agent.run_sync("What's the weather in Paris and Tokyo, in Celsius?")
print(result.output)
```

The model sees `get_weather` and `convert_temp` as callable Python functions, and writes code like:

```python {test="skip" lint="skip"}
# Parallel weather lookups
future_paris = get_weather(city="Paris")
future_tokyo = get_weather(city="Tokyo")
paris = await future_paris
tokyo = await future_tokyo

# Convert both temperatures
paris_c = await convert_temp(fahrenheit=paris["temp_f"])
tokyo_c = await convert_temp(fahrenheit=tokyo["temp_f"])

{"paris": paris_c, "tokyo": tokyo_c}
```

Two cities looked up in parallel, both converted, all in one model call.

## How It Works

[`CodeModeToolset`][pydantic_ai.toolsets.code_mode.CodeModeToolset] wraps any existing [toolset](toolsets.md) and replaces its individual tools with a single `run_code` tool. When the model calls `run_code`, the toolset:

1. **Generates function signatures** from the wrapped tools so the model knows what's available
2. **Builds a prompt** describing the execution model (fire-then-await parallelism, keyword arguments, etc.)
3. **Sends the model's code** to the runtime for execution
4. **Intercepts tool calls** made by the code and routes them through the normal Pydantic AI tool pipeline (with validation, tracing, and dependency injection)
5. **Returns the result** back to the model

Execution errors (syntax, type, or runtime) are automatically retried — the error message is sent back to the model so it can fix its code and try again, up to `max_retries` times (default 3).

## Monty

[Monty](https://github.com/pydantic/monty) is a minimal, secure Python interpreter built by the Pydantic team specifically for code mode. It's the recommended runtime for most use cases.

!!! danger "Early Stage — Not for Untrusted Prompts"
    Monty is under active development. **Do not use it in production systems where untrusted user prompts are passed directly to the model.** While Monty is designed for safe sandboxed execution, it has not yet undergone the level of hardening required for adversarial inputs. This restriction will be relaxed as Monty matures.

**What makes Monty special:**

- **Type checking** — code is type-checked (via [ty](https://github.com/astral-sh/ty)) before execution, catching errors before they happen and giving the model precise feedback to fix its code
- **Snapshot-based checkpointing** — Monty can freeze and restore its full interpreter state, enabling efficient resume without re-executing code from scratch
- **In-process execution** — no containers, no network, no cold starts — Monty runs directly in your Python process
- **Zero configuration** — just `MontyRuntime()`, no infrastructure to manage

```python {title="monty_runtime.py" test="skip"}
from pydantic_ai.runtime import MontyRuntime

runtime = MontyRuntime()
```

Because Monty runs a restricted Python subset, the runtime automatically informs the model of syntax restrictions (no imports, use only provided functions and builtins).

## Other Runtimes

Code mode is designed around a pluggable runtime abstraction. While Monty is the recommended default, the [`DriverBasedRuntime`][pydantic_ai.runtime.DriverBasedRuntime] base class supports executing code in any sandbox that can run a Python process — using a lightweight [driver script](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/runtime/_driver.py) that communicates over stdin/stdout.

### Modal

[`ModalRuntime`][pydantic_ai.runtime.ModalRuntime] runs code in [Modal](https://modal.com/) cloud sandboxes, with gVisor isolation and automatic infrastructure management. This is a good choice when you need full CPython compatibility (arbitrary imports, C extensions) or are running untrusted code.

Install the Modal dependency:

```bash
pip/uv-add "pydantic-ai-slim[modal]"
```

You'll also need a Modal account and API key — see the [Modal Sandboxes guide](https://modal.com/docs/guide/sandboxes) for setup.

```python {title="modal_runtime.py" test="skip" lint="skip"}
from pydantic_ai.runtime import ModalRuntime

runtime = ModalRuntime(
    app_name='my-code-agent',
    timeout=300,  # sandbox lifetime in seconds
)
```

Modal sandboxes are ephemeral — a fresh isolated environment is created per execution. You can provide a custom `image` to pre-install dependencies.

### Docker

[`DockerRuntime`][pydantic_ai.runtime.DockerRuntime] runs code inside an existing Docker container. You manage the container lifecycle; the runtime handles communication.

```python {title="docker_runtime.py" test="skip" lint="skip"}
from pydantic_ai.runtime import DockerRuntime

runtime = DockerRuntime(container_id='my-sandbox-container')

# One-time setup: copy the driver script into the container
await runtime.copy_driver_to_container()
```

### Building a Custom Runtime

To support a new sandbox environment (e.g., E2B, Firecracker, WebAssembly), implement the [`DriverTransport`][pydantic_ai.runtime.DriverTransport] protocol — just four async methods:

```python {test="skip" lint="skip"}
from pydantic_ai.runtime import DriverTransport

class MyTransport(DriverTransport):
    async def read_line(self) -> bytes: ...
    async def write_line(self, data: bytes) -> None: ...
    async def read_stderr(self) -> bytes: ...
    async def kill(self) -> None: ...
```

Then subclass [`DriverBasedRuntime`][pydantic_ai.runtime.DriverBasedRuntime] and implement `_start_driver()` to launch your sandbox and return the transport. All protocol handling, tool dispatch, and checkpoint/resume logic is inherited.

## Customizing the Prompt

The prompt that describes available functions and the execution model to the LLM is generated by [`build_code_mode_prompt`][pydantic_ai.toolsets.code_mode.build_code_mode_prompt]. You can replace it entirely:

```python {test="skip" lint="skip"}
from pydantic_ai.toolsets import CodeModeToolset, FunctionToolset
from pydantic_ai.runtime import MontyRuntime


def my_prompt_builder(*, signatures: list[str], runtime_instructions: str) -> str:
    funcs = '\n'.join(signatures)
    return f'Write Python code using these functions:\n\n{funcs}'


toolset = FunctionToolset(tools=[...])
toolset = CodeModeToolset(
    toolset,
    prompt_builder=my_prompt_builder,
)
```

## MCP Tools

Code mode works with any toolset, including [MCP servers](mcp/client.md). Tool names that aren't valid Python identifiers (e.g. `search-records`) are automatically sanitized (to `search_records`) so the model can call them naturally from code.

## Known Limitations

- **Tool approval and deferral** — [deferred tools](deferred-tools.md) (tools that require approval or external execution) are not yet supported in code mode. This will be added in a future release.
- **Streaming** — code mode does not currently support streaming partial results.
- **Monty's restricted Python** — Monty runs a subset of Python: no imports, no classes, no decorators. The model is instructed about these restrictions, but very complex code may need a driver-based runtime with full CPython.
