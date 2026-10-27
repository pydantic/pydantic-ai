# IPython Code Mode

`IPythonCodeMode` gives the model one `run_code` tool that runs Python in a persistent [IPython](https://ipython.org/) kernel, with your agent's tools defined as functions in it. Use it for local agents, such as a data-analysis or coding assistant on your own machine, that should have full CPython, the packages you have installed, and variables that last for the whole run. It is not a sandbox: for an agent that serves other people, use [Code Mode](../code_mode/).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/ipython_code_mode/)

> [!WARNING]
> **Not a sandbox: do not serve it from a webserver.** Model-written code runs as a full Python process with the permissions of the user running the agent: it can read and write files, reach the network, read environment variables and credentials, and start processes. Do not run `IPythonCodeMode` in a webserver, or anywhere it handles untrusted input such as user prompts, documents, or web pages, unless the whole agent runs inside a sandbox (a container or VM with only the access it needs). For production and web-facing agents, use the Monty-based [Code Mode](../code_mode/), which is strongly recommended there.

## Code Mode or IPython Code Mode

Both capabilities wrap the agent's tools behind a `run_code` tool, so the model writes one Python script instead of one tool call per step. They differ in where that script runs.

| | [Code Mode](../code_mode/) | IPython Code Mode |
|---|---|---|
| Built for | Production and web-facing agents | Local agents you trust to do whatever the code asks |
| Safe in a webserver | Yes | No, unless the agent runs in a sandbox |
| Python | The [Monty](https://github.com/pydantic/monty) subset, with time and memory limits | Full CPython, with any installed package |
| Filesystem and network | Only through `mount`, `os_access`, and tools | Unrestricted, with the agent process's permissions |
| State | Per run, plus an optional `repl_store` | A kernel per run, or one `IPythonKernel` kept across runs |
| Tool functions | `async`, concurrent with `asyncio.gather` | Synchronous, one call at a time |
| Limits | `resource_limits`, `max_tool_calls` | `timeout` per call, `max_tool_calls` |
| Durable execution | Temporal, DBOS, Prefect | Not supported |

## Installation

IPython Code Mode starts a Jupyter kernel through `jupyter-client` and `ipykernel`, available via the `ipython` extra:

uv:

```bash
uv add "pydantic-ai-harness[ipython]"
```

pip:

```bash
pip install "pydantic-ai-harness[ipython]"
```

## Usage

Add `IPythonCodeMode()` to the agent's `capabilities`, then register tools as usual. Every eligible regular tool becomes a function the code can call:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import IPythonCodeMode

# Local agent only: model-written code runs with your permissions.
agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[IPythonCodeMode()])


@agent.tool_plain
def get_sales(region: str) -> list[dict[str, float]]:
    """Get the monthly sales figures for a region."""
    return [{'month': 1, 'total': 1200.0}, {'month': 2, 'total': 1350.0}]


result = agent.run_sync('Compare sales growth in the north and south regions.')
print(result.output)
```

Inside one `run_code` call, the model writes code like the following (illustrative; the code the model writes will vary):

```python
import pandas as pd

north = pd.DataFrame(get_sales(region='north'))
south = pd.DataFrame(get_sales(region='south'))
growth = {name: df['total'].pct_change().mean() for name, df in [('north', north), ('south', south)]}
growth
```

`north`, `south`, and `pd` are still defined in the next `run_code` call. The code can also use IPython syntax: magics such as `%timeit`, shell escapes such as `!ls`, and top-level `await`.

## Selecting tools

The `tools` field takes the same [`ToolSelector`](https://ai.pydantic.dev/api/tools/) as Code Mode: `'all'` (the default), a list of tool names, a `(ctx, tool_def)` predicate, or a metadata dictionary. Tools that don't match stay visible to the model as regular tool calls. Framework control tools such as `search_tools`, tools that tool search hasn't revealed yet, native fallbacks, and other code-execution tools such as `Shell`'s `run_command` stay native whatever the selector says.

```python
from pydantic_ai_harness import IPythonCodeMode

IPythonCodeMode(tools=['get_sales', 'get_targets'])
```

Tool names that aren't Python identifiers are renamed for the kernel: an MCP tool called `get-weather` becomes `get_weather()`.

## Calling tools from code

Tools are defined in the kernel as synchronous functions with keyword-only parameters, and listed with their signatures in `run_code`'s description. Each call goes back to the agent over the kernel's Jupyter stdin channel and runs through Pydantic AI like a direct tool call: arguments are validated, capability hooks run, and approval or deferral handlers apply. Calls run one at a time; the kernel waits for each result.

- A call that fails raises `ToolCallError` in the kernel, with the tool's error message. Code can catch it.
- `max_tool_calls` (default 100) bounds the calls one `run_code` call makes. A call past it raises `ToolCallError`.
- Nested calls count toward the run's [`tool_calls_limit`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits), together with the `run_code` call. A call that would exceed it ends the run with `UsageLimitExceeded`.
- When a tool returns a [`ToolReturn`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#advanced-tool-returns), the code gets its `return_value`. Its `content` is sent to the model alongside the `run_code` result, in call order.

The `run_code` return carries every nested call in its metadata, under `tool_calls` and `tool_returns`, in the same shape as Code Mode.

## Return values

The last expression of the code is returned without a `print()`, following Code Mode's rules. A value with a JSON form is returned as data. Any other value is returned as its `repr`, the text IPython would show, so a DataFrame comes back as its table. An image, such as a PIL image or an `IPython.display.Image`, is returned as an image the model can see, and so is an image the code displays, such as a `matplotlib` figure.

| Scenario | Return |
|---|---|
| A result, nothing printed or displayed | The result |
| No result, nothing printed or displayed | `{}` |
| Printed output, no result | `{'output': '<printed text>'}` |
| Printed output and a result | `{'output': '<printed text>', 'result': <result>}` |
| Displayed images | A list: the printed text if any, the images, then the result if any |
| Printed output and an image result | A list: the printed text, then the image |

Printed output combines `stdout`, `stderr`, and the text of non-image displays, in order. It is capped at 10 MiB per call; the rest is dropped and the output says how much.

## Errors, timeouts, and restarts

Code that raises returns a model retry with the printed output and the traceback, and counts toward `max_retries` (default 3). The kernel keeps the state the code built before the error.

`timeout` (default 60 seconds) bounds each `run_code` call, not counting time spent in tool calls. Code still running then is interrupted with `KeyboardInterrupt`, the model gets a retry saying so, and the kernel keeps its state. Code that ignores the interrupt for 5 more seconds gets its kernel shut down. If the kernel process exits, for example because the code crashed it or called `os._exit()`, the model gets a retry saying its state is gone. Either way, the next call starts a fresh kernel. Pass `timeout=None` to remove the limit.

The model can pass `restart: true` with its code to restart the kernel and clear its state first. `input()` returns an empty string.

## The kernel

Each agent run starts its own kernel when the model first calls `run_code`, and shuts it down when the run ends. A run that never calls `run_code` starts no kernel. The kernel is a separate `ipykernel` process, launched from the `python3` kernelspec with the agent's own Python interpreter, so the code can import whatever the agent's environment has installed. The agent talks to it over Unix sockets in the temp directory. Windows has no such transport, so there the kernel listens on local TCP ports without transport encryption, and other users of the same machine can connect to them and read the code and its output.

### Keep the kernel across runs

To keep variables, imports, and loaded data across the runs of a conversation, create an `IPythonKernel` and pass it to `IPythonCodeMode(kernel=...)`. You own its lifecycle: use it as an async context manager, or call `shutdown()`.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import IPythonCodeMode
from pydantic_ai_harness.ipython_code_mode import IPythonKernel


async def main():
    async with IPythonKernel() as kernel:
        agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[IPythonCodeMode(kernel=kernel)])
        result = await agent.run('Load sales.csv into a DataFrame named `df`.')
        result = await agent.run('Plot monthly totals from `df`.', message_history=result.all_messages())
```

A shared kernel runs one cell at a time, so runs that use it wait for each other. It is bound to the event loop it started on.

`IPythonKernel` also takes:

- `kernel_name`: the Jupyter kernelspec to launch, for code that should run in a different Python environment. It must be an IPython kernel.
- `cwd`: the kernel's working directory; unset, it inherits the agent process's.
- `startup_timeout`: seconds to wait for a new kernel to answer (default 60).

## What carries over from Code Mode

`IPythonCodeMode` takes Code Mode's `tools`, `max_retries`, and `max_tool_calls`, uses the same return shapes and `restart` argument, and records nested calls the same way. Code Mode features that rely on Monty are not available:

- **Eager and speculative execution**: not supported. Both run code while the model is still streaming it, which needs a sandbox that can be stopped at any point.
- **Durable execution**: not supported. A kernel's state can't be replayed, so a run under Temporal, DBOS, or Prefect durability fails with `UserError` before any code runs.
- **`dynamic_catalog`**: not supported. Tool signatures stay in `run_code`'s description, which changes when the toolset does.
- **`mount`, `os_access`, `resource_limits`**: not needed. The kernel has the host's filesystem, environment, and clock, and `timeout` is its one execution limit.

## Observability

`IPythonCodeMode` emits no spans of its own. Core's [instrumentation](https://pydantic.dev/docs/ai/capabilities/instrumentation/) already records the `run_code` call and each tool call made from the kernel, and kernel restarts, timeouts, and crashes are reported to the model in the `run_code` retry, which the `run_code` span records.

## API

- `IPythonCodeMode(tools='all', max_retries=3, *, max_tool_calls=100, timeout=60.0, kernel=None)`: the capability.
- `IPythonKernel(kernel_name='python3', *, cwd=None, startup_timeout=60.0)`: a kernel to keep across runs, from `pydantic_ai_harness.ipython_code_mode`, with `start()`, `restart()`, and `shutdown()`.
- `IPythonCodeModeToolset`: the toolset the capability installs, for use without the capability.
