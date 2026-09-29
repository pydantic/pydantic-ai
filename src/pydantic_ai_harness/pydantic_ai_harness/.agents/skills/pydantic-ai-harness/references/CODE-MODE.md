# Code Mode

`CodeMode` wraps eligible tools into a single `run_code` tool so the model can write Python that loops,
branches, aggregates, and parallelizes multiple tool calls inside a sandbox.

Use it when the agent needs to call several tools, transform intermediate results, run concurrent
tool work with `asyncio.gather`, or anywhere a simple Python script is more reliable than the model alone, such as for mathematics.

## Install

Code Mode needs the Monty sandbox, pulled in by the `codemode` extra:

```bash
uv add "pydantic-ai-harness[codemode,anthropic]"   # `code-mode` is an alias; `anthropic` for the examples
```

Without it, importing `CodeMode` raises `ImportError: pydantic-monty is required for CodeMode`.

## Basic Pattern

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import CodeMode

agent = Agent('anthropic:claude-opus-5-5', capabilities=[CodeMode()])


@agent.tool_plain
def get_weather(city: str) -> dict:
    return {'city': city, 'temp_f': 72, 'condition': 'sunny'}


@agent.tool_plain
def convert_temp(fahrenheit: float) -> float:
    return round((fahrenheit - 32) * 5 / 9, 1)
```

The model could generate code like:

```python {test="skip" lint="skip"}
import asyncio

paris, tokyo = await asyncio.gather(
    get_weather(city='Paris'),
    get_weather(city='Tokyo'),
)
paris_c = await convert_temp(fahrenheit=paris['temp_f'])
tokyo_c = await convert_temp(fahrenheit=tokyo['temp_f'])
{'paris': paris_c, 'tokyo': tokyo_c}
```

## Choose Which Tools Are Sandboxed

The `tools` parameter (a Pydantic AI `ToolSelector`) controls which tools move behind `run_code`.

```python
from pydantic_ai_harness import CodeMode

CodeMode(tools='all')
CodeMode(tools=['search', 'fetch'])
CodeMode(tools=lambda ctx, td: td.name in {'search', 'fetch'})
CodeMode(tools={'code_mode': True})
```

Metadata-based selection is useful when the project already groups tools into toolsets:

```python {test="skip" lint="skip"}
from pydantic_ai import Agent
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import CodeMode

search_tools = FunctionToolset(tools=[search, fetch]).with_metadata(code_mode=True)

agent = Agent(
    'anthropic:claude-opus-5-5',
    toolsets=[search_tools],
    capabilities=[CodeMode(tools={'code_mode': True})],
)
```

Non-matching tools remain regular tool calls. Some tools always stay native even with `tools='all'`:
framework control tools, undiscovered deferred (`defer_loading=True`) tools, native fallbacks, and other
code-execution tools (any tool with `code_arg_name` metadata) -- including `Shell`'s `run_command`, `start_command`, and `shell`, and `CapabilityCreation`'s
`author_capability`. `Shell`'s `check_command`/`stop_command` and all `FileSystem` tools are folded in.
Provider-native tools (`native=True`) execute server-side and never reach `run_code`.

With the default `tools='all'`, approval-gated tools and every tool from MCP or hosted-integration
capabilities (for example `GitHub` or `Linear` write tools) move behind `run_code` too, where an
approval-gated call fails as a retry unless a `HandleDeferredToolCalls` capability resolves it inline. When only some tools should be batched, tag those tools'
toolset with `.with_metadata(code_mode=True)` and use `CodeMode(tools={'code_mode': True})`, so
approval-gated and integration tools stay regular tool calls.

## Return Values

`run_code` captures the last non-`None` expression automatically; an assignment on the last line returns nothing.

| Scenario | Return |
| --- | --- |
| Final expression, no print output | Last expression value |
| Final assignment or `None`, no print output | `{}` |
| Print output, no final expression | `{'output': '<printed text>'}` |
| Print output and final expression | `{'output': '<printed text>', 'result': <last expression>}` |
| Multimodal final expression | Returned natively for model processing (with prints: a list, text first) |

If the user expects a raw dict or list back, avoid unnecessary `print()` statements. Printed output is
capped at 10 MiB. Tool results reach the sandbox as JSON: `Decimal`, `UUID` and `datetime` arrive as
strings and mapping keys are stringified (`bytes` cross unchanged).

## Limits and Retries

```python
from pydantic_ai_harness import CodeMode

CodeMode(
    tools='all',
    max_retries=3,
    max_tool_calls=25,
    resource_limits={'max_duration_secs': 10, 'max_memory': 128 * 1024 * 1024},
)
```

- `max_retries` (default 3): sandbox errors (including syntax/type errors) are sent back to the model to redraft, up to this many times.
- `max_tool_calls` (default 100): nested calls per `run_code` call. A call past the budget fails at its call site; if uncaught, the model gets a retry listing the calls that already started.
- `resource_limits` (default: 30 s execution and 256 MiB heap per snippet): keys `max_duration_secs`, `max_memory`, `max_suspensions` (default 1,000 host interactions per REPL session, shared across snippets). Unknown keys raise `UserError`. Time awaiting tools does not count; sleeps get a separate allowance of `max_duration_secs`. A snippet that hits the time limit is stopped and its REPL session is reset. `'unlimited'` removes time and memory caps only when another boundary supplies them; `max_suspensions` can never be disabled.

## REPL State

State persists between `run_code` calls during the same agent run. Imports, variables, and helper
functions carry over until the run ends. The model passes `restart: true` in the `run_code` arguments to
reset state. A timeout, worker crash, or interrupted eager statement also resets the session, and the
retry message says so -- the next snippet must recreate its state.

## Sandbox Restrictions

Code runs inside Monty, an implementation of Python which intentionally supports only a subset of features.

Key restrictions:

- No third-party imports
- No `import *`
- Only a small stdlib subset is allowed, and each must be imported before use: `sys`, `typing`, `asyncio`, `math`, `json`, `re`, `unicodedata`, `datetime`, `time`, `random`, `os`, `pathlib`
- `asyncio.gather(...)` accepts positional awaitables but no keyword arguments; other task creation and wait APIs are unavailable
- No clock or randomness by default: `datetime.datetime.now()`, `datetime.date.today()`, `time.time()`, and unseeded `random` require an `os_access` handler; `time.sleep` and `asyncio.sleep` really wait, within the sleep allowance
- Filesystem I/O requires an `os_access` handler or a `mount`; `os.getenv` and `os.environ` require an `os_access` handler
- Tools requiring approval or with deferred (`CallDeferred`) execution are sandboxed like any other tool; without a `HandleDeferredToolCalls` (or equivalent) capability to resolve them inline, calling one from `run_code` raises an error that surfaces to the model as a retry

The sandbox constrains the model-generated Python, not the implementation of the tools it calls.
Wrapped tools retain their normal host and network access, so expose only tools with the authority and
input validation appropriate for model-generated calls.

When a generated example keeps failing, check these restrictions before changing the rest of the agent.

## Host Access: `mount` and `os_access`

Leave both unset unless the task requires host access, and grant only what the task needs. Both are fixed
when `CodeMode` is built, so construct it per request to scope access to that request.

```python {test="skip"}
from pathlib import Path

from pydantic_monty import MountDir, OSAccess

from pydantic_ai_harness import CodeMode

work = Path('agent-work')
work.mkdir(exist_ok=True)  # the host directory must exist
CodeMode(mount=MountDir(virtual_path='/work', host_path=str(work), mode='read-write'))
CodeMode(os_access=OSAccess(environ={'API_BASE': 'https://api.example.com'}))
```

- `mount` takes a `MountDir` or a list of them; `host_path` must already exist (`MountDir` raises `TypeError` otherwise). The default `mode='overlay'` is copy-on-write: writes are visible only within the current `run_code` call and never reach the host. Use `mode='read-write'` when writes must persist (including across `run_code` calls), `mode='read-only'` to forbid them.
- `os_access` takes an `AbstractOS` (e.g. `OSAccess`, isolated in-memory filesystem and env, host clock) or a keyword-argument callback `def handler(*, name, args, kwargs, **_)` (may be `async`). Return any value (including `None`) to answer the call; return `pydantic_monty.NOT_HANDLED` to refuse it (raises in the sandbox and burns a retry). The positional `(name, args, kwargs)` form is deprecated.
- Mounts are directories on the machine running the agent, not the run's workspace. With a remote sandbox such as `ModalSandbox`, `FileSystem` and `Shell` act in the sandbox while mounted `pathlib` code still touches the host -- for files the model shares with its commands, sandbox the workspace tools (`FileSystem`) instead of mounting.

## Other Options

```python {test="skip" lint="skip"}
CodeMode(
    tools: ToolSelector = 'all',
    max_retries: int = 3,
    *,
    max_tool_calls: int = 100,
    os_access: CodeModeOS | None = None,
    mount: CodeModeMount | None = None,
    resource_limits: CodeModeResourceLimits | Literal['unlimited'] | None = None,
    eager: bool = False,                   # run complete streamed statements before the call finishes
    speculate: Sequence[str] | Literal['declared'] | None = None,  # start read-only calls while streaming
    monty_sandbox_url: str | None = None,  # ws:// or wss:// remote Monty worker
    dynamic_catalog: bool = False,         # move the tool catalog into dynamic instructions
)
```

- `dynamic_catalog=True`: keeps `run_code`'s description byte-stable and moves sandboxed-tool signatures into dynamic instructions, announcing newly discovered tools with a system prompt part. Worth it only with `ToolSearch`; with a fixed toolset the default is better.
- `eager=True`: side effects from early statements cannot be rolled back, and hooks/approval on `run_code` run only after the call finishes streaming. Only applies when `run_code` is the first tool call in the response.
- `speculate`: pass tool names that are safe to run early, or `'declared'` to trust `Tool(metadata={'read_only': True})` and MCP `readOnlyHint`. Only calls with literal keyword arguments start early; unclaimed launches still cost what they cost. `CodeMode.speculation_stats` reports launched/adopted/evicted.
- `eager` and `speculate` put runs in streaming mode, need asyncio, and are inactive under durable execution (Temporal, DBOS, Prefect).
- `monty_sandbox_url`: only execution moves; tools, mounts, `os_access`, and prints stay host-side. Use `wss://` unless the network is trusted.

## Durable Execution

`CodeMode` works next to `TemporalDurability()` (install `pydantic-ai-harness[codemode,temporal]`), but
`run_code` executes in workflow code and is re-run on replay. Keep snippets deterministic: put external
reads, writes, and clock access in wrapped tools (which become activities), not in `mount`/`os_access`.
Inside a Temporal workflow `max_duration_secs` is disabled (memory and suspension caps still apply) and a
workflow task that does not yield within 2 seconds fails, so move heavy computation into a tool.

## Agent Specs

When the user defines the agent in YAML or JSON, the loader needs to know how to build `CodeMode`.
YAML files also need PyYAML: `uv add "pydantic-ai-harness[codemode,anthropic]" "pydantic-ai-slim[spec]"`.

```yaml
model: anthropic:claude-opus-5-5
capabilities:
  - CodeMode: {}   # or with arguments: CodeMode: {tools: ['search', 'fetch'], max_retries: 5}
```

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import CodeMode

agent = Agent.from_file('agent.yaml', custom_capability_types=[CodeMode])
```

## Observability

With Logfire or another OpenTelemetry backend, nested tool calls inside `run_code` produce child spans.
The `run_code` tool return also carries history-only metadata keyed by nested call id (plus a
`speculation` summary when `speculate` is set):

```python {test="skip" lint="skip"}
for msg in result.all_messages():
    for part in msg.parts:
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
            tool_calls = part.metadata['tool_calls']    # dict[str, ToolCallPart]
            tool_returns = part.metadata['tool_returns'] # dict[str, ToolReturnPart]
```

For an offline test that drives `run_code` with a `FunctionModel`, see [Testing and Debugging](./TESTING-AND-DEBUGGING.md).

## See also

- https://pydantic.dev/docs/ai/harness/code-mode/
- https://pydantic.dev/docs/ai/harness/durable-execution/
- https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/
