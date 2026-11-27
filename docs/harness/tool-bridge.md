---
title: Tool Bridge
description: "Keep a Pydantic AI agent's tool list fixed with three dispatcher tools, so the prompt cache holds however many tools sit behind them and however they change mid-run."
---

# Tool Bridge

Keep the tool list the model sees fixed while the tools behind it are many or change during a run, so the tool definitions at the head of the prompt stay cached on every provider. Use it for large or dynamic tool populations, such as several MCP servers, on providers that have no cache-safe way to reveal a tool mid-run.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ToolBridge

agent = Agent('{{ anthropic:large }}', capabilities=[ToolBridge()])
```

`ToolBridge` replaces the agent's tools with three tools whose definitions don't depend on the tools behind them:

| Tool | Arguments | Returns |
|---|---|---|
| `tool_search` | `queries` (a list of strings) | The name and description of each matching tool, at most `max_results` |
| `tool_describe` | `name` | The tool's name, description and `parameters_json_schema` |
| `tool_call` | `name`, `args_json` | The tool's result |

`tool_call` takes the arguments as a JSON object string rather than a nested object, because some models send nested free-form objects as `null`. An empty string is read as `{}`. `tool_call` has the tool kind `TOOL_CALL_KIND` (`tool_bridge.tool_call`), exported from `pydantic_ai_harness.tool_bridge`, so an approval UI or another capability can recognize it whatever it is named.

A tool added, removed or changed mid-run changes what `tool_search` finds, not the tool definitions sent to the provider. A tool reached through the bridge is validated and run through the same path as a direct call, under the `tool_call` call's ID: argument validation, tool hooks, `ModelRetry` and usage limits apply to it as to any tool (approval works differently, see [What stays native](#what-stays-native)), and an unknown name or invalid arguments ask the model to retry. Both `tool_call` and the bridged tool count as tool calls, and `tool_calls_limit` is checked for both before the bridged tool runs. A bridged tool's retries are counted under its own name, as for a direct call, so its `ctx.retry` and `max_retries` see only its own failures; each one also spends a retry of `tool_call`. If any bridged tool is `sequential`, so is `tool_call`.

## Options

- `tools` (default `'all'`): which tools to bridge, as a list of names, a metadata dict such as `{'bridge': True}` (matching the tools whose metadata contains those entries, for example tools tagged by `SetToolMetadata`; like a list of names, it survives spec serialization), or a `(ctx, tool_def) -> bool` callable, sync or async. The rest stay visible to the model as normal tools.
- `search`: ranks the bridged tools for `tool_search`'s queries, with the signature of a [`ToolSearchFunc`][pydantic_ai.capabilities.ToolSearchFunc]. `None` (default) ranks tools by how many of the queries' words appear in their name or description.
- `max_results` (default `10`): the most tools one `tool_search` returns.
- `max_retries` (default `None`, the agent's tool retry budget, as `Agent(retries={'tools': N})` sets it): retries for each dispatcher tool, spent on unknown tool names, invalid arguments and retries the bridged tools ask for.

## What stays native

These tools are never bridged, so they reach the model as usual:

- Tools that need approval or run outside the agent (`requires_approval=True`, external tools), so approval requests name the real tool. A bridged tool that raises `ApprovalRequired` or `CallDeferred` itself can be resolved by a [`HandleDeferredToolCalls`](../capabilities/handle-deferred-tool-calls.md) handler, which sees the tool's own name; a denial is recorded as the `tool_call` call's denied outcome. Without a handler that resolves it, the run fails with a `UserError` rather than deferring `tool_call`, whose approval would cover whichever tool the model named. Keep such a tool native with `tools`, for example the tools of a toolset wrapped with `approval_required()`, if it needs approval from outside the run.
- Tools with a `tool_kind`, such as tool search, `load_capability` and `FileSystem`'s `read_file`, which the run and other capabilities find by kind (`ToolOutputLimits`, `Memory` and compaction among them), and tools that step aside for one with `unless_tool_kind`, so the run leaves out the same tools it would without the bridge.
- Code-execution tools, such as Code Mode's `run_code` and Shell's `run_command`, which take a program as a string argument, so it isn't sent JSON-escaped inside `args_json`.
- Local fallbacks for a provider-native tool (`unless_native`), so the model can drop them when the provider runs the tool natively.

Tools with `defer_loading=True` are bridged and found with `tool_search`, so [`ToolSearch`](../capabilities/tool-search.md) has nothing left to reveal. Deferred tools that a capability owns, such as the tools of a capability with `defer_loading=True`, stay native, so core's tool search and `load_capability` reveal them as usual. The tool names `tool_search`, `tool_describe` and `tool_call` are reserved: an agent tool with one of them raises a `UserError`.

## Trade-offs

- Each bridged call takes an extra step: the model searches or describes a tool before it calls it, unless it already knows the name and arguments.
- The provider doesn't validate a bridged tool's arguments against its schema, so strict tool calling doesn't apply to them; Pydantic AI still validates them before the tool runs.
- A bridged tool's description and schema reach the model in tool results instead of the tool definitions.

## Composition

Two `ToolBridge` capabilities on one agent merge into one; the later one's settings win. One bridge per agent is supported: given distinct `id`s, the outer bridge refuses the inner one's reserved tool names with a `UserError`. The bridge sits just inside tool search and outside every other capability that wraps the toolset, whatever order the capabilities are listed in, so wrappers such as `SetToolMetadata` and `PrefixTools` apply to the bridged tools. With [`CodeMode`](code-mode.md) too, Code Mode exposes the three dispatchers as sandbox functions rather than the bridged tools; use Code Mode alone to put many tools behind `run_code`.

## Telemetry

`ToolBridge` adds no spans of its own. With instrumentation on, `tool_call` and the bridged tool it runs each get a tool span, since the bridged call goes through the same tool hooks.

## API reference

- [`pydantic_ai_harness.tool_bridge` source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_bridge/)
- [Pydantic AI toolsets](../toolsets.md)

::: pydantic_ai_harness.ToolBridge

::: pydantic_ai_harness.tool_bridge.TOOL_CALL_KIND

::: pydantic_ai_harness.tool_bridge.ToolCallArgs
