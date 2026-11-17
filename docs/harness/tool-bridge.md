---
title: Tool Bridge
description: "Keep a Pydantic AI agent's tool list fixed with three dispatcher tools, so the prompt cache holds however many tools sit behind them and however they change mid-run."
---

# Tool Bridge

Keep the tool list the model sees fixed while the tools behind it are many or change during a run, so the tool definitions at the head of the prompt stay cached on every provider. Use it for large or dynamic tool populations, such as several MCP servers, on providers that have no cache-safe way to reveal a tool mid-run.

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness import ToolBridge

agent = Agent('{{ anthropic:large }}', capabilities=[ToolBridge()])
```

`ToolBridge` replaces the agent's tools with three tools whose definitions don't depend on the tools behind them:

| Tool | Arguments | Returns |
|---|---|---|
| `tool_search` | `query` | The name and description of each matching tool, at most `max_results` |
| `tool_describe` | `name` | The tool's name, description and `parameters_json_schema` |
| `tool_call` | `name`, `args_json` | The tool's result |

`tool_call` takes the arguments as a JSON object string rather than a nested object, because some models send nested free-form objects as `null`. An empty string is read as `{}`.

A tool added, removed or changed mid-run changes what `tool_search` finds, not the tool definitions sent to the provider. A tool reached through the bridge is validated and run through the same path as a direct call: argument validation, tool hooks, `ModelRetry`, approval and usage limits apply to it as to any tool, and an unknown name or invalid arguments ask the model to retry. Both `tool_call` and the bridged tool count as tool calls, and `tool_calls_limit` is checked for both before the bridged tool runs.

## Options

- `tools` (default `'all'`): which tools to bridge, as a list of names or a `(ctx, tool_def) -> bool` callable, sync or async. The rest stay visible to the model as normal tools.
- `search`: ranks the bridged tools for a query, with the signature of a [`ToolSearchFunc`][pydantic_ai.capabilities.ToolSearchFunc]. Defaults to the keyword matching [`ToolSearch`](../capabilities/tool-search.md) uses.
- `max_results` (default `10`): the most tools one `tool_search` returns.
- `max_retries` (default `3`): retries for each dispatcher tool, spent on unknown tool names and invalid arguments.

## What stays native

These tools are never bridged, so they reach the model as usual:

- Tools that need approval or run outside the agent (`requires_approval=True`, external tools), so approval requests name the real tool. A bridged tool that raises `ApprovalRequired` itself is deferred as its `tool_call` call, and runs once that call is approved; a denial is recorded as the `tool_call` call's denied outcome.
- Framework control tools, such as tool search and `load_capability`.
- Local fallbacks for a provider-native tool (`unless_native`), so the model can drop them when the provider runs the tool natively.

Tools with `defer_loading=True` are bridged and found with `tool_search`, so [`ToolSearch`](../capabilities/tool-search.md) has nothing left to reveal. A tool a deferred capability owns joins the bridge once the capability is loaded. The tool names `tool_search`, `tool_describe` and `tool_call` are reserved: a native tool with one of them raises a `UserError`.

## Trade-offs

- Each bridged call takes an extra step: the model searches or describes a tool before it calls it, unless it already knows the name and arguments.
- The provider doesn't validate a bridged tool's arguments against its schema, so strict tool calling doesn't apply to them; Pydantic AI still validates them before the tool runs.
- A bridged tool's description and schema reach the model in tool results instead of the tool definitions.

## Composition

Two `ToolBridge` capabilities on one agent merge into one; the later one's settings win. [`CodeMode`](code-mode.md) wraps the bridge's three tools like any other tools, so to put many tools behind `run_code` instead, use one of the two.

## Telemetry

`ToolBridge` adds no spans of its own. With instrumentation on, `tool_call` and the bridged tool it runs each get a tool span, since the bridged call goes through the same tool hooks.

## API reference

- [`pydantic_ai_harness.tool_bridge` source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_bridge/)
- [Pydantic AI toolsets](../toolsets.md)

::: pydantic_ai_harness.ToolBridge
