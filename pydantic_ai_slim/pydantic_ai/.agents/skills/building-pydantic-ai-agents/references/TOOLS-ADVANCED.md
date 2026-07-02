# Tools Advanced

Read this file when the user wants advanced tool behavior: approval, retries, validation, timeouts, rich tool returns, or tool search/deferred loading.

## Require Tool Approval (Human in the Loop)

Use deferred tools when the run should pause for approval.

```python
from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ToolDenied,
)

agent = Agent('openai:gpt-5.2', name='approval_agent', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    return f'File {path!r} deleted'


result = agent.run_sync('Delete __init__.py')
messages = result.all_messages()

assert isinstance(result.output, DeferredToolRequests)
results = DeferredToolResults()
for call in result.output.approvals:
    results.approvals[call.tool_call_id] = ToolDenied('Deleting files is not allowed')

result = agent.run_sync('Continue', message_history=messages, deferred_tool_results=results)
print(result.output)
```

Two key rules:

- `DeferredToolRequests` must be in the output type
- for conditional approval, raise `ApprovalRequired(...)` instead of marking the whole tool `requires_approval=True`

## Make an Agent Resilient with Retries

Raise `ModelRetry` from inside the tool when the model should correct and try again.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent('openai:gpt-5.2', name='retry_agent', deps_type=dict[str, int])


@agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[dict[str, int]], name: str) -> int:
    user_id = ctx.deps.get(name)
    if user_id is None:
        raise ModelRetry(f'No user found with name {name!r}')
    return user_id
```

Use retries for recoverable model mistakes, not application crashes.

## Validate or Require Approval Before Tool Execution

Use `args_validator=` when arguments are structurally valid but still need business-rule validation before execution or approval.

```python
from pydantic_ai import Agent, DeferredToolRequests, ModelRetry, RunContext

agent = Agent('openai:gpt-5.2', name='validation_agent', deps_type=int, output_type=[str, DeferredToolRequests])


def validate_sum_limit(ctx: RunContext[int], x: int, y: int) -> None:
    if x + y > ctx.deps:
        raise ModelRetry(f'Sum of x and y must not exceed {ctx.deps}')


@agent.tool(requires_approval=True, args_validator=validate_sum_limit)
def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
    return x + y
```

## Use Advanced Tool Features

Reach for these features when the user needs more than a simple function tool:

- `ToolReturn` for rich return values plus separate content/metadata
- `prepare=` for dynamic tool definitions
- `timeout=` for tool execution limits
- `sequential=True` to make a tool a barrier — it runs alone (tools emitted before it finish first, tools after it start once it finishes) while other tools parallelize around it; works on function tools and on output tools via `ToolOutput(sequential=True)`

Example with `ToolReturn`:

```python
from pydantic_ai import Agent, BinaryContent, ToolReturn

agent = Agent('openai:gpt-5.2', name='tool_return_agent')


@agent.tool_plain
def click_and_capture(x: int, y: int) -> ToolReturn:
    return ToolReturn(
        return_value=f'Successfully clicked at ({x}, {y})',
        content=['After:', BinaryContent(data=b'png-data', media_type='image/png')],
        metadata={'coordinates': {'x': x, 'y': y}},
    )
```

## Control Tool Execution When an Output Tool Is Called

When a model calls an output tool (structured output) in the *same* response as other tools, the agent's `end_strategy` controls how those calls run and which one becomes the final result. Most agents never need to touch this, since most responses don't mix an output tool with other tools.

Three strategies (set on the agent, e.g. `Agent(..., end_strategy='exhaustive')`):

- `'graceful'` (default): tools run in emission order; function tools always run (in parallel where possible); the first successful output tool wins, later output tools are skipped. Use when function tool side effects (logging, notifications) should still happen.
- `'early'`: output tools run in emission order, stopping at the first success; function tools in the same response are skipped if an output succeeds, but run if *every* output fails. Fastest when you don't need those function tools once you have a result.
- `'exhaustive'`: every tool runs in parallel; the first valid output by emission order wins; other output tools still execute. Gives the model full visibility that each tool ran, at the cost of discarded output-tool side effects.

Retry-wins (under `'graceful'` / `'exhaustive'`): if a function tool raises `ModelRetry` (or its args fail validation) in the same response as a successful output, the output result is suppressed so the model addresses the retry next round. Does not apply under `'early'`, nor when streaming (`run_stream` commits the first matching output immediately, behaving like `'early'`).

To run a whole run's tools serially, use `with agent.parallel_tool_call_execution_mode('sequential'):` or set `parallel_tool_calls=False` on model settings.

See [Parallel Output Tool Calls](https://ai.pydantic.dev/output/#parallel-output-tool-calls) and [tools-advanced docs](https://ai.pydantic.dev/tools-advanced/#parallel-tool-calls-concurrency).

## Handle Network Errors and Rate Limiting Automatically

For tool-call retries, use `ModelRetry` and tool `retries=...`.

For HTTP request retries at the transport layer, use the library's retry configuration separately. Do not assume `ModelRetry` alone solves provider transport failures.

## Tool Search and Tool-Level Deferred Loading

Use tool-level deferred loading when the agent has many tools and the model should discover individual tools on demand via `search_tools`.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-5.2', name='tool_search_agent')


@agent.tool_plain(defer_loading=True)
def lookup_internal_policy(policy_name: str) -> str:
    return f'policy details for {policy_name}'
```

Good fit:

- large MCP servers
- big tool catalogs
- situations where loading all tool schemas would bloat context

For bundle-level progressive disclosure of instructions plus tools, read [Capabilities on Demand](./ON-DEMAND-CAPABILITIES.md) instead.
