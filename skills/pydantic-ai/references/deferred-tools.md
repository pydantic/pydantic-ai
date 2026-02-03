# Deferred Tools Reference (Human-in-the-Loop)

Source: `pydantic_ai_slim/pydantic_ai/tools.py`, `pydantic_ai_slim/pydantic_ai/output.py`

## Overview

Deferred tools handle two scenarios:
1. **Human-in-the-loop approval** — tools that require user approval before execution
2. **External execution** — tools whose results come from outside the agent run

When a deferred tool is called, the run ends with `DeferredToolRequests`. Resume by providing `DeferredToolResults`.

## Tool Approval Workflow

### Always Require Approval

```python {title="always_require_approval.py"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults

agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file (requires approval)."""
    return f'Deleted {path}'


# First run — tool is deferred
result = agent.run_sync('Delete config.json')
messages = result.all_messages()

assert isinstance(result.output, DeferredToolRequests)
requests = result.output

# Process approvals
results = DeferredToolResults()
for call in requests.approvals:
    # True = approved, False or ToolDenied = denied
    results.approvals[call.tool_call_id] = True

# Resume with approvals
result = agent.run_sync(
    message_history=messages,
    deferred_tool_results=results,
)
print(result.output)
```

### Conditional Approval

```python {title="conditional_approval.py"}
from pydantic_ai import Agent, ApprovalRequired, DeferredToolRequests, RunContext

agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests])

PROTECTED_FILES = {'.env', 'secrets.json'}


@agent.tool
def update_file(ctx: RunContext, path: str, content: str) -> str:
    """Update a file, requiring approval for protected files."""
    if path in PROTECTED_FILES and not ctx.tool_call_approved:
        raise ApprovalRequired(metadata={'reason': 'protected file'})
    return f'Updated {path}'
```

Key context attributes:
- `ctx.tool_call_approved` — `True` if already approved
- `ApprovalRequired(metadata={...})` — attach context for UI display

## External Tool Execution

For tools executed outside the agent process (background tasks, frontend, etc.):

```python {title="external_tool.py" test="skip"}
from pydantic_ai import Agent, CallDeferred, DeferredToolRequests, DeferredToolResults, RunContext

agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests])


@agent.tool
async def long_running_task(ctx: RunContext, query: str) -> str:
    """Execute a task externally."""
    # Schedule background work, tracking by tool_call_id
    task_id = schedule_background_task(ctx.tool_call_id, query)
    raise CallDeferred(metadata={'task_id': task_id})


# First run — tool is deferred
result = await agent.run('Process this data')
messages = result.all_messages()
requests = result.output  # DeferredToolRequests

# Later, when results are ready:
results = DeferredToolResults()
for call in requests.calls:
    task_id = requests.metadata[call.tool_call_id]['task_id']
    results.calls[call.tool_call_id] = get_task_result(task_id)

# Resume with results
result = await agent.run(
    message_history=messages,
    deferred_tool_results=results,
)
```

## DeferredToolRequests Structure

```python
DeferredToolRequests(
    calls=[...],        # External tools (CallDeferred)
    approvals=[...],    # Tools needing approval (ApprovalRequired)
    metadata={          # Keyed by tool_call_id
        'tool_call_id': {'custom': 'data'}
    },
)
```

Each item is a `ToolCallPart` with:
- `tool_name` — name of the tool
- `args` — validated arguments dict
- `tool_call_id` — unique identifier

## DeferredToolResults Structure

```python
from pydantic_ai import DeferredToolResults, ToolApproved, ToolDenied, ModelRetry

results = DeferredToolResults()

# For approvals:
results.approvals['call_id_1'] = True  # Approved
results.approvals['call_id_2'] = False  # Denied
results.approvals['call_id_3'] = ToolDenied('Not allowed')  # Denied with message
results.approvals['call_id_4'] = ToolApproved(override_args={'path': '/safe/path'})

# For external calls:
results.calls['call_id_5'] = 'task result'
results.calls['call_id_6'] = {'structured': 'data'}
results.calls['call_id_7'] = ModelRetry('Task failed, try again')

# Optional metadata for tool execution
results.metadata['call_id_1'] = {'user_id': 'admin'}
```

## Including DeferredToolRequests in Output Type

```python
# At agent construction
agent = Agent('openai:gpt-5', output_type=[str, DeferredToolRequests])

# Or at runtime (overrides construction-time type)
result = agent.run_sync(
    'Do something',
    output_type=[str, DeferredToolRequests],
)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `DeferredToolRequests` | `pydantic_ai.DeferredToolRequests` | Output when tools are deferred |
| `DeferredToolResults` | `pydantic_ai.DeferredToolResults` | Input to resume deferred tools |
| `ApprovalRequired` | `pydantic_ai.ApprovalRequired` | Raise to require approval |
| `CallDeferred` | `pydantic_ai.CallDeferred` | Raise for external execution |
| `ToolApproved` | `pydantic_ai.ToolApproved` | Approval with optional override args |
| `ToolDenied` | `pydantic_ai.ToolDenied` | Denial with optional message |
| `ModelRetry` | `pydantic_ai.ModelRetry` | Signal tool failure for retry |

## See Also

- [tools.md](tools.md) — Basic tool registration
- [toolsets.md](toolsets.md) — `ApprovalRequiredToolset` for toolset-level approval
- [messages.md](messages.md) — Message history for resuming runs
- [observability.md](observability.md) — Logfire debugging
