# Exceptions Reference

Source: `pydantic_ai_slim/pydantic_ai/exceptions.py`

## Exception Hierarchy

```
Exception
├── ModelRetry              # Raise in tools to retry with feedback
├── CallDeferred            # Raise in tools to defer execution
├── ApprovalRequired        # Raise in tools to request approval
├── UserError (RuntimeError)  # Developer usage mistake
├── AgentRunError (RuntimeError)
│   ├── UsageLimitExceeded
│   ├── UnexpectedModelBehavior
│   │   ├── ContentFilterError
│   │   └── IncompleteToolCall
│   └── ModelAPIError
│       └── ModelHTTPError
└── FallbackExceptionGroup (ExceptionGroup)
```

## ModelRetry — Tool Retry

Raise inside a tool function to send an error message back to the model and retry the tool call.

```python {title="tool_retry_example.py"}
from pydantic_ai import Agent, ModelRetry

agent = Agent('openai:gpt-5')


@agent.tool_plain(retries=3)
def validate_input(value: str) -> str:
    """Validate and process input."""
    if not value.strip():
        raise ModelRetry('Input cannot be empty, please provide a valid value.')
    return f'Processed: {value}'


result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

The retry message is sent as a `RetryPromptPart` to the model.
Max retries are controlled by `Tool(max_retries=...)` or `Agent(retries=...)`.

## CallDeferred — Deferred Execution

Raise inside a tool to defer its execution for later (e.g., human-in-the-loop):

```python
from pydantic_ai.exceptions import CallDeferred

@agent.tool_plain
def expensive_action(target: str) -> str:
    raise CallDeferred(metadata={'target': target})
```

The agent run returns with `DeferredToolRequests`. Resume with `DeferredToolResults`.

## ApprovalRequired — Human Approval

Similar to `CallDeferred`, but specifically for tools that need explicit approval:

```python
from pydantic_ai.exceptions import ApprovalRequired

@agent.tool_plain(requires_approval=True)
def delete_record(record_id: int) -> str:
    """Delete a record."""
    return f'Deleted record {record_id}'
```

When `requires_approval=True`, the tool automatically raises `ApprovalRequired`.

## AgentRunError

Base class for errors that occur during an agent run.

```python
from pydantic_ai.exceptions import AgentRunError

try:
    result = await agent.run('prompt')
except AgentRunError as e:
    print(e.message)
```

## UsageLimitExceeded

Raised when token usage exceeds configured limits:

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

result = await agent.run(
    'prompt',
    usage_limits=UsageLimits(request_limit=5, total_token_limit=1000),
)
```

## UnexpectedModelBehavior

Raised when the model behaves unexpectedly (e.g., invalid output, unexpected stop reason):

```python
from pydantic_ai.exceptions import UnexpectedModelBehavior

# Has: message, body (optional raw response body)
```

### ContentFilterError

Subclass of `UnexpectedModelBehavior`. Raised when the model provider's content filter is triggered.

### IncompleteToolCall

Subclass of `UnexpectedModelBehavior`. Raised when the model stops mid-tool-call due to token limits.

## ModelAPIError

Raised when a model provider API request fails:

```python
from pydantic_ai.exceptions import ModelAPIError

# Has: model_name, message
```

### ModelHTTPError

Subclass of `ModelAPIError`. Raised for HTTP 4xx/5xx responses:

```python
from pydantic_ai.exceptions import ModelHTTPError

# Has: status_code, model_name, body
```

## FallbackExceptionGroup

Raised when all models in a `FallbackModel` fail. Contains all individual exceptions:

```python
from pydantic_ai.exceptions import FallbackExceptionGroup

try:
    result = await agent.run('prompt')
except FallbackExceptionGroup as e:
    for exc in e.exceptions:
        print(type(exc).__name__, exc)
```

## UserError

Raised for developer usage mistakes (e.g., invalid configuration):

```python
from pydantic_ai.exceptions import UserError

# Runtime error caused by incorrect API usage
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `ModelRetry` | `pydantic_ai.ModelRetry` | Retry tool with feedback |
| `CallDeferred` | `pydantic_ai.CallDeferred` | Defer tool execution |
| `ApprovalRequired` | `pydantic_ai.ApprovalRequired` | Request human approval |
| `AgentRunError` | `pydantic_ai.AgentRunError` | Base run error |
| `UsageLimitExceeded` | `pydantic_ai.UsageLimitExceeded` | Token limit exceeded |
| `UnexpectedModelBehavior` | `pydantic_ai.UnexpectedModelBehavior` | Unexpected model output |
| `ContentFilterError` | `pydantic_ai.exceptions.ContentFilterError` | Content filter triggered |
| `IncompleteToolCall` | `pydantic_ai.IncompleteToolCall` | Truncated tool call |
| `ModelAPIError` | `pydantic_ai.ModelAPIError` | API request failure |
| `ModelHTTPError` | `pydantic_ai.ModelHTTPError` | HTTP error response |
| `FallbackExceptionGroup` | `pydantic_ai.FallbackExceptionGroup` | All fallbacks failed |
| `UserError` | `pydantic_ai.UserError` | Developer usage error |
