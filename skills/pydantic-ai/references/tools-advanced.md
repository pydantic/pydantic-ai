# Advanced Tools Reference

Source: `pydantic_ai_slim/pydantic_ai/tools.py`

## Tool Output Types

Tools can return anything JSON-serializable, plus multimodal content:

```python {title="tool_outputs.py"}
from datetime import datetime

from pydantic import BaseModel

from pydantic_ai import Agent, DocumentUrl, ImageUrl
from pydantic_ai.models.openai import OpenAIResponsesModel


class User(BaseModel):
    name: str
    age: int


agent = Agent(model=OpenAIResponsesModel('gpt-5'))


@agent.tool_plain
def get_current_time() -> datetime:
    return datetime.now()


@agent.tool_plain
def get_user() -> User:
    return User(name='John', age=30)


@agent.tool_plain
def get_logo() -> ImageUrl:
    return ImageUrl(url='https://example.com/logo.png')


@agent.tool_plain
def get_document() -> DocumentUrl:
    return DocumentUrl(url='https://example.com/doc.pdf')
```

## ToolReturn — Advanced Tool Returns

Separate the return value from rich content sent to the model:

```python {title="tool_return_example.py" test="skip" lint="skip"}
from pydantic_ai import Agent, BinaryContent, ToolReturn

agent = Agent('openai:gpt-5')


@agent.tool_plain
def screenshot_and_click(x: int, y: int) -> ToolReturn:
    """Click at coordinates and show before/after screenshots."""
    before = capture_screen()
    perform_click(x, y)
    after = capture_screen()

    return ToolReturn(
        return_value=f'Clicked at ({x}, {y})',  # Sent to model as tool result
        content=[                                # Additional context for model
            'Before:',
            BinaryContent(data=before, media_type='image/png'),
            'After:',
            BinaryContent(data=after, media_type='image/png'),
        ],
        metadata={'coords': (x, y)},             # Not sent to model (for app use)
    )
```

### ToolReturn Fields

| Field | Sent to Model | Description |
|-------|---------------|-------------|
| `return_value` | Yes (as tool result) | The main tool return value |
| `content` | Yes (as user message) | Additional context (text, images) |
| `metadata` | No | App-only data for logging, analytics |

### When to Use ToolReturn

- **Simple returns**: Just return the value directly (string, dict, Pydantic model)
- **Use ToolReturn when**:
  - You need to send images or rich content alongside the result
  - You want to attach metadata for debugging without sending to the model
  - The "return value" and "context for model" should be different

## Tool.from_schema — Custom JSON Schema

For functions without proper type hints or docstrings:

```python
from pydantic_ai import Agent, Tool


def foobar(**kwargs) -> str:
    return kwargs['a'] + kwargs['b']


tool = Tool.from_schema(
    function=foobar,
    name='sum',
    description='Sum two numbers.',
    json_schema={
        'type': 'object',
        'properties': {
            'a': {'type': 'integer', 'description': 'First number'},
            'b': {'type': 'integer', 'description': 'Second number'},
        },
        'required': ['a', 'b'],
        'additionalProperties': False,
    },
    takes_ctx=False,
)

agent = Agent('openai:gpt-5', tools=[tool])
```

Note: No argument validation is performed with `from_schema`.

## Dynamic Tools with ToolPrepareFunc

Modify or hide tools at runtime based on context:

```python {title="dynamic_tool.py"}
from pydantic_ai import Agent, RunContext, Tool, ToolDefinition


async def only_for_admins(
    ctx: RunContext[str], tool_def: ToolDefinition
) -> ToolDefinition | None:
    """Return None to hide the tool, or modify tool_def."""
    if ctx.deps == 'admin':
        return tool_def
    return None  # Hide for non-admins


def admin_action(command: str) -> str:
    """Perform an admin action."""
    return f'Executed: {command}'


agent = Agent('test', tools=[Tool(admin_action, prepare=only_for_admins)], deps_type=str)

# Tool available for admin
result = agent.run_sync('Do something', deps='admin')

# Tool hidden for regular users
result = agent.run_sync('Do something', deps='user')
```

### Modifying Tool Definitions

```python
async def customize_description(
    ctx: RunContext[str], tool_def: ToolDefinition
) -> ToolDefinition:
    """Customize tool description based on user type."""
    tool_def.parameters_json_schema['properties']['target']['description'] = (
        f'Target for {ctx.deps} user.'
    )
    return tool_def
```

## Agent-Wide prepare_tools

Filter or modify all tools at once:

```python {title="prepare_tools_example.py"}
from dataclasses import replace

from pydantic_ai import Agent, RunContext, ToolDefinition


async def make_all_strict_for_openai(
    ctx: RunContext[None], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    if ctx.model.system == 'openai':
        return [replace(td, strict=True) for td in tool_defs]
    return tool_defs


agent = Agent('openai:gpt-5', prepare_tools=make_all_strict_for_openai)
```

## Tool Strict Mode (OpenAI)

OpenAI's strict mode enforces exact schema compliance. Enable it for guaranteed valid arguments:

```python
from pydantic_ai import Agent, Tool


def my_tool(x: int, y: int) -> int:
    return x + y


# Enable strict mode for a specific tool
tool = Tool(my_tool, strict=True)
agent = Agent('openai:gpt-5', tools=[tool])

# Or enable via prepare_tools for all tools
from dataclasses import replace


async def enable_strict(ctx, tool_defs):
    if ctx.model.system == 'openai':
        return [replace(td, strict=True) for td in tool_defs]
    return tool_defs


agent = Agent('openai:gpt-5', prepare_tools=enable_strict)
```

**When to use strict mode:**
- When you need guaranteed schema compliance
- For production systems where invalid arguments would cause errors
- OpenAI-specific feature; other providers ignore this setting

## Tool Metadata

Attach application metadata to tools (not sent to the model):

```python
from pydantic_ai import Agent, Tool


def database_query(query: str) -> str:
    return 'results'


tool = Tool(
    database_query,
    metadata={'category': 'database', 'requires_auth': True, 'cost': 0.01},
)

agent = Agent('openai:gpt-5', tools=[tool])
```

Access metadata in `prepare_tools` for filtering or logging.

## Tool Timeout

Prevent tools from running indefinitely:

```python
from pydantic_ai import Agent

# Default timeout for all tools
agent = Agent('test', tool_timeout=30)


@agent.tool_plain
async def slow_tool() -> str:
    """Uses agent default timeout (30s)."""
    ...


@agent.tool_plain(timeout=5)
async def fast_tool() -> str:
    """Override with 5 second timeout."""
    ...
```

Timeout triggers a retry prompt: `"Timed out after {timeout} seconds."`

### Timeout Hierarchy

```
Tool-specific timeout > Agent tool_timeout > No timeout (unlimited)
```

```python
agent = Agent('openai:gpt-5', tool_timeout=60)  # Default 60s


@agent.tool_plain(timeout=10)  # Override: 10s
async def quick_lookup(query: str) -> str:
    ...


@agent.tool_plain  # Uses default: 60s
async def slow_analysis(data: str) -> str:
    ...


@agent.tool_plain(timeout=None)  # No timeout
async def long_running_task() -> str:
    ...
```

## Parallel vs Sequential Execution

Tools run concurrently by default. Force sequential execution:

```python
from pydantic_ai import Agent, Tool


@agent.tool_plain(sequential=True)
def must_run_alone(data: str) -> str:
    """This tool will not run in parallel with others."""
    ...


# Or use context manager for entire run
with agent.parallel_tool_call_execution_mode('sequential'):
    result = agent.run_sync('Do things')
```

## Tool Retries

```python
from pydantic_ai import Agent, ModelRetry

agent = Agent('openai:gpt-5')


@agent.tool_plain(retries=3)
def flaky_tool(query: str) -> str:
    if not query.strip():
        raise ModelRetry('Query cannot be empty. Please provide a valid query.')
    return f'Result for: {query}'
```

Retries trigger on:
- `ValidationError` (invalid arguments)
- `ModelRetry` exception
- Timeout

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Tool` | `pydantic_ai.Tool` | Tool wrapper class |
| `ToolDefinition` | `pydantic_ai.ToolDefinition` | Schema + metadata |
| `ToolReturn` | `pydantic_ai.ToolReturn` | Advanced return with content/metadata |
| `ToolPrepareFunc` | `pydantic_ai.tools.ToolPrepareFunc` | Dynamic tool preparation |
| `ToolsPrepareFunc` | `pydantic_ai.tools.ToolsPrepareFunc` | Agent-wide tool preparation |
| `ModelRetry` | `pydantic_ai.ModelRetry` | Exception for tool retry |

## Tool Security Patterns

### Credential Injection via Dependencies

Never hardcode credentials in tools. Use dependency injection:

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class SecureDeps:
    http_client: httpx.AsyncClient
    api_key: str  # Injected at runtime, never in code


agent = Agent('openai:gpt-5', deps_type=SecureDeps)


@agent.tool
async def fetch_data(ctx: RunContext[SecureDeps], endpoint: str) -> str:
    """Fetch data from API with injected credentials."""
    response = await ctx.deps.http_client.get(
        endpoint,
        headers={'Authorization': f'Bearer {ctx.deps.api_key}'},
    )
    response.raise_for_status()
    return response.text


# Runtime: credentials from environment/secrets manager
async def main():
    async with httpx.AsyncClient() as client:
        deps = SecureDeps(client, api_key=os.getenv('API_KEY'))
        result = await agent.run('Fetch user data', deps=deps)
```

### Input Validation via Pydantic

Tool arguments are validated by Pydantic. Use constrained types for security:

```python
from pydantic import BaseModel, Field

from pydantic_ai import Agent


class SafeQuery(BaseModel):
    table: str = Field(pattern=r'^[a-zA-Z_]+$')  # Prevent SQL injection
    limit: int = Field(ge=1, le=100)  # Bounded results


agent = Agent('openai:gpt-5')


@agent.tool_plain
def query_database(params: SafeQuery) -> str:
    """Query with validated, safe parameters."""
    # params.table is guaranteed to match the pattern
    return f'SELECT * FROM {params.table} LIMIT {params.limit}'
```

### Requiring Tool Approval (Human-in-the-Loop)

Use `ApprovalRequiredToolset` for sensitive operations:

```python {title="approval_required_toolset.py" test="skip"}
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults

# Wrap toolset to require approval for dangerous tools
approval_required_toolset = my_toolset.approval_required(
    lambda ctx, tool_def, tool_args: tool_def.name.startswith('delete')
)

agent = Agent(
    'openai:gpt-5',
    toolsets=[approval_required_toolset],
    output_type=[str, DeferredToolRequests],
)

result = agent.run_sync('Delete the old records')

if isinstance(result.output, DeferredToolRequests):
    # Present to user for approval
    for approval in result.output.approvals:
        print(f'Approve {approval.tool_name} with args {approval.args}? (y/n)')

    # Resume with approval decisions
    result = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(
            approvals={'tool_call_id': True}  # or False to deny
        )
    )
```

### Tool Timeout for Safety

Prevent runaway tools from consuming resources:

```python
from pydantic_ai import Agent

# Default timeout for all tools
agent = Agent('openai:gpt-5', tool_timeout=30)


@agent.tool_plain(timeout=5)  # Override: 5 second timeout
async def quick_lookup(query: str) -> str:
    """Must complete quickly or retry."""
    ...


@agent.tool_plain  # Uses agent default: 30 seconds
async def slow_analysis(data: str) -> str:
    """Can take longer."""
    ...
```

Timeout triggers retry with message: `"Timed out after {timeout} seconds."`

### Security Checklist

| Risk | Mitigation |
|------|-----------|
| Credential exposure | Use `RunContext.deps` for injection |
| Injection attacks | Pydantic validation with patterns |
| Unauthorized actions | `ApprovalRequiredToolset` |
| Resource exhaustion | `tool_timeout` parameter |
| Excessive permissions | `FilteredToolset` to limit tools |

## See Also

- [tools.md](tools.md) — Basic tool registration
- [deferred-tools.md](deferred-tools.md) — Human-in-the-loop approval
- [input.md](input.md) — Multimodal content types
- [observability.md](observability.md) — Logfire debugging
- [toolsets.md](toolsets.md) — ApprovalRequiredToolset, FilteredToolset
