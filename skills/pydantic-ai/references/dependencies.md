# Dependencies Reference

Source: `pydantic_ai_slim/pydantic_ai/_run_context.py`

## Overview

PydanticAI uses dependency injection via the `deps_type` parameter and `RunContext[DepsT]`.
Dependencies are passed at runtime via `run(deps=...)` and accessed in tools and system prompts via `RunContext`.

## Basic Pattern

```python {title="deps_basic.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class MyDeps:
    user_id: int
    db_url: str


agent = Agent('openai:gpt-5', deps_type=MyDeps)


@agent.tool
async def get_user_info(ctx: RunContext[MyDeps]) -> str:
    """Get information about the current user."""
    return f'User {ctx.deps.user_id} from {ctx.deps.db_url}'


result = agent.run_sync(
    'Who was Albert Einstein?',
    deps=MyDeps(user_id=42, db_url='postgresql://localhost/db'),
)
print(result.output)
#> Albert Einstein was a German-born theoretical physicist.
```

## RunContext Fields

The `RunContext` dataclass is defined in `pydantic_ai_slim/pydantic_ai/_run_context.py`:

```python
@dataclasses.dataclass(repr=False, kw_only=True)
class RunContext(Generic[RunContextAgentDepsT]):
    deps: RunContextAgentDepsT           # Your dependencies object
    model: Model                          # The model being used
    usage: RunUsage                       # Token usage so far
    prompt: str | Sequence[UserContent] | None  # Original user prompt
    messages: list[ModelMessage]          # Conversation messages so far
    validation_context: Any               # Pydantic validation context
    retries: dict[str, int]              # Retry counts by tool name
    tool_call_id: str | None             # Current tool call ID
    tool_name: str | None                # Current tool name
    retry: int                           # Current retry count
    max_retries: int                     # Max retries configured
    run_step: int                        # Current step number
    run_id: str | None                   # Unique run identifier
    metadata: dict[str, Any] | None      # Run metadata
    partial_output: bool                 # Whether output is partial (streaming)
    tool_call_approved: bool             # Whether tool was approved (deferred)
    tool_call_metadata: Any              # Metadata from deferred tool approval

    @property
    def last_attempt(self) -> bool:
        """True if this is the last retry before error."""
```

## Using Dependencies in System Prompts

```python {title="deps_instructions.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class UserContext:
    name: str
    role: str


agent = Agent('openai:gpt-5', deps_type=UserContext)


@agent.instructions
def add_context(ctx: RunContext[UserContext]) -> str:
    return f'You are helping {ctx.deps.name} who is a {ctx.deps.role}.'


result = agent.run_sync(
    'What is the capital of France?', deps=UserContext(name='Alice', role='student')
)
print(result.output)
#> The capital of France is Paris.
```

## Using Dependencies in Tools

```python {title="deps_in_tools.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext


@dataclass
class ApiConfig:
    base_url: str
    api_key: str


agent = Agent('openai:gpt-5', deps_type=ApiConfig)


@agent.tool
async def call_api(ctx: RunContext[ApiConfig], endpoint: str) -> str:
    """Call an external API endpoint."""
    return f'Called {ctx.deps.base_url}/{endpoint}'


result = agent.run_sync(
    'What is the capital of Italy?',
    deps=ApiConfig(base_url='https://api.example.com', api_key='key123'),
)
print(result.output)
#> The capital of Italy is Rome.
```

## Testing with Dependency Override

Use `Agent.override(deps=...)` to inject test dependencies:

```python {title="deps_testing.py"}
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel


@dataclass
class Deps:
    api_key: str


agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.tool
async def secure_action(ctx: RunContext[Deps]) -> str:
    """Perform a secure action."""
    return f'Done with key={ctx.deps.api_key}'


# Override both model and deps for testing
with agent.override(model=TestModel(), deps=Deps(api_key='test-key')):
    result = agent.run_sync('do something')
    print(result.output)
    #> {"secure_action":"Done with key=test-key"}
```

## Type Safety

The generic type parameter ensures type safety between `deps_type`, `RunContext`, and `run()`:

```python
# Type-safe: deps_type matches RunContext and run() call
agent = Agent('openai:gpt-5', deps_type=MyDeps)

@agent.tool
async def my_tool(ctx: RunContext[MyDeps]) -> str:  # ✓ Matches deps_type
    return ctx.deps.some_field  # ✓ Type-checked

result = agent.run_sync('prompt', deps=MyDeps(...))  # ✓ Must be MyDeps
```

## None Dependencies (Default)

If no `deps_type` is specified, `RunContext[None]` is used and `deps` defaults to `None`:

```python
agent = Agent('openai:gpt-5')  # deps_type defaults to NoneType

@agent.tool
async def my_tool(ctx: RunContext[None]) -> str:
    # ctx.deps is None
    return 'result'
```

## Async vs Sync Dependencies

Dependencies often include async clients. Handle the lifecycle properly:

### Pattern 1: Async Context Manager (Recommended)

```python
from dataclasses import dataclass

import httpx

from pydantic_ai import Agent, RunContext


@dataclass
class Deps:
    client: httpx.AsyncClient


agent = Agent('openai:gpt-5', deps_type=Deps)


@agent.tool
async def fetch_data(ctx: RunContext[Deps], url: str) -> str:
    response = await ctx.deps.client.get(url)
    return response.text


async def main():
    # Create client with proper lifecycle
    async with httpx.AsyncClient() as client:
        deps = Deps(client=client)
        result = await agent.run('Fetch example.com', deps=deps)
```

### Pattern 2: Sync Entry Point with run_sync()

```python
def main():
    # For run_sync(), you need a sync-compatible approach
    # Option A: Create a fresh client per run (simpler, less efficient)
    import httpx
    with httpx.Client() as sync_client:
        # ... use sync client

    # Option B: Use asyncio.run() with async pattern
    import asyncio
    asyncio.run(async_main())
```

### Common Pitfall: Using Closed Clients

```python
# WRONG - client closed before agent runs
async def broken():
    async with httpx.AsyncClient() as client:
        deps = Deps(client=client)
    # client is closed here!
    result = await agent.run('prompt', deps=deps)  # Error!

# CORRECT - agent runs inside context
async def correct():
    async with httpx.AsyncClient() as client:
        deps = Deps(client=client)
        result = await agent.run('prompt', deps=deps)  # Works!
```

## Debugging with Dependencies

When debugging production issues, Logfire instrumentation captures the full `RunContext` state, including:

- The `deps` object values at each tool call
- Message history and usage counters
- Retry state when tools fail

This helps reproduce issues by showing exactly what context the agent had when something went wrong.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `RunContext` | `pydantic_ai.RunContext` | Dependency injection context in tools |
| `AgentDepsT` | `pydantic_ai._run_context.AgentDepsT` | Type variable for deps |

## See Also

- [agents.md](agents.md) — Agent configuration
- [tools.md](tools.md) — Using RunContext in tools
- [testing.md](testing.md) — Testing with dependency override
- [third-party-tools.md](third-party-tools.md) — Integration patterns with databases and APIs
