# Troubleshooting Reference

Common mistakes, anti-patterns, and solutions for PydanticAI development.

## Agent Anti-Patterns

### Don't: Use `run_sync()` Inside an Async Context

```python
# Wrong - blocks the event loop
async def handler():
    result = agent.run_sync('prompt')  # Blocks!
    return result.output
```

```python
# Correct - use async run()
async def handler():
    result = await agent.run('prompt')
    return result.output
```

### Don't: Modify message_history In Place

```python
# Wrong - modifying the list affects future runs
history = result.all_messages()
history.append(custom_message)  # Mutation!
result2 = agent.run_sync('next', message_history=history)
```

```python
# Correct - create a new list
history = list(result.all_messages())  # Copy
history.append(custom_message)
result2 = agent.run_sync('next', message_history=history)
```

### Don't: Create Agents Inside Request Handlers

```python
# Wrong - recreates agent on every request
async def handle_request(user_input: str):
    agent = Agent('openai:gpt-5', instructions='...')  # Expensive!
    return await agent.run(user_input)
```

```python
# Correct - define agent once at module level
agent = Agent('openai:gpt-5', instructions='...')

async def handle_request(user_input: str):
    return await agent.run(user_input)
```

### Don't: Ignore Usage Limits in Production

```python
# Wrong - unbounded token usage
result = await agent.run(user_prompt)  # Could run forever!
```

```python
# Correct - set limits to prevent runaway costs
from pydantic_ai import UsageLimits

result = await agent.run(
    user_prompt,
    usage_limits=UsageLimits(
        request_limit=10,
        total_tokens_limit=10000,
    ),
)
```

## Tool Anti-Patterns

### Don't: Use Mutable Default Arguments

```python
# Wrong - mutable default shared across calls
@agent.tool_plain
def process_items(items: list[str] = []) -> str:  # Danger!
    items.append('new')
    return str(items)
```

```python
# Correct - use None and create inside function
@agent.tool_plain
def process_items(items: list[str] | None = None) -> str:
    if items is None:
        items = []
    items.append('new')
    return str(items)
```

### Don't: Use Overly Complex Nested Models

```python
# Wrong - generates huge JSON schemas, may cause "schema too complex" errors
class DeepNested(BaseModel):
    level1: list[dict[str, list[dict[str, Any]]]]  # Very complex
```

```python
# Correct - flatten or simplify the structure
class SimpleItem(BaseModel):
    key: str
    value: str

@agent.tool_plain
def process_data(items: list[SimpleItem]) -> str:
    return str(items)
```

### Don't: Forget to Pass `ctx.usage` to Delegate Agents

```python
# Wrong - usage not tracked across agents
@parent_agent.tool
async def delegate(ctx: RunContext[None], query: str) -> str:
    result = await child_agent.run(query)  # Usage lost!
    return result.output
```

```python
# Correct - pass usage for combined tracking
@parent_agent.tool
async def delegate(ctx: RunContext[None], query: str) -> str:
    result = await child_agent.run(query, usage=ctx.usage)
    return result.output
```

### Don't: Raise Generic Exceptions in Tools

```python
# Wrong - generic exception crashes the run
@agent.tool_plain
def fetch_data(url: str) -> str:
    if not url.startswith('https'):
        raise ValueError('Invalid URL')  # Crashes agent run!
    return 'data'
```

```python
# Correct - use ModelRetry to let the model try again
from pydantic_ai import ModelRetry

@agent.tool_plain(retries=3)
def fetch_data(url: str) -> str:
    if not url.startswith('https'):
        raise ModelRetry('URL must start with https://')
    return 'data'
```

### Don't: Block the Event Loop in Async Tools

```python
# Wrong - blocks the event loop
@agent.tool
async def slow_tool(ctx: RunContext[None], query: str) -> str:
    import time
    time.sleep(5)  # Blocks everything!
    return 'done'
```

```python
# Correct - use asyncio.sleep or run_in_executor
import asyncio

@agent.tool
async def slow_tool(ctx: RunContext[None], query: str) -> str:
    await asyncio.sleep(5)  # Non-blocking
    return 'done'

# Or for CPU-bound work:
@agent.tool
async def cpu_intensive(ctx: RunContext[None], data: str) -> str:
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, expensive_computation, data)
    return result
```

## Output Anti-Patterns

### Don't: Mix ToolOutput with `json_schema_extra`

```python
# Wrong - json_schema_extra may conflict with tool schema generation
class MyOutput(BaseModel):
    value: int

    class Config:
        json_schema_extra = {'examples': [{'value': 42}]}  # May cause issues
```

```python
# Correct - use Field descriptions instead
from pydantic import Field

class MyOutput(BaseModel):
    value: int = Field(description='The computed value, e.g. 42')
```

### Don't: Use Ambiguous Union Types

```python
# Wrong - model can't distinguish between similar types
agent = Agent('openai:gpt-5', output_type=dict | list)  # Ambiguous!
```

```python
# Correct - use discriminated unions with clear names
from pydantic_ai import ToolOutput

class DictResult(BaseModel):
    data: dict[str, str]

class ListResult(BaseModel):
    items: list[str]

agent = Agent(
    'openai:gpt-5',
    output_type=[
        ToolOutput(DictResult, name='return_dict'),
        ToolOutput(ListResult, name='return_list'),
    ],
)
```

### Don't: Forget Output Validation for Critical Data

```python
# Wrong - no validation of model output
class UserData(BaseModel):
    email: str
    age: int

agent = Agent('openai:gpt-5', output_type=UserData)
# Model might return invalid email or negative age
```

```python
# Correct - add output validators
from pydantic_ai import ModelRetry

@agent.output_validator
def validate_user(ctx: RunContext[None], output: UserData) -> UserData:
    if '@' not in output.email:
        raise ModelRetry('Email must contain @')
    if output.age < 0 or output.age > 150:
        raise ModelRetry('Age must be between 0 and 150')
    return output
```

## Multi-Agent Anti-Patterns

### Don't: Create Infinite Delegation Loops

```python
# Wrong - agents can call each other forever
@agent_a.tool
async def call_b(ctx: RunContext[None]) -> str:
    return (await agent_b.run('help')).output

@agent_b.tool
async def call_a(ctx: RunContext[None]) -> str:
    return (await agent_a.run('help')).output  # Infinite loop!
```

```python
# Correct - use usage limits and design clear delegation hierarchies
@agent_a.tool
async def call_b(ctx: RunContext[None]) -> str:
    # agent_b should NOT have a tool to call agent_a
    return (await agent_b.run('help', usage=ctx.usage)).output

# Also set usage limits at the top level
result = await agent_a.run(
    'prompt',
    usage_limits=UsageLimits(request_limit=20),
)
```

### Don't: Share Mutable State Between Agents

```python
# Wrong - shared mutable state causes race conditions
shared_data = {'count': 0}

@agent_a.tool_plain
def increment_a() -> str:
    shared_data['count'] += 1  # Race condition!
    return str(shared_data['count'])
```

```python
# Correct - pass state through dependencies
from dataclasses import dataclass
import asyncio

@dataclass
class Deps:
    lock: asyncio.Lock
    count: int = 0

@agent_a.tool
async def increment_a(ctx: RunContext[Deps]) -> str:
    async with ctx.deps.lock:
        ctx.deps.count += 1
        return str(ctx.deps.count)
```

## Testing Anti-Patterns

### Don't: Make Real API Calls in Tests

```python
# Wrong - tests are slow, expensive, and flaky
def test_agent():
    result = agent.run_sync('test')  # Real API call!
    assert 'expected' in result.output
```

```python
# Correct - use TestModel and ALLOW_MODEL_REQUESTS
from pydantic_ai.models.test import TestModel
from pydantic_ai import models

models.ALLOW_MODEL_REQUESTS = False  # Block real calls

def test_agent():
    with agent.override(model=TestModel()):
        result = agent.run_sync('test')
        assert result.output == 'success (no tool calls)'
```

### Don't: Test Implementation Details

```python
# Wrong - brittle test that breaks on internal changes
def test_agent():
    with agent.override(model=TestModel()) as m:
        result = agent.run_sync('test')
        # Testing internal message structure
        assert m._internal_messages[0].parts[0].content == 'test'
```

```python
# Correct - test observable behavior
def test_agent():
    with agent.override(model=TestModel()):
        result = agent.run_sync('test')
        assert isinstance(result.output, str)
        # Or use capture_run_messages for message inspection
```

## Debugging Tips

### Enable Logfire for Deep Visibility

When agents behave unexpectedly:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)  # See raw HTTP requests
```

This reveals:
- Exact JSON schemas sent to the model
- Full request/response payloads
- Tool call arguments and return values
- Retry attempts and errors

### Use `capture_run_messages` for Failed Runs

```python
from pydantic_ai import capture_run_messages, UnexpectedModelBehavior

with capture_run_messages() as messages:
    try:
        result = agent.run_sync('prompt')
    except UnexpectedModelBehavior:
        for msg in messages:
            print(f'{type(msg).__name__}: {msg}')
```

### Check Tool Schemas with TestModel

```python
test_model = TestModel()
with agent.override(model=test_model):
    agent.run_sync('test')

# Inspect what tools were exposed to the model
for tool in test_model.last_model_request_parameters.function_tools:
    print(f'{tool.name}: {tool.parameters_json_schema}')
```

## Common Error Messages

| Error | Likely Cause | Solution |
|-------|-------------|----------|
| `ModelHTTPError: 400` | Invalid request (bad schema, too many tokens) | Enable Logfire HTTP capture to see request body |
| `UnexpectedModelBehavior` | Model returned unexpected format | Check model compatibility, add output validators |
| `ModelRetry` exhausted | Tool failed all retry attempts | Increase `retries` or fix underlying tool issue |
| `UsageLimitExceeded` | Exceeded token or request limits | Increase limits or optimize prompts |
| `ValidationError` | Output doesn't match schema | Add output validators with helpful ModelRetry messages |

## See Also

- [testing.md](testing.md) — Test patterns and TestModel
- [exceptions.md](exceptions.md) — Exception hierarchy
- [observability.md](observability.md) — Logfire tracing
- [tools.md](tools.md) — Tool registration
- [agents.md](agents.md) — Agent configuration
