# Testing Reference

Source: `pydantic_ai_slim/pydantic_ai/models/test.py`, `pydantic_ai_slim/pydantic_ai/models/function.py`

## TestModel — Deterministic Testing

`TestModel` calls all tools and returns structured output without API calls:

```python {title="test_weather_app.py" call_name="test_forecast" requires="weather_app.py"}
from datetime import timezone
import pytest

from dirty_equals import IsNow, IsStr

from pydantic_ai import models, capture_run_messages, RequestUsage
from pydantic_ai.models.test import TestModel
from pydantic_ai import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    ModelRequest,
)

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


async def test_forecast():
    conn = DatabaseConn()
    user_id = 1
    with capture_run_messages() as messages:
        with weather_agent.override(model=TestModel()):
            prompt = 'What will the weather be like in London on 2024-11-28?'
            await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == '{"weather_forecast":"Sunny with a chance of rain"}'

    assert messages == [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content='What will the weather be like in London on 2024-11-28?',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            instructions='Providing a weather forecast at the locations the user provides.',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='weather_forecast',
                    args={
                        'location': 'a',
                        'forecast_date': '2024-01-01',
                    },
                    tool_call_id=IsStr(),
                )
            ],
            usage=RequestUsage(
                input_tokens=60,
                output_tokens=7,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name='weather_forecast',
                    content='Sunny with a chance of rain',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ],
            instructions='Providing a weather forecast at the locations the user provides.',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
        ModelResponse(
            parts=[
                TextPart(
                    content='{"weather_forecast":"Sunny with a chance of rain"}',
                )
            ],
            usage=RequestUsage(
                input_tokens=66,
                output_tokens=16,
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            run_id=IsStr(),
        ),
    ]
```

### TestModel Constructor

```python
TestModel(
    *,
    call_tools='all',            # list[str] | 'all' — which tools to call
    custom_output_text=None,     # str | None — override text output
    custom_output_args=None,     # Any — override tool call args
    seed=0,                      # int — deterministic output seed
    model_name='test',           # str — model name for logging
)
```

### Structured Output Testing

```python {title="test_model_structured.py"}
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


class MyOutput(BaseModel):
    name: str
    value: int


agent = Agent('openai:gpt-5', output_type=MyOutput)

# TestModel automatically calls output tools with generated data
with agent.override(model=TestModel()):
    result = agent.run_sync('Create a record with name "test" and value 42')
    print(isinstance(result.output, MyOutput))
    #> True
```

## FunctionModel — Custom Logic

For tests requiring specific tool call behavior:

```python {title="test_weather_app2.py" call_name="test_forecast_future" requires="weather_app.py"}
import re

import pytest

from pydantic_ai import models
from pydantic_ai import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from fake_database import DatabaseConn
from weather_app import run_weather_forecast, weather_agent

pytestmark = pytest.mark.anyio
models.ALLOW_MODEL_REQUESTS = False


def call_weather_forecast(
    messages: list[ModelMessage], info: AgentInfo
) -> ModelResponse:
    if len(messages) == 1:
        # first call, call the weather forecast tool
        user_prompt = messages[0].parts[-1]
        m = re.search(r'\d{4}-\d{2}-\d{2}', user_prompt.content)
        assert m is not None
        args = {'location': 'London', 'forecast_date': m.group()}
        return ModelResponse(parts=[ToolCallPart('weather_forecast', args)])
    else:
        # second call, return the forecast
        msg = messages[-1].parts[0]
        assert msg.part_kind == 'tool-return'
        return ModelResponse(parts=[TextPart(f'The forecast is: {msg.content}')])


async def test_forecast_future():
    conn = DatabaseConn()
    user_id = 1
    with weather_agent.override(model=FunctionModel(call_weather_forecast)):
        prompt = 'What will the weather be like in London on 2032-01-01?'
        await run_weather_forecast([(prompt, user_id)], conn)

    forecast = await conn.get_forecast(user_id)
    assert forecast == 'The forecast is: Rainy with a chance of sun'
```

### FunctionModel Constructor

```python
FunctionModel(
    function,                    # (messages, info) -> ModelResponse
    *,
    stream_function=None,        # Async generator for streaming
    model_name=None,             # str — model name
)
```

## Agent Override

Replace models, dependencies, or toolsets without modifying call sites:

```python {title="test_model_usage.py" call_name="test_my_agent" noqa="I001"}
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

my_agent = Agent('openai:gpt-5', instructions='...')


async def test_my_agent():
    """Unit test for my_agent, to be run by pytest."""
    m = TestModel()
    with my_agent.override(model=m):
        result = await my_agent.run('Testing my agent...')
        assert result.output == 'success (no tool calls)'
    assert m.last_model_request_parameters.function_tools == []
```

## Pytest Fixtures

Reusable model override fixtures:

```python {title="test_agent.py" requires="weather_app.py"}
import pytest

from pydantic_ai.models.test import TestModel

from weather_app import weather_agent


@pytest.fixture
def override_weather_agent():
    with weather_agent.override(model=TestModel()):
        yield


async def test_forecast(override_weather_agent: None):
    ...
    # test code here
```

## ALLOW_MODEL_REQUESTS

Block accidental real API calls in tests:

```python
from pydantic_ai.models import ALLOW_MODEL_REQUESTS

ALLOW_MODEL_REQUESTS = False  # Raises error if real model is used

# In conftest.py for pytest:
import pydantic_ai.models
pydantic_ai.models.ALLOW_MODEL_REQUESTS = False
```

## capture_run_messages

Inspect messages from agent runs:

```python
from pydantic_ai import capture_run_messages

with capture_run_messages() as messages:
    result = agent.run_sync('prompt')

# messages contains all ModelRequest and ModelResponse objects
for msg in messages:
    print(msg.parts)
```

## Inspecting TestModel State

Access tool calls and parameters:

```python
test_model = TestModel()
with agent.override(model=test_model):
    result = agent.run_sync('call tools')

# Last request parameters
params = test_model.last_model_request_parameters
print(params.function_tools)   # Tool definitions sent
print(params.allow_text_output)  # Whether text output allowed

# Custom output
test_model = TestModel(
    custom_output_text='My custom response',
    call_tools=['specific_tool'],  # Only call this tool
)
```

## Testing Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

class User(BaseModel):
    name: str
    age: int

agent = Agent('openai:gpt-5', output_type=User)

# TestModel generates valid data from schema
with agent.override(model=TestModel()):
    result = agent.run_sync('Get user')
    assert isinstance(result.output, User)
    # Generated values won't be meaningful, but will be valid
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `TestModel` | `pydantic_ai.models.test.TestModel` | Deterministic test model |
| `FunctionModel` | `pydantic_ai.models.function.FunctionModel` | Custom function model |
| `AgentInfo` | `pydantic_ai.models.function.AgentInfo` | Info passed to FunctionModel |
| `ALLOW_MODEL_REQUESTS` | `pydantic_ai.models.ALLOW_MODEL_REQUESTS` | Global API block flag |
| `capture_run_messages` | `pydantic_ai.capture_run_messages` | Context manager for messages |

## See Also

- [models.md](models.md) — TestModel and FunctionModel details
- [agents.md](agents.md) — Agent.override() patterns
- [dependencies.md](dependencies.md) — Testing with dependency injection
- [troubleshooting.md](troubleshooting.md) — Testing anti-patterns
