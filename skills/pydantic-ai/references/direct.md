# Direct Model API Reference

Source: `pydantic_ai_slim/pydantic_ai/direct.py`

## Overview

Low-level API for making model requests without the full Agent abstraction. Useful for:
- Custom abstractions on top of model interactions
- Direct control over model requests
- Simple one-off requests without agent overhead

## Functions

| Function | Description |
|----------|-------------|
| `model_request` | Async non-streamed request |
| `model_request_sync` | Sync non-streamed request |
| `model_request_stream` | Async streamed request |
| `model_request_stream_sync` | Sync streamed request |

## Basic Usage

```python {title="direct_basic.py"}
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('What is the capital of France?')],
)

print(response.parts[0].content)
#> The capital of France is Paris.
print(response.usage)
#> RequestUsage(input_tokens=56, output_tokens=7)
```

## Async Usage

```python {title="direct_async.py"}
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request


async def main():
    response = await model_request(
        'openai:gpt-5',
        [ModelRequest.user_text_prompt('Hello!')],
    )
    print(response.parts[0].content)
```

## Tool Calling

```python {title="direct_with_tools.py"}
from typing import Literal

from pydantic import BaseModel

from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import model_request
from pydantic_ai.models import ModelRequestParameters


class Divide(BaseModel):
    """Divide two numbers."""

    numerator: float
    denominator: float
    on_inf: Literal['error', 'infinity'] = 'infinity'


async def main():
    response = await model_request(
        'openai:gpt-5',
        [ModelRequest.user_text_prompt('What is 123 / 456?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name='divide',
                    description='Divide two numbers.',
                    parameters_json_schema=Divide.model_json_schema(),
                )
            ],
            allow_text_output=True,
        ),
    )
    # Response may contain ToolCallPart if model chose to use tool
    print(response)
```

## Streaming

```python {title="direct_streaming.py"}
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream


async def main():
    async with model_request_stream(
        'anthropic:claude-sonnet-4-5',
        [ModelRequest.user_text_prompt('Write a haiku.')],
    ) as stream:
        async for chunk in stream:
            print(chunk, end='', flush=True)
```

## With Instrumentation

```python {title="direct_instrumented.py"}
import logfire

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_sync

logfire.configure()
logfire.instrument_pydantic_ai()

response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('Hello!')],
)
```

Or per-request:

```python
response = model_request_sync(
    'anthropic:claude-haiku-4-5',
    [ModelRequest.user_text_prompt('Hello!')],
    instrument=True,  # Enable instrumentation for this call
)
```

## When to Use Direct API vs Agent

| Use Direct API | Use Agent |
|----------------|-----------|
| Custom abstractions | Standard workflows |
| Simple one-off requests | Tool execution |
| Maximum control | Structured output parsing |
| Building new patterns | Retries and validation |

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `model_request` | `pydantic_ai.direct.model_request` | Async request |
| `model_request_sync` | `pydantic_ai.direct.model_request_sync` | Sync request |
| `model_request_stream` | `pydantic_ai.direct.model_request_stream` | Async streaming |
| `ModelRequest` | `pydantic_ai.ModelRequest` | Request message builder |
| `ModelResponse` | `pydantic_ai.ModelResponse` | Response from model |
| `ModelRequestParameters` | `pydantic_ai.models.ModelRequestParameters` | Tool config, etc. |
| `ToolDefinition` | `pydantic_ai.ToolDefinition` | Tool schema definition |

## See Also

- [models.md](models.md) — Model configuration
- [messages.md](messages.md) — Message types
- [observability.md](observability.md) — Instrumentation details
