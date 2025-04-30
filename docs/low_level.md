# Low-Level Model Requests

The low-level module provides direct access to language models with minimal abstraction. These methods allow you to make requests to LLMs where the only abstraction is input and output schema translation, enabling you to request all models with the same API.

These methods are thin wrappers around the [`Model`][pydantic_ai.models.Model] implementations, offering a simpler interface when you don't need the full functionality of an [`Agent`][pydantic_ai.Agent].

The following functions are available:

- [`model_request`][pydantic_ai.low_level.model_request]: Make a non-streamed async request to a model
- [`model_request_sync`][pydantic_ai.low_level.model_request_sync]: Make a synchronous non-streamed request to a model
- [`model_request_stream`][pydantic_ai.low_level.model_request_stream]: Make a streamed async request to a model

## Basic Example

Here's a simple example demonstrating how to use the low-level API to make a basic request:

```python title="low_level_basic.py"
from pydantic_ai.low_level import model_request_sync
from pydantic_ai.messages import ModelRequest

# Make a synchronous request to the model
model_response, usage_info = model_request_sync(
    'anthropic:claude-3-5-haiku-latest',
    [ModelRequest.user_text_prompt('What is the capital of France?')]
)

print(model_response.parts[0].content)
#> Paris
print(usage_info)
"""
Usage(requests=0, request_tokens=56, response_tokens=1, total_tokens=57, details=None)
"""
```

## Advanced Example with Tool Calling

You can also use the low-level API to work with function/tool calling.

Even here we can use Pydantic to generate the JSON schema for the tool:

```python
from pydantic import BaseModel
from typing_extensions import Literal

from pydantic_ai.low_level import model_request
from pydantic_ai.messages import ModelRequest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition


class Divide(BaseModel):
    """Divide two numbers."""

    numerator: float
    denominator: float
    on_inf: Literal['error', 'infinity'] = 'infinity'


async def main():
    # Make a request to the model with tool access
    model_response, cost = await model_request(
        'openai:gpt-4.1-nano',
        [ModelRequest.user_text_prompt('What is 123 / 456?')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name=Divide.__name__.lower(),
                    description=Divide.__doc__ or '',
                    parameters_json_schema=Divide.model_json_schema(),
                )
            ],
            allow_text_output=True,  # Allow model to either use tools or respond directly
        ),
    )
    print(model_response)
    """
    ModelResponse(
        parts=[
            ToolCallPart(
                tool_name='divide',
                args={'numerator': '123', 'denominator': '456'},
                tool_call_id='pyd_ai_2e0e396768a14fe482df90a29a78dc7b',
                part_kind='tool-call',
            )
        ],
        model_name='gpt-4.1-nano',
        timestamp=datetime.datetime(...),
        kind='response',
    )
    """
    print(cost)
    """
    Usage(
        requests=0, request_tokens=55, response_tokens=7, total_tokens=62, details=None
    )
    """
```

## When to Use Low-Level API vs Agent

The low-level API is ideal when:

1. You need more direct control over model interactions
2. You want to implement custom behavior around model requests
3. You're building your own abstractions on top of model interactions

For most application use cases, the higher-level [`Agent`][pydantic_ai.Agent] API provides a more convenient interface with additional features such as built-in tool execution, structured output parsing, and more.

## OpenTelemetry Instrumentation

As with [agents][pydantic_ai.Agent] you can enable OpenTelemetry/logfire instrumentation with just a few extra lines

```python {title="low_level_instrumented.py" hl_lines="1 6 7"}
import logfire

from pydantic_ai.low_level import model_request_sync
from pydantic_ai.messages import ModelRequest

logfire.configure()
logfire.instrument_pydantic_ai()

# Make a synchronous request to the model
model_response, usage_info = model_request_sync(
    'anthropic:claude-3-5-haiku-latest',
    [ModelRequest.user_text_prompt('What is the capital of France?')]
)

print(model_response.parts[0].content)
#> Paris
print(usage_info)
"""
Usage(requests=0, request_tokens=56, response_tokens=1, total_tokens=57, details=None)
"""
```

See [Debugging and Monitoring](logfire.md) for more details.
