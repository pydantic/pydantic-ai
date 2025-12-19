# Model Integration Guide

This document explains key concepts for PydanticAI model integrations.

## Output Configuration and Its Effects

PydanticAI agents have internal parameters that affect model behavior. These are derived from agent configuration, not set directly.

### `allow_text_output`

Determines whether the model can return plain text as a final response.

**Derived from:** `output_schema.allows_text` (whether a text processor exists)

| Agent Configuration | `allow_text_output` |
|---------------------|---------------------|
| `Agent(model)` (default, output_type=str) | `True` |
| `Agent(model, output_type=str)` | `True` |
| `Agent(model, output_type=MyBaseModel)` | `False` (uses tool output) |
| `Agent(model, output_type=str \| MyBaseModel)` | `True` |

### `allow_image_output`

Determines whether the model can return images as output.

**Derived from:** Whether `BinaryImage` is in the output types

| Agent Configuration | `allow_image_output` |
|---------------------|----------------------|
| `Agent(model)` (default) | `False` |
| `Agent(model, output_type=BinaryImage)` | `True` |
| `Agent(model, output_type=str \| BinaryImage)` | `True` |

### Output Tools vs Function Tools

PydanticAI distinguishes between two types of tools:

1. **Function tools**: User-registered tools via `tools=[...]` or `@agent.tool`
2. **Output tools**: Framework-internal tools created when `output_type` is a BaseModel

When handling tool_choice, these distinctions matter:
- `tool_choice` settings primarily affect function tools
- Output tools may or may not be included depending on the tool_choice variant used

## Testing Different Configurations

```python
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryImage

def get_weather(city: str) -> str:
    '''Get weather for a city.'''
    return f'Sunny in {city}'

class StructuredOutput(BaseModel):
    result: str
    confidence: float

# Scenario 1: Text output allowed, no output tools
agent = Agent(model, tools=[get_weather])

# Scenario 2: Text output NOT allowed, has output tools
agent = Agent(model, output_type=StructuredOutput, tools=[get_weather])

# Scenario 3: Text output allowed, has output tools (union type)
agent = Agent(model, output_type=str | StructuredOutput, tools=[get_weather])

# Scenario 4: Image output allowed
agent = Agent(model, output_type=str | BinaryImage)

# Scenario 5: No function tools, only output tools
agent = Agent(model, output_type=StructuredOutput)
```

## ModelRequestParameters

When a model's `request()` method is called, it receives `ModelRequestParameters` which includes:

- `function_tools`: List of user-registered tool definitions
- `output_tools`: List of framework-generated output tool definitions
- `allow_text_output`: Whether text responses are valid
- `allow_image_output`: Whether image responses are valid
- `tool_choice`: The resolved tool choice setting

These parameters inform how the model should configure its API request.
