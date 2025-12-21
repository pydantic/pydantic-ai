# Tool Choice

Tool choice controls which tools the model can use during a request. This is useful for controlling agent behavior, optimizing for specific use cases, or working around provider limitations.

## Overview

PydanticAI distinguishes between two types of tools:

- **Function tools**: Tools you register with the agent using `@agent.tool` or `tools=[...]`
- **Output tools**: Internal tools used by the framework for structured output (e.g., `final_result`)

The `tool_choice` setting in [`ModelSettings`][pydantic_ai.settings.ModelSettings] controls which **function tools** are available to the model. Output tools are handled separately to ensure agents can always complete their work.

## Options

### `None` or `'auto'` (default)

The model decides whether to use tools. All function tools and output tools are available.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', tools=[get_weather, get_time])

# Both are equivalent - model decides when to use tools
result = agent.run_sync('What is the weather in Paris?')
result = agent.run_sync('What is the weather in Paris?', model_settings={'tool_choice': 'auto'})
```

### `'none'` or `[]`

Disables function tools. The model can only respond with text (or use output tools for structured output).

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', tools=[get_weather])

# Function tools disabled - model responds with text only
result = agent.run_sync(
    'What is the weather in Paris?',
    model_settings={'tool_choice': 'none'}
)
```

This is useful when you want to prevent tool calls before hitting usage limits or when you want the model to respond directly without using tools.

!!! note "Output tools remain available"
    When using structured output, output tools are still available even with `tool_choice='none'`. This ensures agents can complete with the required output type.

### `'required'`

Forces the model to use a function tool. No output tools are sent.

!!! warning "For direct model requests only"
    Use `'required'` only with [direct model requests](direct.md), not with agent runs. Since output tools are excluded, an agent cannot complete normally.

```python
from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import model_request_sync
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

settings: ModelSettings = {'tool_choice': 'required'}
params = ModelRequestParameters(
    function_tools=[
        ToolDefinition(
            name='get_weather',
            description='Get weather for a city',
            parameters_json_schema={
                'type': 'object',
                'properties': {'city': {'type': 'string'}},
                'required': ['city'],
            },
        )
    ],
    allow_text_output=True,
)

response = model_request_sync(
    'openai:gpt-4o',
    [ModelRequest.user_text_prompt('What is the weather in Paris?')],
    model_settings=settings,
    model_request_parameters=params,
)
```

### `list[str]` - Specific tools

Restricts the model to specific function tools by name. The model must use one of the listed tools. No output tools are sent.

!!! warning "For direct model requests only"
    Like `'required'`, use tool lists only with [direct model requests](direct.md).

```python
from pydantic_ai import ModelRequest, ToolDefinition
from pydantic_ai.direct import model_request_sync
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

settings: ModelSettings = {'tool_choice': ['get_weather']}  # Only allow get_weather
params = ModelRequestParameters(
    function_tools=[weather_tool, time_tool, population_tool],
    allow_text_output=True,
)

response = model_request_sync(
    'openai:gpt-4o',
    [ModelRequest.user_text_prompt('Tell me about Paris')],
    model_settings=settings,
    model_request_parameters=params,
)
# Model will use get_weather even though other tools are defined
```

### `ToolsPlusOutput` - Specific tools with output

Restricts function tools while keeping output tools available. This is the recommended way to control tool choice in agent runs with structured output.

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings, ToolsPlusOutput


class CityInfo(BaseModel):
    city: str
    summary: str


agent: Agent[None, CityInfo] = Agent(
    'openai:gpt-4o',
    output_type=CityInfo,
    tools=[get_weather, get_time, get_population],
)

# Only get_weather is available, but output tools (final_result) are preserved
settings: ModelSettings = {'tool_choice': ToolsPlusOutput(function_tools=['get_weather'])}
result = agent.run_sync('Get weather for Tokyo and summarize', model_settings=settings)
```

## Provider Support

| Provider | `'auto'` | `'none'` | `'required'` | Specific tools | Notes |
|----------|:--------:|:--------:|:------------:|:--------------:|-------|
| OpenAI | ✓ | ✓ | ✓ | ✓ | Full support |
| Anthropic | ✓ | ✓ | ⚠️ | ⚠️ | Not supported with thinking enabled |
| Google | ✓ | ✓ | ✓ | ✓ | Uses `allowed_function_names` |
| Bedrock | ✓ | ✓ | ✓ | ⚠️ | Single tool only; no native `'none'` |
| Groq | ✓ | ✓ | ✓ | ⚠️ | Single tool only |
| HuggingFace | ✓ | ✓ | ✓ | ⚠️ | Single tool only |
| Mistral | ✓ | ✓ | ✓ | ⚠️ | Limited specific tool support |

### Provider-Specific Notes

#### OpenAI

Full support for all tool choice options. When specifying multiple tools, OpenAI's `allowed_tools` parameter is used for efficient filtering.

#### Anthropic

`'required'` and specific tool lists are **not supported** when thinking is enabled. Using these options with thinking will raise a `UserError`:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent('anthropic:claude-sonnet-4-5', tools=[my_tool])

# This will raise UserError if thinking is enabled
settings = AnthropicModelSettings(
    tool_choice='required',
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
)
```

#### Google

Full support using Gemini's `allowed_function_names` for tool restrictions.

#### Bedrock

- **No native `'none'` support**: Handled by filtering tools from the request
- **Single tool forcing only**: When specifying multiple tools, falls back to `'any'` mode (must use one of the available tools)

#### Groq

- **Single tool forcing only**: When specifying multiple tools in a list, a warning is issued and the model uses `'any'` mode

#### HuggingFace

- **Single tool forcing only**: Similar to Groq, multiple tools fall back to `'any'` mode with a warning

#### Mistral

- **Limited specific tool support**: Tool lists use `'required'` mode without native filtering

## Common Patterns

### Preventing tool calls before usage limits

Disable tools when approaching token limits to prevent `UsageLimitExceeded` errors:

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

agent = Agent('openai:gpt-4o', tools=[expensive_tool])

# Normal run with tools
result = agent.run_sync('Analyze this data', usage_limits=UsageLimits(request_tokens_limit=1000))

# Disable tools when close to limits
result = agent.run_sync(
    'Summarize your findings',
    message_history=result.all_messages(),
    model_settings={'tool_choice': 'none'},
)
```

### Enforcing tool order

Force specific tools in sequence for controlled workflows:

```python
from pydantic_ai.direct import model_request_sync
from pydantic_ai.settings import ModelSettings

# Step 1: Force search tool
settings: ModelSettings = {'tool_choice': ['search']}
response = model_request_sync('openai:gpt-4o', messages, model_settings=settings, ...)

# Step 2: Force analyze tool with search results
settings = {'tool_choice': ['analyze']}
response = model_request_sync('openai:gpt-4o', updated_messages, model_settings=settings, ...)

# Step 3: Allow model to respond
settings = {'tool_choice': 'none'}
response = model_request_sync('openai:gpt-4o', final_messages, model_settings=settings, ...)
```

### Using structured output with restricted tools

Use `ToolsPlusOutput` when you need both tool restrictions and structured output:

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.settings import ToolsPlusOutput


class Report(BaseModel):
    title: str
    findings: list[str]


agent: Agent[None, Report] = Agent(
    'openai:gpt-4o',
    output_type=Report,
    tools=[fetch_data, analyze_data, format_report],
)

# Only fetch_data available, but final_result output tool is preserved
result = agent.run_sync(
    'Fetch the latest metrics',
    model_settings={'tool_choice': ToolsPlusOutput(function_tools=['fetch_data'])},
)
```
