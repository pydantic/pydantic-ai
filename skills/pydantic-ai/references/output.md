# Output Reference

Source: `pydantic_ai_slim/pydantic_ai/output.py`, `pydantic_ai_slim/pydantic_ai/_output.py`

## Basic Structured Output

Pass a Pydantic model, dataclass, or TypedDict as `output_type`:

```python {title="olympics.py" line_length="90"}
from pydantic import BaseModel

from pydantic_ai import Agent


class CityLocation(BaseModel):
    city: str
    country: str


agent = Agent('google-gla:gemini-2.5-flash', output_type=CityLocation)
result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
print(result.usage())
#> RunUsage(input_tokens=57, output_tokens=8, requests=1)
```

## Output Modes

### ToolOutput (default)

Uses tool calls to extract structured data. Works with all providers.

```python
from pydantic_ai import Agent, ToolOutput

agent = Agent('openai:gpt-4o', output_type=ToolOutput(MyModel))
# Equivalent to: output_type=MyModel (ToolOutput is the default mode)
```

Customize the tool name and description:

```python
agent = Agent(
    'openai:gpt-4o',
    output_type=ToolOutput(MyModel, name='extract_data', description='Extract structured data'),
)
```

### NativeOutput

Uses provider-specific structured output features (e.g., OpenAI's `response_format`).

```python
from pydantic_ai import Agent, NativeOutput

agent = Agent('openai:gpt-4o', output_type=NativeOutput(MyModel))
```

### PromptedOutput

Injects the JSON schema into the system prompt and parses the response.

```python
from pydantic_ai import Agent, PromptedOutput

agent = Agent('openai:gpt-4o', output_type=PromptedOutput(MyModel))
```

### TextOutput

Processes plain text output with a custom function.

```python
from pydantic_ai import Agent, TextOutput


def parse_int(text: str) -> int:
    return int(text.strip())


agent = Agent('openai:gpt-4o', output_type=TextOutput(parse_int))
```

`TextOutput` can also take a function with `RunContext`:

```python
from pydantic_ai import RunContext, TextOutput

def parse_with_ctx(ctx: RunContext[str], text: str) -> int:
    return int(text.strip())
```

## Union Output Types

Pass a list to allow multiple output types:

```python {title="tool_output.py"}
from pydantic import BaseModel

from pydantic_ai import Agent, ToolOutput


class Fruit(BaseModel):
    name: str
    color: str


class Vehicle(BaseModel):
    name: str
    wheels: int


agent = Agent(
    'openai:gpt-5',
    output_type=[
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)
result = agent.run_sync('What is a banana?')
print(repr(result.output))
#> Fruit(name='banana', color='yellow')
```

Use `ToolOutput` with custom names for union types:

```python
from pydantic_ai import Agent, ToolOutput

agent = Agent(
    'openai:gpt-4o',
    output_type=[
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)
```

## Output Validators

Validate or transform the output after the model produces it:

```python {title="output_validator_simple.py"}
from pydantic import BaseModel

from pydantic_ai import Agent, ModelRetry, RunContext


class CityInfo(BaseModel):
    city: str
    country: str


agent = Agent('openai:gpt-5', output_type=CityInfo)


@agent.output_validator
def validate_city(ctx: RunContext[None], output: CityInfo) -> CityInfo:
    if output.city == output.country:
        raise ModelRetry('City name and country cannot be the same.')
    return output


result = agent.run_sync('Where were the olympics held in 2012?')
print(result.output)
#> city='London' country='United Kingdom'
```

Output validators can be sync or async. Raising `ModelRetry` sends feedback to the model.

## StructuredDict

A typed dictionary output for when you want structured data without a full Pydantic model:

```python
from pydantic_ai import Agent, StructuredDict

agent = Agent('openai:gpt-4o', output_type=StructuredDict(name=str, age=int))
```

## Override output_type Per Run

```python
result = await agent.run('prompt', output_type=DifferentModel)
```

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `ToolOutput` | `pydantic_ai.ToolOutput` | Tool-based structured output (default) |
| `NativeOutput` | `pydantic_ai.NativeOutput` | Provider-native structured output |
| `PromptedOutput` | `pydantic_ai.PromptedOutput` | Prompt-injected schema extraction |
| `TextOutput` | `pydantic_ai.TextOutput` | Custom text processing function |
| `StructuredDict` | `pydantic_ai.StructuredDict` | Dict-based structured output |
