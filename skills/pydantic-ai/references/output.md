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

agent = Agent('openai:gpt-5', output_type=ToolOutput(MyModel))
# Equivalent to: output_type=MyModel (ToolOutput is the default mode)
```

Customize the tool name and description:

```python
agent = Agent(
    'openai:gpt-5',
    output_type=ToolOutput(MyModel, name='extract_data', description='Extract structured data'),
)
```

### NativeOutput

Uses provider-specific structured output features (e.g., OpenAI's `response_format`).

```python
from pydantic_ai import Agent, NativeOutput

agent = Agent('openai:gpt-5', output_type=NativeOutput(MyModel))
```

### PromptedOutput

Injects the JSON schema into the system prompt and parses the response.

```python
from pydantic_ai import Agent, PromptedOutput

agent = Agent('openai:gpt-5', output_type=PromptedOutput(MyModel))
```

### TextOutput

Processes plain text output with a custom function.

```python
from pydantic_ai import Agent, TextOutput


def parse_int(text: str) -> int:
    return int(text.strip())


agent = Agent('openai:gpt-5', output_type=TextOutput(parse_int))
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
    'openai:gpt-5',
    output_type=[
        ToolOutput(Fruit, name='return_fruit'),
        ToolOutput(Vehicle, name='return_vehicle'),
    ],
)
```

## Image Output (BinaryImage)

Generate images as agent output using models that support image generation:

```python
from pydantic_ai import Agent, BinaryImage

agent = Agent('openai-responses:gpt-5', output_type=BinaryImage)

result = agent.run_sync('Generate an image of an axolotl.')
assert isinstance(result.output, BinaryImage)
```

When `output_type=BinaryImage`, the `ImageGenerationTool` builtin is enabled automatically if not specified.

### Optional Image Output (Union)

Allow both text and image responses:

```python
from pydantic_ai import Agent, BinaryImage

agent = Agent('openai-responses:gpt-5', output_type=BinaryImage | str)

# Text response
result = agent.run_sync('Tell me about axolotls, no image.')
print(result.output)  # String

# Image response
result = agent.run_sync('Draw an axolotl.')
assert isinstance(result.output, BinaryImage)
# Text is still available:
print(result.response.text)
```

If the model generates both text and image, the image takes precedence as output.

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

### Async Output Validators with Side Effects

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-5', output_type=MyModel)

@agent.output_validator
async def validate_and_log(ctx: RunContext[MyDeps], output: MyModel) -> MyModel:
    # Can perform async operations like database writes
    await ctx.deps.db.log_output(output)
    return output
```

## StructuredDict — Dynamic Schemas

Create structured output without defining a Pydantic model class:

```python
from pydantic_ai import Agent, StructuredDict

# Static schema
agent = Agent('openai:gpt-5', output_type=StructuredDict(name=str, age=int))

# Dynamic schema at runtime
def create_agent(fields: dict[str, type]):
    schema = StructuredDict(**fields)
    return Agent('openai:gpt-5', output_type=schema)

# Create agent with user-defined fields
agent = create_agent({'product': str, 'price': float, 'in_stock': bool})
```

Useful when the schema is determined at runtime or you don't want to define a model class.

## Validation Context

Pass custom context to Pydantic validators during output validation:

```python
from pydantic import BaseModel, field_validator

from pydantic_ai import Agent


class Output(BaseModel):
    value: int

    @field_validator('value')
    @classmethod
    def check_value(cls, v, info):
        max_val = info.context.get('max_value', 100) if info.context else 100
        if v > max_val:
            raise ValueError(f'Value must be <= {max_val}')
        return v


agent = Agent('openai:gpt-5', output_type=Output)

# Pass validation context per-run
result = agent.run_sync(
    'Give me a number',
    validation_context={'max_value': 50}
)
```

Set `validation_context` on the Agent or per-run.

## Override output_type Per Run

```python
result = await agent.run('prompt', output_type=DifferentModel)
```

## prepare_output_tools

Dynamically filter or modify output tools at runtime:

```python
from pydantic_ai import Agent, RunContext, ToolDefinition

async def filter_outputs(
    ctx: RunContext[MyDeps], tool_defs: list[ToolDefinition]
) -> list[ToolDefinition] | None:
    # Only allow certain output tools based on context
    if ctx.deps.restricted_mode:
        return [td for td in tool_defs if td.name != 'detailed_output']
    return tool_defs

agent = Agent(
    'openai:gpt-5',
    output_type=[DetailedOutput, SimpleOutput],
    prepare_output_tools=filter_outputs,
)
```

## NativeOutput Restrictions

Provider-specific limitations when using `NativeOutput`:

| Provider | Limitations |
|----------|-------------|
| OpenAI | Requires JSON mode capable models |
| Anthropic | Limited schema support |
| Google | Works with most Gemini models |

If unsure, use `ToolOutput` (default) which works with all providers.

## Parallel Output Tools & end_strategy

When using union output types with `end_strategy='exhaustive'`:

```python
from pydantic_ai import Agent, ToolOutput

agent = Agent(
    'openai:gpt-5',
    output_type=[ToolOutput(TypeA), ToolOutput(TypeB)],
    end_strategy='exhaustive',  # Process ALL tool calls, including after output
)
```

With `end_strategy='early'` (default), the agent stops after the first output tool is called.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `ToolOutput` | `pydantic_ai.ToolOutput` | Tool-based structured output (default) |
| `NativeOutput` | `pydantic_ai.NativeOutput` | Provider-native structured output |
| `PromptedOutput` | `pydantic_ai.PromptedOutput` | Prompt-injected schema extraction |
| `TextOutput` | `pydantic_ai.TextOutput` | Custom text processing function |
| `StructuredDict` | `pydantic_ai.StructuredDict` | Dict-based structured output |
| `BinaryImage` | `pydantic_ai.BinaryImage` | Image output type |

## See Also

- [agents.md](agents.md) — Agent configuration
- [streaming.md](streaming.md) — Streaming structured output
- [models.md](models.md) — Provider-specific output support
- [troubleshooting.md](troubleshooting.md) — Output anti-patterns
