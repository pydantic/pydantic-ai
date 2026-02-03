# Models Reference

Source: `pydantic_ai_slim/pydantic_ai/models/__init__.py`, `pydantic_ai_slim/pydantic_ai/models/test.py`

## Model String Format

Models are specified as `"provider:model-name"` strings:

```python
agent = Agent('openai:gpt-5')
agent = Agent('anthropic:claude-sonnet-4-5')
agent = Agent('google:gemini-2.5-pro')
agent = Agent('groq:llama-3.3-70b-versatile')
agent = Agent('mistral:mistral-large-latest')
agent = Agent('cohere:command-r-plus')
```

## Model, Provider, and Profile

Three key concepts for model configuration:

| Concept | Description | Example |
|---------|-------------|---------|
| **Model** | How to call a model API | `OpenAIChatModel`, `AnthropicModel` |
| **Provider** | Authentication and endpoint config | `AzureProvider`, `OpenRouterProvider` |
| **Profile** | Model-specific schema/behavior tweaks | JSON schema transformations |

When you use `'openai:gpt-5'`, PydanticAI auto-selects the model class, provider, and profile.

## Providers

| Provider | Prefix | Install Extra | Example |
|----------|--------|--------------|---------|
| OpenAI | `openai:` | `pydantic-ai-slim[openai]` | `openai:gpt-5` |
| Anthropic | `anthropic:` | `pydantic-ai-slim[anthropic]` | `anthropic:claude-sonnet-4-5` |
| Google | `google:` | `pydantic-ai-slim[google]` | `google:gemini-2.5-pro` |
| Groq | `groq:` | `pydantic-ai-slim[groq]` | `groq:llama-3.3-70b-versatile` |
| Mistral | `mistral:` | `pydantic-ai-slim[mistral]` | `mistral:mistral-large-latest` |
| Cohere | `cohere:` | `pydantic-ai-slim[cohere]` | `cohere:command-r-plus` |
| Bedrock | `bedrock:` | `pydantic-ai-slim[bedrock]` | `bedrock:anthropic.claude-sonnet-4-5-v2-0` |
| HuggingFace | `huggingface:` | `pydantic-ai-slim[huggingface]` | `huggingface:meta-llama/...` |

Additional providers using OpenAI-compatible APIs: `azure:`, `openrouter:`, `grok:`, `deepseek:`, `fireworks:`, `together:`, `ollama:`, `github:`, `cerebras:`, `sambanova:`, `nebius:`, `ovhcloud:`, `alibaba:`.

## ModelProfile — OpenAI-Compatible APIs

When using OpenAI-compatible APIs (via `OpenRouterProvider`, custom endpoints, etc.), use `ModelProfile` to configure model-specific behavior:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles import ModelProfile

# Using OpenRouter to access Gemini via OpenAI-compatible API
model = OpenAIChatModel(
    'google/gemini-2.5-pro-preview',
    provider='openrouter',
    profile=ModelProfile.from_profile('google-gla'),  # Use Gemini's profile
)

agent = Agent(model)
```

### Why Profiles Matter

Different models have different:
- JSON schema restrictions for tools
- System prompt handling
- Token counting rules
- Feature support (streaming, function calling, etc.)

The profile tells PydanticAI how to format requests correctly, regardless of which API endpoint you're using.

### Custom Profile

```python
from pydantic_ai.profiles import ModelProfile

custom_profile = ModelProfile(
    supports_json_schema_mode=True,
    supports_strict_tool_mode=True,
    default_max_tokens=4096,
)
```

## ModelSettings

Configure model behavior per-agent or per-run:

```python {title="model_settings_example.py"}
from pydantic_ai import Agent

agent = Agent(
    'openai:gpt-5',
    model_settings={
        'temperature': 0.0,
        'max_tokens': 500,
        'top_p': 0.9,
    },
)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

Override per-run:

```python
result = await agent.run('prompt', model_settings={'temperature': 0.5})
```

Common `ModelSettings` fields:
- `max_tokens: int` — maximum tokens in the response
- `temperature: float` — randomness (0.0 = deterministic)
- `top_p: float` — nucleus sampling threshold
- `timeout: float | Timeout` — request timeout in seconds

## TestModel — For Testing

Deterministic model that returns tool calls and structured output without hitting any API.

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

### TestModel Constructor

```python
TestModel(
    *,
    call_tools='all',            # list[str] | 'all' — which tools to call
    custom_output_text=None,     # str | None — override text output
    custom_output_args=None,     # Any — override tool call args
    seed=0,                      # int — seed for deterministic output
    model_name='test',           # str — model name for logging
    profile=None,                # ModelProfileSpec — optional profile override
)
```

## FunctionModel — Custom Logic

A model controlled by a local function. Useful for testing with complex logic.

```python
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart


def my_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('Custom response')])


agent = Agent(FunctionModel(my_model))
```

### FunctionModel Constructor

```python
FunctionModel(
    function,                    # Sync function (messages, info) -> ModelResponse
    *,
    stream_function=None,        # Async generator for streaming
    model_name=None,             # str — model name
    profile=None,                # ModelProfileSpec
    settings=None,               # ModelSettings
)
```

## FallbackModel

Try multiple models in order, falling back on failure:

```python {title="fallback_model_simple.py"}
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel

agent = Agent(
    FallbackModel('anthropic:claude-sonnet-4-5', 'openai:gpt-5'),
)
result = agent.run_sync('What is the capital of France?')
print(result.output)
#> The capital of France is Paris.
```

By default, falls back on `ModelAPIError`. Customize with `fallback_on`:

```python
from pydantic_ai.exceptions import ModelHTTPError

model = FallbackModel(
    'openai:gpt-5',
    'anthropic:claude-sonnet-4-5',
    fallback_on=(ModelHTTPError,),
)
```

### Per-Model Settings in FallbackModel

Configure different settings for each model in a fallback chain:

```python {title="fallback_model_per_settings.py"}
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

# Configure each model with provider-specific optimal settings
openai_model = OpenAIChatModel(
    'gpt-5',
    settings=ModelSettings(temperature=0.7, max_tokens=1000)  # Higher creativity for OpenAI
)
anthropic_model = AnthropicModel(
    'claude-sonnet-4-5',
    settings=ModelSettings(temperature=0.2, max_tokens=1000)  # Lower temperature for consistency
)

fallback_model = FallbackModel(openai_model, anthropic_model)
agent = Agent(fallback_model)

result = agent.run_sync('Write a creative story about space exploration')
print(result.output)
"""
In the year 2157, Captain Maya Chen piloted her spacecraft through the vast expanse of the Andromeda Galaxy. As she discovered a planet with crystalline mountains that sang in harmony with the cosmic winds, she realized that space exploration was not just about finding new worlds, but about finding new ways to understand the universe and our place within it.
"""
```

## Production Model Patterns

### When to Use FallbackModel

| Use Case | Pattern |
|----------|---------|
| High availability | Primary → Secondary provider |
| Cost optimization | Cheap model → Expensive fallback |
| Rate limit handling | Primary → Different provider |
| Provider outages | Multi-provider redundancy |

### Disabling Provider SDK Retries

Provider SDKs have built-in retries that can delay fallback. Disable them for immediate fallback:

```python
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIChatModel

# Disable OpenAI SDK retries for immediate fallback
client = AsyncOpenAI(max_retries=0)
model = OpenAIChatModel('gpt-5', openai_client=client)
```

### Environment-Based Model Selection

```python
import os
from pydantic_ai import Agent

# Select model based on environment
model = os.getenv('PYDANTIC_AI_MODEL', 'openai:gpt-5')
agent = Agent(model)

# Or use different models for different environments
if os.getenv('ENV') == 'production':
    agent = Agent('anthropic:claude-sonnet-4-5')
else:
    agent = Agent('openai:gpt-5-mini')  # Cheaper for development
```

### Model Override for Testing

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('openai:gpt-5')  # Production model

# In tests, override with TestModel
with agent.override(model=TestModel()):
    result = agent.run_sync('test')
    # No API calls made

# Or override per-run
result = await agent.run('prompt', model='anthropic:claude-sonnet-4-5')
```

### Validation Errors Don't Trigger Fallback

Validation errors use the retry mechanism (re-prompts same model) instead of fallback. This is intentional: validation errors may succeed on retry, while API errors (4xx/5xx) indicate issues that won't resolve by retrying the same request.

## ALLOW_MODEL_REQUESTS

Global flag to prevent accidental real API calls in tests:

```python
from pydantic_ai.models import ALLOW_MODEL_REQUESTS

ALLOW_MODEL_REQUESTS = False  # Raises error if any real model is used
```

In pytest, use the `allow_model_requests` fixture or set this in `conftest.py`.

## Override Model at Runtime

```python
# Via run() parameter
result = await agent.run('prompt', model='anthropic:claude-sonnet-4-5')

# Via override context manager
with agent.override(model=TestModel()):
    result = agent.run_sync('test')
```

## Model Observability

With Logfire instrumentation enabled, every model request is traced. For the deepest visibility, add HTTP instrumentation to see exact request/response payloads:

```python
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)  # See raw HTTP payloads
```

This reveals:

- Request/response latency per model
- Token usage (input/output) for cost tracking
- **Full HTTP request body** — the exact JSON sent to the provider
- **Full HTTP response** — the raw response before parsing
- Fallback sequences when using `FallbackModel`

When debugging provider-specific issues (unexpected errors, malformed responses, rate limits), the HTTP-level view shows exactly what the provider received and returned.

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `Model` | `pydantic_ai.models.Model` | Abstract base class |
| `KnownModelName` | `pydantic_ai.models.KnownModelName` | Literal type of all known models |
| `TestModel` | `pydantic_ai.models.test.TestModel` | Deterministic test model |
| `FunctionModel` | `pydantic_ai.models.function.FunctionModel` | Custom function model |
| `FallbackModel` | `pydantic_ai.models.fallback.FallbackModel` | Multi-model fallback |
| `ModelSettings` | `pydantic_ai.ModelSettings` | Model configuration dict |
| `ModelProfile` | `pydantic_ai.ModelProfile` | Model capability profile |

## See Also

- [agents.md](agents.md) — Using models with agents
- [testing.md](testing.md) — TestModel and FunctionModel patterns
- [observability.md](observability.md) — Model request tracing
- [retries.md](retries.md) — HTTP retry configuration
