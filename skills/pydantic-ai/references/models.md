# Models Reference

Source: `pydantic_ai_slim/pydantic_ai/models/__init__.py`, `pydantic_ai_slim/pydantic_ai/models/test.py`

## Model String Format

Models are specified as `"provider:model-name"` strings:

```python
agent = Agent('openai:gpt-4o')
agent = Agent('anthropic:claude-sonnet-4-5')
agent = Agent('google:gemini-2.5-pro')
agent = Agent('groq:llama-3.3-70b-versatile')
agent = Agent('mistral:mistral-large-latest')
agent = Agent('cohere:command-r-plus')
```

## Providers

| Provider | Prefix | Install Extra | Example |
|----------|--------|--------------|---------|
| OpenAI | `openai:` | `pydantic-ai-slim[openai]` | `openai:gpt-4o` |
| Anthropic | `anthropic:` | `pydantic-ai-slim[anthropic]` | `anthropic:claude-sonnet-4-5` |
| Google | `google:` | `pydantic-ai-slim[google]` | `google:gemini-2.5-pro` |
| Groq | `groq:` | `pydantic-ai-slim[groq]` | `groq:llama-3.3-70b-versatile` |
| Mistral | `mistral:` | `pydantic-ai-slim[mistral]` | `mistral:mistral-large-latest` |
| Cohere | `cohere:` | `pydantic-ai-slim[cohere]` | `cohere:command-r-plus` |
| Bedrock | `bedrock:` | `pydantic-ai-slim[bedrock]` | `bedrock:anthropic.claude-sonnet-4-5-v2-0` |
| HuggingFace | `huggingface:` | `pydantic-ai-slim[huggingface]` | `huggingface:meta-llama/...` |

Additional providers using OpenAI-compatible APIs: `azure:`, `openrouter:`, `grok:`, `deepseek:`, `fireworks:`, `together:`, `ollama:`, `github:`, `cerebras:`, `sambanova:`, `nebius:`, `ovhcloud:`, `alibaba:`.

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
    'openai:gpt-4o',
    'anthropic:claude-sonnet-4-5',
    fallback_on=(ModelHTTPError,),
)
```

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
