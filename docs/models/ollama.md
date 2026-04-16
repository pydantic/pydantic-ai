# Ollama

## Install

To use [`OllamaModel`][pydantic_ai.models.ollama.OllamaModel], you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add "pydantic-ai-slim[openai]"
```

## Configuration

Pydantic AI supports both self-hosted [Ollama](https://ollama.com/) servers (running locally or remotely) and [Ollama Cloud](https://ollama.com/cloud).

For servers running locally, use the `http://localhost:11434/v1` base URL. For Ollama Cloud, use `https://ollama.com/v1` and ensure an API key is set.

[`OllamaModel`][pydantic_ai.models.ollama.OllamaModel] uses Ollama's OpenAI-compatible Chat Completions API (`/v1/chat/completions`), so existing Ollama users can start using Pydantic AI without any migration.

## Environment variable

Set the `OLLAMA_BASE_URL` and (optionally) `OLLAMA_API_KEY` environment variables:

```bash
export OLLAMA_BASE_URL='http://localhost:11434/v1'
export OLLAMA_API_KEY='your-api-key'  # required for Ollama Cloud
```

You can then use `OllamaModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('ollama:qwen3')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

model = OllamaModel('qwen3')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

model = OllamaModel(
    'qwen3', provider=OllamaProvider(base_url='http://localhost:11434/v1')
)
agent = Agent(model)
...
```

For Ollama Cloud, use `base_url='https://ollama.com/v1'` and set the `OLLAMA_API_KEY` environment variable (or pass `api_key=` directly).

## Structured output

Self-hosted Ollama (v0.5.0+, released December 2024) enforces `response_format` with `json_schema` via `llama.cpp`'s grammar-constrained decoder, so [`NativeOutput`][pydantic_ai.output.NativeOutput] produces schema-valid output at generation time:

```python
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.ollama import OllamaProvider


class CityLocation(BaseModel):
    city: str
    country: str


model = OllamaModel(
    'qwen3',
    provider=OllamaProvider(base_url='http://localhost:11434/v1'),
)
agent = Agent(model, output_type=NativeOutput(CityLocation))
...
```

!!! note "Ollama Cloud does not enforce `json_schema` yet"
    Ollama Cloud's inference backend accepts `response_format` with `json_schema` without error but does not apply grammar-constrained decoding, so schemas are silently not enforced. See [ollama/ollama#12362](https://github.com/ollama/ollama/issues/12362) for the upstream tracking issue.

    When [`OllamaModel`][pydantic_ai.models.ollama.OllamaModel] detects a Cloud path — either a `base_url` on `ollama.com` or a model name ending in `-cloud` — it automatically disables `supports_json_schema_output` on the profile.

    If you use [`NativeOutput`][pydantic_ai.output.NativeOutput] with an Ollama Cloud model, you'll get a clear [`UserError`][pydantic_ai.exceptions.UserError] instead of a silent retry loop. Use the default [`ToolOutput`][pydantic_ai.output.ToolOutput] or [`PromptedOutput`][pydantic_ai.output.PromptedOutput] instead — both work on Cloud.

## Tool calling

Tool calling works on both self-hosted and Cloud Ollama via the inherited OpenAI Chat Completions API path. Ollama does not support OpenAI's `strict` mode for tool schemas, so [`openai_supports_strict_tool_definition`][pydantic_ai.profiles.openai.OpenAIModelProfile.openai_supports_strict_tool_definition] is set to `False` in the profile.
