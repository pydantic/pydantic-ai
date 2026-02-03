# Thinking Reference

Source: `pydantic_ai_slim/pydantic_ai/messages.py`

Thinking (reasoning) is step-by-step problem solving by the model before the final answer. Typically disabled by default.

## OpenAI

### OpenAI Responses API

Enable reasoning with model settings:

```python {title="openai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
    openai_reasoning_summary='detailed',
)
agent = Agent(model, model_settings=settings)
...
```

Settings:
- `openai_reasoning_effort`: `'low'` | `'medium'` | `'high'`
- `openai_reasoning_summary`: `'auto'` | `'concise'` | `'detailed'`
- `openai_send_reasoning_ids`: Set `False` to avoid ID mismatch errors with history processors

### OpenAI Chat (compatible APIs)

Text inside `<think>` tags is converted to `ThinkingPart`. Customize tags via model profile:

```python
from pydantic_ai.profiles import ModelProfile

profile = ModelProfile(thinking_tags=('<reasoning>', '</reasoning>'))
```

## Anthropic

Enable extended thinking:

```python {title="anthropic_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-sonnet-4-0')
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
)
agent = Agent(model, model_settings=settings)
...
```

## Google

Enable thinking config:

```python {title="google_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-2.5-pro')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
agent = Agent(model, model_settings=settings)
...
```

## xAI (Grok)

Reasoning models support native thinking. Preserve for multi-turn:

```python {title="xai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.xai import XaiModel, XaiModelSettings

model = XaiModel('grok-4-fast-reasoning')
settings = XaiModelSettings(xai_include_encrypted_content=True)
agent = Agent(model, model_settings=settings)
...
```

## Bedrock

Use provider-specific fields:

```python {title="bedrock_claude_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
model_settings = BedrockModelSettings(
    bedrock_additional_model_requests_fields={
        'thinking': {'type': 'enabled', 'budget_tokens': 1024}
    }
)
agent = Agent(model=model, model_settings=model_settings)

```

For OpenAI on Bedrock:

```python {title="bedrock_openai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

model = BedrockConverseModel('openai.gpt-oss-120b-1:0')
model_settings = BedrockModelSettings(
    bedrock_additional_model_requests_fields={'reasoning_effort': 'low'}
)
agent = Agent(model=model, model_settings=model_settings)

```

## Groq

Choose reasoning format:

```python {title="groq_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('qwen-qwq-32b')
settings = GroqModelSettings(groq_reasoning_format='parsed')
agent = Agent(model, model_settings=settings)
...
```

Formats:
- `'raw'`: Thinking in `<think>` tags (converted to `ThinkingPart`)
- `'hidden'`: Thinking excluded from output
- `'parsed'`: Structured thinking part in response

## OpenRouter

```python {title="openrouter_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

model = OpenRouterModel('openai/gpt-5')
settings = OpenRouterModelSettings(openrouter_reasoning={'effort': 'high'})
agent = Agent(model, model_settings=settings)
...
```

## Mistral

Thinking supported by `magistral` models — no configuration needed.

## Cohere

Thinking supported by `command-a-reasoning-08-2025` — no configuration needed.

## Hugging Face / Outlines

Text in `<think>` tags auto-converted to `ThinkingPart`. Customize via model profile.

## Accessing Thinking Content

Thinking parts are in `ModelResponse.parts`:

```python
from pydantic_ai.messages import ThinkingPart

result = agent.run_sync('Complex problem')

for msg in result.all_messages():
    if hasattr(msg, 'parts'):
        for part in msg.parts:
            if isinstance(part, ThinkingPart):
                print(f'Thinking: {part.content}')
```

## Provider Support Summary

| Provider | Method | Notes |
|----------|--------|-------|
| OpenAI Responses | `openai_reasoning_effort` | Native reasoning |
| OpenAI Chat | `<think>` tags | Via model profile |
| Anthropic | `anthropic_thinking` | Budget tokens required |
| Google | `google_thinking_config` | Gemini 2.5+ |
| xAI | Native | Grok reasoning models |
| Bedrock | `bedrock_additional_model_requests_fields` | Provider-specific |
| Groq | `groq_reasoning_format` | raw/hidden/parsed |
| OpenRouter | `openrouter_reasoning` | Effort setting |
| Mistral | Auto | Magistral models |
| Cohere | Auto | Command-a-reasoning |

## Key Types

| Type | Import | Description |
|------|--------|-------------|
| `ThinkingPart` | `pydantic_ai.ThinkingPart` | Thinking content in response |
| `ModelResponse` | `pydantic_ai.ModelResponse` | Response containing parts |
