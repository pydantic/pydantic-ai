# Thinking

Thinking (or reasoning) is the process by which a model works through a problem step-by-step before
providing its final answer.

This capability is typically disabled by default and depends on the specific model being used.
See the sections below for how to enable thinking for each provider.

## OpenAI

When using the [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], text output inside `<think>` tags are converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).

The [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] can generate native thinking parts.
To enable this functionality, you need to set the `openai_reasoning_effort` and `openai_reasoning_summary` fields in the
[`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings].

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

## Anthropic

To enable thinking, use the `anthropic_thinking` field in the [`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings].

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

To enable thinking, use the `google_thinking_config` field in the
[`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings].

```python {title="google_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-2.5-pro')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
agent = Agent(model, model_settings=settings)
...
```

## Bedrock

## Groq

Groq supports different formats to receive thinking parts:

- `"raw"`: The thinking part is included in the text content inside `<think>` tags, which are automatically converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
- `"hidden"`: The thinking part is not included in the text content.
- `"parsed"`: The thinking part has its own structured part in the response which is converted into a [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] object.

To enable thinking, use the `groq_reasoning_format` field in the
[`GroqModelSettings`][pydantic_ai.models.groq.GroqModelSettings]:

```python {title="groq_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('qwen-qwq-32b')
settings = GroqModelSettings(groq_reasoning_format='parsed')
agent = Agent(model, model_settings=settings)
...
```

## Mistral

Thinking is supported by the `magistral` family of models. It does not need to be specifically enabled.

## Cohere

Thinking is supported by the `command-a-reasoning-08-2025` model. It does not need to be specifically enabled.

## Hugging Face

Text output inside `<think>` tags is automatically converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).
