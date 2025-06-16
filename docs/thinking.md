# Thinking

Thinking or reasoning is the process of using a model's capabilities to reason about a task.

This capability is usually not enabled by default, and how to enable it depends on the model.

## OpenAI

The [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel] doesn't emit thinking (reasoning) parts,
but it's able to receive messages from other models that do. In this case, it will convert the
[`ThinkingPart`][pydantic_ai.messages.ThinkingPart] into [`TextPart`][pydantic_ai.messages.TextPart]s using the
`"<think>"` tag.

If you want to properly emit thinking parts, you'd need to use the
[`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel].

```python {title="openai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('o3-mini')
settings = OpenAIResponsesModelSettings(openai_reasoning_effort='low', openai_reasoning_summary='detailed')
agent = Agent(model, model_settings=settings)
...
```

## Anthropic

Differently than other providers, Anthropic sends back a signature in the thinking part. This signature
is used to make sure that the thinking part was not tampered with.

To enable the thinking part, use the `anthropic_thinking` field on the
[`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings]

```python {title="anthropic_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-3-7-sonnet-latest')
settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
agent = Agent(model, model_settings=settings)
...
```

## Groq

Groq supports different formats to receive thinking parts:

- `"raw"`: The thinking part is included in the text content with the `"<think>"` tag.
- `"hidden"`: The thinking part is not included in the text content.
- `"parsed"`: The thinking part has its own [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] object.

To enable the thinking part, use the `groq_reasoning_format` field on the
[`GroqModelSettings`][pydantic_ai.models.groq.GroqModelSettings]:

```python {title="groq_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel, GroqModelSettings

model = GroqModel('qwen-qwq-32b')
settings = GroqModelSettings(groq_reasoning_format='parsed')
agent = Agent(model, model_settings=settings)
...
```

## Google

To enable the thinking part, use the `google_thinking_config` field on the
[`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings]

```python {title="google_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-2.5-pro-preview-03-25')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
agent = Agent(model, model_settings=settings)
...
```

## Mistral / Cohere

Both Mistral and Cohere don't emit thinking parts, but when the model receives it in the history, it will convert the
[`ThinkingPart`][pydantic_ai.messages.ThinkingPart] into [`TextPart`][pydantic_ai.messages.TextPart]s using the
`"<think>"` tag.
