# Thinking

Thinking (or reasoning) is the process by which a model works through a problem step-by-step before
providing its final answer.

This capability is typically disabled by default and depends on the specific model being used.
See the below sections for how to enable thinking for each provider.

Internally, if the model doesn't provide thinking objects, PydanticAI will convert the thinking blocks
(`"<think>..."</think>"`) in provider specific text parts to PydanticAI `ThinkingPart`s. We also made
the decision to not send back the `ThinkingPart`s to the provider in a multi-turn conversation - the
idea is to save costs on behalf of the user. In the future, we intend to add a setting to customize
this behavior.

## OpenAI

When using the [`OpenAIModel`][pydantic_ai.models.openai.OpenAIModel], thinking objects are not created
by default. However, the text content may contain `"<think>"` tags. When this happens, PydanticAI will
convert them to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.

On the other hand, the [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] does
generate thinking parts. To enable it, you need to set the `openai_reasoning_effort` and
`openai_reasoning_summary` fields in the
[`OpenAIResponsesModelSettings`][pydantic_ai.models.openai.OpenAIResponsesModelSettings].

```python {title="openai_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('o3-mini')
settings = OpenAIResponsesModelSettings(
    openai_reasoning_effort='low',
    openai_reasoning_summary='detailed',
)
agent = Agent(model, model_settings=settings)
...
```

## Anthropic

Unlike other providers, Anthropic includes a signature in the thinking part. This signature is used to
ensure that the thinking part hasn't been tampered with. To enable thinking, use the `anthropic_thinking`
field in the [`AnthropicModelSettings`][pydantic_ai.models.anthropic.AnthropicModelSettings].

```python {title="anthropic_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

model = AnthropicModel('claude-3-7-sonnet-latest')
settings = AnthropicModelSettings(
    anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
)
agent = Agent(model, model_settings=settings)
...
```

## Groq

Groq supports different formats to receive thinking parts:

- `"raw"`: The thinking part is included in the text content with the `"<think>"` tag.
- `"hidden"`: The thinking part is not included in the text content.
- `"parsed"`: The thinking part has its own [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] object.

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

## Google

To enable thinking, use the `google_thinking_config` field in the
[`GoogleModelSettings`][pydantic_ai.models.google.GoogleModelSettings].

```python {title="google_thinking_part.py"}
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

model = GoogleModel('gemini-2.5-pro-preview-03-25')
settings = GoogleModelSettings(google_thinking_config={'include_thoughts': True})
agent = Agent(model, model_settings=settings)
...
```

## Mistral / Cohere

Neither Mistral nor Cohere generate thinking parts.
