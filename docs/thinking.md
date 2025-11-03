# Thinking

Thinking (or reasoning) is the process by which a model works through a problem step-by-step before
providing its final answer.

This capability is typically disabled by default and depends on the specific model being used.
See the sections below for how to enable thinking for each provider.

## OpenAI

When using the [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], text output inside `<think>` tags are converted to [`ThinkingPart`][pydantic_ai.messages.ThinkingPart] objects.
You can customize the tags using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field on the [model profile](models/openai.md#model-profile).

### OpenAI Responses

The [`OpenAIResponsesModel`][pydantic_ai.models.openai.OpenAIResponsesModel] can generate native thinking parts.
To enable this functionality, you need to set the
[`OpenAIResponsesModelSettings.openai_reasoning_effort`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_effort] and [`OpenAIResponsesModelSettings.openai_reasoning_summary`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_reasoning_summary] [model settings](agents.md#model-run-settings).

By default, the unique IDs of reasoning, text, and function call parts from the message history are sent to the model, which can result in errors like `"Item 'rs_123' of type 'reasoning' was provided without its required following item."`
if the message history you're sending does not match exactly what was received from the Responses API in a previous response, for example if you're using a [history processor](message-history.md#processing-message-history).
To disable this, you can disable the [`OpenAIResponsesModelSettings.openai_send_reasoning_ids`][pydantic_ai.models.openai.OpenAIResponsesModelSettings.openai_send_reasoning_ids] [model setting](agents.md#model-run-settings).

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

To enable thinking, use the [`AnthropicModelSettings.anthropic_thinking`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_thinking] [model setting](agents.md#model-run-settings).

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

To enable thinking, use the [`GoogleModelSettings.google_thinking_config`][pydantic_ai.models.google.GoogleModelSettings.google_thinking_config] [model setting](agents.md#model-run-settings).

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

To enable thinking, use the [`GroqModelSettings.groq_reasoning_format`][pydantic_ai.models.groq.GroqModelSettings.groq_reasoning_format] [model setting](agents.md#model-run-settings):

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

## Outlines

Some local models run through Outlines include in their text output a thinking part delimited by tags. In that case, it will be handled by Pydantic AI that will separate the thinking part from the final answer without the need to specifically enable it. The thinking tags used by default are `"<think>"` and `"</think>"`. If your model uses different tags, you can specify them in the [model profile](models/openai.md#model-profile) using the [`thinking_tags`][pydantic_ai.profiles.ModelProfile.thinking_tags] field.

Outlines currently does not support thinking along with structured output. If you provide an `output_type`, the model text output will not contain a thinking part with the associated tags, and you may experience degraded performance.
